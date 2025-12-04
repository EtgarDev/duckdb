#include "duckdb/common/type_visitor.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/common/string_util.hpp"

// Optional symbol provenance check (POSIX)
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <atomic>
#include <chrono>
#include <thread>
#include <cstring>

namespace duckdb {

ExecuteFunctionState::ExecuteFunctionState(const Expression &expr, ExpressionExecutorState &root)
    : ExpressionState(expr, root) {
	// Check if the expression is eligible for dictionary optimization
	if (!expr.IsConsistent() || expr.IsVolatile() || expr.CanThrow()) {
		return; // Needs to be consistent, non-volatile, and non-throwing
	}

	if (expr.return_type.InternalType() == PhysicalType::STRUCT) {
		return; // FIXME: get this working for STRUCT
	}

	// Set input_col_idx accordingly, marking the expression as eligible for dictionary optimization
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_FUNCTION: {
		auto &bound_function = expr.Cast<BoundFunctionExpression>();
		auto &children = bound_function.children;
		for (idx_t child_idx = 0; child_idx < children.size(); child_idx++) {
			auto &child = *children[child_idx];
			if (child.IsFoldable()) {
				continue; // Constant
			}
			if (input_col_idx.IsValid()) {
				input_col_idx.SetInvalid(); // Found more than 1 non-constant
				break;
			}
			if (child.return_type.InternalType() == PhysicalType::STRUCT) {
				break; // FIXME
			}
			input_col_idx = child_idx;
		}
		break;
	}
	default:
		break;
	}
}

ExecuteFunctionState::~ExecuteFunctionState() {
}

bool ExecuteFunctionState::TryExecuteDictionaryExpression(const BoundFunctionExpression &expr, DataChunk &args,
                                                          ExpressionState &state, Vector &result) {
	static constexpr idx_t MAX_DICTIONARY_SIZE_THRESHOLD = 20000;
	static constexpr double CHUNK_FILL_RATIO_THRESHOLD = 0.5;

	if (!input_col_idx.IsValid()) {
		return false; // This expression is not eligible for dictionary optimization
	}

	// Figure out if we can do the optimization
	const auto &unary_input = args.data[input_col_idx.GetIndex()];
	if (unary_input.GetVectorType() != VectorType::DICTIONARY_VECTOR) {
		return false; // Not a dictionary
	}

	const auto input_dictionary_size_opt = DictionaryVector::DictionarySize(unary_input);
	const auto &input_dictionary_id = DictionaryVector::DictionaryId(unary_input);
	if (!input_dictionary_size_opt.IsValid() || input_dictionary_id.empty()) {
		return false; // Not a dictionary that comes from storage
	}

	const auto input_dictionary_size = input_dictionary_size_opt.GetIndex();
	if (input_dictionary_size >= MAX_DICTIONARY_SIZE_THRESHOLD) {
		return false; // Dictionary is too large, bail
	}

	if (!output_dictionary || current_input_dictionary_id != input_dictionary_id) {
		// We haven't seen this dictionary before
		const auto chunk_fill_ratio = static_cast<double>(args.size()) / STANDARD_VECTOR_SIZE;
		if (input_dictionary_size > STANDARD_VECTOR_SIZE && chunk_fill_ratio <= CHUNK_FILL_RATIO_THRESHOLD) {
			// If the dictionary size is <= STANDARD_VECTOR_SIZE, we always do the optimization
			// If it's greater, we only do the optimization if the chunk is more than 50% full
			// This protects the optimization against selective filters
			return false;
		}

		// We can do dictionary optimization! Re-initialize
		output_dictionary = DictionaryVector::CreateReusableDictionary(result.GetType(), input_dictionary_size);
		current_input_dictionary_id = input_dictionary_id;

		// Set up the input chunk
		DataChunk input_chunk;
		input_chunk.InitializeEmpty(args.GetTypes());
		for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
			if (col_idx != input_col_idx.GetIndex()) {
				input_chunk.data[col_idx].Reference(args.data[col_idx]);
			}
		}

		// Loop over the dictionary, executing at most STANDARD_VECTOR_SIZE at a time
		for (idx_t offset = 0; offset < input_dictionary_size; offset += STANDARD_VECTOR_SIZE) {
			const auto count = MinValue<idx_t>(input_dictionary_size - offset, STANDARD_VECTOR_SIZE);

			// Offset the input dictionary
			Vector offset_input(DictionaryVector::Child(unary_input), offset, offset + count);
			input_chunk.data[input_col_idx.GetIndex()].Reference(offset_input);
			input_chunk.SetCardinality(count);

			// Execute, storing the result in an intermediate vector, and copying it to the output dictionary
			Vector output_intermediate(result.GetType());
			expr.function.GetFunctionCallback()(input_chunk, state, output_intermediate);
			VectorOperations::Copy(output_intermediate, output_dictionary->data, count, 0, offset);
		}
	}

	// Result references the dictionary
	result.Dictionary(output_dictionary, DictionaryVector::SelVector(unary_input));

	return true;
}

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundFunctionExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExecuteFunctionState>(expr, root);
	for (auto &child : expr.children) {
		result->AddChild(*child);
	}

	result->Finalize();
	if (expr.function.HasInitStateCallback()) {
		result->local_state = expr.function.GetInitStateCallback()(*result, expr, expr.bind_info.get());
	}
	return std::move(result);
}

static void VerifyNullHandling(const BoundFunctionExpression &expr, DataChunk &args, Vector &result) {
#ifdef DEBUG
	if (args.data.empty() || expr.function.GetNullHandling() != FunctionNullHandling::DEFAULT_NULL_HANDLING) {
		return;
	}

	// Combine all the argument validity masks into a flat validity mask
	idx_t count = args.size();
	ValidityMask combined_mask(count);
	for (auto &arg : args.data) {
		UnifiedVectorFormat arg_data;
		arg.ToUnifiedFormat(count, arg_data);

		for (idx_t i = 0; i < count; i++) {
			auto idx = arg_data.sel->get_index(i);
			if (!arg_data.validity.RowIsValid(idx)) {
				combined_mask.SetInvalid(i);
			}
		}
	}

	// Default is that if any of the arguments are NULL, the result is also NULL
	UnifiedVectorFormat result_data;
	result.ToUnifiedFormat(count, result_data);
	for (idx_t i = 0; i < count; i++) {
		if (!combined_mask.RowIsValid(i)) {
			auto idx = result_data.sel->get_index(i);
			D_ASSERT(!result_data.validity.RowIsValid(idx));
		}
	}
#endif
}

void ExpressionExecutor::Execute(const BoundFunctionExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	state->intermediate_chunk.Reset();
	auto &arguments = state->intermediate_chunk;
	if (!state->types.empty()) {
		for (idx_t i = 0; i < expr.children.size(); i++) {
			D_ASSERT(state->types[i] == expr.children[i]->return_type);
			Execute(*expr.children[i], state->child_states[i].get(), sel, count, arguments.data[i]);
#ifdef DEBUG
			if (expr.children[i]->return_type.id() == LogicalTypeId::VARCHAR) {
				arguments.data[i].UTFVerify(count);
			}
#endif
		}
	}
	arguments.SetCardinality(count);
	arguments.Verify();

	D_ASSERT(expr.function.HasFunctionCallback());
	auto &execute_function_state = state->Cast<ExecuteFunctionState>();
	if (!execute_function_state.TryExecuteDictionaryExpression(expr, arguments, *state, result)) {
		// Optional sandboxing of UDF execution: enabled via environment variable DUCKDB_UDF_SANDBOX=1
		// IMPORTANT: Only apply to external (non-built-in) scalar functions. Built-ins are never sandboxed.
		// Strategy (POSIX): use dladdr to compare the shared object of the callback against a known DuckDB symbol.
		bool sandbox_should_probe = false;
		if (HasContext()) {
			const char *env = std::getenv("DUCKDB_UDF_SANDBOX");
			const bool sandbox_env = env && (StringUtil::CIEquals(env, "1") || StringUtil::CIEquals(env, "true") ||
			                                 StringUtil::CIEquals(env, "on"));
			if (sandbox_env) {
#if defined(__unix__) || defined(__APPLE__)
				// Identify external functions by shared object provenance
				using FnPtr = void (*)(DataChunk &, ExpressionState &, Vector &);
				scalar_function_t cb = expr.function.GetFunctionCallback();
				// Attempt to extract a raw function pointer from std::function
				FnPtr const *fptr = cb.target<FnPtr>();
				if (fptr && *fptr) {
					Dl_info info_target {};
					Dl_info info_duck {};
					(void)dladdr(reinterpret_cast<void *>(*fptr), &info_target);
					(void)dladdr(reinterpret_cast<void *>(&ScalarFunction::NopFunction), &info_duck);
					// If we can resolve both, and the shared objects differ, treat as external → sandbox
					if (info_target.dli_fname && info_duck.dli_fname &&
					    strcmp(info_target.dli_fname, info_duck.dli_fname) != 0) {
						sandbox_should_probe = true;
					}
					// If they are the same SO, it's built-in → do not sandbox
					// If we cannot classify (dladdr failure), we default to not sandboxing to avoid penalizing
					// built-ins.
				}
#else
				(void)sandbox_env; // unused on non-POSIX
#endif
			}
		}

		if (sandbox_should_probe) {
			// Thread-based probe: run the UDF once in a separate thread to allow the main thread
			// to observe interrupts. Note: threads cannot be forcibly terminated portably.
			// On interrupt, we join the worker to avoid use-after-free, then throw InterruptException.
			std::atomic<bool> worker_done {false};
			std::exception_ptr worker_exception = nullptr;

			// Launch worker thread: execute the real function into `result` exactly once
			std::thread worker([&] {
				try {
					// Execute the UDF once, writing directly into the actual result vector
					expr.function.GetFunctionCallback()(arguments, *state, result);
				} catch (...) {
					worker_exception = std::current_exception();
				}
				worker_done.store(true, std::memory_order_release);
			});

			// Poll loop: wait for completion or interrupt
			while (!worker_done.load(std::memory_order_acquire)) {
				if (HasContext() && GetContext().IsInterrupted()) {
					// Cooperative cancellation only: wait for the worker to finish to keep memory safe
					worker.join();
					if (worker_exception) {
						std::rethrow_exception(worker_exception);
					}
					throw InterruptException();
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}

			// Ensure thread is finished
			worker.join();

			if (worker_exception) {
				std::rethrow_exception(worker_exception);
			}
		} else {
			// Sandbox disabled or not applicable on this platform
			expr.function.GetFunctionCallback()(arguments, *state, result);
		}
	}

	VerifyNullHandling(expr, arguments, result);
	D_ASSERT(result.GetType() == expr.return_type);
}

} // namespace duckdb
