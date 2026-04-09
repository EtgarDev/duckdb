#include "duckdb/function/table/system_functions.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/built_in_functions.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/trigger_executor.hpp"

namespace duckdb {

struct TriggerRowScanBindData : public FunctionData {
	explicit TriggerRowScanBindData(TriggerRowBindingContext *ctx_p) : ctx(ctx_p) {
	}

	// Non-owning pointer - lifetime managed by TriggerExecutor
	TriggerRowBindingContext *ctx;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<TriggerRowScanBindData>(ctx);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<TriggerRowScanBindData>();
		return ctx == other.ctx;
	}
};

struct TriggerRowScanGlobalState : public GlobalTableFunctionState {
	bool done = false;
};

static unique_ptr<FunctionData> TriggerRowScanBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	auto *ctx = reinterpret_cast<TriggerRowBindingContext *>(input.inputs[0].GetPointer());
	D_ASSERT(ctx);

	return_types = ctx->column_types;
	names = ctx->column_names;

	return make_uniq<TriggerRowScanBindData>(ctx);
}

static unique_ptr<GlobalTableFunctionState> TriggerRowScanInit(ClientContext &context,
                                                               TableFunctionInitInput &input) {
	return make_uniq<TriggerRowScanGlobalState>();
}

static void TriggerRowScanExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<TriggerRowScanBindData>();
	auto &state = data_p.global_state->Cast<TriggerRowScanGlobalState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}

	D_ASSERT(bind_data.ctx->current_row);
	output.Reference(*bind_data.ctx->current_row);
	state.done = true;
}

void TriggerRowScanFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunction func("__trigger_row_scan", {LogicalType::POINTER}, TriggerRowScanExecute, TriggerRowScanBind,
	                   TriggerRowScanInit);
	set.AddFunction(func);
}

} // namespace duckdb
