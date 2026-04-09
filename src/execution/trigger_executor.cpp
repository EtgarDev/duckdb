#include "duckdb/execution/trigger_executor.hpp"

#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/query_node/insert_query_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/query_node/update_query_node.hpp"
#include "duckdb/parser/query_node/delete_query_node.hpp"
#include "duckdb/parser/statement/insert_statement.hpp"
#include "duckdb/parser/statement/update_statement.hpp"
#include "duckdb/parser/statement/delete_statement.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/planner.hpp"

namespace duckdb {

constexpr idx_t TriggerExecutor::MAX_TRIGGER_DEPTH;

struct TriggerDepthGuard {
	explicit TriggerDepthGuard(idx_t &depth) : depth(depth) {
		depth++;
	}
	~TriggerDepthGuard() {
		depth--;
	}
	idx_t &depth;
};

// Inject a "FROM __trigger_row_scan(ptr) AS <alias>" clause into a SelectNode that has no real FROM clause.
// This makes NEW/OLD column references resolvable when the trigger body uses them without an explicit FROM.
static void InjectRowScanIntoSelect(SelectNode &sel, TriggerRowBindingContext *ctx, const string &alias) {
	D_ASSERT(!sel.from_table || sel.from_table->type == TableReferenceType::EMPTY_FROM);
	vector<unique_ptr<ParsedExpression>> args;
	args.push_back(make_uniq<ConstantExpression>(Value::POINTER(CastPointerToValue(ctx))));
	auto func_expr = make_uniq<FunctionExpression>("__trigger_row_scan", std::move(args));
	auto func_ref = make_uniq<TableFunctionRef>();
	func_ref->function = std::move(func_expr);
	func_ref->alias = alias;
	sel.from_table = std::move(func_ref);
}

// Planning requires a SQLStatement,
// but the trigger body is stored as a QueryNode (since SQLStatement is not serializable)
static unique_ptr<SQLStatement> WrapQueryNode(unique_ptr<QueryNode> body) {
	switch (body->type) {
	case QueryNodeType::INSERT_QUERY_NODE: {
		auto stmt = make_uniq<InsertStatement>();
		stmt->node = unique_ptr_cast<QueryNode, InsertQueryNode>(std::move(body));
		return stmt;
	}
	case QueryNodeType::UPDATE_QUERY_NODE: {
		auto stmt = make_uniq<UpdateStatement>();
		stmt->node = unique_ptr_cast<QueryNode, UpdateQueryNode>(std::move(body));
		return stmt;
	}
	case QueryNodeType::DELETE_QUERY_NODE: {
		auto stmt = make_uniq<DeleteStatement>();
		stmt->node = unique_ptr_cast<QueryNode, DeleteQueryNode>(std::move(body));
		return stmt;
	}
	default:
		throw InternalException("Unexpected trigger body query node type");
	}
}

PlannedRowTrigger TriggerExecutor::PlanTriggerBody(ClientContext &context, const TriggerInfo &trigger,
                                                   const vector<string> &column_names,
                                                   const vector<LogicalType> &column_types) {
	PlannedRowTrigger planned;
	planned.uses_new_row = trigger.uses_new_row;

	if (trigger.uses_new_row) {
		planned.binding_ctx = make_uniq<TriggerRowBindingContext>();
		planned.binding_ctx->column_names = column_names;
		planned.binding_ctx->column_types = column_types;
	}

	// Copy the body and inject the NEW row scan before wrapping into a SQLStatement.
	// Injection happens at the QueryNode level so we can work directly with the SelectNode
	// instead of navigating back through the SQLStatement after wrapping.
	auto body_copy = trigger.body->Copy();

	if (trigger.uses_new_row) {
		switch (body_copy->type) {
		case QueryNodeType::INSERT_QUERY_NODE: {
			auto &ins = body_copy->Cast<InsertQueryNode>();
			D_ASSERT(ins.select_statement && ins.select_statement->node &&
			         ins.select_statement->node->type == QueryNodeType::SELECT_NODE);
			auto &sel = ins.select_statement->node->Cast<SelectNode>();
			InjectRowScanIntoSelect(sel, planned.binding_ctx.get(), "new");
			break;
		}
		case QueryNodeType::UPDATE_QUERY_NODE: {
			auto &upd = body_copy->Cast<UpdateQueryNode>();
			D_ASSERT(!upd.from_table);
			vector<unique_ptr<ParsedExpression>> args;
			args.push_back(
			    make_uniq<ConstantExpression>(Value::POINTER(CastPointerToValue(planned.binding_ctx.get()))));
			auto func_ref = make_uniq<TableFunctionRef>();
			func_ref->function = make_uniq<FunctionExpression>("__trigger_row_scan", std::move(args));
			func_ref->alias = "new";
			upd.from_table = std::move(func_ref);
			break;
		}
		case QueryNodeType::DELETE_QUERY_NODE:
			throw NotImplementedException("NEW references in DELETE trigger bodies are not yet supported");
		default:
			throw InternalException("Unexpected trigger body query node type");
		}
	}

	auto stmt = WrapQueryNode(std::move(body_copy));

	Planner planner(context);
	planner.CreatePlan(std::move(stmt));
	if (!planner.plan) {
		planned.physical_plan = nullptr;
		return planned;
	}

	Optimizer optimizer(*planner.binder, context);
	auto logical_plan = optimizer.Optimize(std::move(planner.plan));

	PhysicalPlanGenerator physical_generator(context);
	planned.physical_plan = physical_generator.Plan(std::move(logical_plan));

	return planned;
}

void TriggerExecutor::ExecutePlannedTriggerBody(ClientContext &context, PlannedRowTrigger &planned,
                                                DataChunk *row_data) {
	if (!planned.physical_plan) {
		return;
	}

	// Point the binding context at the current row before running the body
	if (planned.uses_new_row && planned.binding_ctx) {
		planned.binding_ctx->current_row = row_data;
	}

	auto trigger_exec = make_uniq<Executor>(context);
	trigger_exec->Initialize(planned.physical_plan->Root());

	while (!trigger_exec->ExecutionIsFinished()) {
		trigger_exec->WorkOnTasks();
		if (trigger_exec->HasError()) {
			trigger_exec->ThrowException();
		}
	}
	// CancelTasks() spins until all worker threads release their tasks,
	// preventing a race with ~Executor() which asserts executor_tasks == 0.
	trigger_exec->CancelTasks();
}

void TriggerExecutor::Fire(ClientContext &context, const vector<TriggerInfo> &triggers, idx_t row_count,
                           const vector<string> &column_names, const vector<LogicalType> &column_types,
                           DataChunk *new_rows_chunk) {
	if (triggers.empty()) {
		return;
	}
	if (context.trigger_depth >= MAX_TRIGGER_DEPTH) {
		throw InvalidInputException("Trigger recursion depth limit (%llu) exceeded.", MAX_TRIGGER_DEPTH);
	}
	TriggerDepthGuard depth_guard(context.trigger_depth);

	for (auto &trigger_info : triggers) {
		if (trigger_info.for_each == TriggerForEach::ROW && row_count == 0) {
			continue;
		}

		auto planned = PlanTriggerBody(context, trigger_info, column_names, column_types);

		if (trigger_info.for_each == TriggerForEach::ROW) {
			if (trigger_info.uses_new_row && new_rows_chunk) {
				for (idx_t row_idx = 0; row_idx < new_rows_chunk->size(); row_idx++) {
					SelectionVector sel(1);
					sel.set_index(0, row_idx);
					DataChunk single_row;
					single_row.Initialize(Allocator::DefaultAllocator(), new_rows_chunk->GetTypes());
					single_row.Slice(*new_rows_chunk, sel, 1);
					single_row.Flatten();
					ExecutePlannedTriggerBody(context, planned, &single_row);
				}
			} else {
				for (idx_t i = 0; i < row_count; i++) {
					ExecutePlannedTriggerBody(context, planned, nullptr);
				}
			}
		} else {
			ExecutePlannedTriggerBody(context, planned, nullptr);
		}
	}
}

} // namespace duckdb
