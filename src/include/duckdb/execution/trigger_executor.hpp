//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/trigger_executor.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb/common/common.hpp"
#include "duckdb/common/enums/trigger_type.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/parser/query_node.hpp"

namespace duckdb {
class ClientContext;
class TableCatalogEntry;
class PhysicalPlan;

//! Context passed to __trigger_row_scan to make NEW/OLD column references resolvable.
//! current_row is null at bind time and set per-row at execution time.
struct TriggerRowBindingContext {
	DataChunk *current_row = nullptr;
	vector<string> column_names;
	vector<LogicalType> column_types;
};

struct TriggerInfo {
	unique_ptr<QueryNode> body;
	TriggerForEach for_each;
	bool uses_new_row = false;
};

struct PlannedRowTrigger {
	bool uses_new_row;
	unique_ptr<PhysicalPlan> physical_plan;
	unique_ptr<TriggerRowBindingContext> binding_ctx;
};

class TriggerExecutor {
public:
	// Each recursive trigger level pushes Parser, Planner, Optimizer, and PlanGenerator onto the call stack.
	// Keep depth small enough that stack overflow cannot occur before the guard fires.
	static constexpr idx_t MAX_TRIGGER_DEPTH = 8;

	//! Plan a trigger body once, setting up a TriggerRowBindingContext if uses_new_row is set.
	static PlannedRowTrigger PlanTriggerBody(ClientContext &context, const TriggerInfo &trigger,
	                                         const vector<string> &column_names,
	                                         const vector<LogicalType> &column_types);

	//! Execute a pre-planned trigger body. row_data is the current row chunk (used when uses_new_row is true).
	static void ExecutePlannedTriggerBody(ClientContext &context, PlannedRowTrigger &planned, DataChunk *row_data);

	//! Fire pre-collected triggers.
	//! new_rows_chunk: if non-null, contains the actual inserted rows (one chunk, N rows).
	//!                 Required when any trigger has uses_new_row = true.
	static void Fire(ClientContext &context, const vector<TriggerInfo> &triggers, idx_t row_count,
	                 const vector<string> &column_names, const vector<LogicalType> &column_types,
	                 DataChunk *new_rows_chunk = nullptr);
};
} // namespace duckdb
