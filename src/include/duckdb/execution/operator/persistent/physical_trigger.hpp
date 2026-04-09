//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/persistent/physical_trigger.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/trigger_executor.hpp"

namespace duckdb {
class TableCatalogEntry;

//! PhysicalTrigger fires triggers after a statement completes
class PhysicalTrigger : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::TRIGGER;

public:
	PhysicalTrigger(PhysicalPlan &physical_plan, TableCatalogEntry &table, vector<TriggerInfo> triggers,
	                bool needs_row_data, idx_t estimated_cardinality);

	//! The table the triggers belong to
	TableCatalogEntry &table;
	vector<TriggerInfo> triggers;
	//! True when at least one trigger references NEW - child INSERT emits actual rows instead of a count
	bool needs_row_data;

	// Sink interface - receives rows (needs_row_data=true) or a row count (needs_row_data=false)
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;
	bool IsSink() const override {
		return true;
	}

	// Source interface - re-emits the row count
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;
	bool IsSource() const override {
		return true;
	}
};

} // namespace duckdb
