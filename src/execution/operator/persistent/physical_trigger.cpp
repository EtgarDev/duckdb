#include "duckdb/execution/operator/persistent/physical_trigger.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

namespace duckdb {

struct TriggerGlobalSinkState : public GlobalSinkState {
	atomic<idx_t> row_count {0};
	//! Collected rows - used in row-data mode (needs_row_data=true)
	unique_ptr<ColumnDataCollection> collected_rows;
};

PhysicalTrigger::PhysicalTrigger(PhysicalPlan &physical_plan, TableCatalogEntry &table, vector<TriggerInfo> triggers,
                                 bool needs_row_data, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::TRIGGER, {LogicalType::BIGINT}, estimated_cardinality),
      table(table), triggers(std::move(triggers)), needs_row_data(needs_row_data) {
}

unique_ptr<GlobalSinkState> PhysicalTrigger::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<TriggerGlobalSinkState>();
	if (needs_row_data) {
		vector<LogicalType> types;
		for (auto &col : table.GetColumns().Physical()) {
			types.push_back(col.Type());
		}
		state->collected_rows = make_uniq<ColumnDataCollection>(context, types);
	}
	return std::move(state);
}

SinkResultType PhysicalTrigger::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<TriggerGlobalSinkState>();
	if (needs_row_data) {
		D_ASSERT(gstate.collected_rows);
		gstate.collected_rows->Append(chunk);
		gstate.row_count += chunk.size();
	} else {
		D_ASSERT(chunk.ColumnCount() == 1);
		D_ASSERT(chunk.size() == 1);
		gstate.row_count += NumericCast<idx_t>(chunk.GetValue(0, 0).GetValue<int64_t>());
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType PhysicalTrigger::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                           OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<TriggerGlobalSinkState>();

	vector<string> column_names;
	vector<LogicalType> column_types;
	for (auto &col : table.GetColumns().Physical()) {
		column_names.push_back(col.Name());
		column_types.push_back(col.Type());
	}

	if (needs_row_data && gstate.collected_rows && gstate.collected_rows->Count() > 0) {
		DataChunk scan_chunk;
		scan_chunk.Initialize(Allocator::Get(context), column_types);
		ColumnDataScanState scan_state;
		gstate.collected_rows->InitializeScan(scan_state);
		while (gstate.collected_rows->Scan(scan_state, scan_chunk)) {
			TriggerExecutor::Fire(context, triggers, scan_chunk.size(), column_names, column_types, &scan_chunk);
		}
	} else {
		TriggerExecutor::Fire(context, triggers, gstate.row_count.load(), column_names, column_types, nullptr);
	}

	return SinkFinalizeType::READY;
}

SourceResultType PhysicalTrigger::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                  OperatorSourceInput &input) const {
	auto &gstate = sink_state->Cast<TriggerGlobalSinkState>();
	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::BIGINT(NumericCast<int64_t>(gstate.row_count.load())));
	return SourceResultType::FINISHED;
}

} // namespace duckdb
