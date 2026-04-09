//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_trigger.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/common/enums/trigger_type.hpp"
#include "duckdb/parser/query_node.hpp"

namespace duckdb {
class ClientContext;
class TableCatalogEntry;

void CollectTriggers(ClientContext &context, TableCatalogEntry &table, TriggerTiming timing,
                     TriggerEventType event_type, vector<unique_ptr<QueryNode>> &trigger_bodies,
                     vector<TriggerForEach> &trigger_for_each, vector<bool> &trigger_uses_new_row);

//! LogicalTrigger represents trigger firing for a statement
class LogicalTrigger : public LogicalOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_TRIGGER;

public:
	LogicalTrigger(TableCatalogEntry &table, TriggerTiming timing, TriggerEventType event_type,
	               vector<unique_ptr<QueryNode>> trigger_bodies, vector<TriggerForEach> trigger_for_each,
	               vector<bool> trigger_uses_new_row, bool needs_row_data);

	TableCatalogEntry &table;
	TriggerTiming timing;
	TriggerEventType event_type;
	//! Trigger bodies - parallel to trigger_for_each and trigger_uses_new_row
	vector<unique_ptr<QueryNode>> trigger_bodies;
	vector<TriggerForEach> trigger_for_each;
	vector<bool> trigger_uses_new_row;
	//! True when at least one trigger references NEW - child INSERT emits actual rows instead of a count
	bool needs_row_data;

public:
	void Serialize(Serializer &serializer) const override;
	static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);

	idx_t EstimateCardinality(ClientContext &context) override;
	string GetName() const override;

protected:
	vector<ColumnBinding> GetColumnBindings() override;
	void ResolveTypes() override;

private:
	LogicalTrigger(ClientContext &context, const unique_ptr<CreateInfo> &table_info, TriggerTiming timing,
	               TriggerEventType event_type);
};

} // namespace duckdb
