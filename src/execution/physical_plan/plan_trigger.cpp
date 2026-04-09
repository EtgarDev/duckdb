#include "duckdb/execution/operator/persistent/physical_trigger.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_trigger.hpp"

namespace duckdb {

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalTrigger &op) {
	D_ASSERT(op.children.size() == 1);
	D_ASSERT(!op.trigger_bodies.empty());
	D_ASSERT(op.trigger_bodies.size() == op.trigger_for_each.size());
	D_ASSERT(op.trigger_bodies.size() == op.trigger_uses_new_row.size());

	auto &child = CreatePlan(*op.children[0]);

	vector<TriggerInfo> triggers;
	triggers.reserve(op.trigger_bodies.size());
	for (idx_t i = 0; i < op.trigger_bodies.size(); i++) {
		TriggerInfo info;
		info.body = std::move(op.trigger_bodies[i]);
		info.for_each = op.trigger_for_each[i];
		info.uses_new_row = op.trigger_uses_new_row[i];
		triggers.push_back(std::move(info));
	}

	auto &trigger = Make<PhysicalTrigger>(op.table, std::move(triggers), op.needs_row_data, 1);
	trigger.children.push_back(child);
	return trigger;
}

} // namespace duckdb
