#include "duckdb/execution/operator/helper/physical_result_collector.hpp"

#include "duckdb/execution/operator/helper/physical_batch_collector.hpp"
#include "duckdb/execution/operator/helper/physical_buffered_batch_collector.hpp"
#include "duckdb/execution/operator/helper/physical_materialized_collector.hpp"
#include "duckdb/execution/operator/helper/physical_buffered_collector.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/main/client_context.hpp"
#include <cstdio>
#include <cstdlib>

namespace {
static bool DuckDBDebugStepsEnabled_PhysCollector() {
    static int enabled = []() {
        const char *v = std::getenv("DUCKDB_DEBUG_STEPS");
        return v && v[0] != '\0' && v[0] != '0' ? 1 : 0;
    }();
    return enabled != 0;
}
}

namespace duckdb {

PhysicalResultCollector::PhysicalResultCollector(PhysicalPlan &physical_plan, PreparedStatementData &data)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::RESULT_COLLECTOR, {LogicalType::BOOLEAN}, 0),
      statement_type(data.statement_type), properties(data.properties), memory_type(data.memory_type),
      plan(data.physical_plan->Root()), names(data.names) {
	types = data.types;
}

PhysicalOperator &PhysicalResultCollector::GetResultCollector(ClientContext &context, PreparedStatementData &data) {
    auto &physical_plan = *data.physical_plan;
    auto &root = physical_plan.Root();

    auto preserve_order = PhysicalPlanGenerator::PreserveInsertionOrder(context, root);
    auto use_batch_index = PhysicalPlanGenerator::UseBatchIndex(context, root);
    if (DuckDBDebugStepsEnabled_PhysCollector()) {
        std::fprintf(stderr,
                     "[DuckDBDebug] PhysicalResultCollector::GetResultCollector - preserve_order=%s, use_batch_index=%s, streaming=%s\n",
                     preserve_order ? "true" : "false", use_batch_index ? "true" : "false",
                     data.output_type == QueryResultOutputType::ALLOW_STREAMING ? "true" : "false");
        std::fflush(stderr);
    }

    if (!preserve_order) {
        // Not an order-preserving plan: use the parallel materialized collector.
        if (data.output_type == QueryResultOutputType::ALLOW_STREAMING) {
            return physical_plan.Make<PhysicalBufferedCollector>(data, true);
        }
        return physical_plan.Make<PhysicalMaterializedCollector>(data, true);
    }

    if (!use_batch_index) {
        // Order-preserving plan, and we cannot use the batch index: use single-threaded result collector.
        if (data.output_type == QueryResultOutputType::ALLOW_STREAMING) {
            return physical_plan.Make<PhysicalBufferedCollector>(data, false);
        }
        return physical_plan.Make<PhysicalMaterializedCollector>(data, false);
    }

	// Order-preserving plan, and we can use the batch index: use a batch collector.
	if (data.output_type == QueryResultOutputType::ALLOW_STREAMING) {
		return physical_plan.Make<PhysicalBufferedBatchCollector>(data);
	}
	return physical_plan.Make<PhysicalBatchCollector>(data);
}

vector<const_reference<PhysicalOperator>> PhysicalResultCollector::GetChildren() const {
	return {plan};
}

void PhysicalResultCollector::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
    // operator is a sink, build a pipeline
    sink_state.reset();

    D_ASSERT(children.empty());

    if (DuckDBDebugStepsEnabled_PhysCollector()) {
        std::fprintf(stderr, "[DuckDBDebug] PhysicalResultCollector::BuildPipelines - enter\n");
        std::fflush(stderr);
    }

    // single operator: the operator becomes the data source of the current pipeline
    auto &state = meta_pipeline.GetState();
    state.SetPipelineSource(current, *this);

    // we create a new pipeline starting from the child
    auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
    child_meta_pipeline.Build(plan);

    if (DuckDBDebugStepsEnabled_PhysCollector()) {
        std::fprintf(stderr, "[DuckDBDebug] PhysicalResultCollector::BuildPipelines - child built\n");
        std::fflush(stderr);
    }
}

unique_ptr<ColumnDataCollection> PhysicalResultCollector::CreateCollection(ClientContext &context) const {
	switch (memory_type) {
	case QueryResultMemoryType::IN_MEMORY:
		return make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	case QueryResultMemoryType::BUFFER_MANAGED:
		// Use the DatabaseInstance BufferManager because the query result can outlive the ClientContext
		return make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(*context.db), types,
		                                       ColumnDataCollectionLifetime::THROW_ERROR_AFTER_DATABASE_CLOSES);
	default:
		throw NotImplementedException("PhysicalResultCollector::CreateCollection for %s",
		                              EnumUtil::ToString(memory_type));
	}
}

} // namespace duckdb
