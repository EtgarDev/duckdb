//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/schema/physical_create_trigger.cpp
//
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/operator/schema/physical_create_trigger.hpp"
#include "duckdb/catalog/catalog.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalCreateTrigger::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                        OperatorSourceInput &input) const {
	auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
	catalog.CreateTrigger(context.client, *info);

	return SourceResultType::FINISHED;
}

} // namespace duckdb
