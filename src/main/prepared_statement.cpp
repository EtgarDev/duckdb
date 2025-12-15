#include "duckdb/main/prepared_statement.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/prepared_statement_data.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
static bool DuckDBDebugStepsEnabled_Prepared() {
    static int enabled = []() {
        const char *v = std::getenv("DUCKDB_DEBUG_STEPS");
        return v && v[0] != '\0' && v[0] != '0' ? 1 : 0;
    }();
    return enabled != 0;
}
}

namespace duckdb {

PreparedStatement::PreparedStatement(shared_ptr<ClientContext> context, shared_ptr<PreparedStatementData> data_p,
                                     string query, case_insensitive_map_t<idx_t> named_param_map_p)
    : context(std::move(context)), data(std::move(data_p)), query(std::move(query)), success(true),
      named_param_map(std::move(named_param_map_p)) {
	D_ASSERT(data || !success);
}

PreparedStatement::PreparedStatement(ErrorData error) : context(nullptr), success(false), error(std::move(error)) {
}

PreparedStatement::~PreparedStatement() {
}

const string &PreparedStatement::GetError() {
	D_ASSERT(HasError());
	return error.Message();
}

ErrorData &PreparedStatement::GetErrorObject() {
	return error;
}

bool PreparedStatement::HasError() const {
	return !success;
}

idx_t PreparedStatement::ColumnCount() {
	D_ASSERT(data);
	return data->types.size();
}

StatementType PreparedStatement::GetStatementType() {
	D_ASSERT(data);
	return data->statement_type;
}

StatementProperties PreparedStatement::GetStatementProperties() {
	D_ASSERT(data);
	return data->properties;
}

const vector<LogicalType> &PreparedStatement::GetTypes() {
	D_ASSERT(data);
	return data->types;
}

const vector<string> &PreparedStatement::GetNames() {
	D_ASSERT(data);
	return data->names;
}

case_insensitive_map_t<LogicalType> PreparedStatement::GetExpectedParameterTypes() const {
	D_ASSERT(data);
	case_insensitive_map_t<LogicalType> expected_types(data->value_map.size());
	for (auto &it : data->value_map) {
		auto &identifier = it.first;
		D_ASSERT(data->value_map.count(identifier));
		D_ASSERT(it.second);
		expected_types[identifier] = it.second->GetValue().type();
	}
	return expected_types;
}

unique_ptr<QueryResult> PreparedStatement::Execute(case_insensitive_map_t<BoundParameterData> &named_values,
                                                   bool allow_stream_result) {
    if (DuckDBDebugStepsEnabled_Prepared()) {
        std::fprintf(stderr,
                     "[DuckDBDebug] PreparedStatement::Execute(named_values) - enter (stream=%s, stmt_type=%s)\n",
                     allow_stream_result ? "true" : "false",
                     duckdb::StatementTypeToString(GetStatementType()).c_str());
        std::fflush(stderr);
    }
    auto pending = PendingQuery(named_values, allow_stream_result);
    if (pending->HasError()) {
        return make_uniq<MaterializedQueryResult>(pending->GetErrorObject());
    }
    auto res = pending->Execute();
    if (DuckDBDebugStepsEnabled_Prepared()) {
        std::fprintf(stderr, "[DuckDBDebug] PreparedStatement::Execute(named_values) - exit\n");
        std::fflush(stderr);
    }
    return res;
}

unique_ptr<QueryResult> PreparedStatement::Execute(vector<Value> &values, bool allow_stream_result) {
    if (DuckDBDebugStepsEnabled_Prepared()) {
        std::fprintf(stderr,
                     "[DuckDBDebug] PreparedStatement::Execute(values) - enter (n_values=%llu, stream=%s, stmt_type=%s)\n",
                     (unsigned long long)values.size(), allow_stream_result ? "true" : "false",
                     duckdb::StatementTypeToString(GetStatementType()).c_str());
        std::fflush(stderr);
    }
    auto pending = PendingQuery(values, allow_stream_result);
    if (pending->HasError()) {
        return make_uniq<MaterializedQueryResult>(pending->GetErrorObject());
    }
    auto res = pending->Execute();
    if (DuckDBDebugStepsEnabled_Prepared()) {
        std::fprintf(stderr, "[DuckDBDebug] PreparedStatement::Execute(values) - exit\n");
        std::fflush(stderr);
    }
    return res;
}

unique_ptr<PendingQueryResult> PreparedStatement::PendingQuery(vector<Value> &values, bool allow_stream_result) {
	case_insensitive_map_t<BoundParameterData> named_values;
	for (idx_t i = 0; i < values.size(); i++) {
		auto &val = values[i];
		named_values[std::to_string(i + 1)] = BoundParameterData(val);
	}
	return PendingQuery(named_values, allow_stream_result);
}

unique_ptr<PendingQueryResult> PreparedStatement::PendingQuery(case_insensitive_map_t<BoundParameterData> &named_values,
                                                               bool allow_stream_result) {
    if (!success) {
        auto exception = InvalidInputException("Attempting to execute an unsuccessfully prepared statement!");
        return make_uniq<PendingQueryResult>(ErrorData(exception));
    }
    if (DuckDBDebugStepsEnabled_Prepared()) {
        std::fprintf(stderr,
                     "[DuckDBDebug] PreparedStatement::PendingQuery(named_values) - enter (stream=%s)\n",
                     allow_stream_result ? "true" : "false");
        std::fflush(stderr);
    }
    PendingQueryParameters parameters;
    parameters.parameters = &named_values;

	try {
		VerifyParameters(named_values, named_param_map);
	} catch (const std::exception &ex) {
		return make_uniq<PendingQueryResult>(ErrorData(ex));
	}

    D_ASSERT(data);
    parameters.query_parameters.output_type =
        allow_stream_result && data->properties.output_type == QueryResultOutputType::ALLOW_STREAMING
            ? QueryResultOutputType::ALLOW_STREAMING
            : QueryResultOutputType::FORCE_MATERIALIZED;
    auto result = context->PendingQuery(query, data, parameters);
    if (DuckDBDebugStepsEnabled_Prepared()) {
        std::fprintf(stderr, "[DuckDBDebug] PreparedStatement::PendingQuery(named_values) - exit (pending ready)\n");
        std::fflush(stderr);
    }
    // The result should not contain any reference to the 'vector<Value> parameters.parameters'
    return result;
}

} // namespace duckdb
