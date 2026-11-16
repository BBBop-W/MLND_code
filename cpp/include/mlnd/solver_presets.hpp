#pragma once

#include "mlnd/linear_model.hpp"

namespace mlnd {
// Current fixed-task model with two solver methods.
// Configuration is selected before applying explicit runtime overrides.
struct SolverConfiguration {
    SolverOptions options;
    std::string method = "benders_cg";
    Json auto_disabled_accelerations = Json::array();
    std::size_t path_column_limit = 0;

    SolverConfiguration();
    void apply_method(const std::string& name);
    void adapt_instance(const Input& data);
    void set_acceleration(const std::string& name,bool enabled);
    void validate() const;
    Json metadata() const;
};
} // namespace mlnd
