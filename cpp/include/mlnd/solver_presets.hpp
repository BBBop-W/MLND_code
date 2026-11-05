#pragma once

#include "mlnd/linear_model.hpp"

namespace mlnd {
// The unselected configuration preserves the historical CLI. Production
// commands name a versioned preset, then apply their explicit overrides.
struct SolverConfiguration {
    SolverOptions options;
    std::string preset = "legacy", method = "full_path";
    std::size_t path_column_limit = 500000;

    void apply_preset(const std::string& name);
    bool acceleration_flag(const std::string& command);
    void set_acceleration(const std::string& name,bool enabled);
    void plain();
    void validate() const;
    Json metadata() const;
};
} // namespace mlnd
