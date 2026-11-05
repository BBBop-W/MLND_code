#pragma once

#include "mlnd/formulation.hpp"

namespace mlnd {
struct StabilizedMasterResult {
    // A validated candidate for subsequent full recourse evaluation only.
    // The objective is restored to the original master objective. No trial
    // objective bound or optimality claim is exported through this member.
    SolveOutcome candidate;
    Json metadata = Json::object();
    double seconds = 0;
};

// Copy all currently active master rows, add an original-cost level constraint,
// and minimize categorical head/capacity distance to an integer design center.
// Pair assignments and conic masters are deliberately unsupported. The caller
// keeps its unrestricted master as the sole source of global lower bounds.
StabilizedMasterResult select_stabilized_design(const Input&, const Formulation& master,
    int theta, const std::vector<double>& binary_center, double global_lb,
    double incumbent_ub, const SolverOptions&, double deadline);
} // namespace mlnd
