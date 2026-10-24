#pragma once

#include "mlnd/bdd.hpp"
#include <functional>

namespace mlnd {
struct BddSearchOptions {
    std::string method = "search"; // fixed ordinary multipliers, or boxed Kelley search
    double started = 0, deadline = 0;
    double inner_seconds = 30, master_seconds = 3, box_radius = 50;
    int multipliers_per_root = 5, max_evaluations = 100000;
    std::function<void(const Json&)> checkpoint;
};

// Independent full-domain integer-aware root diagnostic. The boxed Kelley
// method is a simplified stabilization, not the published quadratic-proximal
// BDD algorithm and not a claim of unrestricted dual convergence.
Json search_bdd_root_bounds(const Input&, const PathCollection&, BddFullDomain&,
    const std::vector<BddAnchor>&, double certified_total_cost_floor,
    const SolverOptions&, const BddSearchOptions&);
} // namespace mlnd
