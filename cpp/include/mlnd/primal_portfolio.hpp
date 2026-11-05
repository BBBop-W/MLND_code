#pragma once

#include "mlnd/heuristics.hpp"

namespace mlnd {
struct PrimalPortfolioOptions {
    std::size_t max_heads = 4, max_lockers = 6, max_candidates = 3;
    // Shortlist breadth before choosing one physical path per positive branch.
    std::size_t paths_per_branch = 2;
    // Seed-portfolio per-domain cap. Coupled search uses its caller deadline.
    double candidate_seconds = 10;
};

// Primal-only searches in the original K2/preprocessed/origin two-base model.
// deadline is absolute and includes selection, construction and validation.
// Results use the restricted-heuristic wrapper; no bound is returned.
Json solve_primal_seed_portfolio(const Input&, const PathCollection&,
    const SolverOptions&, double deadline, const Json* incumbent = nullptr,
    const PrimalPortfolioOptions& = {});

// Release independent task-assignment groups with their resource decisions.
// guide is optional and uses the DesignLayout prefix of a master/LP point.
// rotation changes the chosen local block without changing model semantics.
Json solve_primal_coupled_neighborhood(const Input&, const PathCollection&,
    const SolverOptions&, double deadline, const Json& incumbent,
    const std::vector<double>* guide = nullptr, std::size_t rotation = 0,
    const PrimalPortfolioOptions& = {});
} // namespace mlnd
