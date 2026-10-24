#pragma once

#include "mlnd/formulation.hpp"

namespace mlnd {
// Native master / conic recourse / explicit-dual decomposition. Every issued
// cut requires an optimal RMP, tight conic dual and complete exact label pricing.
Json solve_benders(const Input&, PathCollection&, const SolverOptions&,
                   const Json* start_solution = nullptr);
// Same-design validation entry point: no outer master or incumbent cleanup.
Json solve_fixed_design_cg(const Input&, PathCollection&, const SolverOptions&,
                          const std::vector<double>& design);
// Full continuous relaxation with free strategic variables. A retained
// certified_lower_bound is global for this model/path domain, even when CG
// has not converged. continuous_objective is never an integer incumbent.
Json solve_continuous_cg(const Input&, PathCollection&, const SolverOptions&);
// Fixed fractional strategic design, with the same complete-recourse CG and
// ordinary globally valid Benders cut. No integer solution is packaged.
Json evaluate_fractional_recourse_cg(const Input&, PathCollection&, const SolverOptions&,
                                    const std::vector<double>& design);
} // namespace mlnd
