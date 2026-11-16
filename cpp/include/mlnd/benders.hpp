#pragma once

#include "mlnd/formulation.hpp"
#include <limits>

namespace mlnd {
// Physical columns only: no design fixings, objective values or dual state are
// transferred. The input and catalogue must outlive this cache and registered
// paths remain immutable; append-only catalogue growth is allowed.
class CgColumnSeeds {
public:
    void include(const Input&, const PathCollection&, const std::vector<int>& service,
                 const std::vector<int>& empty);
    void validate(const Input&, const PathCollection&) const;
    const std::vector<int>& service() const { return service_; }
    const std::vector<int>& empty() const { return empty_; }
private:
    const Input* input_ = nullptr;
    const PathCollection* paths_ = nullptr;
    std::string input_key_, scope_;
    std::size_t service_count_ = 0, empty_count_ = 0;
    std::vector<int> service_, empty_;
    std::vector<std::string> service_keys_, empty_keys_;
};
// Native master / conic recourse / CPLEX dual extraction. Every recourse cut
// requires a valid full-domain dual bound after complete exact label pricing
// and conservative correction. Optional weak cuts need not be tight;
// recourse optimality is certified separately under the original tolerances.
Json solve_benders(const Input&, PathCollection&, const SolverOptions&,
                   const Json* start_solution = nullptr);
// Same-design validation entry point: no outer master or incumbent cleanup.
Json solve_fixed_design_cg(const Input&, PathCollection&, const SolverOptions&,
                          const std::vector<double>& design, CgColumnSeeds* seeds = nullptr);
// Full continuous relaxation with free strategic variables. A retained
// certified_lower_bound is global for this model/path domain, even when CG
// has not converged. continuous_objective is never an integer incumbent.
Json solve_continuous_cg(const Input&, PathCollection&, const SolverOptions&,
                         CgColumnSeeds* seeds = nullptr);
// Conditional total-cost bound with one head in state 0/unassigned, 1/base 0,
// or 2/base 1. Every other strategic coordinate stays continuous and free.
// Native dual + complete pricing only; this is NOT a global bound by itself.
// With cg_seed_reuse enabled, a finite no_gain_upper must be the caller's
// certified global integer lower bound. A validated face feasible objective
// below it can stop work without yielding any new lower-bound certificate.
Json solve_design_face_cg(const Input&, PathCollection&, const SolverOptions&,
                          int head, int state, CgColumnSeeds* seeds = nullptr,
                          double no_gain_upper = std::numeric_limits<double>::quiet_NaN());
// Fixed fractional strategic design, with the same complete-recourse CG and
// ordinary globally valid Benders cut. No integer solution is packaged.
Json evaluate_fractional_recourse_cg(const Input&, PathCollection&, const SolverOptions&,
                                    const std::vector<double>& design);
} // namespace mlnd
