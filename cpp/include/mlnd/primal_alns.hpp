#pragma once

#include "mlnd/primal_support.hpp"
#include <functional>
#include <memory>
#include <optional>

namespace mlnd {
// Feasible-only acceptance probability. Used primal wall time controls cooling;
// no objective here is a lower bound or an infeasibility certificate.
double primal_alns_acceptance_probability(double current, double candidate,
    double initial_best, double used_seconds, double budget_seconds,
    double reheat_temperature_fraction = 0);

// Persistent upper-bound search. Construction uses an internally generated
// all-courier fallback; supplied/observed points must already have passed the
// same-model native and physical validation. PathCollection is append-only and
// must outlive this object. Every local deadline is absolute wall_time().
// The callback receives verified improving points before optional bookkeeping.
// Global budget ownership and certified Benders lower bounds stay with caller.
class PrimalAlnsSearch {
public:
    using Submit = std::function<void(const Json&)>;
    PrimalAlnsSearch(const Input&, const PathCollection&, const SolverOptions&,
                    const Json& validated_initial, double primal_budget_seconds);
    ~PrimalAlnsSearch();
    PrimalAlnsSearch(PrimalAlnsSearch&&) noexcept;
    PrimalAlnsSearch& operator=(PrimalAlnsSearch&&) noexcept;
    PrimalAlnsSearch(const PrimalAlnsSearch&) = delete;
    PrimalAlnsSearch& operator=(const PrimalAlnsSearch&) = delete;

    // Up to three genuinely different joint seed directions; support selection
    // shares this allowance. Normal step executes at most one repair.
    void startup(double deadline, const Submit& submit = {});
    void step(double deadline, const std::vector<double>* master_guide = nullptr,
              const Submit& submit = {});
    void observe_validated(const Json&);
    // Bounded CG-to-primal ranking feedback. Never replaces best/current or
    // produces a lower bound. Caller accounts this full wall time globally.
    Json observe_cg_columns(const PrimalCgFeedback&,double deadline);
    const Json& best() const;
    const Json& current() const;
    const Json& records() const;
    Json summary() const;
    double used_seconds() const;
    // Between calls, synchronize the authoritative shared primal wall time,
    // including caller-owned elite CG and bookkeeping. Never moves backward.
    void synchronize_used_seconds(double shared_used_seconds);
    const std::vector<int>& service_columns() const;
    const std::vector<int>& empty_columns() const;
    // At most two distinct promising designs, each returned once. Caller may
    // refine them with complete-domain CG under the shared elite allowance.
    std::optional<Json> take_elite_refinement();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace mlnd
