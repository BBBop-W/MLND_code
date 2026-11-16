#pragma once

#include <algorithm>
#include <cmath>
#include <optional>

namespace mlnd {
struct MasterSolveBudget {
    double seconds = 0, relative_gap = 0;
    bool exact_phase = false;
};

struct MasterTargetBudget {
    MasterSolveBudget budget;
    double target_bound = 0, bound_deficit = 0, candidate_margin = 0;
};

struct RestrictedSearchWindow {
    double total_seconds = 0, remaining = 0, heuristic_remaining = 0, since_last = 0;
    int completed_windows = 0, local_branching_radius = 0;
    bool pending_design = false, local_branching = false;
};

// A single existing primal window, sharing the ordinary heuristic allowance.
// -1 requests unrestricted column search; a nonnegative value is its Hamming radius.
inline std::optional<int> scheduled_restricted_radius(const RestrictedSearchWindow& state) {
    if(state.pending_design || !std::isfinite(state.total_seconds) ||
       !std::isfinite(state.remaining) || !std::isfinite(state.heuristic_remaining) ||
       !std::isfinite(state.since_last) || state.total_seconds<=0 || state.remaining<=10 ||
       state.heuristic_remaining<=.01 || state.since_last<std::max(30.0,.20*state.total_seconds) ||
       state.completed_windows<0 || state.local_branching_radius<0)return std::nullopt;
    // The initial unrestricted solve precedes these windows. Alternate one
    // local and one unrestricted call; disabling the flag preserves all free calls.
    return state.local_branching && state.completed_windows%2==0
        ? state.local_branching_radius : -1;
}

struct RestrictedSearchMode {
    bool coupled = false, fixed_assignment = false, fixed_infrastructure = false;
};

// Ordinary neighborhoods keep their own cycle when local calls are interleaved.
inline RestrictedSearchMode restricted_search_mode(int ordinary_attempts,int radius,
        bool polish,bool coupled_enabled,bool asset_enabled,bool positive_service) {
    RestrictedSearchMode mode;
    if(radius>=0 || !positive_service)return mode;
    mode.coupled=coupled_enabled && ordinary_attempts%3==1;
    mode.fixed_assignment=!mode.coupled && polish && ordinary_attempts%2==0;
    mode.fixed_infrastructure=!mode.coupled && polish && asset_enabled && !mode.fixed_assignment;
    return mode;
}

inline const char* restricted_search_policy(bool alns,bool restricted,bool scheduled,bool local) {
    if(alns)return "alns";
    if(!restricted)return "disabled";
    if(!scheduled)return "legacy_periodic";
    return local ? "alternating_local_branching_and_column_mip" : "column_mip";
}

// Charge the complete restricted-search call, including postprocessing and
// exceptional exits. Clock is a nonthrowing monotonic source, injected in probes.
template<class Clock> class RestrictedSearchCharge {
public:
    RestrictedSearchCharge(double& total,double& restricted,double& last,
                           double started,Clock clock)
        :total_(total),restricted_(restricted),last_(last),started_(started),clock_(clock) {}
    RestrictedSearchCharge(const RestrictedSearchCharge&)=delete;
    RestrictedSearchCharge& operator=(const RestrictedSearchCharge&)=delete;
    ~RestrictedSearchCharge() { finish(); }
    double finish() {
        if(!finished_) {
            last_=clock_();seconds_=std::max(0.0,last_-started_);
            total_+=seconds_;restricted_+=seconds_;finished_=true;
        }
        return seconds_;
    }
private:
    double &total_,&restricted_,&last_;
    double started_,seconds_=0;
    Clock clock_;
    bool finished_=false;
};

// A bounded attempt to certify an already-near outer gap. The candidate is
// only the finite-cut master's incumbent, never a global lower bound. The
// caller owns maturity/stagnation, cooldown and the cumulative 2% allowance,
// and must use min(existing MIP tolerance, budget.relative_gap) for this call.
// No global feasibility, recourse or outer-gap tolerance is changed here.
inline std::optional<MasterTargetBudget> target_master_budget(
        std::optional<double> candidate_objective,double upper,double lower,
        double target_gap,double remaining,double target_budget_remaining,
        bool stalled_and_mature) {
    if(!candidate_objective || !stalled_and_mature ||
       !std::isfinite(*candidate_objective) || !std::isfinite(upper) ||
       !std::isfinite(lower) || !std::isfinite(target_gap) ||
       !std::isfinite(remaining) || !std::isfinite(target_budget_remaining) ||
       upper<=0 || lower>upper || target_gap<=0 || target_gap>=1 ||
       remaining<=0 || target_budget_remaining<=0)return std::nullopt;
    const long double scale=std::max(1.0L,std::abs(static_cast<long double>(upper)));
    const long double allowed_gap=static_cast<long double>(target_gap)*scale;
    const long double open_gap=static_cast<long double>(upper)-lower;
    if(open_gap<=allowed_gap || open_gap>1.25L*allowed_gap)return std::nullopt;
    const long double target=static_cast<long double>(upper)-allowed_gap;
    MasterTargetBudget result;
    // Round the requested bound upward and the tolerance downward. Refuse an
    // unrepresentably small margin instead of silently requesting zero gap.
    result.target_bound=std::nextafter(static_cast<double>(target),INFINITY);
    const long double margin=static_cast<long double>(*candidate_objective)-result.target_bound;
    if(!std::isfinite(result.target_bound) || !(margin>0))return std::nullopt;
    result.bound_deficit=result.target_bound-lower;
    result.candidate_margin=static_cast<double>(margin);
    const double tolerance=std::nextafter(static_cast<double>(.5L*margin/
        (1+std::abs(static_cast<long double>(*candidate_objective)))),0.0);
    if(!std::isfinite(result.bound_deficit) || !std::isfinite(result.candidate_margin) ||
       !std::isfinite(tolerance) || !(tolerance>0))return std::nullopt;
    result.budget={std::min({5.0,remaining,target_budget_remaining}),
        tolerance,false};
    return result;
}

// Inexact master solves produce candidates, while only their certified best
// bound contributes to the global lower bound. A fully priced candidate that
// no longer separates must trigger tighter / longer master searches. After
// six stalled passes use the remaining budget and zero relative MIP gap.
inline MasterSolveBudget master_solve_budget(bool adaptive, double remaining,
        double initial_gap, double slice_seconds, double outer_gap,
        double target_gap, int stalled_passes) {
    if (!adaptive) return {remaining, target_gap, true};
    const int stalled = std::max(0, stalled_passes);
    const bool exact = stalled >= 3 || outer_gap <= std::max(2 * target_gap, 1e-6);
    const double tolerance = exact ? 0.0 :
        std::min(initial_gap, .25 * outer_gap) * std::pow(.1, std::min(stalled, 3));
    const double allowance = stalled >= 6 ? remaining :
        std::min(remaining, slice_seconds * std::pow(2.0, std::min(stalled, 6)));
    return {std::max(0.0, allowance), tolerance, exact};
}

// A finite-cut master need not be solved to zero gap merely because a better
// incumbent makes the outer gap small. Refine it after supported candidates
// have actually reached recourse; exact searches require repeated closed
// recourse without a separating cut. Every returned best bound stays valid.
inline MasterSolveBudget scheduled_master_budget(bool adaptive, double remaining,
        double initial_gap, double slice_seconds, double outer_gap,
        double target_gap, int bound_stalls, int supported_passes,
        int closed_without_progress) {
    if (!adaptive) return {remaining, target_gap, true};
    const bool mature = supported_passes >= 3;
    const bool exact = mature && closed_without_progress >= 3;
    const int level = mature ? std::min(3, std::max(0, bound_stalls) / 8) : 0;
    const double tolerance = exact ? 0.0 :
        std::min(initial_gap, std::max(.25 * target_gap, .25 * outer_gap)) * std::pow(.5, level);
    const double slice = exact ? std::min(60.0, 2 * slice_seconds) :
        std::min(slice_seconds, (mature ? 5.0 : 2.0) * std::pow(2.0, level));
    return {std::max(0.0, std::min(remaining, slice)), tolerance, exact};
}

inline int updated_bound_stalls(double before, double after, int stalls) {
    return after > before + 1e-7 * (1 + std::abs(after)) ? 0 : stalls + 1;
}

inline bool defer_resource_recourse(bool cost_cut_added, int consecutive_restarts,
                                   int max_restarts = 2) {
    return cost_cut_added && consecutive_restarts < max_restarts;
}
} // namespace mlnd
