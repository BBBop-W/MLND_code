#pragma once

#include <algorithm>
#include <cmath>

namespace mlnd {
struct MasterSolveBudget {
    double seconds = 0, relative_gap = 0;
    bool exact_phase = false;
};

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
} // namespace mlnd
