#pragma once

#include <algorithm>
#include <cmath>

namespace mlnd {
struct ContinuousRootProgress {
    double spent = 0, since_last = 0;
    int integer_passes = 0;
    bool initialized = false, closed = false, pending = false;
};

// Split the existing continuous-root allowance, without reserving extra time.
// A resumed slice waits for integer work, never interrupts pending recourse,
// and leaves the final thirty seconds to the ordinary search. Column/bound
// progress is retained by the caller even when a stage does not close CG.
inline double continuous_root_slice_budget(double total_limit,double global_remaining,
                                           const ContinuousRootProgress& progress) {
    if(!std::isfinite(total_limit) || !std::isfinite(global_remaining) ||
       !std::isfinite(progress.spent) || !std::isfinite(progress.since_last) ||
       total_limit<=0 || global_remaining<=0 || progress.spent<0 ||
       progress.since_last<0 || progress.integer_passes<0 || progress.closed || progress.pending)
        return 0;
    const double left=std::max(0.0,total_limit-progress.spent);
    if(!progress.initialized)
        return std::min({left,global_remaining,30.0,.25*total_limit});
    if(progress.integer_passes==0 || progress.since_last<60 || global_remaining<=30)return 0;
    return std::min({left,15.0,.10*global_remaining,global_remaining-30.0});
}
} // namespace mlnd
