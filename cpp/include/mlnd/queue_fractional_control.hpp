#pragma once

#include <algorithm>
#include <cmath>
#include <optional>

namespace mlnd {
struct QueueFractionalBudget {
    double seconds = 0, lp_seconds = 0, price_seconds = 0;
};

// Reserve both phases before starting a call. The caller consumes the newest
// donor once, charges actual wall time, and measures cooldown since the last
// call finished. A pending recourse design takes priority over this work.
// The module must also honor this total deadline after any LP solver overrun;
// its price phase cannot spend the original reservation past that deadline.
inline std::optional<QueueFractionalBudget> queue_fractional_budget(
        double remaining,double total_seconds,double lp_spent,double price_spent,
        double elapsed_since_last,bool fresh_donor,bool pending_design) {
    if(!fresh_donor || pending_design || !std::isfinite(remaining) ||
       !std::isfinite(total_seconds) || !std::isfinite(lp_spent) ||
       !std::isfinite(price_spent) || !std::isfinite(elapsed_since_last) ||
       remaining<=0 || total_seconds<=0 || lp_spent<0 || price_spent<0 ||
       elapsed_since_last<5)return std::nullopt;
    // Round the dedicated allowance down so arithmetic cannot authorize more
    // than 2% of the total budget. Splitting the global remainder equally keeps
    // both phases available without exceeding the two-second call ceiling.
    const double dedicated=std::nextafter(total_seconds/50.0,0.0);
    const double share=std::min(1.0,remaining/2);
    QueueFractionalBudget result;
    result.lp_seconds=std::min(share,dedicated-lp_spent);
    result.price_seconds=std::min(share,dedicated-price_spent);
    if(!(result.lp_seconds>0) || !(result.price_seconds>0))return std::nullopt;
    result.seconds=result.lp_seconds+result.price_seconds;
    return result;
}
} // namespace mlnd
