#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace mlnd {
// Discovery/column-batch policy only. No member is an optimality certificate.
// A controller belongs to exactly one fixed design and row layout.
struct AdaptivePricingControl {
    double weight=.5, exact_seconds=0, certified_lower=-std::numeric_limits<double>::infinity();
    int interval=4, since_exact=0, max_interval=4;
    std::size_t batch=64,max_batch=64;
    bool initialized=false, certified_progress=false;
    void initialize(double initial_weight,int exact_interval,std::size_t columns) {
        if(initialized)return;
        weight=initial_weight;interval=max_interval=std::max(1,exact_interval);
        batch=max_batch=std::max<std::size_t>(1,columns);initialized=true;
    }
    bool exact_due(double seconds_left) const {
        return since_exact>=interval-1 || (exact_seconds>0 && seconds_left<=1.5*exact_seconds);
    }
    void discovery(std::size_t proposed,std::size_t accepted) {
        ++since_exact;
        const double ratio=proposed?static_cast<double>(accepted)/proposed:0;
        if(ratio<.25) {weight=std::min(1.0,weight+.2);interval=std::max(1,interval-1);}
        else if(ratio>=.75 && certified_progress) {
            weight=std::max(.25,weight*.9);interval=std::min(max_interval,interval+1);
        }
    }
    void exact(double seconds) {
        since_exact=0;
        if(std::isfinite(seconds)&&seconds>=0)exact_seconds=exact_seconds?(.7*exact_seconds+.3*seconds):seconds;
    }
    void certificate(double bound) {
        if(!std::isfinite(bound))return;
        certified_progress=!std::isfinite(certified_lower) || bound>certified_lower+1e-7*(1+std::abs(bound));
        certified_lower=std::max(certified_lower,bound);
        // Recovery from an interval-one fallback requires an exact certificate,
        // rather than a discovery event that can no longer occur at interval 1.
        if(certified_progress && interval==1 && max_interval>1)interval=2;
    }
    void participation(std::size_t offered,std::size_t positive,double rmp_seconds,double pricing_seconds) {
        if(!offered)return;
        const double ratio=static_cast<double>(positive)/offered;
        if(ratio<.25 && rmp_seconds>pricing_seconds)
            batch=std::max(std::min<std::size_t>(16,max_batch),batch/2);
        else if(ratio>=.6)batch=std::min(max_batch,batch>max_batch/2?max_batch:2*batch);
    }
};
} // namespace mlnd
