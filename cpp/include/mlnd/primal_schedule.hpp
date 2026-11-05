#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace mlnd {
enum class PrimalPhase { startup, regular, elite };
// Wall time is charged by the caller after all construction, solve, validation
// and submission work, including failed calls. No pending-design gate exists.
class PrimalSchedule {
public:
    PrimalSchedule(double seconds,double started): total_(.30*seconds),last_(started) {
        if(!std::isfinite(seconds)||seconds<=0||!std::isfinite(started))
            throw std::invalid_argument("Invalid primal schedule budget");
    }
    double spent() const {return used_[0]+used_[1]+used_[2];}
    double limit() const {return total_;}
    double used(PrimalPhase phase) const {return used_[static_cast<int>(phase)];}
    double allowance(PrimalPhase phase,double now,double remaining) const {
        if(!std::isfinite(now)||!std::isfinite(remaining)||remaining<=20)return 0;
        const auto index=static_cast<int>(phase);
        const double portion=index==1?total_*2/3:total_/6;
        if(index!=0 && now-last_<45)return 0;
        const double cap=index==0?30:index==1?6:15;
        return std::max(0.0,std::min({cap,portion-used_[index],total_-spent(),remaining-10}));
    }
    void charge(PrimalPhase phase,double seconds,double now) {
        if(!std::isfinite(seconds)||seconds<0||!std::isfinite(now))
            throw std::invalid_argument("Invalid primal wall accounting");
        used_[static_cast<int>(phase)]+=seconds;last_=now;
    }
    // Ranking/feedback consumes the shared primal allowance but must not
    // postpone the next search window by resetting its cooldown clock.
    void charge_auxiliary(double seconds) {
        if(!std::isfinite(seconds)||seconds<0)
            throw std::invalid_argument("Invalid primal auxiliary accounting");
        used_[static_cast<int>(PrimalPhase::regular)]+=seconds;
    }
    double cg_slice(double now,double remaining) const {
        if(remaining<=20 || total_*2/3-used_[1]<=.5 || total_-spent()<=.5)return std::max(0.0,remaining);
        return std::max(0.0,std::min({30.0,remaining,last_+45-now}));
    }
private:
    double total_,last_;
    std::array<double,3> used_{{0,0,0}};
};
}
