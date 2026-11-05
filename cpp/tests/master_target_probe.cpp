#include "mlnd/benders_control.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    int checks=0;
    auto require=[&](bool ok,const char* message){if(!ok)throw std::runtime_error(message);++checks;};
    // Saved M08 queue run: the incumbent of the finite-cut master is above
    // the 1% target, but its certified global lower bound is still below it.
    // These decimal expectations were calculated independently at 40 digits.
    const auto target=mlnd::target_master_budget(3872.6132832385156,
        3911.2860808041387,3871.205912275554,.01,40,12,true);
    require(target.has_value(),"Near-target M08 master was not eligible");
    require(std::abs(target->target_bound-3872.173219996097313)<1e-9,
            "M08 target bound is wrong");
    require(std::abs(target->bound_deficit-.967307720543313)<1e-9,
            "M08 remaining certificate deficit is wrong");
    require(std::abs(target->candidate_margin-.440063242418287)<1e-9,
            "M08 master candidate margin is wrong");
    require(target->budget.relative_gap>0 &&
            target->budget.relative_gap<=.000056802681404,
            "M08 refinement tolerance cannot certify the requested target");
    require(target->budget.seconds==5 && !target->budget.exact_phase,
            "Target refinement became an unbounded exact solve");
    // The API must reject work that cannot serve a still-open near target.
    auto request=[](std::optional<double> candidate,double upper=100,double lower=98.9,
                    double goal=.01,double wall=40,double dedicated=12,bool ready=true) {
        return mlnd::target_master_budget(candidate,upper,lower,goal,wall,dedicated,ready);
    };
    require(!request(99.4,100,99),"Already reached target launched another solve");
    require(!request(99.4,100,98.74),"Far-from-target master used the dedicated refinement");
    require(!request(99),"Candidate at target was treated as a lower-bound certificate");
    require(!request(98.99),"Candidate below target enabled refinement");
    require(!request(std::nullopt),"Missing master incumbent enabled refinement");
    require(!request(99.4,100,98.9,.01,40,12,false),"Immature or unstalled search enabled refinement");
    require(!request(99.4,100,98.9,.01,0),"Exhausted global wall budget enabled refinement");
    require(!request(99.4,100,98.9,.01,40,0),"Exhausted dedicated budget enabled refinement");
    require(!request(99.4,100,98.9,.01,-1),"Negative global budget enabled refinement");
    require(!request(99.4,100,98.9,.01,40,-1),"Negative dedicated budget enabled refinement");
    require(!request(99.4,100,98.9,0) && !request(99.4,100,98.9,-.01) &&
            !request(99.4,100,98.9,1),"Invalid target gap enabled refinement");
    require(!request(99.4,100,101),"Contradictory global bounds enabled refinement");
    require(!request(.1,0,-.01) && !request(-1,-1,-1.011),
            "Nonpositive objective scale enabled refinement");
    const double bad=std::numeric_limits<double>::quiet_NaN();
    const double inf=std::numeric_limits<double>::infinity();
    for(double value:{bad,inf,-inf}) {
        require(!request(value),"Nonfinite candidate enabled refinement");
        require(!request(99.4,value),"Nonfinite upper bound enabled refinement");
        require(!request(99.4,100,value),"Nonfinite lower bound enabled refinement");
        require(!request(99.4,100,98.9,value),"Nonfinite goal enabled refinement");
        require(!request(99.4,100,98.9,.01,value),"Nonfinite wall budget enabled refinement");
        require(!request(99.4,100,98.9,.01,40,value),"Nonfinite dedicated budget enabled refinement");
    }
    const auto short_wall=request(99.4,100,98.9,.01,1.25,12);
    require(short_wall && short_wall->budget.seconds==1.25,
            "Refinement exceeded the global time left");
    const auto short_allowance=request(99.4,100,98.9,.01,40,.4);
    require(short_allowance && short_allowance->budget.seconds==.4,
            "Refinement exceeded its dedicated allowance");
    require(request(99.4,100,98.75).has_value(),"Closed near-target boundary was rejected");
    require(!request(99.4,100,std::nextafter(98.75,-INFINITY)),
            "Open far-side boundary was accepted");
    const auto small_objective=request(.095,.1,.089);
    require(small_objective && std::abs(small_objective->target_bound-.09)<1e-14,
            "Target differs from the outer gap definition for objectives below one");
    std::cout<<"{\"passed\":true,\"checks\":"<<checks<<"}\n";
}
