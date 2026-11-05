#include "mlnd/benders_control.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace mlnd;
    auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
    // A better feasible upper bound must not turn an immature master into
    // repeated zero-gap searches.
    const auto immature=scheduled_master_budget(true,600,.05,30,.015,.01,100,0,0);
    require(!immature.exact_phase && immature.relative_gap>0 && immature.seconds<=2,
            "Small outer gap caused premature exact search");
    const auto ready=scheduled_master_budget(true,600,.05,30,.015,.01,24,3,0);
    require(!ready.exact_phase && ready.relative_gap<immature.relative_gap && ready.seconds<=30,
            "Supported stagnant master was not progressively refined");
    const auto closed=scheduled_master_budget(true,600,.05,30,.015,.01,24,3,3);
    require(closed.exact_phase && closed.relative_gap==0 && closed.seconds<=60,
            "Closed recourse cannot lead to an exact bounded master search");
    const auto exhausted=scheduled_master_budget(true,.1,.05,30,.015,.01,24,3,3);
    require(exhausted.seconds<=.1,"Master exceeded remaining budget");
    const auto legacy=master_solve_budget(true,600,.05,30,.015,.01,0);
    require(legacy.exact_phase && legacy.relative_gap==0,"Legacy policy changed");
    int stalls=0;
    for(int i=0;i<20;++i)stalls=updated_bound_stalls(100,100,stalls);
    require(stalls==20,"Structural restarts lost bound stagnation");
    require(updated_bound_stalls(100,101,stalls)==0,"True LB progress was ignored");
    require(defer_resource_recourse(true,0) && defer_resource_recourse(true,1) &&
            !defer_resource_recourse(true,2) && !defer_resource_recourse(false,0),
            "Resource cost cuts can indefinitely starve complete recourse");
    RestrictedSearchWindow window;
    window.total_seconds=600;window.remaining=400;window.heuristic_remaining=90;
    window.since_last=120;window.local_branching=true;window.local_branching_radius=7;
    require(scheduled_restricted_radius(window)==7,
            "Enabled local branching never reaches the scheduled primal window");
    window.completed_windows=1;
    require(scheduled_restricted_radius(window)==-1,
            "Local branching displaced unrestricted column search in every window");
    window.completed_windows=2;
    require(scheduled_restricted_radius(window)==7,"Local branching was not revisited");
    window.local_branching=false;
    require(scheduled_restricted_radius(window)==-1,"Disabled local branching still ran");
    window.since_last=119;
    require(!scheduled_restricted_radius(window),"Primal window ignored its cooldown");
    window.since_last=120;window.pending_design=true;
    require(!scheduled_restricted_radius(window),"Scheduled primal displaced pending recourse");
    window.pending_design=false;window.heuristic_remaining=0;
    require(!scheduled_restricted_radius(window),"Primal window exceeded its shared budget");
    window.heuristic_remaining=90;window.remaining=10;
    require(!scheduled_restricted_radius(window),"Primal window consumed final reserve");
    int ordinary_attempts=0;
    for(int radius:{-1,7,-1,7,-1,7,-1}) {
        const auto mode=restricted_search_mode(ordinary_attempts,radius,true,false,true,true);
        if(radius>=0) {
            require(!mode.coupled && !mode.fixed_assignment && !mode.fixed_infrastructure,
                    "Local branching inherited an ordinary fixing restriction");
        } else {
            require(mode.fixed_assignment==(ordinary_attempts%2==0) &&
                    mode.fixed_infrastructure==(ordinary_attempts%2==1),
                    "Interleaved local calls starved ordinary assignment or infrastructure search");
            ++ordinary_attempts;
        }
    }
    const auto coupled=restricted_search_mode(1,-1,true,true,true,true);
    require(coupled.coupled && !coupled.fixed_assignment && !coupled.fixed_infrastructure,
            "Ordinary coupled cycle lost priority over fixing modes");
    const auto no_service=restricted_search_mode(0,-1,true,true,true,false);
    require(!no_service.coupled && !no_service.fixed_assignment && !no_service.fixed_infrastructure,
            "Empty service incumbent was fixed before positive support was found");
    require(std::string(restricted_search_policy(false,false,true,true))=="disabled" &&
            std::string(restricted_search_policy(false,true,false,true))=="legacy_periodic" &&
            std::string(restricted_search_policy(true,true,true,true))=="alns" &&
            std::string(restricted_search_policy(false,true,true,true))=="alternating_local_branching_and_column_mip" &&
            std::string(restricted_search_policy(false,true,true,false))=="column_mip",
            "Restricted policy summary misreported the active control path");
    double clock=100,total=5,restricted=2,last=90;
    {
        RestrictedSearchCharge charge(total,restricted,last,clock,[&](){return clock;});
        clock=103; // restricted solve returned
        clock=107; // validation, pool handling and cleanup finished
        require(charge.finish()==7 && total==12 && restricted==9 && last==107,
                "Postprocessing was not charged to the shared heuristic budget");
        require(charge.finish()==7 && total==12,"Repeated finalization double charged a call");
    }
    require(total==12 && restricted==9,"Destructor charged a completed call twice");
    try {
        RestrictedSearchCharge charge(total,restricted,last,clock,[&](){return clock;});
        clock=111;
        throw std::runtime_error("simulated budget interruption during cleanup");
    } catch(const std::runtime_error&) {}
    require(total==16 && restricted==13 && last==111,
            "Interrupted restricted search lost its actual wall time");
    std::cout<<"{\"passed\":true,\"checks\":30}\n";
}
