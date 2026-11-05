#include "mlnd/heuristics.hpp"
#include "mlnd/output.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> m(rows,cols);std::fill(m.values.begin(),m.values.end(),value);return m;
}
Input fixture() {
    Input d;d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=2;d.max_orders=2;
    d.name="alns_repair_fixture";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";
    d.demand={4,7};d.ground_cost={20,20};d.deadline={1,1};
    d.flight_times=filled(4,4,.05);d.allowed_links=filled(4,4,1);
    for(int i=0;i<4;++i){d.flight_times(i,i)=0;d.allowed_links(i,i)=0;}
    d.flight_times(2,3)=d.flight_times(3,2)=.1;
    d.link_cost=filled(4,4,.4);d.base_cost={.5,.5};d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,2,.08);d.first_cost=filled(2,2,1.0);
    d.terminal_cost=filled(2,2,.5);d.receipt_mean=filled(2,2,.08);
    d.removal_mean=filled(2,2,.03);d.slot_counts=filled(2,2,20);
    d.slot_cost=filled(2,2,.1);d.occupancy_limit=filled(2,2,14.0);
    for(int h=0;h<2;++h){d.slot_counts(h,1)=40;d.slot_cost(h,1)=.2;d.occupancy_limit(h,1)=28;}
    d.kappa=filled(2,2,1.0);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;d.preprocessed_demand=true;
    d.single_rates={.5,4.5};d.pair_rates=filled(2,2,0.0);
    d.pair_rates(0,0)=.5;d.pair_rates(0,1)=2;d.pair_rates(1,0)=.5;
    d.head_rates={3,5};d.common_origin_support=filled(2,2,1);return d;
}
}
int main(){try {
    auto d=fixture();auto paths=enumerate_paths(d);
    SolverOptions options;options.seconds=10;options.gap=1e-6;options.neighborhood_cuts=false;
    options.grouped_local_branching=true;
    options.local_branching_head_radius=d.task_count();
    auto initial=solve_restricted_heuristic(d,paths,options,wall_time()+10);
    require(initial.at("objective").get<double>()<100,"Fixture has no useful feasible incumbent");
    auto incumbent=initial.at("solution");const double saved=incumbent.at("objective");
    CoupledPrimalNeighborhood block;
    for(int q=0;q<d.task_count();++q)if(d.task_rate(q)>0)block.heads.push_back(q);
    block.lockers={0,1};block.bases={0,1};
    for(int u=0;u<4;++u)for(int v=0;v<4;++v)if(u!=v)block.airlines.emplace_back(u,v);
    PrimalRepairRestrictions explore;explore.change_heads=block.heads;explore.closed_bases={0,1};
    explore.grouped_radius=true;
    explore.objective_upper=221;int callbacks=0;
    explore.retain_validated=[&](const Json& candidate) {
        require(std::abs(candidate.at("objective").get<double>()-220)<1e-5,
                "Callback received a candidate violating the closed-base neighborhood");++callbacks;
    };
    auto worse=solve_restricted_heuristic(d,paths,options,wall_time()+10,&incumbent,-1,
        nullptr,nullptr,false,false,&block,&explore);
    require(worse.at("candidate_validated"),"Forced exploration did not return its feasible neighbor");
    require(worse.at("grouped_local_branching").at("head_radius")==options.local_branching_head_radius &&
            worse.at("grouped_local_branching").at("capacity_radius")==options.local_branching_capacity_radius,
            "Coupled repair omitted one of the configured group distances");
    require(std::abs(worse.at("candidate").at("objective").get<double>()-220)<1e-5,
            "Forced both-base closure must produce all-courier cost 220");
    require(std::abs(worse.at("objective").get<double>()-saved)<1e-6,
            "Worse neighbor overwrote globally retained incumbent");
    require(worse.at("diagnostics").at("candidate_validation").at("physical").at("passed"),
            "Exploration neighbor did not pass original physical validation");
    require(callbacks==1,"Candidate was not submitted exactly once before optional work");
    const auto design=solution_design(d,worse.at("candidate"));DesignLayout layout(d);
    require(design.at(layout.facility)==0&&design.at(layout.facility+1)==0,
            "Candidate ignored forced base closure");
    require(!worse.contains("bound")&&!worse.at("candidate").contains("bound")&&
            !worse.at("global_lower_bound_available").get<bool>(),"Local bound leaked into primal result");
    auto ground=all_courier_result(d,paths);explore.retain_validated={};
    auto impossible=solve_restricted_heuristic(d,paths,options,wall_time()+10,&ground,-1,
        nullptr,nullptr,false,false,&block,&explore);
    require(!impossible.at("candidate_validated").get<bool>()&&!impossible.contains("candidate"),
            "Global fallback was misreported as a forced-change neighbor");
    require(std::abs(impossible.at("objective").get<double>()-220)<1e-5,
            "Infeasible neighborhood lost valid global fallback");
    PrimalRepairRestrictions capacity;capacity.change_lockers={0};capacity.closed_bases={0,1};
    auto capacity_neighbor=solve_restricted_heuristic(d,paths,options,wall_time()+10,&ground,-1,
        nullptr,nullptr,false,false,&block,&capacity);
    require(capacity_neighbor.at("candidate_validated"),"Closed locker could not change its capacity state");
    require(std::abs(capacity_neighbor.at("candidate").at("objective").get<double>()-220.1)<1e-5,
            "Capacity group exploration failed its independently known cheapest-state cost");
    require(std::abs(capacity_neighbor.at("objective").get<double>()-220)<1e-5,
            "Capacity exploration overwrote a better global fallback");
    explore.objective_upper=saved*1.03;
    auto capped=solve_restricted_heuristic(d,paths,options,wall_time()+10,&incumbent,-1,
        nullptr,nullptr,false,false,&block,&explore);
    require(!capped.at("candidate_validated").get<bool>(),"Exploration ignored its temporary objective cap");
    auto unchanged=solve_restricted_heuristic(d,paths,options,wall_time()+10,&incumbent,-1,
        nullptr,nullptr,false,false,&block);
    require(unchanged.at("objective").get<double>()<=saved+1e-6,
            "Temporary exploration constraints leaked to a later repair");
    require(worse.at("repair_restrictions").at("validation_reserve_seconds").get<double>()>=.5,
            "Repair did not reserve validation time");
    std::cout<<Json({{"passed",true},{"incumbent_objective",saved},
        {"exploration_objective",worse.at("candidate").at("objective")},
        {"fallback_preserved",true},{"impossible_neighbor_rejected",true},
        {"objective_cap_respected",true},{"capacity_group_exploration",true},
        {"temporary_constraints_isolated",true}}).dump(2)<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
