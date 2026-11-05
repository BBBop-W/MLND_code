#include "mlnd/benders.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> m(rows,cols);std::fill(m.values.begin(),m.values.end(),value);return m;
}
Input fixture() {
    Input d;d.name="cg_reuse_physical_k2";d.model_version="consecutive";
    d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand={4,7};d.single_rates={.5,4.5};d.head_rates={3,5};
    d.pair_rates=filled(2,2,0.0);d.pair_rates(0,0)=.5;d.pair_rates(0,1)=2;d.pair_rates(1,0)=.5;
    d.common_origin_support=filled(2,2,1);d.ground_cost={20,20};d.deadline={1,1};
    d.flight_times=filled(4,4,.05);d.allowed_links=filled(4,4,1);
    for(int u=0;u<4;++u){d.allowed_links(u,u)=0;d.flight_times(u,u)=0;}
    d.flight_times(2,3)=d.flight_times(3,2)=.1;
    d.link_cost=filled(4,4,.01);d.base_cost={.5,.7};d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,2,.08);d.first_cost=filled(2,2,1.0);
    d.terminal_cost=filled(2,2,.5);d.receipt_mean=filled(2,2,.08);
    d.removal_mean=filled(2,2,.03);d.slot_counts=filled(2,1,20);
    d.slot_cost=filled(2,1,.1);d.occupancy_limit=filled(2,1,14.0);d.kappa=filled(2,2,1.0);
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    return d;
}

int count_branch_records(const Json& j) {
    int count=0;
    if(j.is_object()) {
        if(j.contains("branch_correction_complete") && j.at("branch_correction_complete")==true)++count;
        for(const auto& item:j.items())count+=count_branch_records(item.value());
    } else if(j.is_array())for(const auto& item:j)count+=count_branch_records(item);
    return count;
}
}
int main(){try {
    const auto d=fixture();auto paths=enumerate_paths(d);
    SolverOptions options;options.seconds=20;options.threads=1;options.seed=43;
    options.preprocessed_demand=true;options.pareto=false;options.continuous_root_seconds=8;
    options.continuous_root_budget_fraction=.4;options.branch_pricing_correction=true;
    options.cg_seed_reuse=true;options.staged_continuous_root=true;
    options.design_face_cuts=true;options.queue_polymatroid_cuts=true;options.queue_joint_cuts=true;
    auto full=build_formulation(d,paths);NativeSolver solver(full.matrix,options,"lb_integration_reference");
    const auto solved=solver.optimize(10);
    require(solved.optimal&&solved.has_solution&&point_residuals(full.matrix,solved.values).at("passed"),
        "Tiny reference did not certify the full integer optimum");
    const double optimum=objective_value(full.matrix,solved.values);
    const auto result=solve_benders(d,paths,options);
    const auto& stages=result.at("continuous_root_stage_records");
    require(!stages.empty(),"Production stage integration did not execute");
    require(stages.at(0).at("allocated_seconds").get<double>()<=2.0,"Initial stage ignored quarter-budget cap");
    require(result.at("continuous_root_schedule").at("seconds").get<double>()>=
        stages.at(0).at("stage_wall_seconds").get<double>(),"Continuous stage time disappeared");
    require(stages.at(0).at("reused_service_columns").get<int>()>0, "Main solver did not retain generated CG columns");
    require(count_branch_records(result)>0,"Native main solve did not consume complete branch pricing");
    require(result.at("bound").get<double>()<=optimum+1e-5*(1+std::abs(optimum)),"Main solver produced invalid global LB");
    require(result.at("objective").get<double>()>=optimum-1e-5*(1+std::abs(optimum)),"Main solver produced invalid integer objective");
    std::cout<<Json({{"passed",true},{"full_optimum",optimum},{"lower_bound",result.at("bound")},
        {"upper_bound",result.at("objective")},{"branch_records",count_branch_records(result)},
        {"schedule",result.at("continuous_root_schedule")}}).dump(2)<<'\n';return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
