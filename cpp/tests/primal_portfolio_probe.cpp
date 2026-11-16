#include "mlnd/primal_portfolio.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

using namespace mlnd;
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class T> Matrix<T> filled(int rows, int cols, T value) {
    Matrix<T> out(rows, cols);
    std::fill(out.values.begin(), out.values.end(), value);
    return out;
}
Input fixture() {
    Input d;
    d.name="primal_portfolio_fixture"; d.model_version="consecutive";
    d.n_l=2; d.n_h=3; d.n_od=2; d.n_levels=2; d.max_orders=2;
    d.route_mode="direct"; d.flow_policy="aggregate";
    d.demand={4,7}; d.ground_cost={20,20}; d.deadline={1,1};
    d.flight_times=filled(5,5,.05); d.allowed_links=filled(5,5,1);
    for(int i=0;i<5;++i) {d.flight_times(i,i)=0;d.allowed_links(i,i)=0;}
    d.link_cost=filled(5,5,.4); d.base_cost={.5,.5};
    d.slot_epsilon={.05,.05,.05}; d.rider_mean=filled(2,3,.08);
    d.first_cost=filled(2,3,1.0); d.terminal_cost=filled(2,3,.5);
    d.receipt_mean=filled(2,3,.08); d.removal_mean=filled(2,3,.03);
    d.slot_counts=filled(3,2,20); d.slot_cost=filled(3,2,.1);
    d.occupancy_limit=filled(3,2,14.0);
    for(int h=0;h<3;++h) {
        d.slot_counts(h,1)=40; d.slot_cost(h,1)=.2; d.occupancy_limit(h,1)=28;
    }
    d.kappa=filled(2,2,1.0); d.delta=.04; d.operating_limit=1.5;
    d.fleet_cost=1; d.flight_cost=.2; d.waiting_cost=.3;
    d.preprocessed_demand=true; d.single_rates={.5,4.5}; d.head_rates={3,5};
    d.pair_rates=filled(2,2,0.0);
    d.pair_rates(0,0)=.5; d.pair_rates(0,1)=2; d.pair_rates(1,0)=.5;
    d.common_origin_support=filled(2,2,1);
    return d;
}
Json zero_budget() {
    const auto d=fixture(); const auto paths=enumerate_paths(d);
    SolverOptions options; options.seconds=0; options.neighborhood_cuts=false;
    const auto result=solve_primal_seed_portfolio(d,paths,options,wall_time());
    require(result.at("validated"),"Zero-budget fallback was not validated");
    require(std::abs(result.at("objective").get<double>()-220)<1e-10,
        "Zero budget did not retain the physical all-courier cost");
    require(result.at("attempts").empty(),"Zero budget attempted optimization");
    require(!result.at("global_lower_bound_available").get<bool>() &&
        !result.contains("bound") && !result.at("solution").contains("bound"),
        "Primal fallback exposed a lower bound");
    return {{"case","zero_budget"},{"objective",result.at("objective")},{"passed",true}};
}
Json complete_catalogue_seed() {
    const auto d=fixture(); const auto paths=enumerate_paths(d);
    SolverOptions options; options.seconds=12; options.gap=1e-8;
    options.threads=1; options.neighborhood_cuts=false;
    PrimalPortfolioOptions settings; settings.max_heads=2; settings.max_lockers=2;
    settings.max_candidates=2; settings.candidate_seconds=5;
    const auto result=solve_primal_seed_portfolio(d,paths,options,wall_time()+12,nullptr,settings);
    require(paths.complete,"Seed witness must use the complete catalogue branch");
    require(result.at("validated") && result.at("objective").get<double>()<200,
        "Complete-catalogue portfolio failed to improve the 220 all-courier cost");
    require(result.at("solution").at("served").get<double>()>1,
        "Seed portfolio returned no actual drone service");
    require(!result.at("attempts").empty(),"Seed portfolio did not expose its attempts");
    bool physically_checked=false;
    for(const auto& attempt:result.at("attempts")) {
        if(attempt.contains("diagnostics") && attempt.at("diagnostics").contains("candidate_validation"))
            physically_checked=physically_checked ||
                attempt.at("diagnostics").at("candidate_validation").at("physical").at("passed").get<bool>();
        const auto& selection=attempt.at("selection");
        require(selection.at("lockers").size()<=2,"Seed portfolio exceeded its declared asset block");
        require(selection.at("service_indices").size()<paths.service_paths.size(),
            "Seed portfolio silently used the entire service catalogue");
        for(const auto& group:selection.at("head_groups")) {
            const int r=group.at("head"),l=group.at("departure");
            std::set<int> branches;
            for(int index:selection.at("service_indices").get<std::vector<int>>()) {
                const auto& p=paths.service_paths.at(index);
                if(d.task_index(p.first_od,p.second_od)==r && p.b[0]==l)branches.insert(p.second_od);
            }
            require(d.task_rate(r)<=0 || branches.count(d.task_second(r)),
                "Selected independent task lost its physical witness");
        }
    }
    require(physically_checked,"Seed gain bypassed original physical and cost validation");
    require(!result.contains("bound") && !result.at("solution").contains("bound"),
        "Seed portfolio promoted its restricted bound");
    return {{"case","complete_catalogue_seed"},{"objective",result.at("objective")},
        {"served",result.at("solution").at("served")},{"passed",true}};
}
Json coupled_resource_repair() {
    auto d=fixture();d.fleet_cost=10;
    for(int h=0;h<d.n_h;++h)d.flight_times(1,d.n_l+h)=.5;
    const auto paths=enumerate_paths(d);const DesignLayout layout(d);
    std::vector<double> design(layout.count,0);
    for(int i=0;i<d.n_nodes();++i)design[layout.facility+i]=1;
    for(int h=0;h<d.n_h;++h)design[layout.capacity+h*d.n_levels]=1;
    // Keep the stable head at the fast base. Moving the released head there
    // consolidates queues as well as shortening flights; an all-slow-base
    // center would trade flight savings against opening a second queue.
    design[layout.assignment_index(0,1)]=1;
    design[layout.assignment_index(1,0)]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        design[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    SolverOptions options;options.seconds=12;options.gap=1e-8;
    options.threads=1;options.neighborhood_cuts=false;
    const auto incumbent=solve_full(d,paths,options,&design);
    require(incumbent.at("served").get<double>()>1,"Coupled fixture lacks a positive-service incumbent");
    auto guide=design;guide[layout.assignment_index(0,0)]=1;guide[layout.assignment_index(0,1)]=0;
    PrimalPortfolioOptions settings;settings.max_heads=1;settings.max_lockers=3;
    const auto repaired=solve_primal_coupled_neighborhood(d,paths,options,wall_time()+12,
        incumbent,&guide,0,settings);
    require(repaired.at("validated") && repaired.at("objective").get<double>()+1<incumbent.at("objective").get<double>(),
        "Coupled resource repair did not escape the slow-base assignment");
    const auto moved=solution_design(d,repaired.at("solution"));
    if(moved[layout.assignment_index(0,0)]!=1)
        std::cerr<<Json({{"diagnostic","coupled_assignment_witness"},
            {"incumbent_objective",incumbent.at("objective")},{"objective",repaired.at("objective")},
            {"assignment",repaired.at("solution").at("_solution").at("assignment")},
            {"block",repaired.at("coupled_neighborhood")},
            {"selection",repaired.at("attempts").at(0).at("selection")},
            {"status",repaired.at("status")}}).dump(2)<<'\n';
    require(moved[layout.assignment_index(0,0)]==1,"Released head did not move to the useful base");
    for(int l=0;l<d.n_l;++l)
        require(moved[layout.assignment_index(1,l)]==design[layout.assignment_index(1,l)],
            "Coupled repair changed the stable head outside its block");
    const auto free=repaired.at("coupled_neighborhood").at("released_design_indices").get<std::vector<int>>();
    const std::set<int> released(free.begin(),free.end());
    for(int j=0;j<layout.count;++j)if(!released.count(j))
        require(moved[j]==design[j],"Coupled repair changed a stable resource outside its declared block");
    for(int l=0;l<d.n_l;++l)
        require(released.count(layout.assignment_index(0,l)),"Released head lost one of its assignment choices");
    for(int h:repaired.at("coupled_neighborhood").at("lockers").get<std::vector<int>>())
        for(int z=0;z<d.n_levels;++z)
            require(released.count(layout.capacity+h*d.n_levels+z),"Released locker lost a capacity level");
    require(repaired.at("attempts").at(0).at("diagnostics").at("incumbent_validation").at("passed"),
        "Coupled search dropped some positive incumbent support");
    require(repaired.at("attempts").at(0).at("diagnostics").at("candidate_validation").at("physical").at("passed"),
        "Coupled gain bypassed physical validation");
    require(!repaired.contains("bound") && !repaired.at("solution").contains("bound"),
        "Coupled repair leaked a neighborhood lower bound");
    return {{"case","coupled_resource_repair"},{"incumbent",incumbent.at("objective")},
        {"objective",repaired.at("objective")},{"passed",true}};
}
Json nonfinite_center() {
    const auto d=fixture();const auto paths=enumerate_paths(d);
    auto invalid=all_courier_result(d,paths);
    invalid["_solution"]["assignment"][0][0]=std::numeric_limits<double>::quiet_NaN();
    SolverOptions options;options.seconds=5;options.neighborhood_cuts=false;
    const DesignLayout layout(d);const std::vector<double> guide(layout.count,0);
    const auto result=solve_primal_coupled_neighborhood(d,paths,options,wall_time()+5,invalid,&guide);
    require(result.at("status")=="invalid_incumbent_design" && result.at("attempts").empty(),
        "Nonfinite center reached block selection or model construction");
    require(result.at("validated") && result.at("objective").get<double>()==220,
        "Invalid center contaminated the physical fallback");
    return {{"case","nonfinite_center"},{"passed",true}};
}
Json retire_incumbent_only_resources() {
    auto d=fixture();
    for(int z=0;z<d.n_levels;++z)d.slot_cost(2,z)=5+z;
    const auto paths=enumerate_paths(d);const DesignLayout layout(d);
    std::vector<int> old_service,old_empty;
    for(std::size_t i=0;i<paths.service_paths.size();++i) {
        const auto& p=paths.service_paths[i];std::set<int> lockers;
        for(const auto& leg:p.legs)for(int node:leg)if(node>=d.n_l)lockers.insert(node-d.n_l);
        const bool old_head=p.first_od==0 && p.b==BasePair{0,1} &&
            lockers.count(2) && !lockers.count(1);
        const bool stable_head=p.first_od==1 && p.b==BasePair{0,0} &&
            lockers.count(1) && !lockers.count(2);
        if(old_head || stable_head)old_service.push_back(static_cast<int>(i));
    }
    for(std::size_t i=0;i<paths.empty_paths.size();++i)
        if(paths.empty_paths[i].b==BasePair{1,0})old_empty.push_back(static_cast<int>(i));
    SolverOptions options;options.seconds=10;options.gap=1e-8;
    options.threads=1;options.neighborhood_cuts=false;
    const auto old=solve_restricted_heuristic(d,paths,options,wall_time()+10,nullptr,-1,&old_service,&old_empty);
    const auto& incumbent=old.at("solution");
    const auto center=solution_design(d,incumbent);
    require(old.at("validated") && incumbent.at("served").get<double>()>1 &&
        center[layout.facility+d.n_l+2]==1 && center[layout.assignment_index(0,0)]==1,
        "Retirement witness did not use its old exclusive locker");
    require(!old_empty.empty() && incumbent.at("_solution").at("empty_flow").at(old_empty[0]).get<double>()>1e-4,
        "Retirement witness did not use its connected return flight");
    PrimalPortfolioOptions settings;settings.max_heads=3;settings.max_lockers=3;
    auto guide=center;
    for(int q=0;q<d.task_count();++q)if(d.task_first(q)==0 && d.task_rate(q)>0)
        for(int l=0;l<d.n_l;++l)guide[layout.assignment_index(q,l)]=1-center[layout.assignment_index(q,l)];
    const auto repaired=solve_primal_coupled_neighborhood(d,paths,options,wall_time()+10,
        incumbent,&guide,0,settings);
    const auto moved=solution_design(d,repaired.at("solution"));
    require(repaired.at("validated") && moved[layout.facility+d.n_l+2]==0,
        "Coupled repair could not retire an incumbent-only locker of its released head");
    require(moved[layout.airline+d.n_nodes()]==0,
        "Coupled repair could not retire the connected incumbent return airline");
    require(repaired.at("objective").get<double>()+4<incumbent.at("objective").get<double>(),
        "Retiring the old resource block did not realize its fixed-cost saving");
    for(int l=0;l<d.n_l;++l)
        require(moved[layout.assignment_index(1,l)]==center[layout.assignment_index(1,l)],
            "Resource retirement widened the stable head assignment block");
    require(repaired.at("diagnostics").at("candidate_validation").at("physical").at("passed"),
        "Resource retirement bypassed original physical validation");
    return {{"case","retire_incumbent_only_resources"},{"incumbent",incumbent.at("objective")},
        {"objective",repaired.at("objective")},{"passed",true}};
}
Json missing_branches_and_invalid_incumbent() {
    auto missing=fixture();missing.deadline[0]=1e-6;
    const auto incomplete_groups=enumerate_paths(missing);
    SolverOptions options;options.seconds=5;options.neighborhood_cuts=false;
    const auto seed=solve_primal_seed_portfolio(missing,incomplete_groups,options,wall_time()+5);
    require(seed.at("objective").get<double>()<220 && !seed.at("attempts").empty(),
        "Unsupported sibling tasks incorrectly blocked a supported independent task");
    for(const auto& attempt:seed.at("attempts"))for(int id:attempt.at("selection").at("service_indices").get<std::vector<int>>()) {
        const auto& path=incomplete_groups.service_paths.at(id);
        require(path.first_od!=0 && path.second_od!=0,"Deadline-infeasible task entered a primal proposal");
    }
    const auto d=fixture();auto paths=enumerate_paths(d);
    auto invalid=all_courier_result(d,paths);invalid["_solution"]["courier"][0]=-1;
    const auto checked=solve_primal_coupled_neighborhood(d,paths,options,wall_time()+5,invalid);
    require(checked.at("status")=="coupled_polish_requires_valid_incumbent" &&
        checked.at("objective").get<double>()==220 && !checked.contains("coupled_neighborhood"),
        "Unvalidated physical incumbent supplied local fixings");
    paths.complete=false;
    const auto skipped=solve_primal_seed_portfolio(d,paths,options,wall_time()+5);
    require(skipped.at("status")=="unsupported_scope" && skipped.at("attempts").empty(),
        "Complete-catalogue seed search accepted an incomplete path domain");
    options.seconds=0;
    const auto zero=solve_primal_coupled_neighborhood(d,paths,options,wall_time(),invalid);
    require(zero.at("attempts").empty() && zero.at("objective").get<double>()==220,
        "Zero-budget coupled search accepted invalid incumbent data");
    return {{"case","missing_branches_and_invalid_incumbent"},{"passed",true}};
}
}
int main() {
    try {
        std::cout<<Json({{"passed",true},{"cases",Json::array({zero_budget(),complete_catalogue_seed(),
            coupled_resource_repair(),nonfinite_center(),missing_branches_and_invalid_incumbent(),
            retire_incumbent_only_resources()})}}).dump(2)<<'\n';
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
