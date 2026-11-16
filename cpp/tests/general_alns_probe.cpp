#include "mlnd/primal_alns.hpp"
#include "mlnd/primal_support.hpp"
#include "mlnd/output.hpp"
#include "mlnd/solver_presets.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace mlnd;
namespace {
void require(bool condition,const std::string& message) {
    if(!condition)throw std::runtime_error(message);
}
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> out(rows,cols);std::fill(out.values.begin(),out.values.end(),value);return out;
}
Input fixture(int bases,int k) {
    Input d;d.name="general_alns_last_base";d.model_version="consecutive";
    d.n_l=bases;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=k;
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand={1,1};d.ground_cost={40,40};d.deadline={1,1};
    d.single_rates=k==1?std::vector<double>{1,1}:std::vector<double>{.5,.5};
    d.head_rates=k==1?std::vector<double>{1,1}:std::vector<double>{1,.5};
    d.pair_rates=filled(2,2,0.);if(k==2)d.pair_rates(0,1)=.5;
    d.common_origin_support=filled(2,bases,1);
    const int nodes=d.n_nodes();d.flight_times=filled(nodes,nodes,.04);d.allowed_links=filled(nodes,nodes,1);
    for(int i=0;i<nodes;++i){d.flight_times(i,i)=0;d.allowed_links(i,i)=0;}
    d.base_cost.assign(bases,.01);d.link_cost=filled(nodes,nodes,.001);d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,2,.05);d.first_cost=filled(2,2,.1);d.terminal_cost=filled(2,2,.1);
    d.receipt_mean=filled(2,2,.03);d.home_receipt_mean=filled(2,2,.06);d.removal_mean=filled(2,2,.02);
    d.slot_counts=filled(2,1,20);d.slot_cost=filled(2,1,.01);d.occupancy_limit=filled(2,1,15.);
    d.kappa=filled(2,2,1.);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=.1;d.flight_cost=.1;d.waiting_cost=.1;return d;
}
PathCollection single_base_catalogue(const Input& d,int base) {
    auto paths=enumerate_paths(d);
    paths.service_paths.erase(std::remove_if(paths.service_paths.begin(),paths.service_paths.end(),[&](const auto& p) {
        return p.b[0]!=base || p.b[1]!=base;
    }),paths.service_paths.end());
    for(std::size_t i=0;i<paths.service_paths.size();++i)paths.service_paths[i].id=static_cast<int>(i);
    paths.complete=false; // Generated witnesses do not certify absent physical paths.
    return paths;
}
Json support_case(int bases,int k) {
    auto d=fixture(bases,k);auto paths=single_base_catalogue(d,bases-1);
    const std::string label="L"+std::to_string(bases)+" K"+std::to_string(k);
    PrimalSupportBank bank(d,paths);bank.extend(wall_time()+2,true);
    require(bank.metadata().at("supported"),label+": support bank rejected the independent-task scope");
    auto baseline=all_courier_result(d,paths);auto state=bank.inspect(baseline,wall_time()+2);
    require(state.ready && state.head_state.size()==static_cast<std::size_t>(k==1?2:6),label+": task-state shape");
    std::mt19937 random(43);
    auto request=bank.request(0,state,8,random,wall_time()+2);
    std::vector<int> all_bases(bases);std::iota(all_bases.begin(),all_bases.end(),0);
    require(request.allowed_bases==all_bases,label+": regular search omits a base");
    request.complete_missing_support=true;request.indexed_support_completion=true;
    auto block=bank.close(state,request,wall_time()+2);
    require(block.ready && !block.service_indices.empty(),label+": support closure lost last-base paths");
    for(int id:block.service_indices)require(paths.service_paths[id].b[0]==bases-1,label+": wrong witness origin");
    for(auto forbidden:block.restrictions.forbidden_origins)
        require(forbidden.second!=bases-1,label+": local completion forbids the only supported base");
    for(int direction=0;direction<3;++direction) {
        auto seed=bank.seed(direction,state,random,wall_time()+2);
        for(int base:seed.allowed_bases)require(base>=0 && base<bases,label+": seed base is out of range");
        if(direction==2) {
            require(seed.allowed_bases==all_bases,label+": mixed seed omits bases");
            require(!seed.heads.empty(),label+": mixed seed cannot use last-base support");
        }
    }
    PrimalBlockRequest default_request;default_request.heads=request.heads;
    default_request.complete_missing_support=true;default_request.indexed_support_completion=true;
    auto defaults=bank.close(state,default_request,wall_time()+2);
    require(defaults.ready && !defaults.service_indices.empty(),label+": default allowed bases omit valid support");
    auto feedback=make_primal_cg_feedback(d,paths,solution_design(d,baseline),{{0,-1.}},{});
    auto accepted=bank.observe_cg_columns(feedback,wall_time()+2);
    require(accepted.at("accepted")==1 && !accepted.at("certificate_eligible").get<bool>(),label+": CG ranking feedback rejected");
    return {{"L",bases},{"K",k},{"support_passed",true},{"last_base",bases-1}};
}
void configuration_case(int bases,int k) {
    SolverConfiguration config;config.apply_method("benders_cg");
    const auto original=config.metadata();config.adapt_instance(fixture(bases,k));config.validate();
    const auto adapted=config.metadata();
    for(const char* flag:{"primal_alns","primal_support_completion","indexed_support_completion","cg_primal_feedback"})
        require(adapted.at("flags").at(flag),"Instance adaptation disabled "+std::string(flag));
    require(adapted.at("flags")==original.at("flags"),"Instance adaptation changed unrelated accelerations");
    for(const char* flag:{"catalogue_pricing","cache_pricing_graph","path_support_cuts",
                         "fractional_path_support","airline_resource_cuts","airline_resource_balance"})
        require(!adapted.at("flags").at(flag).get<bool>(),"Catalogue-dependent acceleration was enabled");
}
Json search_case(int bases,int k,int service_base) {
    auto d=fixture(bases,k);auto paths=single_base_catalogue(d,service_base);
    SolverConfiguration config;config.apply_method("benders_cg");
    PrimalAlnsSearch search(d,paths,config.options,all_courier_result(d,paths),10);
    require(search.summary().at("supported"),"ALNS search rejects L"+std::to_string(bases)+" K"+std::to_string(k));
    const auto feedback=make_primal_cg_feedback(d,paths,solution_design(d,search.best()),{{0,-1.}},{});
    const auto observed=search.observe_cg_columns(feedback,wall_time()+1);
    require(observed.at("accepted")==1 && !observed.at("certificate_eligible").get<bool>(),
        "ALNS search cannot use ranking-only CG feedback");
    search.startup(wall_time()+8);
    const auto& result=search.best();
    require(result.value("native_model_validated",false) && result.at("objective").get<double>()<80-1e-6,
        "L"+std::to_string(bases)+" K"+std::to_string(k)+" base "+std::to_string(service_base)+
        ": ALNS startup did not produce a physically validated service improvement: "+search.records().dump());
    const auto& assignment=result.at("_solution").at("assignment");
    int served_tasks=0;
    for(int task=0;task<d.task_count();++task)if(d.task_rate(task)>0) {
        if(assignment.at(task).at(service_base).get<double>()>.5)++served_tasks;
        for(int base=0;base<bases;++base)if(base!=service_base)
            require(assignment.at(task).at(base).get<double>()<1e-8,"ALNS assigned an unsupported base");
    }
    require(served_tasks>0,"ALNS repair did not activate the only supported base");
    if(k==2)require(assignment.at(d.task_index(0,1)).at(service_base).get<double>()>.5,
        "ALNS repair did not activate the positive pair task at the supported base");
    require(!result.contains("bound") && !search.summary().at("global_lower_bound_available").get<bool>(),
        "ALNS search leaked a restricted lower bound");
    return {{"L",bases},{"K",k},{"objective",result.at("objective")},
        {"service_base",service_base},{"repair_calls",search.summary().at("repair_calls")},{"passed",true}};
}
void indexed_completion_case(int bases,int k) {
    auto d=fixture(bases,k);d.slot_cost(1,0)=10;
    const int task=k==1?d.task_index(0):d.task_index(0,1);
    PathCollection paths;paths.complete=false;
    auto append=[&](int target,int locker,double cost) {
        ServicePath path;path.id=static_cast<int>(paths.service_paths.size());path.cost=cost;
        path.first_od=d.task_first(target);path.second_od=d.task_second(target);
        path.b={bases-1,bases-1};path.pickups={locker};path.deliveries={locker};
        path.legs={{bases-1,bases+locker,bases-1}};path.slot_times={{locker,.1+cost}};
        path.arc_counts={{bases-1,bases+locker,1},{bases+locker,bases-1,1}};
        paths.service_paths.push_back(std::move(path));
    };
    // Synthetic registered incidence exercises only support indexing/ranking.
    // No physical feasibility or objective claim is made for these witnesses;
    // search_case above separately exercises real enumerated paths and repair.
    for(int i=0;i<10;++i)append(d.task_index(1),0,i);
    append(task,0,0);append(task,0,.1);append(task,1,20);
    PrimalSupportBank bank(d,paths);bank.extend(wall_time()+2,true);
    const auto state=bank.inspect(all_courier_result(d,paths),wall_time()+1);
    PrimalBlockRequest request;request.heads={task};request.allowed_bases={bases-1};
    request.paths_per_branch=1;request.complete_missing_support=true;request.indexed_support_completion=true;
    request.restrictions.closed_lockers={0};
    const auto completed=bank.close(state,request,wall_time()+1);
    require(completed.ready && completed.service_indices==std::vector<int>{12},
        "Indexed completion did not rescue the unranked admissible witness");
    require(completed.diagnostics.at("support_completion").at("catalogue_paths_scanned")==3,
        "Indexed completion scanned unrelated task branches");
    request.restrictions.closed_lockers.clear();
    auto feedback=make_primal_cg_feedback(d,paths,state.design,{{12,-1.}},{});
    bank.observe_cg_columns(feedback,wall_time()+1);
    const auto informed=bank.close(state,request,wall_time()+1);
    require(informed.ready && informed.service_indices==std::vector<int>({10,12}),
        "CG feedback failed to add an unranked later-base witness without replacing the cheap path");
    require(!completed.diagnostics.at("support_completion").at("global_inference").get<bool>(),
        "Registered-pool completion claimed a physical-domain certificate");
}
}
int main() {try {
    Json cases=Json::array(),repairs=Json::array();
    for(int bases:{1,2,3,5})for(int k:{1,2}) {
        cases.push_back(support_case(bases,k));configuration_case(bases,k);indexed_completion_case(bases,k);
        repairs.push_back(search_case(bases,k,bases-1));
    }
    for(int k:{1,2})repairs.push_back(search_case(5,k,2));
    std::cout<<Json({{"passed",true},{"cases",cases},{"repairs",repairs}}).dump(2)<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
