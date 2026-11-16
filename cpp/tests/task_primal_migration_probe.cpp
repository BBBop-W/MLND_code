#include "mlnd/primal_support.hpp"
#include "mlnd/output.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
template<class T> Matrix<T> filled(int rows,int cols,T value) {
    Matrix<T> out(rows,cols);std::fill(out.values.begin(),out.values.end(),value);return out;
}
Input fixture() {
    Input d;d.name="independent_task_primal";d.model_version="consecutive";
    d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.route_mode="direct";d.flow_policy="aggregate";d.preprocessed_demand=true;
    d.demand={1,1};d.ground_cost={40,40};d.deadline={1,1};
    d.single_rates={.5,.5};d.head_rates={1,.5};d.pair_rates=filled(2,2,0.);
    d.pair_rates(0,1)=.5;d.common_origin_support=filled(2,2,1);
    d.flight_times=filled(4,4,.04);d.allowed_links=filled(4,4,1);
    for(int i=0;i<4;++i){d.flight_times(i,i)=0;d.allowed_links(i,i)=0;}
    d.base_cost={.01,.01};d.link_cost=filled(4,4,.001);d.slot_epsilon={.05,.05};
    d.rider_mean=filled(2,2,.05);d.first_cost=filled(2,2,.1);
    d.terminal_cost=filled(2,2,.1);d.receipt_mean=filled(2,2,.03);
    d.home_receipt_mean=filled(2,2,.06);d.removal_mean=filled(2,2,.02);
    d.slot_counts=filled(2,1,20);d.slot_cost=filled(2,1,.01);d.occupancy_limit=filled(2,1,15.);
    d.kappa=filled(2,2,1.);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=.1;d.flight_cost=.1;d.waiting_cost=.1;return d;
}
Json independent_support() {
    auto pending=fixture();pending.single_rates.clear();pending.head_rates.clear();pending.pair_rates={};
    PathCollection no_paths;
    const auto early=all_courier_result(pending,no_paths);
    require(early.at("objective")==80 && !early.at("task_rates_available").get<bool>(),
        "Before-preprocessing courier checkpoint was lost");
    require(early.at("task_table").at(0).at("rate").is_null(),
        "Before-preprocessing checkpoint invented task demand");
    auto d=fixture();auto paths=enumerate_paths(d);
    const int single=d.task_index(0),pair=d.task_index(0,1);
    // Restrict the registered witnesses so siblings have disjoint origins.
    paths.service_paths.erase(std::remove_if(paths.service_paths.begin(),paths.service_paths.end(),[&](const auto& p) {
        const int q=d.task_index(p.first_od,p.second_od);
        return !((q==single && p.b[0]==0)||(q==pair && p.b[0]==1));
    }),paths.service_paths.end());
    for(std::size_t i=0;i<paths.service_paths.size();++i)paths.service_paths[i].id=static_cast<int>(i);
    paths.complete=false;
    d.common_origin_support(0,0)=0; // Reconstruction metadata is not an execution constraint.
    PrimalSupportBank bank(d,paths);bank.extend(wall_time()+2,true);
    auto ground=all_courier_result(d,paths);auto state=bank.inspect(ground,wall_time()+2);
    require(state.ready && state.head_state.size()==static_cast<std::size_t>(d.task_count()),"Task support shape is wrong");
    PrimalBlockRequest request;request.heads={single};request.allowed_bases={0};
    auto alone=bank.close(state,request,wall_time()+2);
    require(alone.ready && !alone.service_indices.empty(),"A supported single task depends on its pair sibling");
    for(int id:alone.service_indices)require(d.task_index(paths.service_paths[id].first_od,paths.service_paths[id].second_od)==single,"Single repair released a sibling task");
    request.heads={single,pair};request.allowed_bases={0,1};request.complete_missing_support=true;
    request.indexed_support_completion=true;
    auto together=bank.close(state,request,wall_time()+2);
    require(together.ready,"Independent-origin block was rejected");
    const std::vector<std::pair<int,int>> expected={{single,1},{pair,0}};
    require(together.restrictions.forbidden_origins==expected,"Support closure confused sibling task origins");
    require(together.diagnostics.at("support_completion").at("unknown_support"),"Partial support pool claimed a global absence certificate");
    auto design=solution_design(d,ground);
    Formulation f(d);design[f.layout.assignment_index(pair,0)]=1;
    add_grouped_primal_radius(d,f,design,{pair},{},1,0);
    auto moved=design;moved[f.layout.assignment_index(pair,0)]=0;moved[f.layout.assignment_index(pair,1)]=1;
    double value=0;for(auto [j,a]:f.matrix.rows.front().coefficients)value+=a*moved[j];
    require(value<=f.matrix.rows.front().rhs+1e-10,"Pair-task origin switch counted twice");
    require(ground.at("_solution").at("assignment").size()==static_cast<std::size_t>(d.task_count()),"Exported task assignment has OD shape");
    auto old=ground;old["_solution"]["assignment_scope"]="origin";
    bool rejected=false;try{solution_design(d,old);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Old coupled-head warm start was silently reinterpreted");
    auto stale=ground;stale["_solution"]["task_table"][pair]["rate"]=.6;
    rejected=false;try{solution_design(d,stale);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Mismatched task-rate identity was accepted");
    return {{"case","independent_support_and_encoding"},{"passed",true}};
}
Json physical_repair() {
    auto d=fixture();auto paths=enumerate_paths(d);
    auto incumbent=all_courier_result(d,paths);const int single=d.task_index(0),pair=d.task_index(0,1);
    CoupledPrimalNeighborhood block;
    for(int q=0;q<d.task_count();++q)if(d.task_rate(q)>0)block.heads.push_back(q);
    block.bases={0,1};block.lockers={0,1};
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)if(d.allowed_links(u,v))block.airlines.emplace_back(u,v);
    PrimalRepairRestrictions restrictions;restrictions.forbidden_origins={{single,1},{pair,0}};
    SolverOptions options;options.seconds=10;options.gap=1e-7;options.threads=1;options.neighborhood_cuts=false;
    auto solved=solve_restricted_heuristic(d,paths,options,wall_time()+10,&incumbent,-1,
        nullptr,nullptr,false,false,&block,&restrictions);
    require(solved.value("candidate_validated",false),"Independent-task repair produced no validated candidate");
    const auto& result=solved.at("candidate");const auto& assignment=result.at("_solution").at("assignment");
    require(assignment.at(single).at(0).get<double>()>.5 && assignment.at(pair).at(1).get<double>()>.5,
        "Repair failed to assign a single and its sibling pair to different bases");
    require(solved.at("diagnostics").at("candidate_validation").at("physical").at("passed"),"Task repair bypassed physical validation");
    require(result.at("objective").get<double>()<20,"Task repair did not improve the courier fallback");
    const auto saved=solution_design(d,result);
    require(saved.size()==static_cast<std::size_t>(DesignLayout(d).count),"Independent-task incumbent failed round trip");
    require(!solved.contains("bound") && !solved.at("global_lower_bound_available").get<bool>(),"Restricted repair leaked a global bound");
    return {{"case","physically_valid_independent_assignment"},{"objective",result.at("objective")},{"passed",true}};
}
}
int main(){try {
    std::cout<<Json({{"passed",true},{"cases",Json::array({independent_support(),physical_repair()})}}).dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
