#include "mlnd/primal_support.hpp"
#include "mlnd/output.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
template<class T> Matrix<T> fill(int rows,int cols,T value) {
    Matrix<T> result(rows,cols);std::fill(result.values.begin(),result.values.end(),value);return result;
}
Input fixture() {
    Input d;d.name="primal_neighborhood_domain";d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=2;
    d.max_orders=2;d.preprocessed_demand=true;d.flow_policy="aggregate";d.route_mode="direct";
    d.demand={1,1};d.ground_cost={6,6};d.single_rates={1,1};d.head_rates={1,1};
    d.pair_rates=fill(2,2,0.);d.common_origin_support=fill(2,2,1);d.base_cost={1,1};
    d.slot_cost=fill(2,2,0.);d.link_cost=fill(4,4,0.);d.allowed_links=fill(4,4,1);
    d.slot_counts=fill(2,2,10);d.occupancy_limit=fill(2,2,8.);return d;
}
void append(PathCollection& paths,int head,int base,int locker,double cost=0,int branch=-1) {
    ServicePath p;p.id=static_cast<int>(paths.service_paths.size());p.first_od=head;p.second_od=branch;
    p.b={base,base};p.pickups={locker};p.deliveries={locker};p.legs={{base,2+locker,base}};
    p.served_counts={{head,1}};p.slot_times={{locker,.1+cost}};
    p.arc_counts={{base,2+locker,1},{2+locker,base,1}};p.cost=cost;
    paths.service_paths.push_back(std::move(p));
}
Json completion() {
    auto d=fixture();PathCollection paths;
    // Both cheap paths dominate the surviving expensive route in every
    // ranking stored by the bank; closing locker 0 removes that shortlist.
    append(paths,0,0,0);append(paths,0,0,0,.1);append(paths,0,0,1,10);
    PrimalSupportBank bank(d,paths);bank.extend(wall_time()+1);
    auto state=bank.inspect(all_courier_result(d,paths),wall_time()+1);
    PrimalBlockRequest request;request.heads={0};request.allowed_bases={0};
    request.restrictions.closed_lockers={0};request.complete_missing_support=true;
    auto block=bank.close(state,request,wall_time()+1);
    require(block.ready,"On-demand completion discarded a usable neighborhood");
    require(block.service_indices==std::vector<int>{2},"Completion did not find the route outside the shortlist");
    require(block.restrictions.forbidden_origins.empty(),"A completed origin was still forbidden");
    require(block.diagnostics.at("support_completion").at("completed_branches")==1,
            "Completion metadata did not record the recovered branch");
    require(block.coupled.heads==std::vector<int>{0},"Completion released an unrelated fixed head");
    // With the feature off the historical shortlist behavior is unchanged.
    request.complete_missing_support=false;auto legacy=bank.close(state,request,wall_time()+1);
    require(!legacy.ready && legacy.status=="missing_branch","Disabled completion changed the old domain policy");
    // An unindexed catalogue is a partial bank, not proof of absent support.
    PrimalSupportBank partial(d,paths);auto zero=partial.inspect(all_courier_result(d,paths),wall_time()+1);
    request.complete_missing_support=true;auto completed=partial.close(zero,request,wall_time()+1);
    require(completed.ready && completed.service_indices==std::vector<int>{2},"Partial-bank completion failed");
    return {{"case","missing_shortlist_witness_and_partial_bank"},{"passed",true}};
}
Json local_origin_closure() {
    auto d=fixture();d.pair_rates(0,1)=.25;PathCollection paths;
    append(paths,0,0,0);append(paths,0,1,1);append(paths,0,1,1,0,1);
    PrimalSupportBank bank(d,paths);bank.extend(wall_time()+1);
    auto zero=all_courier_result(d,paths);auto state=bank.inspect(zero,wall_time()+1);
    const int pair=d.task_index(0,1);
    PrimalBlockRequest request;request.heads={pair};request.complete_missing_support=true;
    auto block=bank.close(state,request,wall_time()+1);
    require(block.ready,"One unsupported origin discarded the other origin");
    require(block.restrictions.forbidden_origins==std::vector<std::pair<int,int>>{{pair,0}},
            "Support loss was not confined to the affected origin");
    require(block.service_indices==std::vector<int>({2}),"Complete alternative origin lost a task branch");
    require(block.coupled.heads==std::vector<int>{pair},"Local support closure changed the fixed region");
    require(!block.diagnostics.at("support_completion").at("global_inference").get<bool>(),
            "Local catalogue loss claimed a global inference");
    paths.complete=false;
    auto incomplete=bank.close(state,request,wall_time()+1);
    require(!incomplete.diagnostics.at("support_completion").at("scan_domain_complete").get<bool>() &&
            incomplete.diagnostics.at("support_completion").at("unknown_support").get<bool>(),
            "Complete task-row support was confused with a complete physical path universe");
    // A feasible zero-service head may carry a redundant assignment. The
    // support API reads incidence only; no fabricated objective is certified.
    zero["_solution"]["assignment"][pair][0]=1;
    auto incumbent=bank.inspect(zero,wall_time()+1);auto retained=bank.close(incumbent,request,wall_time()+1);
    require(retained.ready && retained.restrictions.forbidden_origins.empty(),
            "Intensification excluded an incumbent zero-throughput origin");
    require(retained.diagnostics.at("support_completion").at("retained_incumbent_origins_without_full_support").size()==1,
            "Retained unsupported incumbent origin was not diagnosed");
    // Positive flows remain registered even when forced closure sets them to
    // zero, and every head using the closed resource is released together.
    zero["_solution"]["flow"][0]=.1;append(paths,1,1,0);
    zero["_solution"]["flow"].push_back(.1);zero["_solution"]["assignment"][1][1]=1;
    auto dependent=bank.inspect(zero,wall_time()+1);request.restrictions.closed_lockers={0};
    auto closed=bank.close(dependent,request,wall_time()+1);
    require(closed.ready && closed.coupled.heads==std::vector<int>({0,1,pair}),"Closed asset stranded a fixed dependent head");
    for(int id:{0,3})require(std::find(closed.service_indices.begin(),closed.service_indices.end(),id)!=closed.service_indices.end(),
            "Forced closure discarded incumbent positive support");
    PrimalRepairRestrictions invalid;invalid.forbidden_origins={{1,0}};
    CoupledPrimalNeighborhood fixed_zone;fixed_zone.heads={0};SolverOptions options;
    bool rejected=false;
    try {solve_restricted_heuristic(d,paths,options,wall_time()-1,nullptr,-1,
        nullptr,nullptr,false,false,&fixed_zone,&invalid);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"An origin restriction silently modified a fixed head outside its neighborhood");
    return {{"case","local_origin_loss_incumbent_preservation_and_dependency_closure"},{"passed",true}};
}
std::vector<double> design(const Input& d,const std::array<int,4>& states) {
    const DesignLayout layout(d);std::vector<double> x(layout.count,0);
    for(int r=0;r<2;++r)if(states[r])x[layout.assignment_index(r,states[r]-1)]=1;
    for(int h=0;h<2;++h)if(states[2+h])x[layout.capacity+h*d.n_levels+states[2+h]-1]=1;
    return x;
}
bool satisfies(const RowRecord& row,const std::vector<double>& x) {
    double value=0;for(auto [j,a]:row.coefficients)value+=a*x[j];return value<=row.rhs+1e-10;
}
Json group_distances() {
    const auto d=fixture();std::size_t checked=0;
    for(int c=0;c<81;++c) {
        std::array<int,4> center_states{};int code=c;
        for(auto& s:center_states){s=code%3;code/=3;}
        const auto center=design(d,center_states);Formulation f(d);
        const auto report=add_grouped_primal_radius(d,f,center,{0,1,1},{0,1},1,1);
        require(f.matrix.rows.size()==2,"Grouped neighborhood did not add exactly two separate radii");
        require(!report.at("base_facilities_counted").get<bool>(),"Base opening entered a categorical distance");
        for(int p=0;p<81;++p) {
            std::array<int,4> states{};code=p;for(auto& s:states){s=code%3;code/=3;}
            auto x=design(d,states);
            // All infrastructure bits may change without an additional cost
            // in either group metric. Original physical rows still apply.
            for(int j=0;j<d.n_nodes();++j)x[j]=1;
            for(int j=f.layout.airline;j<f.layout.count;++j)x[j]=1;
            const int heads=(states[0]!=center_states[0])+(states[1]!=center_states[1]);
            const int caps=(states[2]!=center_states[2])+(states[3]!=center_states[3]);
            require(satisfies(f.matrix.rows[0],x)==(heads<=1),"Task migration does not count exactly once");
            require(satisfies(f.matrix.rows[1],x)==(caps<=1),"Capacity transition does not count exactly once");
            ++checked;
        }
    }
    Formulation f(d);const auto center=design(d,{1,2,1,2});
    add_grouped_primal_radius(d,f,center,{0},{0},0,0);
    auto fixed_other=design(d,{1,0,1,0});
    require(satisfies(f.matrix.rows[0],fixed_other) && satisfies(f.matrix.rows[1],fixed_other),
            "Grouped radius unexpectedly included groups outside the released domain");
    auto changed=design(d,{2,2,2,2});
    require(!satisfies(f.matrix.rows[0],changed) && !satisfies(f.matrix.rows[1],changed),
            "Zero group radius admitted a changed task or capacity state");
    return {{"case","exhaustive_categorical_distance"},{"center_candidate_pairs",checked},{"passed",true}};
}
}
int main() {try {
    Json cases=Json::array({completion(),local_origin_closure(),group_distances()});
    std::cout<<Json({{"passed",true},{"cases",cases},{"solver_called",false}}).dump(2)<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
