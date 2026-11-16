#include "mlnd/primal_support.hpp"
#include "mlnd/output.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
template<class T> Matrix<T> fill(int rows,int cols,T value) {Matrix<T> m(rows,cols);std::fill(m.values.begin(),m.values.end(),value);return m;}
Input fixture() {
    Input d;d.name="alns_support_joint_fixed_cost";d.n_l=2;d.n_h=8;d.n_od=2;d.n_levels=2;
    d.max_orders=2;d.preprocessed_demand=true;d.flow_policy="aggregate";d.route_mode="direct";
    d.demand={1,1};d.ground_cost={6,6};d.single_rates={1,1};d.head_rates={1,1};
    d.pair_rates=fill(2,2,0.);d.common_origin_support=fill(2,2,1);d.base_cost={10,10};
    d.slot_cost=fill(8,2,0.);d.link_cost=fill(10,10,0.);d.allowed_links=fill(10,10,1);
    d.slot_counts=fill(8,2,10);d.occupancy_limit=fill(8,2,8.);return d;
}
PathCollection catalogue(const Input& d) {
    PathCollection paths;
    for(int r=0;r<2;++r)for(int base=0;base<2;++base) {
        ServicePath p;p.id=static_cast<int>(paths.service_paths.size());p.first_od=r;p.b={base,base};
        p.second_od=-1;p.pickups={0};p.deliveries={7};p.legs={{base,2,3,4,5,6,7,8,9,base}};
        p.served_counts={{r,1}};for(int h=0;h<8;++h)p.slot_times.emplace_back(h,.1);
        for(std::size_t j=1;j<p.legs.front().size();++j)p.arc_counts.emplace_back(p.legs.front()[j-1],p.legs.front()[j],1);
        paths.service_paths.push_back(p);
    }
    (void)d;return paths;
}
Json joint_activation() {
    const auto d=fixture();const auto paths=catalogue(d);PrimalSupportBank bank(d,paths);bank.extend(wall_time()+1);
    const auto state=bank.inspect(all_courier_result(d,paths),wall_time()+1);
    std::mt19937 rng(43);const auto req=bank.seed(0,state,rng,wall_time()+1);
    const auto block=bank.close(state,req,wall_time()+1);
    // One head saves 6 and pays 10 (-4). Two sharing the same base save 12
    // and pay 10 (+2). A positive-single-head gate would return an empty set.
    require(req.heads.size()==2,"Joint activation rejected negative intermediate head savings");
    require(block.ready,"Joint seed lacks a complete resource block");
    require(block.coupled.lockers.size()==8,"Support closure retained the legacy six-locker cap");
    require(block.restrictions.change_heads.size()==2,"All-courier seed did not force a head-group change");
    require(block.diagnostics.at("all_positive_branches_preserved"),"Joint seed lost branch coverage");
    return {{"case","joint_activation_without_single_head_gate"},{"passed",true}};
}
Json missing_branch_and_partial_bank() {
    auto d=fixture();d.pair_rates(0,1)=.25;const auto paths=catalogue(d);PrimalSupportBank bank(d,paths);
    bank.extend(wall_time()-1);require(bank.metadata().at("partial"),"Unscanned bank was called complete");
    bank.extend(wall_time()+1);const auto state=bank.inspect(all_courier_result(d,paths),wall_time()+1);
    PrimalBlockRequest request;request.heads={d.task_index(0,1)};request.allowed_bases={0,1};
    const auto block=bank.close(state,request,wall_time()+1);
    require(!block.ready && block.status=="missing_branch","Missing positive pair branch was accepted as complete");
    require(!block.diagnostics.at("all_positive_branches_preserved").get<bool>(),"Missing support was mislabeled covered");
    return {{"case","missing_branch_and_partial_support_are_unknown"},{"passed",true}};
}
Json closure_and_cross_base() {
    const auto d=fixture();auto paths=catalogue(d);
    for(int i=0;i<4;++i) {auto p=paths.service_paths[i];p.id=static_cast<int>(paths.service_paths.size());
        p.legs={{p.b[0],3,4,5,6,7,8,9,p.b[1]}};p.slot_times.erase(p.slot_times.begin());p.arc_counts.clear();
        for(std::size_t j=1;j<p.legs[0].size();++j)p.arc_counts.emplace_back(p.legs[0][j-1],p.legs[0][j],1);
        p.pickups={1};paths.service_paths.push_back(p);}
    PrimalSupportBank bank(d,paths);bank.extend(wall_time()+1);
    // Synthetic incidence-only witness. The public support API never treats
    // this view as a physically validated objective or returns an upper bound.
    auto encoded=all_courier_result(d,paths);encoded["_solution"]["flow"][0]=1.;encoded["_solution"]["flow"][2]=1.;
    encoded["_solution"]["assignment"][0][0]=1;encoded["_solution"]["assignment"][1][0]=1;
    const auto state=bank.inspect(encoded,wall_time()+1);require(state.ready,"Incidence witness was not readable");
    PrimalBlockRequest request;request.heads={0};request.allowed_bases={0,1};request.restrictions.closed_lockers={0};
    const auto block=bank.close(state,request,wall_time()+1);
    require(block.ready,"Alternative branch supports failed closed-asset closure");
    require(block.coupled.heads==std::vector<int>({0,1}),"Closure omitted a second head sharing the closed locker");
    for(int id:{0,2})require(std::find(block.service_indices.begin(),block.service_indices.end(),id)!=block.service_indices.end(),"Mandatory old positive path was dropped");
    require(block.coupled.lockers.size()==8,"Closure omitted an old incumbent resource");
    auto cross=catalogue(d);for(auto& p:cross.service_paths)p.b[1]=1-p.b[0];
    PrimalSupportBank cross_bank(d,cross);cross_bank.extend(wall_time()+1);
    const auto zero=cross_bank.inspect(all_courier_result(d,cross),wall_time()+1);request.restrictions={};
    const auto cross_block=cross_bank.close(zero,request,wall_time()+1);
    require(cross_block.ready && cross_block.empty_indices.empty(),"Cross-base witness was rejected without paired empty relocation");
    return {{"case","resource_dependency_closure_and_service_balance"},{"passed",true}};
}
Json reject_changed_service_identity() {
    const auto d=fixture();auto paths=catalogue(d);PrimalSupportBank bank(d,paths);bank.extend(wall_time()+1);
    const auto state=bank.inspect(all_courier_result(d,paths),wall_time()+1);paths.service_paths[0].cost=1;
    PrimalBlockRequest request;request.heads={0};request.allowed_bases={0};
    bool rejected=false;try {bank.close(state,request,wall_time()+1);}catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"Changed selected canonical path reused a stale bank identity");
    return {{"case","changed_service_identity_is_rejected_on_use"},{"passed",true}};
}
Json reject_changed_empty_identity() {
    const auto d=fixture();auto paths=catalogue(d);EmptyPath empty;empty.id=0;empty.b={0,1};empty.cost=1;empty.arc_counts={{0,1,1}};paths.empty_paths.push_back(empty);
    PrimalSupportBank bank(d,paths);bank.extend(wall_time()+1);paths.empty_paths[0].cost=2;
    bool rejected=false;try {bank.extend(wall_time()+1);}catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"Changed canonical empty path reused a stale bank identity");
    return {{"case","changed_empty_identity_is_rejected"},{"passed",true}};
}
}
int main() {try {Json cases=Json::array();cases.push_back(joint_activation());cases.push_back(missing_branch_and_partial_bank());cases.push_back(closure_and_cross_base());cases.push_back(reject_changed_empty_identity());cases.push_back(reject_changed_service_identity());std::cout<<Json({{"passed",true},{"cases",cases}}).dump(2)<<'\n';return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
