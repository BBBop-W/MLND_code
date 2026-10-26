#include "mlnd/formulation.hpp"
#include "mlnd/preprocessed_demand.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok,const std::string& why) { if(!ok) throw std::runtime_error(why); }
template<class T> Matrix<T> constant(int rows,int cols,T value) {
    Matrix<T> result(rows,cols);std::fill(result.values.begin(),result.values.end(),value);return result;
}
Input fixture() {
    Input d;d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=2;
    d.name="preprocessed_demand_witness";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";d.demand={4,7};d.ground_cost={5,5};d.deadline={1,1};
    d.flight_times=constant(4,4,0.0);d.flight_times.values={0,.03,.15,.17,.03,0,.16,.14,.15,.16,0,.1,.17,.14,.1,0};
    d.allowed_links=constant(4,4,1);for(int i=0;i<4;++i)d.allowed_links(i,i)=0;
    d.link_cost=constant(4,4,0.0);d.base_cost={.5,.4};d.slot_epsilon={.05,.05};
    d.rider_mean=constant(2,2,.08);d.first_cost=constant(2,2,1.0);d.terminal_cost=constant(2,2,.5);
    d.receipt_mean=constant(2,2,.08);d.removal_mean=constant(2,2,.03);
    d.slot_counts=constant(2,1,20);d.slot_cost=constant(2,1,.1);d.occupancy_limit=constant(2,1,14.0);
    d.kappa=constant(2,2,0.0);d.kappa(0,1)=1;
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=10;d.waiting_cost=.3;return d;
}
Json witness() {
    auto d=fixture();auto paths=enumerate_paths(d);
    const auto legacy=build_formulation(d,paths);
    require(legacy.layout.airline-legacy.layout.assignment==8,"Legacy K2 assignment changed");
    preprocess_generalized_demand(d,paths);
    require(d.pair_rates(0,1)>0,"Fixture needs an admitted pair");
    const double expected=4*(-std::expm1(-7*d.delta));
    require(std::abs(d.pair_rates(0,1)-expected)<1e-12,"Pair opportunity rate is incorrect");
    auto f=build_formulation(d,paths);
    require(f.layout.airline-f.layout.assignment==4,"Preprocessed assignment is not origin-only");
    std::map<std::string,int> var;
    for(std::size_t j=0;j<f.matrix.variables.size();++j) {
        const auto name=f.matrix.variables[j].name;var[name]=j;
        require(name.rfind("eta_",0)!=0&&name.rfind("u_",0)!=0&&name.rfind("Y_",0)!=0&&name.rfind("Z_",0)!=0,
                "Fixed demand retained a lifted variable");
    }
    require(f.lift_groups.size()==4,"Return bases were not merged in demand groups");
    std::vector<double> x(f.matrix.variables.size(),0);
    for(int n=0;n<d.n_nodes();++n)x[n]=1;
    for(int h=0;h<d.n_h;++h)x[f.layout.capacity+h]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[f.layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    const double rho=.1;
    x[f.layout.assignment_index(0,0)]=1;x[f.rho[0]]=rho;
    x[f.idle[0]]=rho/(1-rho);x[var.at("v_0_0")]=rho;
    std::vector<double> served(2),heads(f.lift_groups.size());
    double busy=0,return_one=0;
    for(int second:{-1,1})for(int destination=0;destination<2;++destination) {
        const auto p=std::find_if(paths.service_paths.begin(),paths.service_paths.end(),[&](const ServicePath& p) {
            return p.first_od==0&&p.second_od==second&&p.b==BasePair{0,destination};
        });
        require(p!=paths.service_paths.end(),"Missing witness path");
        const double flow=rho*(second<0?d.single_rates[0]:d.pair_rates(0,1))/2;
        x[f.flow[p->id]]=flow;x[f.head[p->segment_id]]+=flow;
        heads[f.segment_group[p->segment_id]]+=flow;busy+=flow*p->time;
        for(auto [r,count]:p->served_counts)served[r]+=flow*count;
        if(destination==1)return_one+=flow;
    }
    const auto empty=std::find_if(paths.empty_paths.begin(),paths.empty_paths.end(),[](const auto& p){return p.b==BasePair{1,0};});
    require(empty!=paths.empty_paths.end(),"Missing empty return");
    x[f.empty_flow[empty->id]]=return_one;busy+=return_one*empty->time;
    for(std::size_t g=0;g<heads.size();++g)x[var.at("group_head_"+std::to_string(g))]=heads[g];
    for(int r=0;r<2;++r)x[f.courier[r]]=d.demand[r]-served[r];
    x[f.fleet]=x[f.idle[0]]+busy;
    for(std::size_t l=0;l<f.rho.size();++l) {
        const auto c=f.matrix.cones[l];x[c.up]=x[f.idle[l]]+1;x[c.down]=1-x[f.rho[l]];x[c.one]=1;
    }
    const auto residual=point_residuals(f.matrix,x);
    require(residual.at("passed"),"Preprocessed multi-return witness failed: "+residual.dump());
    require(served[1]>0&&x[f.layout.assignment_index(1,0)]==0&&x[f.layout.assignment_index(1,1)]==0,
            "Second OD needs no own departure assignment");
    auto bad=x;bad[var.at("v_1_0")]=rho;bad[f.layout.assignment_index(1,0)]=1;
    require(!point_residuals(f.matrix,bad).at("passed").get<bool>(),"Missing positive first-task flow was accepted");
    return {{"residuals",residual},{"pair_rate",d.pair_rates(0,1)},
        {"served_second_without_own_assignment",served[1]},{"assignment_variables",d.assignment_count()},
        {"legacy_assignment_variables",8},{"return_bases_merged",true}};
}
Json edge_cases() {
    auto d=fixture();d.kappa(0,1)=0;d.kappa(0,0)=1;auto p=enumerate_paths(d);
    preprocess_generalized_demand(d,p);
    const double expected=2*(-std::expm1(-4*d.delta));
    require(std::abs(d.pair_rates(0,0)-expected)<1e-12,"Self pair rate wrong");
    require(std::abs(d.single_rates[0]+2*d.pair_rates(0,0)-4)<1e-12,"Self pair not counted twice");
    auto zero=fixture();zero.delta=0;auto zp=enumerate_paths(zero);preprocess_generalized_demand(zero,zp);
    require(zero.single_rates==zero.demand,"Zero window changed single demand");
    require(std::all_of(zero.pair_rates.values.begin(),zero.pair_rates.values.end(),[](double v){return v==0;}),"Zero-window pair positive");
    auto fractional=fixture();fractional.kappa(0,1)=.5;bool rejected=false;
    try{preprocess_generalized_demand(fractional,enumerate_paths(fractional));}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Fractional kappa silently accepted");
    auto missing=fixture();missing.allowed_links=constant(4,4,0);auto mp=enumerate_paths(missing);
    preprocess_generalized_demand(missing,mp);const auto mf=build_formulation(missing,mp);
    int rows=0;
    for(const auto& row:mf.matrix.rows)if(row.name.rfind("orders_",0)==0) {
        ++rows;require(row.coefficients.size()==1,"No-support head must force v to zero");
        require(row.coefficients[0].second<0,"No-support head coefficient missing");
    }
    require(rows==4,"Missing universal head rows for unsupported groups");
    return {{"self_pair_rate",d.pair_rates(0,0)},{"zero_window_preserves_demand",true},
        {"fractional_kappa_rejected",true},{"unsupported_head_rows",rows}};
}
Json first_leg_witness(bool restricted,bool proportional) {
    auto d=fixture();auto paths=enumerate_paths(d);preprocess_generalized_demand(d,paths);
    const Json reconstruction=d.preprocessed_demand_report;
    d.first_leg_consistency=restricted;
    preprocess_generalized_demand(d,paths);
    require(reconstruction==d.preprocessed_demand_report,"First-leg flag changed reconstructed demand");
    d.first_leg_support_report=first_leg_support(d,paths);
    auto f=build_formulation(d,paths);
    std::map<std::string,int> var;
    for(std::size_t j=0;j<f.matrix.variables.size();++j)var[f.matrix.variables[j].name]=j;
    std::vector<double>x(f.matrix.variables.size()),served(2),heads(f.lift_groups.size());
    for(int n=0;n<d.n_nodes();++n)x[n]=1;
    for(int h=0;h<d.n_h;++h)x[f.layout.capacity+h]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[f.layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    const double rho=.1;
    x[f.layout.assignment_index(0,0)]=1;x[f.rho[0]]=rho;
    x[f.idle[0]]=rho/(1-rho);x[var.at("v_0_0")]=rho;
    double busy=0,return_one=0;
    for(int second:{-1,1})for(int pickup=0;pickup<2;++pickup)for(int destination=0;destination<2;++destination) {
        const auto p=std::find_if(paths.service_paths.begin(),paths.service_paths.end(),[&](const ServicePath& p){
            return p.first_od==0&&p.second_od==second&&p.pickups[0]==pickup&&p.b==BasePair{0,destination};
        });
        require(p!=paths.service_paths.end(),"Missing mixed-first-leg witness path");
        const double first_mix=(second<0||proportional)?.8:.2;
        const double weight=pickup==0?first_mix:1-first_mix;
        const double flow=rho*(second<0?d.single_rates[0]:d.pair_rates(0,1))*weight/2;
        x[f.flow[p->id]]=flow;x[f.head[p->segment_id]]+=flow;
        heads[f.segment_group[p->segment_id]]+=flow;busy+=flow*p->time;
        for(auto [r,count]:p->served_counts)served[r]+=flow*count;
        if(destination==1)return_one+=flow;
    }
    const auto empty=std::find_if(paths.empty_paths.begin(),paths.empty_paths.end(),[](const auto& p){return p.b==BasePair{1,0};});
    require(empty!=paths.empty_paths.end(),"Missing mixed-first-leg balancing path");
    x[f.empty_flow[empty->id]]=return_one;busy+=return_one*empty->time;
    for(std::size_t g=0;g<heads.size();++g)x[var.at("group_head_"+std::to_string(g))]=heads[g];
    for(int r=0;r<2;++r)x[f.courier[r]]=d.demand[r]-served[r];
    x[f.fleet]=x[f.idle[0]]+busy;
    for(std::size_t l=0;l<f.rho.size();++l) {
        const auto c=f.matrix.cones[l];x[c.up]=x[f.idle[l]]+1;x[c.down]=1-x[f.rho[l]];x[c.one]=1;
    }
    const auto residual=point_residuals(f.matrix,x);
    require(residual.at("passed").get<bool>()==(!restricted||proportional),
            "First-leg branch consistency did not distinguish different branch mixtures");
    require(f.lift_groups.size()==static_cast<std::size_t>(restricted?8:4),"Incorrect physical first-leg grouping");
    require(d.first_leg_support_report.at("first_leg_groups")==8,"Support counted return bases separately");
    require(d.first_leg_support_report.at("complete_first_leg_groups")==8,"All fixture first legs support both branches");
    return {{"restricted",restricted},{"proportional",proportional},{"groups",f.lift_groups.size()},
        {"point_feasible",residual.at("passed")},{"max_linear_violation",residual.at("max_linear_violation")},
        {"demand_unchanged",true},{"return_bases_merged",true}};
}
}
int main(){try{
    Json first_leg=Json::array();
    for(bool restricted:{false,true})for(bool proportional:{false,true})
        first_leg.push_back(first_leg_witness(restricted,proportional));
    std::cout<<Json{{"passed",true},{"witness",witness()},{"edge_cases",edge_cases()},{"first_leg_cases",first_leg}}.dump(2)<<'\n';return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
