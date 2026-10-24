// Structural K=1 regression: one OD may use two physical return bases.
#include "mlnd/formulation.hpp"
#include "mlnd/strengthening.hpp"
#include "mlnd/assignment_support_cuts.hpp"
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
Input fixture(int k,const std::string& policy) {
    Input d;d.n_l=2;d.n_h=2;d.n_od=2;d.n_levels=2;d.max_orders=k;
    d.name="structural_strengthening_fixture";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy=policy;d.demand={4,7};d.ground_cost={5,5};d.deadline={1,1};
    d.flight_times=constant(4,4,0.0);d.flight_times.values={0,.03,.05,.07,.03,0,.06,.04,.05,.06,0,.1,.07,.04,.1,0};
    d.allowed_links=constant(4,4,1);for(int i=0;i<4;++i)d.allowed_links(i,i)=0;
    d.link_cost=constant(4,4,0.0);d.base_cost={.5,.4};d.slot_epsilon={.05,.05};
    d.rider_mean=constant(2,2,0.0);d.rider_mean.values={.08,.12,.1,.06};
    d.first_cost=constant(2,2,1.0);d.terminal_cost=constant(2,2,.5);
    d.receipt_mean=constant(2,2,.08);d.removal_mean=constant(2,2,.03);
    d.slot_counts=constant(2,2,0);d.slot_counts.values={10,20,10,20};
    d.slot_cost=constant(2,2,0.0);d.slot_cost.values={.1,.2,.1,.2};
    d.occupancy_limit=constant(2,2,0.0);d.occupancy_limit.values={6.1690072893953225,14.072024748341311,6.1690072893953225,14.072024748341311};
    d.kappa=constant(2,2,0.0);d.kappa.values={.3,.9,.2,.7};
    d.delta=.04;d.operating_limit=1;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;return d;
}

Json check(const std::string& policy) {
    const auto d=fixture(1,policy);
    auto paths=enumerate_paths(d);
    auto f=build_formulation(d,paths);
    require(f.layout.airline-f.layout.assignment==d.n_od*d.n_l,"K1 assignment was not reduced");
    std::vector<double> x(f.matrix.variables.size(),0);
    std::map<std::string,int> var;
    for(std::size_t j=0;j<f.matrix.variables.size();++j) var[f.matrix.variables[j].name]=j;
    for(int u=0;u<d.n_nodes();++u) x[u]=1;
    for(int i=0;i<d.n_h;++i) x[f.layout.capacity+i*d.n_levels]=1;
    for(int u=0;u<d.n_nodes();++u) for(int v=0;v<d.n_nodes();++v)
        x[f.layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    x[f.layout.assignment_index(0,0)]=1;
    const double flow=.25,rho=2*flow/d.demand[0];
    x[f.rho[0]]=rho;x[f.idle[0]]=rho/(1-rho);
    x[var.at("v_0_0")]=rho;
    double busy=0;
    std::vector<double> heads(f.lift_groups.size());
    for(int m=0;m<2;++m) {
        auto p=std::find_if(paths.service_paths.begin(),paths.service_paths.end(),[&](const auto& p){
            return p.first_od==0&&p.second_od<0&&p.b==BasePair{0,m};});
        require(p!=paths.service_paths.end(),"Missing return-base route");
        x[f.flow[p->id]]=flow;x[f.head[p->segment_id]]+=flow;
        heads[f.segment_group[p->segment_id]]+=flow;busy+=flow*p->time;
    }
    auto e=std::find_if(paths.empty_paths.begin(),paths.empty_paths.end(),[](const auto& p){return p.b==BasePair{1,0};});
    require(e!=paths.empty_paths.end(),"Missing balancing relocation");
    x[f.empty_flow[e->id]]=flow;busy+=flow*e->time;
    if(policy=="aggregate")for(std::size_t g=0;g<heads.size();++g)x[var.at("group_head_"+std::to_string(g))]=heads[g];
    for(int r=0;r<d.n_od;++r)x[f.courier[r]]=d.demand[r]-(r==0?2*flow:0);
    x[f.fleet]=x[f.idle[0]]+busy;
    for(std::size_t l=0;l<f.rho.size();++l){const auto c=f.matrix.cones[l];
        x[c.up]=x[f.idle[l]]+1;x[c.down]=1-x[f.rho[l]];x[c.one]=1;}
    const auto residuals=point_residuals(f.matrix,x);
    require(residuals.at("passed"),"Two-return origin-assignment witness failed: "+residuals.dump());
    std::size_t order_rows=0;
    for(const auto& row:f.matrix.rows) if(row.name.rfind("orders_",0)==0) ++order_rows;
    require(order_rows==static_cast<std::size_t>(d.n_od*d.n_l),"Duplicate K1 OD-origin rows");
    const auto result=solution_json(d,paths,f,x);
    require(result.at("assignment_scope")=="origin"&&result.at("_solution").at("assignment").at(0).at(0).is_number(),"Wrong K1 result shape");
    auto fixed=solution_design(d,result);
    auto compact=build_formulation(d,paths,&fixed,nullptr,nullptr,false,true);
    require(compact.layout.count==f.layout.count,"Conditional model changed strategic layout");
    auto legacy=result;
    legacy["_solution"]["assignment"]=Json::array();
    for(int r=0;r<d.n_od;++r){Json row=Json::array();for(int l=0;l<d.n_l;++l)row.push_back(std::vector<double>(d.n_l));legacy["_solution"]["assignment"].push_back(row);}
    bool rejected=false;
    try{solution_design(d,legacy);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Legacy 3D assignment was silently interpreted as K1");
    const auto bounds=analytical_master_bounds(d);
    require(bounds.unit_cost.size()==static_cast<std::size_t>(d.assignment_count()),"Analytical bounds use legacy groups");
    auto rm=resource_savings_model(d,true,true,true);
    require(rm.group_rhs.size()==static_cast<std::size_t>(d.assignment_count()),"Resource LP uses legacy groups");
    auto bad=d;bad.max_orders=2;const DesignLayout k2(bad);
    require(k2.airline-k2.assignment==bad.n_od*bad.n_l*bad.n_l,"K2 shape changed");
    for(int r=0;r<bad.n_od;++r)for(int l=0;l<bad.n_l;++l)for(int m=0;m<bad.n_l;++m)
        require(k2.assignment_index(r,l,m)==k2.assignment+(r*bad.n_l+l)*bad.n_l+m,"K2 indexing changed");
    rejected=false;try{build_assignment_support_cache(d,paths,INFINITY);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Pair-specific optional cuts accepted K1");
    return {{"policy",policy},{"assignment_variables",d.assignment_count()},{"order_rows",order_rows},
        {"two_return_flows",{flow,flow}},{"residuals",residuals},{"legacy_3d_rejected",true},
        {"k2_layout_preserved",true},{"pair_cut_rejected",true}};
}
}
int main(){try{Json cases=Json::array();for(const std::string policy:{"aggregate","prefix"})cases.push_back(check(policy));
    std::cout<<Json{{"passed",true},{"cases",cases}}.dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
