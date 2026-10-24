#include "mlnd/formulation.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace {
using namespace mlnd;
void require(bool condition,const std::string& message) { if(!condition)throw std::runtime_error(message); }
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
std::map<std::string,int> indices(const Formulation& f) {
    std::map<std::string,int> result;
    for(std::size_t j=0;j<f.matrix.variables.size();++j)
        result.emplace(f.matrix.variables[j].name,static_cast<int>(j));
    return result;
}
std::vector<double> design(const Input& d,int base_mask,int first,int second,int level=0) {
    DesignLayout layout(d);std::vector<double> x(layout.count,0);
    for(int l=0;l<2;++l)x[l]=(base_mask>>l)&1;
    if(base_mask)for(int h=0;h<2;++h){x[2+h]=1;x[layout.capacity+h*2+level]=1;}
    for(int r=0;r<2;++r){const int pair=r==0?first:second;
        if(pair>=0){require(x[pair/2]&&x[pair%2],"Illegal test assignment");x[layout.assignment+r*4+pair]=1;}}
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)x[layout.airline+i*4+j]=d.allowed_links(i,j)*x[i]*x[j];
    return x;
}
void set_cones(const Formulation& f,std::vector<double>& x) {
    for(std::size_t l=0;l<f.rho.size();++l){const double open=f.strengthening.queue_perspective?x[l]:1;
        x[f.matrix.cones[l].up]=x[f.idle[l]]+open;x[f.matrix.cones[l].down]=open-x[f.rho[l]];
        x[f.matrix.cones[l].one]=open;}
}
std::vector<double> mapped(const Formulation& source,const std::vector<double>& values,const Formulation& target) {
    const auto source_indices=indices(source);std::vector<double> result(target.matrix.variables.size());
    for(std::size_t j=0;j<result.size();++j)result[j]=values.at(source_indices.at(target.matrix.variables[j].name));
    set_cones(target,result);return result;
}
std::vector<double> courier_point(const Input& d,const Formulation& f,const std::vector<double>& design) {
    std::vector<double> x(f.matrix.variables.size(),0);std::copy(design.begin(),design.end(),x.begin());
    for(int r=0;r<d.n_od;++r)x[f.courier[r]]=d.demand[r];set_cones(f,x);return x;
}
void relax(Formulation& f) { for(auto& v:f.matrix.variables)v.binary=false; }
double row_violation(const LinearModel& model,const std::vector<double>& x,const std::string& name) {
    for(const auto& row:model.rows)if(row.name==name){long double value=-row.rhs;
        for(auto [j,a]:row.coefficients)value+=static_cast<long double>(a)*x[j];
        return row.sense=='L'?static_cast<double>(value):row.sense=='G'?-static_cast<double>(value):std::abs(static_cast<double>(value));}
    throw std::runtime_error("Missing row "+name);
}
Json fractional_checks(const Input& d,const PathCollection& paths) {
    auto old=build_formulation(d,paths), joint=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,false,{false,true,false});
    auto queue=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,false,{false,false,true});
    relax(old);relax(joint);relax(queue);const auto ids=indices(old);
    auto z=design(d,3,-1,-1);z[0]=.5;
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)z[old.layout.airline+i*4+j]=d.allowed_links(i,j)*std::min(z[i],z[j]);
    auto x=courier_point(d,old,z);x[old.rho[0]]=.25;x[old.idle[0]]=1.0/3;x[old.fleet]=1.0/3;set_cones(old,x);
    require(point_residuals(old.matrix,x).at("passed").get<bool>(),"Old fractional inventory witness invalid");
    auto stronger=mapped(old,x,queue);
    require(!point_residuals(queue.matrix,stronger).at("passed").get<bool>(),"Perspective failed to remove old fractional inventory witness");
    stronger[queue.idle[0]]=.5;stronger[queue.fleet]=.5;set_cones(queue,stronger);
    require(point_residuals(queue.matrix,stronger).at("passed").get<bool>(),"Perspective threshold witness invalid");
    // Full operational witness: split an OD between two returns while each
    // separate McCormick product claims rho. Balance with one empty return.
    for(int b=0;b<2;++b){x[old.layout.assignment+b]=.25;x[ids.at("v_0_0_"+std::to_string(b))]=.25;}
    std::vector<double> group_heads(old.lift_groups.size(),0);
    double busy=0;
    for(int b=0;b<2;++b){auto it=std::find_if(paths.service_paths.begin(),paths.service_paths.end(),[&](const auto& p){
            return p.first_od==0&&p.second_od<0&&p.b==BasePair{0,b};});
        require(it!=paths.service_paths.end(),"No single path for joint witness");
        x[old.flow[it->id]]=1;x[old.head[it->segment_id]]=1;group_heads[old.segment_group[it->segment_id]]+=1;busy+=it->time;
    }
    auto empty=std::find_if(paths.empty_paths.begin(),paths.empty_paths.end(),[](const auto& p){return p.b==BasePair{1,0};});
    require(empty!=paths.empty_paths.end(),"No empty balance path");x[old.empty_flow[empty->id]]=1;busy+=empty->time;
    for(std::size_t g=0;g<group_heads.size();++g){
        if(d.flow_policy=="aggregate")x[ids.at("group_head_"+std::to_string(g))]=group_heads[g];
        if(d.effective_delta()>0)for(int r=0;r<=d.n_od;++r)x[ids.at("Z_"+std::to_string(g)+"_"+std::to_string(r))]=group_heads[g];
    }
    x[old.courier[0]]=d.demand[0]-2;x[old.fleet]=x[old.idle[0]]+busy;
    const auto residuals=point_residuals(old.matrix,x);
    require(residuals.at("passed").get<bool>(),"Old full shared-rho witness invalid: "+residuals.dump());
    const auto shared=mapped(old,x,joint);
    const double joint_violation=row_violation(joint.matrix,shared,"v_joint_executed_0_0");
    require(joint_violation>.249999,"Joint row did not remove split-return witness");
    // A separate design-only witness defeats both departure/return pairwise
    // links but is removed by the union incident clique (count diagonal once).
    auto weak_design=design_model(d,false),clique=design_model(d,false,{true,false,false});
    auto a=design(d,3,-1,-1);a[0]=.5;a[weak_design.layout.assignment+1]=.5;a[weak_design.layout.assignment+2]=.5;
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)a[weak_design.layout.airline+i*4+j]=0;
    require(point_residuals(weak_design.matrix,a).at("passed").get<bool>(),"Old incident witness invalid");
    const double incident=row_violation(clique.matrix,a,"assign_incident_0_0");
    require(incident>.499999,"Incident clique failed");
    auto joint_design=design_model(d,false,{false,true,false});
    auto both_design=design_model(d,false,{true,true,false});
    auto outside=design(d,3,-1,-1);outside[0]=.5;
    outside[old.layout.assignment]=.375;outside[old.layout.assignment+1]=.375;
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)outside[old.layout.airline+i*4+j]=0;
    require(point_residuals(weak_design.matrix,outside).at("passed").get<bool>(),"Joint-only domain witness not old-feasible");
    const double domain_violation=row_violation(joint_design.matrix,outside,"assign_shared_rho_domain_0_0");
    require(domain_violation>.249999,"Joint-only master misses projected recourse domain");
    require(joint_design.strengthening_metadata.at("shared_rho_domain_rows").get<int>()==d.n_od*d.n_l&&
        both_design.strengthening_metadata.at("shared_rho_domain_rows").get<int>()==0,"Joint-only domain row metadata or deduplication invalid");
    outside[old.layout.assignment]=.25;outside[old.layout.assignment+1]=.25;
    require(point_residuals(joint_design.matrix,outside).at("passed").get<bool>(),"Joint-only projected boundary not feasible");
    auto domain_fixed=build_formulation(d,paths,&outside,nullptr,nullptr,true,false,false,{false,true,false});
    require(point_residuals(domain_fixed.matrix,courier_point(d,domain_fixed,outside)).at("passed").get<bool>(),
        "Joint-only strategic domain lacks all-courier extension");
    return {{"perspective_old_idle",1.0/3},{"perspective_new_idle",.5},
        {"joint_full_operational_witness_old_feasible",true},{"joint_violation",joint_violation},
        {"incident_clique_violation",incident},{"joint_only_master_domain_violation",domain_violation},
        {"joint_only_master_domain_boundary_has_all_courier_extension",true},
        {"joint_only_domain_rows",joint_design.strengthening_metadata.at("shared_rho_domain_rows")},
        {"joint_domain_rows_not_duplicated_with_incident",true}};
}
Json repair_and_bound_checks(const Input& d,const PathCollection& paths) {
    const auto z=design(d,1,-1,-1);
    auto f=build_formulation(d,paths,&z,nullptr,nullptr,false,false,false,{true,true,true});
    auto x=courier_point(d,f,z);x[f.rho[0]]=.5;x[f.idle[0]]=1;x[f.idle[1]]=.5;x[f.fleet]=1.5;set_cones(f,x);
    require(point_residuals(f.matrix,x).at("passed").get<bool>(),"Closed-base positive idle no longer feasible");
    x[f.idle[0]]=.8;set_cones(f,x);const auto damaged=x;
    require(!point_residuals(f.matrix,x).at("passed").get<bool>(),"Repair witness not damaged");
    const auto repaired=repair_inventory_primal(f,x);
    require(repaired.at("passed").get<bool>()&&repaired.at("applied").get<bool>(),"Perspective repair failed: "+repaired.dump());
    require(x[f.idle[0]]>=1&&x[f.idle[1]]==.5,"Repair changed closed idle or did not repair inventory");
    for(std::size_t j=0;j<x.size();++j){bool allowed=static_cast<int>(j)==f.fleet;
        for(int i:f.idle)allowed=allowed||static_cast<int>(j)==i;
        for(const auto& c:f.matrix.cones)allowed=allowed||static_cast<int>(j)==c.up||static_cast<int>(j)==c.down||static_cast<int>(j)==c.one;
        require(allowed||x[j]==damaged[j],"Repair changed unrelated variable");}
    const double M=f.matrix.variables[f.idle[0]].upper;
    const long double exact=static_cast<long double>(M)/(1.0L+M);
    require(f.rho_upper[0]>=exact&&f.rho_upper[0]<=static_cast<double>(exact)+4*std::numeric_limits<double>::epsilon(),"Finite rho bound not conservatively rounded");
    auto boundary=courier_point(d,f,z);boundary[f.rho[0]]=static_cast<double>(exact);boundary[f.idle[0]]=M;boundary[f.fleet]=M;set_cones(f,boundary);
    require(point_residuals(f.matrix,boundary).at("passed").get<bool>(),"Finite inventory upper boundary excluded");
    auto impossible=x;impossible[f.rho[0]]=1;set_cones(f,impossible);const auto saved=impossible;
    require(!repair_inventory_primal(f,impossible).at("passed").get<bool>()&&impossible==saved,"Invalid perspective boundary repaired nonatomically");
    return {{"closed_base_positive_idle_preserved",true},{"repair",repaired},
        {"idle_bound",M},{"rho_upper",f.rho_upper[0]},{"finite_upper_boundary_feasible",true},{"invalid_boundary_rejected_atomically",true}};
}
Json run() {
    const FormulationStrengthening on{true,true,true};SolverOptions options;options.gap=0;options.threads=1;
    Json report={{"schema","mlnd-structural-strengthening-regression/1"},{"cases",Json::array()}};
    int paired_solves=0,mappings=0;double maximum_difference=0;
    for(const std::string policy:{"aggregate","prefix"})for(int k:{1,2}) {
        const auto d=fixture(k,policy);const auto paths=enumerate_paths(d);
        std::vector<std::vector<double>> designs;
        for(int r=-1;r<4;++r)for(int s=-1;s<4;++s)designs.push_back(design(d,3,r,s,(r+s+10)%2));
        for(int mask:{1,2})for(int r:{-1,mask==1?0:3})for(int s:{-1,mask==1?0:3})designs.push_back(design(d,mask,r,s));
        designs.push_back(design(d,0,-1,-1));
        double case_difference=0;
        for(const auto& z:designs){double lifted_reference=0;
            for(bool conditional:{false,true}) {
                auto old=build_formulation(d,paths,&z,nullptr,nullptr,false,conditional);
                auto strong=build_formulation(d,paths,&z,nullptr,nullptr,false,conditional,false,on);
                NativeSolver old_solver(old.matrix,options,"structure_old_fixed"),new_solver(strong.matrix,options,"structure_new_fixed");
                const auto a=old_solver.optimize(10),b=new_solver.optimize(10);
                require(a.optimal&&b.optimal&&a.has_solution&&b.has_solution,"Fixed SOCP not optimal");
                require(point_residuals(old.matrix,a.values).at("passed").get<bool>()&&
                    point_residuals(strong.matrix,b.values).at("passed").get<bool>(),"Original fixed primal invalid");
                const double difference=std::abs(a.objective-b.objective);maximum_difference=std::max(maximum_difference,difference);case_difference=std::max(case_difference,difference);
                require(difference<=5e-6*(1+std::abs(a.objective)),"Integer operating objective changed");
                require(point_residuals(strong.matrix,mapped(old,a.values,strong)).at("passed").get<bool>(),"Old integer feasible point excluded");
                require(point_residuals(old.matrix,mapped(strong,b.values,old)).at("passed").get<bool>(),"New integer point not old-feasible");
                if(!conditional)lifted_reference=a.objective;
                else require(std::abs(lifted_reference-a.objective)<=5e-6*(1+std::abs(lifted_reference)),"Conditional recourse mismatch");
                ++paired_solves;mappings+=2;
            }
        }
        auto old=build_formulation(d,paths),strong=build_formulation(d,paths,nullptr,nullptr,nullptr,false,false,false,on);
        relax(old);relax(strong);NativeSolver old_lp(old.matrix,options,"structure_old_continuous"),new_lp(strong.matrix,options,"structure_new_continuous");
        const auto a=old_lp.optimize(20),b=new_lp.optimize(20);
        require(a.optimal&&b.optimal&&a.has_solution&&b.has_solution,"Continuous comparison not optimal");
        require(point_residuals(old.matrix,a.values).at("passed").get<bool>()&&point_residuals(strong.matrix,b.values).at("passed").get<bool>(),"Continuous comparison primal invalid");
        require(b.objective>=a.objective-1e-5,"Structural strengthening lowered continuous optimum");
        report["cases"].push_back({{"K",k},{"policy",policy},{"integer_designs",designs.size()},
            {"maximum_integer_objective_difference",case_difference},{"old_continuous_objective",a.objective},
            {"strengthened_continuous_objective",b.objective},{"fractional",fractional_checks(d,paths)},
            {"repair_and_bound",repair_and_bound_checks(d,paths)},{"metadata",strong.strengthening_metadata}});
    }
    report["paired_fixed_socps"]=paired_solves;report["integer_bidirectional_point_checks"]=mappings;
    report["maximum_integer_objective_difference"]=maximum_difference;report["passed"]=true;return report;
}
}
int main(int argc,char** argv) {try{if(argc!=2)throw std::invalid_argument("Usage: mlnd_structural_strengthening_probe OUTPUT_JSON");
    const auto output=std::filesystem::absolute(argv[1]);require(!std::filesystem::exists(output),"Refusing existing audit output");
    const auto result=run();mlnd::atomic_json_file(output,result);std::cout<<result.dump(2)<<'\n';return 0;
}catch(const IloException& e){std::cerr<<e.getMessage()<<'\n';return 2;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
