// Independent full-path fixed-design validation of assignment-support cost cuts.
// This probe does not call the separator, reuse its cached support, or change the model.
#include "mlnd/formulation.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
namespace {
using namespace mlnd;
void require(bool condition,const std::string& message) { if(!condition) throw std::runtime_error(message); }
template<class T> Matrix<T> filled(int rows, int columns, T value) {
    Matrix<T> result(rows, columns);
    std::fill(result.values.begin(), result.values.end(), value);
    return result;
}
// Frozen physical fixture shared with conditional_formulation_probe, extended
// to two capacity levels to cover both sparse networks and relocation.
Input fixture(int k, const std::string& policy, bool two_bases = false) {
    Input d;
    d.n_l = two_bases ? 2 : 1; d.n_h = d.n_od = d.n_levels = 2;
    d.max_orders = k; d.name = "assignment_support_validity_fixture";
    d.model_version = "consecutive"; d.route_mode = "direct"; d.flow_policy = policy;
    d.demand = {4, 7}; d.ground_cost = {5, 5}; d.deadline = {1, 1};
    d.base_cost.assign(d.n_l, .5); d.slot_epsilon = {.05, .05};
    d.flight_times = Matrix<double>(d.n_nodes(), d.n_nodes());
    if (two_bases)
        d.flight_times.values = {0,.03,.05,.07, .03,0,.06,.04, .05,.06,0,.1, .07,.04,.1,0};
    else d.flight_times.values = {0,.05,.07, .05,0,.1, .07,.1,0};
    d.allowed_links = filled(d.n_nodes(), d.n_nodes(), 1);
    for (int i = 0; i < d.n_nodes(); ++i) d.allowed_links(i,i) = 0;
    d.link_cost = filled(d.n_nodes(), d.n_nodes(), 0.0);
    d.rider_mean = Matrix<double>(2,2); d.rider_mean.values = {.08,.12,.1,.06};
    d.first_cost = filled(2,2,1.0); d.terminal_cost = filled(2,2,.5);
    d.receipt_mean = filled(2,2,.08); d.removal_mean = filled(2,2,.03);
    d.slot_counts = Matrix<int>(2,2); d.slot_counts.values = {10,20,10,20};
    d.slot_cost = Matrix<double>(2,2); d.slot_cost.values = {.1,.2,.1,.2};
    d.occupancy_limit = Matrix<double>(2,2);
    d.occupancy_limit.values = {6.1690072893953225,14.072024748341311,
                              6.1690072893953225,14.072024748341311};
    d.kappa = Matrix<double>(2,2); d.kappa.values = {.3,.9,.2,.7};
    d.delta = .04; d.operating_limit = 1; d.fleet_cost = 1;
    d.flight_cost = .2; d.waiting_cost = .3;
    return d;
}

struct Cut {
    int r=0,l=0,b=0;
    std::vector<std::pair<int,int>> blocked;
};
bool open_path(const Input& d,const ServicePath& p,const std::vector<double>& design) {
    const DesignLayout layout(d);
    for(auto [u,v,count]:p.arc_counts)
        if(count>0 && design[layout.airline+u*d.n_nodes()+v]<.5) return false;
    return true;
}
bool support_absent(const Input& d,const PathCollection& paths,const Cut& cut) {
    for(const auto& p:paths.service_paths) {
        if(p.second_od>=0 || p.first_od!=cut.r || p.b[0]!=cut.l || p.b[1]!=cut.b) continue;
        bool blocked=false;
        for(auto [u,v,count]:p.arc_counts)
            if(count>0 && std::find(cut.blocked.begin(),cut.blocked.end(),std::make_pair(u,v))!=cut.blocked.end()) blocked=true;
        if(!blocked) return false;
    }
    return true;
}
double cost_floor(const Input& d,const std::vector<double>& x,const Cut& cut) {
    const DesignLayout layout(d);
    const double baseline=std::inner_product(d.demand.begin(),d.demand.end(),d.ground_cost.begin(),0.0);
    auto a=[&](int r,int l,int b) { return layout.assignment+(r*d.n_l+l)*d.n_l+b; };
    double rhs=baseline*x[a(cut.r,cut.l,cut.b)];
    for(int r=0;r<d.n_od;++r) for(int l=0;l<d.n_l;++l) if(l!=cut.l)
        for(int b=0;b<d.n_l;++b) rhs-=d.demand[r]*d.ground_cost[r]*x[a(r,l,b)];
    for(auto [u,v]:cut.blocked) rhs-=baseline*x[layout.airline+u*d.n_nodes()+v];
    return rhs;
}
Cut bad_cut(const Input& d,const PathCollection& paths,const std::vector<double>& design) {
    const DesignLayout layout(d); const int N=d.n_nodes(),L=d.n_l;
    for(int r=0;r<d.n_od;++r) for(int l=0;l<L;++l) for(int b=0;b<L;++b) {
        if(d.demand[r]<=0 || design[layout.assignment+(r*L+l)*L+b]<.5) continue;
        Cut c; c.r=r;c.l=l;c.b=b;
        for(int u=0;u<N;++u) for(int v=0;v<N;++v)
            if(u!=v && d.allowed_links(u,v) && design[layout.airline+u*N+v]<.5) c.blocked.emplace_back(u,v);
        if(support_absent(d,paths,c)) return c;
    }
    throw std::runtime_error("No assigned unsupported OD in saved bad design");
}
std::vector<double> tiny_design(const Input& d,int assignment0,int assignment1,bool block) {
    const DesignLayout layout(d); std::vector<double> x(layout.count,0);
    for(int i=0;i<d.n_nodes();++i) x[i]=1;
    for(int i=0;i<d.n_h;++i) x[layout.capacity+i*d.n_levels+d.n_levels-1]=1;
    for(int r=0;r<2;++r) {
        int a=r==0?assignment0:assignment1;
        if(a>=0) x[layout.assignment+r*d.n_l*d.n_l+a]=1;
    }
    for(int u=0;u<d.n_nodes();++u) for(int v=0;v<d.n_nodes();++v)
        x[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    if(block) x[layout.airline+d.n_l*d.n_nodes()+d.n_l+1]=0;
    return x;
}
Json compare(const Input& d,const PathCollection& paths,const std::vector<double>& design,
             const Cut& cut,const std::string& label,bool conditional) {
    require(paths.complete && paths.support_complete,"Need complete physical catalogue");
    require(support_absent(d,paths,cut),"Independent full catalogue invalidates blocking set");
    require(point_residuals(design_model(d).matrix,design).at("passed"),"Invalid test strategic design");
    auto f=build_formulation(d,paths,&design,nullptr,nullptr,false,conditional,true);
    const double rhs=cost_floor(d,design,cut);
    SolverOptions options; options.seconds=30;options.threads=1;options.gap=1e-9;
    NativeSolver old_solver(f.matrix,options,"assignment_support_original");
    auto old=old_solver.optimize(30);
    require(old.has_solution && old.optimal,label+": original full recourse not optimal: "+old.status);
    require(repair_inventory_primal(f,old.values).at("passed"),label+": original repair failed");
    const auto old_residual=point_residuals(f.matrix,old.values);
    require(old_residual.at("passed"),label+": original residual check failed");
    const double old_cost=objective_value(f.matrix,old.values);
    Coefficients cost;
    for(int i=0;i<static_cast<int>(f.matrix.variables.size());++i)
        if(f.matrix.variables[i].cost!=0) cost.emplace_back(i,f.matrix.variables[i].cost);
    f.matrix.row("independent_assignment_support_cost",'G',rhs,std::move(cost));
    const double tolerance=2e-7*(1+std::abs(old_cost));
    require(old_cost+tolerance>=rhs,label+": cost cut excludes original optimal solution");
    NativeSolver new_solver(f.matrix,options,"assignment_support_added");
    auto strengthened=new_solver.optimize(30);
    require(strengthened.has_solution && strengthened.optimal,label+": strengthened full recourse not optimal: "+strengthened.status);
    require(repair_inventory_primal(f,strengthened.values).at("passed"),label+": strengthened repair failed");
    auto strengthened_residual=point_residuals(f.matrix,strengthened.values);
    require(strengthened_residual.at("passed"),label+": strengthened residual check failed");
    const double new_cost=objective_value(f.matrix,strengthened.values);
    require(std::abs(old_cost-new_cost)<=tolerance,label+": full recourse objective changed");
    double flow=0;for(int v:f.flow) if(v>=0) flow+=old.values[v];
    Json result={{"case",label},{"conditional",conditional},{"K",d.max_orders},{"bases",d.n_l},
      {"path_catalogue_complete",true},{"service_paths",paths.service_paths.size()},
      {"anchor",{cut.r,cut.l,cut.b}},{"blocked_count",cut.blocked.size()},
      {"original_objective",old_cost},{"strengthened_objective",new_cost},{"absolute_difference",std::abs(old_cost-new_cost)},
      {"cut_rhs",rhs},{"original_slack",old_cost-rhs},{"objective_tolerance",tolerance},
      {"original_service_task_flow",flow},{"original_rho",Json::array()},
      {"original_residual",old_residual},{"strengthened_residual",strengthened_residual},
      {"original_status",old.status},{"strengthened_status",strengthened.status}};
    for(int v:f.rho) result["original_rho"].push_back(old.values[v]);
    std::cout<<label<<" conditional="<<conditional<<" cost="<<old_cost<<" rhs="<<rhs<<" diff="<<std::abs(old_cost-new_cost)<<std::endl;
    return result;
}
}
int main(int argc,char**argv) {
 try {
    if(argc!=3) throw std::invalid_argument("Usage: probe FIXED_DESIGN_CASES_JSON OUTPUT_JSON");
    Json checks=Json::array();
    for(bool two_bases:{false,true}) for(int k:{1,2}) {
        auto d=fixture(k,"aggregate",two_bases);
        // OD 0 can only be picked up at locker 0; OD 1 only at locker 1.
        d.rider_mean.values={.02,1.,1.,.02}; d.deadline={.25,.25};
        auto paths=enumerate_paths(d);
        Cut cut;cut.blocked={{d.n_l,d.n_l+1}};
        require(support_absent(d,paths,cut),"Fixture did not create a true OD-specific obstruction");
        const std::vector<std::pair<int,int>> assignments=two_bases ?
            std::vector<std::pair<int,int>>{{0,0},{0,2},{0,-1},{-1,0},{2,0},{1,0},{-1,-1}}:
            std::vector<std::pair<int,int>>{{0,0},{0,-1},{-1,0},{-1,-1}};
        for(auto [a,b]:assignments) for(bool blocked:{false,true}) for(bool conditional:{false,true})
            checks.push_back(compare(d,paths,tiny_design(d,a,b,blocked),cut,
                "tiny_L"+std::to_string(d.n_l)+"_K"+std::to_string(k)+"_a"+std::to_string(a)+"_b"+std::to_string(b)+"_block"+std::to_string(blocked),conditional));
    }
    Json cases;std::ifstream input(argv[1]);input>>cases;
    for(const auto& c:cases) {
        const auto d=read_input(c.at("input").get<std::string>(),2);
        const auto paths=enumerate_paths(d);
        std::vector<double> bad; for(char bit:c.at("bad_design_key").get<std::string>()) {require(bit=='0'||bit=='1',"Bad design encoding");bad.push_back(bit-'0');}
        const Cut cut=bad_cut(d,paths,bad);
        const auto positive=solution_design(d,c.at("positive_solution"));
        for(bool conditional:{false,true}) {
            checks.push_back(compare(d,paths,bad,cut,c.at("label").get<std::string>()+"_saved_bad",conditional));
            checks.push_back(compare(d,paths,positive,cut,c.at("label").get<std::string>()+"_saved_positive",conditional));
        }
    }
    Json output={{"passed",true},{"scope","Independent full-path fixed binary designs; original model versus added cost inequality; not a benchmark"},{"comparisons",checks},{"comparison_count",checks.size()}};
    std::ofstream out(argv[2]);out<<output.dump(2)<<'\n';
 }catch(const IloException&e){std::cerr<<e<<'\n';return 1;}
 catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
