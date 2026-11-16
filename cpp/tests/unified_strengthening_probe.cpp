// Regression over the existing two-OD physical fixture, with fixed task rates.
// Unit fixtures are deliberately not production experiment instances.
#include "mlnd/strengthening.hpp"
#include "mlnd/topology_cuts.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool condition,const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> Matrix<T> constant(int rows,int cols,T value) {
    Matrix<T> out(rows,cols);std::fill(out.values.begin(),out.values.end(),value);return out;
}
Input fixture(int bases,int cardinality) {
    Input d;d.n_l=bases;d.n_h=2;d.n_od=2;d.n_levels=1;d.max_orders=cardinality;
    d.name="unified_strengthening_fixture";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";d.demand={4,7};d.ground_cost={2,20};d.deadline={1,1};
    const int n=bases+2;
    d.flight_times=constant(n,n,.05);
    for(int u=0;u<n;++u)d.flight_times(u,u)=0;
    d.flight_times(bases,bases+1)=d.flight_times(bases+1,bases)=.1;
    d.allowed_links=constant(n,n,1);for(int u=0;u<n;++u)d.allowed_links(u,u)=0;
    d.link_cost=constant(n,n,0.0);d.base_cost=std::vector<double>(bases,.5);d.slot_epsilon={.05,.05};
    d.rider_mean=constant(2,2,.08);d.first_cost=constant(2,2,1.0);d.terminal_cost=constant(2,2,.5);
    d.receipt_mean=constant(2,2,.08);d.removal_mean=constant(2,2,.03);
    d.slot_counts=constant(2,1,20);d.slot_cost=constant(2,1,.1);d.occupancy_limit=constant(2,1,14.0);
    d.kappa=constant(2,2,1.0);d.delta=.04;d.operating_limit=1.5;
    d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    d.preprocessed_demand=true;d.pair_rates=constant(2,2,0.0);
    if(cardinality==1) { d.single_rates=d.demand;d.head_rates=d.demand; }
    else {
        // Includes both directions and AA. Each raw OD is counted exactly once
        // per occurrence: singles + outgoing + incoming = {4,7}.
        d.single_rates={.5,4.5};d.pair_rates(0,0)=.5;
        d.pair_rates(0,1)=2;d.pair_rates(1,0)=.5;d.head_rates={3,5};
    }
    d.common_origin_support=constant(2,bases,1);
    return d;
}
double affine(const Coefficients& coefficients,const std::vector<double>& x) {
    double out=0;for(auto [j,c]:coefficients)out+=c*x.at(j);return out;
}
double violation(const RowRecord& row,const std::vector<double>& point) {
    return row.rhs-affine(row.coefficients,point);
}
std::vector<double> design(const Input& d,int first,int second,double infrastructure=1) {
    const DesignLayout layout(d);std::vector<double> x(layout.count);
    for(int l=0;l<d.n_l;++l)x[l]=1;
    for(int h=0;h<d.n_h;++h)x[d.n_l+h]=x[layout.capacity+h]=infrastructure;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        if(d.allowed_links(u,v))x[layout.airline+u*d.n_nodes()+v]=(u>=d.n_l||v>=d.n_l)?infrastructure:1;
    if(first>=0)x[layout.assignment_index(0,first)]=1;
    if(second>=0)x[layout.assignment_index(1,second)]=1;
    return x;
}
Json explicit_witness(bool second_assigned_elsewhere) {
    const auto d=fixture(2,2);const auto paths=enumerate_paths(d);auto f=build_formulation(d,paths);
    std::map<std::string,int> vars;for(std::size_t j=0;j<f.matrix.variables.size();++j)vars[f.matrix.variables[j].name]=j;
    auto x=design(d,0,second_assigned_elsewhere?1:-1);x.resize(f.matrix.variables.size());
    std::vector<double> heads(f.lift_groups.size()),served(2),net(2);double busy=0;
    for(int r=0;r<2;++r) {
        if(r==1&&!second_assigned_elsewhere)continue;
        const int l=r;const double rho=r==0?.3:.2;
        x[f.rho[l]]=rho;x[f.idle[l]]=rho/(1-rho);x[vars.at("v_"+std::to_string(r)+"_"+std::to_string(l))]=rho;
        for(int s=-1;s<2;++s) {
            const double rate=s<0?d.single_rates[r]:d.pair_rates(r,s);if(rate==0)continue;
            for(int m=0;m<2;++m) {
                const auto p=std::find_if(paths.service_paths.begin(),paths.service_paths.end(),[&](const ServicePath& p){
                    return p.first_od==r&&p.second_od==s&&p.b==BasePair{l,m};});
                require(p!=paths.service_paths.end(),"Witness is missing a physical route");
                const double flow=rate*rho/2;x[f.flow[p->id]]=flow;x[f.head[p->segment_id]]+=flow;
                heads[f.segment_group[p->segment_id]]+=flow;busy+=flow*p->time;
                net[l]+=flow;net[m]-=flow;for(auto [od,count]:p->served_counts)served[od]+=flow*count;
            }
        }
    }
    if(std::abs(net[0])>1e-12) {
        const BasePair b=net[0]>0?BasePair{1,0}:BasePair{0,1};
        const auto e=std::find_if(paths.empty_paths.begin(),paths.empty_paths.end(),[&](const EmptyPath& p){return p.b==b;});
        require(e!=paths.empty_paths.end(),"Witness missing balancing relocation");
        x[f.empty_flow[e->id]]=std::abs(net[0]);busy+=std::abs(net[0])*e->time;
    }
    for(std::size_t g=0;g<heads.size();++g)x[vars.at("group_head_"+std::to_string(g))]=heads[g];
    for(int r=0;r<2;++r)x[f.courier[r]]=d.demand[r]-served[r];
    x[f.fleet]=busy+std::accumulate(f.idle.begin(),f.idle.end(),0.0,[&](double a,int v){return a+x[v];});
    for(std::size_t l=0;l<f.rho.size();++l) {
        const auto c=f.matrix.cones[l];x[c.up]=x[f.idle[l]]+1;x[c.down]=1-x[f.rho[l]];x[c.one]=1;
    }
    const auto residual=point_residuals(f.matrix,x);
    require(residual.at("passed"),"Explicit incoming/AA witness failed: "+residual.dump());
    const auto resource=resource_savings_model(d,true,true,true);
    require(affine(resource.group_rhs[d.assignment_offset(1,0)],x)>0&&x[f.layout.assignment_index(1,0)]==0,
            "Incoming OD occurrence incorrectly requires its own departure assignment");
    require(std::abs(affine(resource.group_rhs[d.assignment_offset(0,0)],x)-3.5)<1e-12,
            "AA occurrence was not counted twice");
    double operating=0;for(std::size_t j=f.layout.count;j<x.size();++j)operating+=f.matrix.variables[j].cost*x[j];
    const auto bounds=analytical_master_bounds(d);double legacy_rhs=bounds.courier_cost;
    for(int r=0;r<2;++r)for(int l=0;l<2;++l)
        legacy_rhs-=d.demand[r]*bounds.unit_saving[d.assignment_offset(r,l)]*x[f.layout.assignment_index(r,l)];
    auto master=design_model(d,false);const int theta=master.matrix.variable("theta",0,INFINITY,1);
    add_master_strengthening(d,paths,master,theta);
    auto projected=x;projected.resize(master.matrix.variables.size());projected[theta]=operating;
    require(point_residuals(master.matrix,projected).at("passed"),"New analytical cuts exclude explicit witness");
    if(!second_assigned_elsewhere)require(legacy_rhs-operating>1,"Witness does not expose the legacy own-alpha error");
    return {{"second_assigned_elsewhere",second_assigned_elsewhere},{"residuals",residual},
        {"served",served},{"legacy_assignment_bound_violation",legacy_rhs-operating},
        {"operating_cost",operating},{"incoming_supply_without_own_assignment",true},{"AA_counted_twice",true}};
}
Json evaluate(int bases,int cardinality) {
    auto d=fixture(bases,cardinality);const auto paths=enumerate_paths(d);
    const auto allocations=capacity_savings_network(d);const auto bounds=analytical_master_bounds(d);
    const auto rm=resource_savings_model(d,allocations,true,true,false);
    const auto od_rm=resource_savings_model(d,allocations,true,true,true);
    const DesignLayout layout(d);double max_cost_error=0,max_load_error=0;
    for(int r=0;r<2;++r)for(int l=0;l<bases;++l) {
        std::map<int,double> actual;for(auto [j,c]:rm.group_rhs[d.assignment_offset(r,l)])actual[j]+=c;
        for(int h=0;h<2;++h)require(std::abs(actual[layout.assignment_index(h,l)]-
            (d.pair_rates(h,r)+(h==r?d.head_rates[r]:0)))<1e-12,"Incorrect generalized occurrence coefficient");
    }
    for(const auto& p:paths.service_paths) {
        double cost=0;std::map<int,double> load,actual;
        for(auto [j,t]:p.slot_times)actual[j]+=t;
        for(std::size_t k=0;k<p.deliveries.size();++k) {
            const int r=k==0?p.first_od:p.second_od,j=p.deliveries[k];
            const int g=d.assignment_offset(r,p.b[0])*d.n_h+j;
            require(allocations.eligible[g],"A physical occurrence was omitted by the relaxation");
            cost+=allocations.unit_cost[g];load[j]+=allocations.slot_time[g];
        }
        max_cost_error=std::max(max_cost_error,cost-p.cost-d.fleet_cost*p.time);
        for(auto [j,t]:load)max_load_error=std::max(max_load_error,t-actual[j]);
    }
    require(max_cost_error<1e-9&&max_load_error<1e-9,"Physical occurrence allocation is invalid");
    auto master=design_model(d,false);const int theta=master.matrix.variable("theta",0,INFINITY,1);
    add_master_strengthening(d,paths,master,theta);
    SolverOptions options;options.seconds=10;options.threads=1;options.gap=1e-9;
    ResourceSavingsRelaxation resource(rm,options),od_resource(od_rm,options);
    struct Point {std::vector<double> x;bool integer;};std::vector<Point> points;
    int exact_solves=0;double min_improvement=INFINITY;
    auto solve_point=[&](std::vector<double> fixed,bool integer) {
        auto f=build_formulation(d,paths,&fixed,nullptr,nullptr,!integer,false,true);
        NativeSolver solver(f.matrix,options);auto solution=solver.optimize(options.seconds);
        require(solution.has_solution&&solution.optimal,"Tiny full recourse did not solve optimally: "+solution.status);
        const auto residual=point_residuals(f.matrix,solution.values);
        require(residual.at("passed"),"Tiny full recourse residual failure: "+residual.dump());
        fixed.resize(master.matrix.variables.size());fixed[theta]=solution.objective;
        require(point_residuals(master.matrix,fixed).at("passed"),"Analytical strengthening excludes exact recourse");
        min_improvement=std::min(min_improvement,bounds.courier_cost-solution.objective);
        points.push_back({std::move(fixed),integer});++exact_solves;
    };
    for(int a=-1;a<bases;++a)for(int b=-1;b<bases;++b)solve_point(design(d,a,b),true);
    solve_point(design(d,0,bases-1,0),true);
    for(double infrastructure:{.02,.25,.7}) {
        auto fixed=design(d,0,bases-1,infrastructure);
        for(int r=0;r<2;++r)for(int l=0;l<bases;++l)fixed[layout.assignment_index(r,l)]=.7/bases;
        solve_point(fixed,false);
    }
    // Convex mixtures are also valid targets for the integer-convex-hull cuts.
    const auto original=points.size();
    for(std::size_t k=1;k<original;++k)if(points[k].integer) {
        auto mix=points[k].x;for(std::size_t j=0;j<mix.size();++j)mix[j]=.37*mix[j]+.63*points[0].x[j];
        points.push_back({std::move(mix),true});
    }
    double max_cut_error=0;std::size_t cross_checks=0;
    for(const auto& source:points) {
        const auto capacity=allocations.separate(source.x,0,theta);
        const auto weighted=resource.separate(source.x,0,theta,5);
        const auto strong=od_resource.separate(source.x,0,theta,5);
        require(weighted.valid&&strong.valid&&weighted.max_dual_violation_after==0&&strong.max_dual_violation_after==0,
                "Resource cut did not have a valid repaired dual certificate");
        for(const auto& target:points) {
            max_cut_error=std::max({max_cut_error,violation(capacity.row,target.x),violation(weighted.row,target.x)});
            cross_checks+=2;
            if(target.integer) {max_cut_error=std::max(max_cut_error,violation(strong.row,target.x));++cross_checks;}
        }
    }
    require(max_cut_error<1e-6,"A strengthened cut excludes exact recourse at another design");
    // Enumerate all physical service-link networks independently of the
    // topology oracle. Positive-rate filtering is intentionally absent: the
    // oracle searches this physical superset, which is safe for zero-service
    // certificates under fixed generalized task demands.
    std::vector<std::pair<int,int>> arcs;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        if(d.allowed_links(u,v)&&(u>=d.n_l||v>=d.n_l))arcs.emplace_back(u,v);
    const unsigned networks=1U<<arcs.size();std::size_t topology_cuts=0,topology_checks=0;
    for(unsigned mask=0;mask<networks;++mask) {
        auto anchor=design(d,0,-1);
        for(std::size_t k=0;k<arcs.size();++k)
            anchor[layout.airline+arcs[k].first*d.n_nodes()+arcs[k].second]=(mask>>k)&1U;
        const bool exists=std::any_of(paths.service_paths.begin(),paths.service_paths.end(),[&](const ServicePath& p) {
            return std::all_of(p.arc_counts.begin(),p.arc_counts.end(),[&](const auto& edge) {
                const auto [u,v,count]=edge;(void)count;return anchor[layout.airline+u*d.n_nodes()+v]>.5;
            });
        });
        const auto cut=separate_topology_cost(d,paths,anchor,theta,0,INFINITY);
        require(!cut.timed_out&&cut.valid==!exists&&cut.zero_service_certified==!exists,
                "Generalized topology oracle disagrees with complete physical catalogue");
        if(!cut.valid)continue;
        ++topology_cuts;
        for(const auto& p:paths.service_paths) {
            const bool hit=std::any_of(p.arc_counts.begin(),p.arc_counts.end(),[&](const auto& edge) {
                const auto [u,v,count]=edge;(void)count;
                return std::binary_search(cut.blocking_arcs.begin(),cut.blocking_arcs.end(),std::make_pair(u,v));
            });
            require(hit,"Topology blocking set misses a possible generalized route");
        }
        for(const auto& target:points)if(target.integer) {
            max_cut_error=std::max(max_cut_error,violation(cut.row,target.x));++topology_checks;
        }
    }
    require(max_cut_error<1e-6,"A topology cut excludes full integer recourse or its convex projection");
    return {{"bases",bases},{"task_cardinality",cardinality},{"paths_checked",paths.service_paths.size()},
        {"exact_full_recourse_solves",exact_solves},{"fractional_recourse_designs",3},
        {"cross_projection_checks",cross_checks},{"maximum_cut_violation",max_cut_error},
        {"maximum_cost_allocation_violation",max_cost_error},{"maximum_load_allocation_violation",max_load_error},
        {"minimum_improvement_over_all_courier",min_improvement},{"occurrence_supply_scope",rm.metadata.at("assignment_supply_scope")},
        {"topology_networks_exhaustively_checked",networks},{"topology_absence_cuts",topology_cuts},
        {"topology_cross_projection_checks",topology_checks}};
}
}
int main() {try {
    Json cases=Json::array();for(int cardinality:{1,2})for(int bases:{1,2})cases.push_back(evaluate(bases,cardinality));
    std::cout<<Json{{"passed",true},{"cases",cases},{"explicit_witnesses",{explicit_witness(false),explicit_witness(true)}}}.dump(2)<<'\n';return 0;
}catch(const IloException& e){std::cerr<<e<<'\n';return 1;}
 catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
