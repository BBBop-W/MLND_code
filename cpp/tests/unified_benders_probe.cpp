// Fixed-task recourse, exact pricing, and global Benders cut regression.
#include "mlnd/benders.hpp"
#include "mlnd/heuristics.hpp"
#include "mlnd/preprocessed_demand.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace mlnd;
namespace {
void require(bool ok,const std::string& why) { if(!ok) throw std::runtime_error(why); }
template<class T> Matrix<T> fill(int r,int c,T v) {
    Matrix<T> a(r,c);std::fill(a.values.begin(),a.values.end(),v);return a;
}
Input fixture(int k) {
    Input d;d.n_l=2;d.n_h=d.n_od=2;d.n_levels=1;d.max_orders=k;
    d.name="unified_benders_witness";d.model_version="consecutive";
    d.route_mode="direct";d.flow_policy="aggregate";
    d.demand={4,7};d.ground_cost={5,5};d.deadline={1,1};
    d.flight_times=fill(4,4,0.0);
    d.flight_times.values={0,.03,.05,.07,.03,0,.06,.04,.05,.06,0,.1,.07,.04,.1,0};
    d.allowed_links=fill(4,4,1);for(int i=0;i<4;++i)d.allowed_links(i,i)=0;
    d.link_cost=fill(4,4,0.0);d.base_cost={.5,.4};d.slot_epsilon={.05,.05};
    d.rider_mean=fill(2,2,.08);d.first_cost=fill(2,2,1.0);d.terminal_cost=fill(2,2,.5);
    d.receipt_mean=fill(2,2,.08);d.removal_mean=fill(2,2,.03);
    d.slot_counts=fill(2,1,20);d.slot_cost=fill(2,1,.1);d.occupancy_limit=fill(2,1,14.0);
    d.kappa=fill(2,2,0.0);d.kappa(0,1)=1;
    d.delta=.04;d.operating_limit=1.5;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    return d;
}
std::vector<double> design(const Input& d,int a,int b) {
    DesignLayout layout(d);std::vector<double>x(layout.count);
    std::fill(x.begin(),x.begin()+d.n_nodes(),1);
    for(int h=0;h<d.n_h;++h)x[layout.capacity+h]=1;
    if(a>=0)x[layout.assignment_index(0,a)]=1;
    if(b>=0)x[layout.assignment_index(1,b)]=1;
    for(int u=0;u<d.n_nodes();++u)for(int v=0;v<d.n_nodes();++v)
        x[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    return x;
}
double exact_cost(const Input& d,const PathCollection& p,const std::vector<double>& x,const SolverOptions& o) {
    auto f=build_formulation(d,p,&x,nullptr,nullptr,false,false,o.tight_airline_bounds,formulation_strengthening(o));
    NativeSolver solver(f.matrix,o,"unified_probe_reference");auto solved=solver.optimize(15);
    require(solved.optimal&&solved.has_solution,"Full recourse reference did not solve");
    require(point_residuals(f.matrix,solved.values).at("passed").get<bool>(),"Invalid full reference");
    return solved.objective;
}
}
int main() { try {
    Json cases=Json::array();
    for(int k:{1,2})for(bool strengthened:{false,true}) {
        auto d=fixture(k);auto p=enumerate_paths(d);auto legacy=d;
        // Singles do not use compatibility, even when legacy input supplies
        // a fractional matrix: the same fixed-task model simply has no pairs.
        if(k==1)d.kappa(0,1)=.4;
        preprocess_generalized_demand(d,p);
        require(d.assignment_count()==4,"Unified model lost origin assignment");
        require(k==2||d.single_rates==d.demand,"Singles reconstruction changed demand");
        SolverOptions o;o.seconds=30;o.gap=1e-8;o.threads=1;o.pareto=false;
        o.cg_max_columns=4;o.preprocessed_demand=true;
        o.assignment_incident_clique=o.shared_rho_joint=o.queue_perspective=strengthened;
        auto x=design(d,0,-1);const double ref=exact_cost(d,p,x,o);
        if(k==1)require(std::abs(ref-exact_cost(legacy,p,x,o))<1e-5,"Unified singles differ from legacy K1");
        const auto cg=solve_fixed_design_cg(d,p,o,x);
        require(cg.at("pricing_complete").get<bool>(),"Fixed-task CG was not certified: "+cg.at("termination_reason").dump());
        require(std::abs(cg.at("continuous_cost").get<double>()-ref)<1e-4,"CG/full recourse mismatch");
        require(cg.contains("benders_cuts"),"Missing globally valid cut");
        if(k==2&&strengthened) {
            auto heuristic_options=o;heuristic_options.seconds=1;
            const auto recovery=solve_restricted_heuristic(d,p,heuristic_options,wall_time()+1,&cg,-1);
            require(recovery.at("diagnostics").at("incumbent_validation").at("passed").get<bool>(),
                    "Native heuristic rejected valid incoming-second service");
        }
        double worst=0;int checked=0;
        for(int a:{-1,0,1})for(int b:{-1,0,1}) {
            auto point=design(d,a,b);const double value=exact_cost(d,p,point,o);
            for(const auto& cut:cg.at("benders_cuts")) {
                const auto beta=cut.at("coefficients").get<std::vector<double>>();
                const double bound=cut.at("intercept").get<double>()+std::inner_product(beta.begin(),beta.end(),point.begin(),0.0);
                worst=std::max(worst,bound-value);++checked;
                require(bound<=value+1e-4,"Benders cut invalid at another assignment");
                require(!cut.contains("condition")||cut.at("condition").is_null(),"Fixed task cut was assignment-gated");
            }
        }
        cases.push_back({{"k",k},{"strengthened",strengthened},{"full_recourse",ref},
            {"cg_recourse",cg.at("continuous_cost")},{"cut_checks",checked},{"worst_cut_excess",worst},
            {"paths",p.service_paths.size()},{"cg_columns",cg.at("service_columns")}});
    }
    auto d=fixture(2);auto paths=enumerate_paths(d);preprocess_generalized_demand(d,paths);
    SolverOptions o;o.seconds=10;o.gap=1e-8;o.preprocessed_demand=true;o.pareto=false;
    o.initial_design=o.restricted_mip=o.primal_search=o.neighborhood_cuts=false;
    o.root_cuts=o.root_projection=o.capacity_cuts=o.resource_lp_cuts=false;
    o.continuous_root_seconds=4;o.continuous_root_max_columns=4;o.max_iterations=1;
    o.cg_max_columns=4;o.root_budget_fraction=0;
    const auto outer=solve_benders(d,paths,o);
    const auto& root=outer.at("continuous_root");
    require(root.value("model_construction_complete",false),"Root reuse probe did not construct root");
    require(root.at("reused_service_columns")==root.at("service_columns"),"Root reuse includes ungenerated catalogue columns");
    require(root.at("reused_service_columns").get<std::size_t>()<paths.service_paths.size(),"Fixture did not produce a strict root-column subset");
    require(!outer.at("iterations").empty(),"Root reuse probe did not reach integer recourse");
    const auto& first=outer.at("iterations").front().at("cg_trace").front();
    require(first.at("service_columns")==root.at("reused_service_columns"),"Integer recourse did not start from actual root columns");
    Json reuse={{"catalogue",paths.service_paths.size()},{"root_rmp",root.at("service_columns")},
        {"first_integer_rmp",first.at("service_columns")},{"generated_columns",outer.at("generated_columns")},
        {"generated_columns_scope",outer.at("generated_columns_scope")}};
    std::cout<<Json{{"passed",true},{"cases",cases},{"root_reuse",reuse}}.dump(2)<<'\n';return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
