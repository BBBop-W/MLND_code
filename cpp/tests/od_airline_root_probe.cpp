#include "mlnd/benders.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace mlnd;
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> Matrix<T> constant(int rows, int columns, T value) {
    Matrix<T> result(rows, columns);
    std::fill(result.values.begin(), result.values.end(), value);
    return result;
}
// A complete physical example with queueing, not an arbitrary resource LP:
// demand 1 can use the only three-leg route; demand 9 cannot meet its deadline.
// Fractional original links cost only 3*s/10 (K=1), while integer-valid OD
// linking strengthens the projected root above the original continuous UB.
Input fixture(int k) {
    Input d;
    d.n_l=1; d.n_h=2; d.n_od=2; d.n_levels=1; d.max_orders=k;
    d.name="od_airline_root_scope_fixture"; d.model_version="consecutive";
    d.route_mode="direct"; d.flow_policy="aggregate";
    d.demand={1,9}; d.ground_cost={10,10}; d.deadline={1,.05}; d.base_cost={0};
    d.flight_times=constant<double>(3,3,.1);
    d.allowed_links=constant<int>(3,3,0);
    d.allowed_links(0,1)=d.allowed_links(1,2)=d.allowed_links(2,0)=1;
    for(int i=0;i<3;++i) d.flight_times(i,i)=0;
    d.link_cost=constant<double>(3,3,1);
    d.rider_mean=constant<double>(2,2,0);
    d.first_cost=constant<double>(2,2,0); d.terminal_cost=constant<double>(2,2,0);
    d.receipt_mean=constant<double>(2,2,0); d.removal_mean=constant<double>(2,2,0);
    d.slot_counts=constant<int>(2,1,100); d.slot_cost=constant<double>(2,1,0);
    d.occupancy_limit=constant<double>(2,1,100); d.slot_epsilon={.05,.05};
    d.kappa=constant<double>(2,2,0); d.delta=.04; d.operating_limit=1;
    d.fleet_cost=1; d.flight_cost=0; d.waiting_cost=0;
    return d;
}
SolverOptions root_options(bool conditional, bool od) {
    SolverOptions o;
    o.seconds=30; o.threads=1; o.gap=1e-8; o.max_iterations=0;
    o.root_budget_fraction=1; o.root_cut_rounds=100;
    o.continuous_root_seconds=10; o.continuous_root_budget_fraction=.3;
    o.initial_design=o.restricted_mip=o.neighborhood_cuts=false;
    o.pareto=o.primal_search=o.local_branching=false;
    o.warm_start=o.incumbent_cleanup=o.incumbent_cutoff=false;
    o.root_inout=false; o.resource_lp_seconds=1; o.resource_lp_cut_limit=500;
    o.capacity_cuts=false; o.od_airline_bounds=od; o.conditional_demand=conditional;
    return o;
}
}

int main(int argc, char** argv) {
    try {
        if(argc>2) throw std::invalid_argument("Usage: mlnd_od_airline_root_probe [OUTPUT_JSON]");
        Json rows=Json::array();
        for(int k:{1,2}) {
            const auto d=fixture(k);
            const auto catalogue=enumerate_paths(d);
            SolverOptions full_options; full_options.seconds=10; full_options.gap=1e-8;
            const auto full=solve_full(d,catalogue,full_options);
            require(full.at("gap").get<double>()<1e-7,"Integer oracle was not certified");
            const double integer_optimum=full.at("objective");
            if(k==1) require(std::abs(integer_optimum-(103-std::pow(std::sqrt(9.7)-1,2)))<1e-5,
                "Complete integer fixture disagrees with analytical queueing optimum");
            for(bool conditional:{false,true}) for(bool od:{false,true}) {
                auto pool=prepare_path_support(d);
                const auto result=solve_benders(d,pool,root_options(conditional,od));
                const auto& original=result.at("continuous_root");
                const auto& root=result.at("root_diagnostics");
                require(original.value("bound_valid",false) && original.value("primal_valid",false),
                    "Original continuous CG reference lacks valid lower/primal certificates");
                const double old_upper=original.at("continuous_objective");
                const double root_lower=root.at("lower_bound");
                require(root_lower<=integer_optimum+3e-5,"Root lower bound exceeds complete integer optimum");
                require(root.at("original_continuous_value_closed").get<bool>(),
                    "Original continuous closure was not retained independently");
                require(result.at("master_structure").at("operational_auxiliary_variables")==0,
                    "OD root strengthening introduced operational master variables");
                if(od) {
                    require(root_lower>old_upper+.2,
                        "Regression fixture did not exercise strengthened LB above original continuous UB");
                    require(root.at("scope")=="integer_valid_strengthened_root" &&
                        root.at("continuous_upper_bound").is_null() && root.at("gap").is_null(),
                        "Original continuous UB/gap leaked into strengthened root diagnostics");
                    require(!root.at("converged").get<bool>() &&
                        !root.at("candidate_epigraph_certified").get<bool>(),
                        "Original recourse closure was mislabeled as strengthened-root closure");
                    require(result.at("od_airline_summary").at("cuts_with_active_od_prices").get<int>()>0,
                        "Strengthened test generated no active OD gate certificate");
                    const auto& rounds=root.at("rounds");
                    for(std::size_t i=0;i<rounds.size();++i) {
                        const auto& round=rounds[i];
                        require(round.at("continuous_upper_bound").is_null() && round.at("continuous_gap").is_null(),
                            "A root round incorrectly mixed original and strengthened bounds");
                        if(round.value("original_candidate_recourse_consistent",false) &&
                            (round.value("resource_lp_cuts_added",0)>0 || round.value("capacity_cuts_added",0)>0))
                            require(i+1<rounds.size() || root.at("termination_reason")!="original_candidate_recourse_consistent",
                                "Root stopped for candidate consistency after adding a violated resource cut");
                    }
                } else {
                    require(root.at("scope")=="full_path_continuous_relaxation" &&
                        root.at("continuous_upper_bound").is_number() && root.at("gap").is_number(),
                        "Disabling OD gates changed legacy root diagnostic scope");
                    require(root_lower<=old_upper+3e-5,"Unstrengthened root exceeds original continuous UB");
                }
                rows.push_back({{"K",k},{"conditional",conditional},{"od_airline_bounds",od},
                    {"integer_oracle",integer_optimum},{"original_continuous_upper",old_upper},
                    {"root_lower",root_lower},{"root_diagnostics",root},
                    {"od_airline_summary",result.at("od_airline_summary")}});
                std::cerr<<"K="<<k<<" conditional="<<conditional<<" od="<<od
                         <<" original_UB="<<old_upper<<" strengthened_LB="<<root_lower<<'\n';
            }
        }
        Json report={{"passed",true},{"comparisons",rows.size()},{"rows",rows}};
        if(argc==2) { std::ofstream file(argv[1]); file<<report.dump(2)<<'\n'; }
        std::cout<<Json{{"passed",true},{"comparisons",rows.size()}}.dump()<<'\n';
    } catch(const IloException& error) { std::cerr<<error<<'\n'; return 1; }
      catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
