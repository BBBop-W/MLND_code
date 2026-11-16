#include "mlnd/benders.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

namespace {
using namespace mlnd;
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> Matrix<T> constant(int rows, int columns, T value) {
    Matrix<T> matrix(rows, columns);
    std::fill(matrix.values.begin(), matrix.values.end(), value);
    return matrix;
}
// Frozen copy of the established K2 benders_probe fixture, including the
// independently computed Poisson compartment thresholds.
Input fixture(const std::string& policy) {
    Input d;
    d.n_l=1; d.n_h=2; d.n_od=2; d.n_levels=2; d.max_orders=2;
    d.name="conditional_benders_test"; d.model_version="consecutive";
    d.route_mode="direct"; d.flow_policy=policy; d.demand={4,7};
    d.flight_times=constant<double>(3,3,0);
    d.flight_times.values={0,.05,.07,.05,0,.1,.07,.1,0};
    d.rider_mean=constant<double>(2,2,0); d.rider_mean.values={.08,.12,.1,.06};
    d.first_cost=constant<double>(2,2,1); d.terminal_cost=constant<double>(2,2,.5);
    d.receipt_mean=constant<double>(2,2,.08); d.removal_mean=constant<double>(2,2,.03);
    d.ground_cost={5,5}; d.deadline={1,1}; d.base_cost={.5};
    d.slot_counts=constant<int>(2,2,0); d.slot_counts.values={10,20,10,20};
    d.slot_cost=constant<double>(2,2,0); d.slot_cost.values={.1,.2,.1,.2};
    d.slot_epsilon={.05,.05}; d.link_cost=constant<double>(3,3,0);
    d.kappa=constant<double>(2,2,0); d.kappa.values={.3,.9,.2,.7};
    d.delta=.04; d.operating_limit=1; d.fleet_cost=1; d.flight_cost=.2; d.waiting_cost=.3;
    d.allowed_links=constant<int>(3,3,1);
    for(int i=0;i<3;++i) d.allowed_links(i,i)=0;
    d.occupancy_limit=constant<double>(2,2,0);
    d.occupancy_limit.values={6.1690072893953225,14.072024748341311,
                            6.1690072893953225,14.072024748341311};
    return d;
}
SolverOptions plain() {
    SolverOptions o;
    o.seconds=30; o.gap=1e-6; o.max_iterations=1000;
    o.reuse_dual_lp=o.pareto=o.reuse_columns=o.courier_savings_bound=false;
    o.initial_design=o.warm_start=o.incumbent_cleanup=o.primal_search=false;
    o.branch_covering_seeds=o.resource_bounds=o.capacity_cuts=false;
    o.resource_lp_cuts=o.adaptive_master=o.neighborhood_cuts=false;
    o.continuous_root=o.root_projection=o.root_inout=o.auxiliary_valid_cuts=false;
    o.resource_airline_bounds=o.root_cuts=o.restricted_mip=o.local_branching=o.incumbent_cutoff=false;
    return o;
}
std::vector<double> design(const Input& d, int assignment, int level=0,
                           int link_mask=63, int closed_locker=-1, bool all_closed=false) {
    DesignLayout l(d);
    std::vector<double> x(l.count,0);
    if(all_closed) return x;
    x[l.facility]=1;
    for(int h=0;h<d.n_h;++h) if(h!=closed_locker) {
        x[l.facility+d.n_l+h]=1;
        x[l.capacity+h*d.n_levels+level]=1;
    }
    for(int r=0;r<d.n_od;++r) x[l.assignment+r]=(assignment>>r)&1;
    int bit=0;
    for(int u=0;u<d.n_nodes();++u) for(int v=0;v<d.n_nodes();++v) if(d.allowed_links(u,v)) {
        if(x[l.facility+u] && x[l.facility+v]) x[l.airline+u*d.n_nodes()+v]=(link_mask>>bit)&1;
        ++bit;
    }
    require(point_residuals(design_model(d).matrix,x).at("passed").get<bool>(),"Fixture produced illegal design");
    return x;
}
double cut_value(const Json& cut,const std::vector<double>& point) {
    long double value=cut.at("intercept").get<double>();
    const auto coefficients=cut.at("coefficients").get<std::vector<double>>();
    require(coefficients.size()==point.size(),"Conditional cut has wrong dimensions");
    for(std::size_t i=0;i<point.size();++i) value+=static_cast<long double>(coefficients[i])*point[i];
    return static_cast<double>(value);
}
std::string assignment_key(const Input& d,const std::vector<double>& point) {
    const DesignLayout l(d); std::string key;
    for(int j=l.assignment;j<l.airline;++j) key.push_back(point[j]==1?'1':'0');
    return key;
}
void matching_objective(const Json& result,double reference,const char* label) {
    const double tolerance=3e-5;
    require(std::abs(result.at("objective").get<double>()-reference)<=tolerance,
            std::string(label)+" changed the original optimal objective");
    require(result.at("bound").get<double>()<=reference+tolerance,
            std::string(label)+" has an invalid global bound");
    require(result.at("gap").get<double>()<=4e-6,
            std::string(label)+" did not certify the requested tiny-instance gap");
}
}

int main(int argc,char** argv) {
    try {
        Json records=Json::array();
        for(const std::string policy:{"aggregate","prefix"}) {
            const auto d=fixture(policy);
            const auto catalogue=enumerate_paths(d);
            auto base=plain();
            const auto full=solve_full(d,catalogue,base);
            const double optimum=full.at("objective");
            require(full.at("gap").get<double>()<=1e-6,"Full MISOCP reference was not certified");
            Json variants=Json::object();
            for(const std::string mode:{"lifted_plain","conditional_plain","conditional_accelerated","conditional_face_pareto"}) {
                SolverOptions o=mode=="conditional_accelerated"?SolverOptions{}:base;
                o.seconds=30; o.gap=1e-6; o.max_iterations=1000;
                o.conditional_demand=mode!="lifted_plain";
                if(mode=="conditional_face_pareto") {
                    o.pareto=o.auxiliary_valid_cuts=true;
                    o.pareto_frequency=1; o.pareto_budget_fraction=.3; o.pareto_call_seconds=5;
                }
                auto pool=prepare_path_support(d);
                auto answer=solve_benders(d,pool,o);
                matching_objective(answer,optimum,mode.c_str());
                const auto& structure=answer.at("master_structure");
                require(structure.at("variables").get<int>()==DesignLayout(d).count+1 &&
                        structure.at("cones").get<int>()==0 &&
                        structure.at("operational_auxiliary_variables").get<int>()==0,
                        "Conditional mode changed the lean MILP master");
                if(o.conditional_demand) {
                    require(answer.at("demand_formulation_scope").at("integer_recourse")=="conditional" &&
                            answer.at("demand_formulation_scope").at("fractional_recourse")=="lifted" &&
                            answer.at("demand_formulation_scope").at("continuous_root")=="lifted" &&
                            answer.at("demand_formulation_scope").at("restricted_mip")=="lifted",
                            "Conditional mode leaked into a free or fractional lifted model");
                    for(const auto& cut:answer.at("benders_cuts")) {
                        if(cut.at("kind")=="root")
                            require(cut.at("condition").is_null(),"Global root cut was incorrectly assignment-gated");
                        else require(!cut.at("condition").is_null() &&
                            cut.at("condition").at("cut_scope")=="fixed_assignment_all_facility_designs",
                            "Integer-recourse cut escaped without assignment gating");
                    }
                }
                if(mode=="conditional_accelerated") {
                    require(answer.at("continuous_root").at("bound_valid").get<bool>(),"Lifted continuous root was not retained");
                    require(answer.at("continuous_root").at("certified_lower_bound").get<double>()<=optimum+3e-5,
                            "Retained root bound exceeds integer optimum");
                    bool excluded=false;
                    for(const auto& round:answer.at("root_diagnostics").at("rounds"))
                        excluded=excluded || round.value("conditional_rows_excluded",0)>0;
                    require(excluded,"Root test did not exclude previously generated integer-only conditional cuts");
                }
                if(mode=="conditional_face_pareto") {
                    require(answer.at("acceleration_statistics").at("pareto_attempts").get<int>()>0,
                            "Conditional Pareto test did not exercise auxiliary solves");
                    for(const auto& iteration:answer.at("iterations"))
                        if(iteration.at("pareto").value("attempted",false))
                            require(iteration.at("pareto").at("scope")=="fixed_assignment_face",
                                    "Conditional Pareto was labeled global");
                }
                std::cerr<<policy<<" "<<mode<<" objective="<<answer.at("objective")
                         <<" gap="<<answer.at("gap")<<'\n';
                variants[mode]=std::move(answer);
            }

            // Direct old full-path solves provide independent operating values
            // for all cross-design certificate checks below.
            std::set<std::vector<double>> unique;
            for(int assignment=0;assignment<4;++assignment) {
                unique.insert(design(d,assignment));
                for(int bit=0;bit<6;++bit) unique.insert(design(d,assignment,0,63^(1<<bit)));
                unique.insert(design(d,assignment,1));
                unique.insert(design(d,assignment,0,63,0));
                unique.insert(design(d,assignment,0,63,1));
            }
            unique.insert(design(d,0,0,0,-1,true));
            std::vector<std::vector<double>> targets(unique.begin(),unique.end());
            std::vector<double> operating;
            for(const auto& target:targets) {
                const auto reference=solve_full(d,catalogue,base,&target);
                require(reference.at("gap").get<double>()<=1e-6,"Fixed-design old full reference not certified");
                operating.push_back(reference.at("continuous_cost"));
            }
            // Check all actual outer-loop cuts, including the face-Pareto
            // auxiliaries and the retained globally valid lifted root cuts.
            std::size_t outer_inequalities=0;
            for(const auto& answer:variants) for(const auto& cut:answer.at("benders_cuts"))
                for(std::size_t j=0;j<targets.size();++j) {
                    const double rhs=cut_value(cut,targets[j]);
                    require(rhs<=operating[j]+3e-6,"Outer ordinary/Pareto/root cut exceeds old recourse value");
                    if(!cut.at("condition").is_null() &&
                        cut.at("condition").at("assignment_fingerprint")!=assignment_key(d,targets[j]))
                        require(rhs<=1e-8,"Outer conditional/Pareto cut escaped its assignment face");
                    ++outer_inequalities;
                }
            const std::vector<std::vector<double>> anchors{
                design(d,3),design(d,3,0,62),design(d,1),design(d,2,1),design(d,0,0,0,-1,true)};
            Json cut_checks=Json::array();
            std::size_t inequalities=0;
            for(const auto& anchor:anchors) {
                auto o=base; o.conditional_demand=true; o.column_generation=false;
                auto pool=catalogue;
                const auto compact=solve_fixed_design_cg(d,pool,o,anchor);
                require(compact.at("pricing_complete").get<bool>() && compact.at("benders_cuts").size()==1,
                        "Full compact subproblem did not emit a certified conditional cut");
                const auto old=solve_full(d,catalogue,base,&anchor);
                matching_objective(compact,old.at("objective"),"fixed compact full");
                const auto& cut=compact.at("benders_cuts").front();
                require(cut.at("condition").at("assignment_fingerprint")==assignment_key(d,anchor),
                        "Conditional cut uses a different anchor assignment");
                double worst=-1e100,wrong_pattern=-1e100;
                for(std::size_t j=0;j<targets.size();++j) {
                    const double rhs=cut_value(cut,targets[j]);
                    worst=std::max(worst,rhs-operating[j]);
                    require(rhs<=operating[j]+3e-6,"Conditional cut exceeds independent old recourse value");
                    if(assignment_key(d,targets[j])!=assignment_key(d,anchor)) {
                        wrong_pattern=std::max(wrong_pattern,rhs);
                        require(rhs<=1e-8,"Conditional cut remains active outside its assignment pattern");
                    }
                    ++inequalities;
                }
                cut_checks.push_back({{"anchor",anchor},{"worst_overestimate",worst},
                    {"other_assignment_max_rhs",wrong_pattern},{"cut",cut}});
            }
            auto fixed_options=base; fixed_options.conditional_demand=true; fixed_options.cg_max_columns=1;
            auto pool=prepare_path_support(d);
            const auto fixed=solution_design(d,full);
            const auto fixed_cg=solve_fixed_design_cg(d,pool,fixed_options,fixed);
            matching_objective(fixed_cg,optimum,"fixed compact implicit CG");
            require(fixed_cg.at("pricing_complete").get<bool>() && fixed_cg.at("cg_iterations").get<int>()>1,
                    "Fixed compact test did not complete actual incremental pricing");
            auto stopped_options=fixed_options; stopped_options.seconds=0;
            auto stopped_pool=prepare_path_support(d);
            const auto stopped=solve_fixed_design_cg(d,stopped_pool,stopped_options,fixed);
            require(!stopped.at("pricing_complete").get<bool>() && stopped.at("benders_cuts").empty(),
                    "Stopped compact subproblem emitted a cut");
            require(solution_design(d,stopped)==fixed,"Stopped compact subproblem changed its integer design");
            records.push_back({{"policy",policy},{"K",2},{"full",full},{"variants",variants},
                {"cross_design_count",targets.size()},{"cut_inequalities_checked",inequalities},
                {"outer_cut_inequalities_checked",outer_inequalities},
                {"cut_checks",cut_checks},{"fixed_implicit_cg",fixed_cg},{"zero_budget",stopped},{"passed",true}});
            std::cerr<<policy<<" cross-design inequalities="<<inequalities<<" passed\n";
        }
        if(argc>1) {
            std::ofstream output(argv[1]); require(static_cast<bool>(output),"Cannot write conditional probe output");
            output<<Json{{"passed",true},{"cases",records}}.dump(2)<<'\n';
        } else std::cout<<Json{{"passed",true},{"cases",records}}.dump()<<'\n';
    } catch(const IloException& e) { std::cerr<<e<<'\n'; return 1; }
      catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
