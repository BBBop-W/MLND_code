// Independent end-to-end CG / conic-dual / conditional-gate certification.
// Uses full old-formulation fixed-design values only as reference evidence.
#include "mlnd/benders.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace mlnd;
void require(bool condition,const std::string& reason) {
    if(!condition) throw std::runtime_error(reason);
}
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
    d.max_orders = k; d.name = "structural_cg_certificate_fixture";
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

SolverOptions options(bool strengthened,bool conditional) {
    SolverOptions o; o.seconds=12; o.threads=1; o.gap=1e-8;
    o.cg_max_columns=3; o.reuse_dual_lp=true; o.pareto=false;
    o.branch_covering_seeds=false; o.conditional_demand=conditional;
    o.assignment_incident_clique=strengthened;
    o.shared_rho_joint=strengthened; o.queue_perspective=strengthened;
    return o;
}
std::vector<double> design(const Input& d,int choice0,int choice1,bool closed=false) {
    const DesignLayout l(d); std::vector<double> x(l.count,0);
    if(closed) return x;
    std::fill(x.begin(),x.begin()+d.n_nodes(),1);
    for(int h=0;h<d.n_h;++h) x[l.capacity+h*d.n_levels]=1;
    if(choice0>=0) x[l.assignment+choice0]=1;
    if(choice1>=0) x[l.assignment+4+choice1]=1;
    for(int u=0;u<d.n_nodes();++u) for(int v=0;v<d.n_nodes();++v)
        x[l.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
    return x;
}
double cut_value(const Json& cut,const std::vector<double>& x) {
    const auto beta=cut.at("coefficients").get<std::vector<double>>();
    require(beta.size()==x.size(),"Cut coefficient layout mismatch");
    long double value=cut.at("intercept").get<double>();
    for(std::size_t j=0;j<x.size();++j) value+=static_cast<long double>(beta[j])*x[j];
    return static_cast<double>(value);
}
Json run() {
    Json out={{"schema","mlnd-structural-cg-cut-regression/1"},{"cases",Json::array()}};
    std::size_t checks=0,changed_assignment_checks=0,conditional_cuts=0,closed_checks=0;
    double worst=0,max_objective_change=0;
    for(const auto& policy:{std::string("aggregate"),std::string("prefix")}) for(int k:{1,2}) {
        const auto d=fixture(k,policy,true);const auto catalogue=enumerate_paths(d);
        auto reference_options=options(false,false);
        const auto original=build_formulation(d,catalogue);
        NativeSolver integer_solver(original.matrix,reference_options,"structural_cg_integer_reference");
        const auto integer=integer_solver.optimize(12);
        require(integer.optimal && integer.has_solution &&
            point_residuals(original.matrix,integer.values).at("passed").get<bool>(),"Integer reference not validated");
        std::vector<std::vector<double>> designs;
        for(int a=-1;a<4;++a) for(int b=-1;b<4;++b) designs.push_back(design(d,a,b));
        designs.push_back(design(d,-1,-1,true));
        // Close one base while retaining a legitimate assigned pool; tests
        // normal zero-apex perspective cones beside an active positive cone.
        auto one_closed=design(d,0,0);one_closed[1]=0;
        for(int u=0;u<d.n_nodes();++u) {
            one_closed[original.layout.airline+u*d.n_nodes()+1]=0;
            one_closed[original.layout.airline+d.n_nodes()+u]=0;
        }
        designs.push_back(one_closed);
        std::vector<double> recourse;Json references=Json::array();
        for(const auto& x:designs) {
            const auto fixed=build_formulation(d,catalogue,&x);
            NativeSolver solver(fixed.matrix,reference_options,"structural_cg_fixed_reference");
            auto solved=solver.optimize(12);
            require(solved.optimal && solved.has_solution,"Old fixed reference failed to solve");
            require(repair_inventory_primal(fixed,solved.values).at("passed").get<bool>(),"Old fixed reference primal invalid");
            const double q=objective_value(fixed.matrix,solved.values);recourse.push_back(q);
            references.push_back({{"design",x},{"operating_cost",q}});
        }
        Json records=Json::array();
        // all-closed, assigned with a closed base, same-base, cross-base,
        // and different OD pools exercise distinct conditional gate patterns.
        const std::vector<int> anchor_indices={25,26,6,12,9};
        for(bool strengthened:{false,true}) for(bool conditional:{false,true}) for(int a:anchor_indices) {
            auto pool=prepare_path_support(d);const auto x=designs.at(a);
            const auto answer=solve_fixed_design_cg(d,pool,options(strengthened,conditional),x);
            require(answer.at("pricing_complete").get<bool>(),"CG lacks full-domain certificate: K="+std::to_string(k)+
                " policy="+policy+" strengthened="+std::to_string(strengthened)+" conditional="+
                std::to_string(conditional)+" anchor="+std::to_string(a)+" reason="+answer.value("termination_reason",std::string()));
            require(answer.at("native_model_validated").get<bool>(),"CG primal not validated");
            const double objective=answer.at("continuous_cost").get<double>();
            max_objective_change=std::max(max_objective_change,std::abs(objective-recourse.at(a)));
            require(std::abs(objective-recourse.at(a))<=3e-5,"Integer-design CG changed original operating objective");
            require(!answer.at("benders_cuts").empty(),"Certified CG has no cut");
            for(const auto& cut:answer.at("benders_cuts")) {
                require(cut.at("pricing_complete").get<bool>() && cut.at("support_complete").get<bool>(),"Cut lacks path certificate");
                if(conditional) {
                    require(cut.at("condition").is_object() &&
                        cut.at("condition").at("cut_scope")=="fixed_assignment_all_facility_designs","Conditional cut escaped assignment gate");
                    ++conditional_cuts;
                } else require(cut.at("condition").is_null(),"Lifted global cut was assignment gated");
                for(std::size_t b=0;b<designs.size();++b) {
                    const double violation=cut_value(cut,designs[b])-recourse[b];
                    worst=std::max(worst,violation);++checks;
                    bool same=true;
                    for(int j=original.layout.assignment;j<original.layout.airline;++j) same=same && x[j]==designs[b][j];
                    if(conditional && !same) ++changed_assignment_checks;
                    require(violation<=3e-6*(1+std::abs(recourse[b])),"Certified cut excludes an old-formulation feasible strategy");
                }
            }
            if(a==25 || a==26) ++closed_checks;
            records.push_back({{"strengthened",strengthened},{"conditional",conditional},{"anchor_index",a},
                {"objective",objective},{"reference",recourse.at(a)}, {"pricing_complete",true},
                {"status",answer.at("termination_reason")},{"cg_iterations",answer.at("cg_iterations")},
                {"subproblem_model_size",answer.at("subproblem_model_size")},{"cuts",answer.at("benders_cuts")},
                {"cg_trace",answer.at("cg_trace")}});
        }
        Json roots=Json::array();double old_root=NAN;
        for(bool strengthened:{false,true}) {
            std::array<double,2> objectives{};
            for(bool conditional:{false,true}) {
                auto pool=prepare_path_support(d);
                const auto answer=solve_continuous_cg(d,pool,options(strengthened,conditional));
                require(answer.at("bound_valid").get<bool>() && answer.at("primal_valid").get<bool>() &&
                    answer.at("pricing_complete").get<bool>(),"Shared free continuous CG was not certified");
                const double q=answer.at("continuous_objective").get<double>();objectives[conditional]=q;
                require(answer.at("certified_lower_bound").get<double>()<=integer.objective+3e-5,"Global strengthened CG LB exceeds old integer optimum");
                if(!strengthened && !conditional) old_root=q;
                if(strengthened) require(q>=old_root-3e-5,"Strengthening decreased free continuous optimum");
                roots.push_back({{"strengthened",strengthened},{"conditional_option",conditional},{"result",answer}});
            }
            require(std::abs(objectives[0]-objectives[1])<=3e-5,"Conditional option changed common free lifted root");
        }
        out["cases"].push_back({{"K",k},{"policy",policy},{"reference_integer_objective",integer.objective},
            {"references",references},{"cg_certificates",records},{"continuous_roots",roots}});
    }
    out["cross_design_cut_checks"]=checks;out["conditional_changed_assignment_checks"]=changed_assignment_checks;
    out["conditional_cuts"]=conditional_cuts;out["closed_base_cg_checks"]=closed_checks;
    out["max_cross_design_violation"]=worst;out["max_fixed_objective_change"]=max_objective_change;
    out["end_to_end_benders"]=Json::array();
    for(int k:{1,2}) {
        const auto d=fixture(k,"aggregate");const auto catalogue=enumerate_paths(d);
        auto reference_options=options(false,false);
        const auto full=solve_full(d,catalogue,reference_options);
        require(full.at("gap").get<double>()<=1e-7,"End-to-end old full reference not optimal");
        const double optimum=full.at("objective").get<double>();
        for(bool conditional:{false,true}) {
            SolverOptions o;o.seconds=30;o.threads=1;o.gap=1e-6;o.max_iterations=1000;
            o.assignment_incident_clique=o.shared_rho_joint=o.queue_perspective=true;
            o.conditional_demand=conditional;
            auto pool=prepare_path_support(d);
            const auto answer=solve_benders(d,pool,o,&full);
            const auto& master=answer.at("master_structure");
            require(master.at("variables").get<int>()==DesignLayout(d).count+1 &&
                master.at("operational_auxiliary_variables")==0 && master.at("cones")==0,
                "End-to-end master added operational variables or cones");
            require(std::abs(answer.at("objective").get<double>()-optimum)<=3e-5,
                "End-to-end strengthened Benders differs from original integer optimum");
            require(answer.at("bound").get<double>()<=optimum+3e-5 &&
                answer.at("gap").get<double>()<=2e-6,"End-to-end Benders did not close certified gap");
            const auto& root=answer.at("continuous_root");
            require(root.at("bound_valid").get<bool>() && root.at("primal_valid").get<bool>() &&
                root.at("pricing_complete").get<bool>(),"End-to-end common root was not certified");
            require(root.at("scope")=="structurally_strengthened_continuous_relaxation",
                "End-to-end root strengthening scope omitted");
            const auto& scope=answer.at("demand_formulation_scope");
            require(scope.at("integer_recourse")== (conditional?"conditional":"lifted") &&
                scope.at("fractional_recourse")=="lifted" && scope.at("continuous_root")=="lifted" &&
                scope.at("restricted_mip")=="lifted","End-to-end demand formulation wiring incorrect");
            bool supplied=false,restricted_validated=false;
            for(const auto& record:answer.at("acceleration_records")) {
                supplied=supplied || record.value("stage",std::string())=="supplied_design";
                if(record.value("stage",std::string())=="initial_restricted_mip") {
                    restricted_validated=record.value("validated",false) &&
                        record.at("diagnostics").at("incumbent_validation").at("passed").get<bool>() &&
                        record.at("diagnostics").at("fallback_validation").at("passed").get<bool>();
                }
            }
            require(supplied,"Supplied full-model design was not evaluated by CG");
            require(restricted_validated &&
                answer.at("acceleration_statistics").at("master_starts").get<int>()>0,
                "Restricted-MIP start mapping or actual master warm start not exercised");
            require(answer.at("accelerations").at("warm_start").get<bool>() &&
                answer.at("accelerations").at("restricted_mip").get<bool>(),
                "End-to-end accelerated defaults were disabled");
            out["end_to_end_benders"].push_back({{"K",k},{"conditional",conditional},
                {"old_full_reference_objective",optimum},{"supplied_design_evaluated",supplied},
                {"result",answer}});
        }
    }
    out["passed"]=true;
    return out;
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::invalid_argument("Usage: mlnd_structural_cg_cut_probe OUTPUT_JSON");
        const auto output=std::filesystem::absolute(argv[1]);
        if(std::filesystem::exists(output)) throw std::invalid_argument("Refusing to overwrite existing report");
        const auto result=run();mlnd::atomic_json_file(output,result);
        std::cout<<"Structural CG cut checks="<<result.at("cross_design_cut_checks")<<" passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
