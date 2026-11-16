// End-to-end regression for valid full-domain cuts whose conservative
// floating-point correction prevents a tight recourse certificate.
#define main primal_polish_probe_original_main
#include "primal_polish_probe.cpp"
#undef main
#include "mlnd/benders.hpp"

namespace {
double full_recourse_value(const Input& d,const PathCollection& paths,
                          const SolverOptions& options,const std::vector<double>& design) {
    const auto f=build_formulation(d,paths,&design,nullptr,nullptr,true);
    NativeSolver solver(f.matrix,options,"weak_dual_reference");
    const auto solved=solver.optimize(15);
    require(solved.optimal && solved.has_solution,"Independent full recourse did not solve optimally");
    require(point_residuals(f.matrix,solved.values).at("passed"),"Independent recourse failed original residuals");
    return objective_value(f.matrix,solved.values);
}
double affine_value(const Json& cut,const std::vector<double>& design) {
    const auto coefficients=cut.at("coefficients").get<std::vector<double>>();
    require(coefficients.size()==design.size(),"Weak cut lost strategic coefficients");
    long double value=cut.at("intercept").get<double>();
    for(std::size_t j=0;j<design.size();++j)value+=static_cast<long double>(coefficients[j])*design[j];
    return static_cast<double>(value);
}
}

int main() {
    try {
        Json records=Json::array();
        std::size_t cross_design_checks=0,native_reads=0;
        bool missing_column_correction_exercised=false;
        for(int k:{1,2}) {
            const auto d=fixture(2,k);const auto catalogue=enumerate_paths(d);
            const DesignLayout layout(d);
            auto generous=generous_design(d),swapped=generous,no_assignment=generous,sparse=generous;
            for(int r=0;r<d.n_od;++r)for(int l=0;l<d.n_l;++l) {
                swapped[layout.assignment_index(r,l)]=l==(r+1)%d.n_l;
                no_assignment[layout.assignment_index(r,l)]=0;
            }
            sparse[layout.airline+d.n_l*d.n_nodes()+d.n_l+1]=0;
            const std::vector<double> closed(layout.count,0);
            const std::vector<std::vector<double>> designs={closed,generous,swapped,no_assignment,sparse};
            SolverOptions options;options.seconds=15;options.threads=1;options.pareto=false;
            options.weak_dual_cuts=true;
            options.branch_covering_seeds=false;options.cg_max_columns=1;
            // Test-only: the existing positive correction margin must exceed
            // this threshold even for an otherwise exact native dual. The
            // production tolerance is unchanged by this regression.
            options.cg_duality_tolerance=1e-12;
            std::vector<double> references;
            for(const auto& design:designs)references.push_back(full_recourse_value(d,catalogue,options,design));
            for(std::size_t a=0;a<designs.size();++a) {
                auto paths=catalogue;
                const auto result=solve_fixed_design_cg(d,paths,options,designs[a]);
                if(!result.value("bound_valid",false)) {
                    std::cerr<<Json{{"failed_case",a},{"k",k},{"termination_reason",result.at("termination_reason")},
                        {"pricing_complete",result.at("pricing_complete")},{"cg_trace",result.at("cg_trace")}}.dump(2)<<'\n';
                }
                require(result.value("bound_valid",false),"Valid but non-tight native dual was discarded before full-domain certification");
                require(result.value("pricing_complete",false),"Weak cut skipped complete path pricing");
                require(!result.value("recourse_optimal",true),"Weak certificate incorrectly claims tight recourse optimality");
                require(!result.at("benders_cuts").empty(),"Full-domain weak certificate produced no master cut");
                const auto& cut=result.at("benders_cuts").front();
                for(std::size_t b=0;b<designs.size();++b) {
                    const double scale=1+std::abs(references[b]);
                    require(affine_value(cut,designs[b])<=references[b]+1e-6*scale,
                            "Weak native-dual cut exceeds independent recourse on another design");
                    ++cross_design_checks;
                }
                require(result.at("continuous_cost").get<double>()>=references[a]-1e-6*(1+std::abs(references[a])),
                        "Weak-certificate run returned an infeasible improvement over independent full recourse");
                for(const auto& iteration:result.at("cg_trace")) {
                    for(const auto& attempt:iteration.at("dual_attempts")) {
                        require(attempt.at("method")=="native_cplex_dual","Weak certificate introduced auxiliary dual optimization");
                        require(attempt.value("auxiliary_optimization_calls",-1)==0,"Weak certificate solved another dual problem");
                        ++native_reads;
                    }
                    if(iteration.contains("pricing")) {
                        const auto& pricing=iteration.at("pricing");
                        missing_column_correction_exercised=missing_column_correction_exercised ||
                            (pricing.value("complete",false) && pricing.value("dual_correction",0.0)>1e-7);
                    }
                }
                records.push_back({{"k",k},{"anchor",a},{"reference_recourse",references[a]},
                    {"certified_recourse_lb",result.at("continuous_bound")},{"primal_recourse_ub",result.at("continuous_cost")},
                    {"termination_reason",result.at("termination_reason")},{"recourse_optimal",result.at("recourse_optimal")},
                    {"cg_iterations",result.at("cg_iterations")}});
            }
            // The same non-tight certificate is valid at an interior anchor;
            // this is the public seam used by the scheduled interior cuts.
            auto interior=generous;
            for(std::size_t j=0;j<interior.size();++j)interior[j]=.7*generous[j]+.3*closed[j];
            const double interior_reference=full_recourse_value(d,catalogue,options,interior);
            auto fractional_paths=catalogue;
            const auto fractional=evaluate_fractional_recourse_cg(d,fractional_paths,options,interior);
            require(fractional.value("bound_valid",false) && fractional.value("pricing_complete",false),
                    "Interior non-tight certificate did not survive complete pricing");
            require(!fractional.value("recourse_optimal",true),"Interior weak certificate claims recourse optimality");
            require(fractional.at("certified_lower_bound").get<double>()<=interior_reference+1e-6*(1+std::abs(interior_reference)),
                    "Interior cut contradicts independent full recourse");
            for(std::size_t b=0;b<designs.size();++b) {
                require(affine_value(fractional.at("ordinary_cut"),designs[b])<=references[b]+1e-6*(1+std::abs(references[b])),
                        "Interior weak cut is not globally valid across integer designs");
                ++cross_design_checks;
            }
            records.push_back({{"k",k},{"anchor","interior"},{"reference_recourse",interior_reference},
                {"certified_recourse_lb",fractional.at("certified_lower_bound")},
                {"primal_recourse_ub",fractional.at("operating_objective")},
                {"termination_reason",fractional.at("termination_reason")},
                {"recourse_optimal",fractional.at("recourse_optimal")}});
            auto stopped_options=options;stopped_options.seconds=0;
            auto stopped_paths=catalogue;
            const auto stopped=evaluate_fractional_recourse_cg(d,stopped_paths,stopped_options,interior);
            require(!stopped.value("bound_valid",true) && !stopped.value("pricing_complete",true) &&
                    !stopped.value("recourse_optimal",true) && stopped.at("ordinary_cut").is_null(),
                    "No-budget interior pricing fabricated a full-domain certificate");
        }
        require(missing_column_correction_exercised,"Weak-cut regression did not exercise negative omitted columns");
        std::cout<<Json{{"passed",true},{"cases",records},{"cross_design_checks",cross_design_checks},
            {"native_dual_reads",native_reads},{"missing_column_correction_exercised",missing_column_correction_exercised}}.dump(2)<<'\n';
    } catch(const IloException& e) {std::cerr<<e<<'\n';return 1;}
      catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
