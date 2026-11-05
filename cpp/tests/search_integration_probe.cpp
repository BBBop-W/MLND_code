// Reuse the physical fixtures and independent primal checks.
#define main primal_polish_probe_original_main
#include "primal_polish_probe.cpp"
#undef main
#include "mlnd/benders.hpp"
#include "mlnd/task_savings.hpp"

int main() {
    try {
        Json records=Json::array();
        bool exercised_combined_root=false;
        bool exercised_fractional_queue=false;
        for(bool fractional_mode:{false,true})for(int k:{1,2}) {
            auto d=fixture(2,k);auto paths=enumerate_paths(d);
            SolverOptions options;options.seconds=20;options.gap=.005;options.threads=1;
            options.pareto=false;
            const auto full=solve_full(d,paths,options);
            require(full.at("has_solver_solution"),"Full integration reference has no solution");
            options.path_support_cuts=options.fractional_path_support=true;
            options.task_savings_bound=options.task_resource_cuts=options.task_airline_bounds=true;
            options.airline_connectivity_cuts=options.airline_resource_cuts=options.airline_resource_balance=true;
            options.search_scheduling=options.dual_lp_first=options.primal_polish=true;
            options.weak_dual_cuts=options.interior_cuts=options.primal_asset_neighborhood=true;
            options.queue_polymatroid_cuts=true;
            options.queue_price_search=options.target_master_refinement=true;
            options.queue_fractional_separation=fractional_mode;
            options.shared_rho_joint=options.queue_perspective=true;
            const auto prepared=build_task_airline_bounds(d,paths);
            options.task_airline_upper=prepared.upper;options.task_airline_metadata=prepared.metadata;
            auto result=solve_benders(d,paths,options);
            int native_reads=0;
            std::function<void(const Json&)> check_dual_source=[&](const Json& node) {
                if(node.is_object()) {
                    if(node.contains("dual_attempts")) for(const auto& attempt:node.at("dual_attempts")) {
                        require(attempt.at("method")=="native_cplex_dual",
                                "Ordinary Benders CG optimized an auxiliary dual");
                        require(attempt.value("auxiliary_optimization_calls",-1)==0,
                                "Native dual retrieval launched an optimization");
                        ++native_reads;
                    }
                    for(const auto& child:node.items()) if(child.key()!="dual_attempts" && child.key()!="pareto")
                        check_dual_source(child.value());
                } else if(node.is_array()) for(const auto& child:node)check_dual_source(child);
            };
            check_dual_source(result);
            require(native_reads>0,"Benders did not exercise native dual extraction");
            const double scale=1+std::abs(full.at("objective").get<double>());
            require(result.at("bound").get<double>()<=full.at("objective").get<double>()+1e-5*scale,
                    "Optimized Benders LB exceeds full-path feasible UB");
            require(result.at("objective").get<double>()>=full.at("bound").get<double>()-1e-5*scale,
                    "Optimized feasible solution contradicts full-path bound");
            require(result.at("master_structure").at("operational_auxiliary_variables")==0,
                    "Search strengthening added operating variables to master");
            require(result.at("search_schedule_summary").at("enabled"),"Missing search scheduling diagnostics");
            // These fast fixtures finish before the 30-second scheduled window.
            // Radius selection and local execution are exercised separately in
            // search_schedule_probe and primal_polish_probe, respectively.
            require(result.at("local_branching_summary").at("enabled"),
                    "Requested scheduled local branching is not reported as enabled");
            double restricted_wall=0;int complete_calls=0;
            for(const auto& call:result.at("acceleration_records"))
                if(call.value("time_scope",std::string())=="complete_restricted_search_including_postprocessing") {
                    const double wall=call.at("seconds").get<double>();
                    require(wall>=call.at("restricted_solver_seconds").get<double>() &&
                            wall>=call.at("timings").at("total").get<double>(),
                            "Restricted accounting omitted work after the solver returned");
                    restricted_wall+=wall;++complete_calls;
                    if(call.value("local_branching",false))
                        require(call.at("local_branching_radius").get<int>()>=0,
                                "Local branching call lost its radius");
                }
            require(complete_calls>0,"Fixture did not exercise full-call restricted accounting");
            require(result.at("heuristic_budget_used").get<double>()+1e-9>=restricted_wall &&
                    result.at("timings").at("restricted_mip").get<double>()+1e-9>=restricted_wall,
                    "Restricted records were not charged to the actual budget totals");
            require(result.at("interior_cut_summary").at("enabled"),"Missing interior cut diagnostics");
            require(result.at("accelerations").at("weak_dual_cuts"),"Weak cuts were not enabled");
            require(result.at("queue_polymatroid_summary").at("cuts").get<int>()>0,
                    "Queue cuts were enabled but never inserted into the master");
            require(!result.at("queue_polymatroid_summary").at("direct_global_bound_update").get<bool>(),
                    "Queue cut value was promoted directly to global lower bound");
            const auto& price_search=result.at("queue_price_search_summary");
            require(price_search.at("enabled") && (fractional_mode || price_search.at("calls").get<int>()>0),
                    "Native prices never reached queue separation");
            require(!price_search.at("auxiliary_dual_optimization").get<bool>(),
                    "Native price search launched an auxiliary dual optimization");
            for(const auto& call:result.at("queue_price_search_records")) {
                if(fractional_mode)require(call.at("stage")=="fractional_queue_root" ||
                    call.at("stage")=="fractional_queue_integer","Price search remained at a closed recourse point");
                require(call.at("diagnostics").at("evaluation_count").get<int>()<=7,
                        "Price search exceeded the full-scan evaluation limit");
                require(call.at("time_budget").get<double>()<=.5,
                        "Price search exceeded its per-call allowance");
            }
            const auto& refinement=result.at("target_master_refinement_summary");
            require(refinement.at("enabled") && !refinement.at("direct_candidate_bound_update").get<bool>(),
                    "Target refinement lacks certified-bound-only diagnostics");
            const auto& fractional=result.at("queue_fractional_summary");
            require(fractional.at("enabled")==fractional_mode,"Fractional queue flag changed outside its mode");
            if(fractional_mode) {
                exercised_fractional_queue=exercised_fractional_queue || fractional.at("calls").get<int>()>0;
                require(!fractional.at("direct_cut_value_bound_update").get<bool>(),"Queue point value became a global bound");
                std::size_t donor=0;
                for(const auto& call:result.at("queue_fractional_records")) {
                    require(call.at("donor_serial").get<std::size_t>()>donor,"Fractional call reused a consumed donor");
                    donor=call.at("donor_serial").get<std::size_t>();
                    require(call.at("diagnostics").at("accepted_rows").get<int>()<=2,"Too many fractional rows");
                    require(call.at("diagnostics").at("lp_solves").get<int>()<=3,"Too many fractional master solves");
                    for(const auto& round:call.at("diagnostics").at("rounds"))if(!round.at("certified_lower").is_null())
                        require(round.at("optimal") && round.at("feasible"),"Uncertified LP changed the global bound");
                }
            } else require(fractional.at("calls")==0,"Disabled fractional mode performed LP work");
            require(result.at("fractional_path_support_summary").at("enabled"),"Missing fractional support diagnostics");
            bool combined_root_round=false;
            for(const auto& round:result.at("root_diagnostics").at("rounds"))
                if(round.value("fractional_support_added",false)) {
                    require(round.contains("evaluations"),"Support-only separation consumed a root CG round");
                    combined_root_round=combined_root_round || !round.at("evaluations").empty();
                }
            exercised_combined_root=exercised_combined_root || combined_root_round;
            records.push_back({{"k",k},{"fractional_mode",fractional_mode},{"native_dual_reads",native_reads},{"reference_ub",full.at("objective")},{"reference_lb",full.at("bound")},
                {"ub",result.at("objective")},{"lb",result.at("bound")},{"gap",result.at("gap")},
                {"termination",result.at("termination_reason")},
                {"combined_support_recourse_round",combined_root_round},
                {"search_schedule",result.at("search_schedule_summary")},
                {"queue_price_search",price_search},{"target_master_refinement",refinement},
                {"queue_fractional",fractional},
                {"fractional_support",result.at("fractional_path_support_summary")}});
        }
        require(exercised_combined_root,"Neither fixture exercised joint support and recourse separation");
        require(exercised_fractional_queue,"Neither physical fixture exercised current-master LP separation");
        std::cout<<Json{{"passed",true},{"cases",records}}.dump(2)<<'\n';
    } catch(const IloException& e) {std::cerr<<e<<'\n';return 1;}
      catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
