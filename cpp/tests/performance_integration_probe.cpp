#define main primal_polish_fixture_main
#include "primal_polish_probe.cpp"
#undef main
#include "mlnd/benders.hpp"
#include "mlnd/solver_presets.hpp"

int main() {
    Json cases=Json::array();
    for(int k:{1,2}) {
        auto d=fixture(2,k);auto paths=enumerate_paths(d);
        SolverOptions reference_options;reference_options.seconds=20;reference_options.threads=1;
        reference_options.gap=1e-8;
        const auto full=solve_full(d,paths,reference_options);
        require(full.at("has_solver_solution"),"Full-path reference has no feasible solution");
        SolverConfiguration configuration;configuration.apply_preset("performance-20261105");
        // ALNS has always been an L2 K2 module. The shared compression,
        // pricing, cache and cut strategies also apply to standard K1 Benders.
        if(k==1) {
            configuration.options.primal_alns=configuration.options.primal_support_completion=false;
            configuration.options.cg_primal_feedback=configuration.options.indexed_support_completion=false;
        }
        configuration.options.seconds=20;configuration.options.threads=1;configuration.validate();
        const auto airline=build_task_airline_bounds(d,paths);
        configuration.options.task_airline_upper=airline.upper;
        configuration.options.task_airline_metadata=airline.metadata;
        auto result=solve_benders(d,paths,configuration.options);
        const double scale=1+std::abs(full.at("objective").get<double>());
        require(result.at("bound").get<double>()<=full.at("objective").get<double>()+1e-5*scale,
                "Combined acceleration produced a lower bound above a full-domain feasible solution");
        require(result.at("objective").get<double>()>=full.at("bound").get<double>()-1e-5*scale,
                "Combined acceleration produced an impossible primal objective");
        require(result.at("bound").get<double>()<=result.at("objective").get<double>()+1e-5*scale,
                "Combined acceleration inverted its certified bounds");
        for(const char* flag:{"eliminate_zero_task_columns","compact_task_heads","cache_pricing_graph",
            "persist_recourse_model","stabilized_pricing","grouped_local_branching",
            "cut_pool_deduplication","cut_pool_management","streamlined_recourse","catalogue_pricing",
            "diverse_pricing","reuse_recourse_rhs","design_certificate_cache","adaptive_pricing","stabilized_master"})
            require(result.at("accelerations").at(flag),"Combined configuration lost a requested acceleration");
        require(result.at("accelerations").at("primal_support_completion")==Json(k==2),
                "Support completion ignored the ALNS model scope");
        require(result.at("certified_cut_pool_summary").at("dormant")==0,
                "Final cut pool did not restore every original certified row");
        const auto fixed=solution_design(d,result);
        const auto checked=solve_full(d,paths,reference_options,&fixed,&result);
        require(checked.at("has_solver_solution") && checked.at("objective").get<double>()<=
                result.at("objective").get<double>()+1e-5*scale,
                "Compressed result could not be recovered in the original full formulation");
        cases.push_back({{"k",k},{"objective",result.at("objective")},{"bound",result.at("bound")},
            {"full_objective",full.at("objective")},{"termination_reason",result.at("termination_reason")},
            {"graph_cache",result.at("pricing_graph_cache_summary")},
            {"recourse_cache",result.at("recourse_state_cache_summary")},
            {"cut_pool",result.at("certified_cut_pool_summary")},
            {"memo",result.at("design_certificate_cache_summary")},
            {"proximity",result.at("stabilized_master_summary")},
            {"feedback",result.at("cg_primal_feedback_records")}});
    }
    // Keep the outer integer loop observable: a strong production root can
    // close these tiny fixtures before any memo/proximity/feedback call.
    auto d=fixture(2,2);auto paths=enumerate_paths(d);
    SolverConfiguration controlled;controlled.apply_preset("performance-20261105");controlled.plain();
    for(const char* name:{"reuse_columns","primal_alns","weak_dual_cuts","adaptive_master","incumbent_cutoff",
        "fast_pricing","batch_columns","eliminate_zero_task_columns","compact_task_heads","cache_pricing_graph",
        "persist_recourse_model","stabilized_pricing","primal_support_completion","grouped_local_branching",
        "cut_pool_deduplication","cut_pool_management","streamlined_recourse","catalogue_pricing","diverse_pricing",
        "reuse_recourse_rhs","design_certificate_cache","adaptive_pricing","cg_primal_feedback",
        "indexed_support_completion","stabilized_master"})controlled.set_acceleration(name,true);
    controlled.options.seconds=30;controlled.options.threads=1;controlled.options.gap=0;
    controlled.options.max_iterations=12;controlled.validate();
    const auto combined=solve_benders(d,paths,controlled.options);
    require(!combined.at("cg_primal_feedback_records").empty(),"Controlled outer loop did not deliver CG feedback to ALNS");
    require(combined.at("stabilized_master_summary").at("calls").get<int>()>0,
            "Controlled outer loop did not exercise the proximity helper");
    std::size_t trial_candidates=0,accepted_feedback=0;
    for(const auto& trial:combined.at("stabilized_master_records")) {
        require(trial.at("global_lower_before")==trial.at("global_lower_after"),"Trial solve modified global lower bound");
        trial_candidates+=trial.at("used_candidate").get<bool>();
    }
    for(const auto& record:combined.at("cg_primal_feedback_records"))accepted_feedback+=record.value("accepted",0);
    require(trial_candidates>0,"Controlled loop called proximity but never used its validated candidate");
    require(accepted_feedback>0,"Controlled loop never accepted a nonempty CG feedback batch");
    SolverOptions reference;reference.seconds=10;reference.threads=1;reference.gap=1e-8;
    const auto full=solve_full(d,paths,reference);
    const double tolerance=1e-5*(1+std::abs(full.at("objective").get<double>()));
    require(combined.at("bound").get<double>()<=full.at("objective").get<double>()+tolerance,
            "Controlled policies reported an invalid global lower bound");
    const auto x=solution_design(d,combined);
    const auto lifted=solve_full(d,paths,reference,&x,&combined);
    require(lifted.at("has_solver_solution") && lifted.at("objective").get<double>()<=
            combined.at("objective").get<double>()+tolerance,"Controlled incumbent failed original-model recovery");
    cases.push_back({{"case","controlled_integer_loop"},{"proximity_candidates",trial_candidates},{"feedback_columns_accepted",accepted_feedback},
        {"objective",combined.at("objective")},
        {"bound",combined.at("bound")},{"full_objective",full.at("objective")},
        {"termination_reason",combined.at("termination_reason")},{"memo",combined.at("design_certificate_cache_summary")},
        {"proximity",combined.at("stabilized_master_summary")},{"feedback",combined.at("cg_primal_feedback_records")},
        {"recourse_cache",combined.at("recourse_state_cache_summary")}});
    std::cout<<Json{{"passed",true},{"cases",cases}}.dump()<<'\n';
}
