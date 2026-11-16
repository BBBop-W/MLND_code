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
        SolverConfiguration configuration;configuration.apply_preset("accelerated-20261105");
        // ALNS has always been an L2 K2 module. The shared compression,
        // pricing, cache and cut strategies also apply to standard K1 Benders.
        if(k==1)configuration.options.primal_alns=configuration.options.primal_support_completion=false;
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
            "cut_pool_deduplication","cut_pool_management"})
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
            {"cut_pool",result.at("certified_cut_pool_summary")}});
    }
    std::cout<<Json{{"passed",true},{"cases",cases}}.dump()<<'\n';
}
