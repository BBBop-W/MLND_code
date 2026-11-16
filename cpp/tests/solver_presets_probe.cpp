#include "mlnd/solver_presets.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    int checks=0;
    auto require=[&](bool ok,const char* message){if(!ok)throw std::runtime_error(message);++checks;};
    auto rejects=[&](auto action,const char* message){
        bool rejected=false;try{action();}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,message);
    };
    mlnd::SolverConfiguration config;
    const auto historical=config.metadata();
    require(config.preset=="legacy" && config.method=="full_path" &&
            !config.options.preprocessed_demand && config.options.gap==.005 &&
            config.options.cg_max_columns==64 && config.path_column_limit==500000 &&
            !config.options.export_catalogue,"Unselected configuration changed the historical CLI");
    config.apply_preset("baseline");
    require(config.preset=="baseline-20261029" && config.method=="benders_cg",
            "Baseline alias does not select the versioned Benders configuration");
    require(config.options.preprocessed_demand && config.options.queue_fractional_separation &&
            config.options.queue_price_search && config.options.target_master_refinement,
            "Baseline omitted the verified fractional queue strategy");
    require(config.options.seconds==600 && config.options.gap==.01 &&
            config.options.cg_max_columns==512 && config.options.max_iterations==1000000 &&
            config.path_column_limit==0,"Baseline differs from the frozen production settings");
    require(config.options.shared_rho_joint && config.options.queue_perspective &&
            config.options.tight_airline_bounds && config.options.task_airline_bounds &&
            config.options.path_support_cuts && config.options.fractional_path_support &&
            config.options.airline_resource_balance && config.options.search_scheduling,
            "Baseline omitted a shared strengthening or separator dependency");
    require(!config.options.assignment_incident_clique && !config.options.dual_lp_first &&
            !config.options.primal_asset_neighborhood && !config.options.first_leg_consistency,
            "Baseline enabled a feature absent from the frozen production command");
    config.validate();
    const auto baseline_metadata=config.metadata();
    require(baseline_metadata.at("schema")=="mlnd-solver-configuration/1" &&
            baseline_metadata.at("preset")=="baseline-20261029" &&
            baseline_metadata.at("seconds")==600 &&
            baseline_metadata.at("flags").at("queue_fractional_separation")==true &&
            baseline_metadata.at("flags").at("dual_lp_first")==false,
            "Resolved configuration omitted the requested budget or policy");
    config.apply_preset("full-cplex-20261029");
    require(config.method=="full_path" && config.options.preprocessed_demand &&
            config.options.shared_rho_joint && config.options.queue_perspective &&
            config.options.tight_airline_bounds && config.options.task_airline_bounds,
            "Full CPLEX comparator does not share the baseline's physical formulation");
    require(!config.options.queue_polymatroid_cuts && !config.options.queue_price_search &&
            !config.options.queue_fractional_separation && !config.options.path_support_cuts &&
            !config.options.target_master_refinement,"Full CPLEX preset retained Benders-only work");
    config.validate();
    config.apply_preset("legacy");
    require(config.method=="full_path" && !config.options.preprocessed_demand &&
            config.options.gap==.005 && config.options.cg_max_columns==64 &&
            config.options.max_iterations==1000 && config.path_column_limit==500000 &&
            !config.options.queue_fractional_separation,
            "Legacy preset changed historical CLI defaults");
    require(config.metadata()==historical,"Explicit legacy reset differs from the unselected configuration");

    config.options.output="outputs/preset_probe_not_written";
    config.options.log=true;config.options.export_catalogue=true;
    config.apply_preset("baseline-20261029");
    require(config.options.output=="outputs/preset_probe_not_written" && config.options.log &&
            config.options.export_catalogue,"Preset reset an output-only request");
    config.options.seconds=123;config.options.gap=.02;config.path_column_limit=456;
    config.set_acceleration("interior_cuts",false);
    require(config.metadata().at("seconds")==123 && config.metadata().at("gap")==.02 &&
            config.metadata().at("path_column_limit_argument")==456 &&
            config.metadata().at("flags").at("interior_cuts")==false,
            "Metadata lost explicit overrides after a preset");
    const auto unchanged=config.metadata();
    rejects([&]{config.apply_preset("unknown");},"Unknown preset was accepted");
    require(config.metadata()==unchanged,"Failed preset selection partially mutated the configuration");
    rejects([&]{config.set_acceleration("unknown",false);},"Unknown acceleration was accepted");
    require(!config.acceleration_flag("--seconds") && config.acceleration_flag("--interior-cuts") &&
            config.options.interior_cuts,"Enable registry failed to distinguish scalar and boolean flags");

    config.plain();
    bool all_disabled=true;
    const auto plain_metadata=config.metadata();
    for(const auto& flag:plain_metadata.at("flags").items())all_disabled=all_disabled&&!flag.value().get<bool>();
    require(all_disabled && config.options.preprocessed_demand && config.method=="benders_cg" &&
            config.options.export_catalogue,"Named plain retained acceleration or changed model/output policy");
    config.validate();
    config.apply_preset("legacy");
    config.acceleration_flag("--tight-airline-bounds");
    config.acceleration_flag("--path-support-cuts");
    config.acceleration_flag("--queue-polymatroid-cuts");
    config.plain();
    require(config.options.tight_airline_bounds && config.options.path_support_cuts &&
            !config.options.queue_polymatroid_cuts && !config.options.pareto &&
            !config.options.reuse_dual_lp,"Legacy plain changed its historical switch membership");

    config.apply_preset("baseline");config.set_acceleration("path_support_cuts",false);
    rejects([&]{config.validate();},"Fractional support accepted a disabled prerequisite");
    config.apply_preset("baseline");config.set_acceleration("queue_price_search",false);
    rejects([&]{config.validate();},"Fractional queue accepted a disabled price prerequisite");
    config.apply_preset("baseline");config.set_acceleration("search_scheduling",false);
    rejects([&]{config.validate();},"Target refinement accepted a disabled scheduling prerequisite");
    config.apply_preset("full-cplex-20261029");config.acceleration_flag("--queue-polymatroid-cuts");
    rejects([&]{config.validate();},"Full CPLEX accepted a Benders-only queue policy");
    config.apply_preset("baseline");config.options.first_leg_consistency=true;
    rejects([&]{config.validate();},"Baseline cuts accepted a different demand-consistency model");
    config.apply_preset("baseline");config.options.seconds=std::numeric_limits<double>::infinity();
    rejects([&]{config.validate();},"Infinite solver budget was accepted");
    config.apply_preset("baseline");config.options.heuristic_budget_fraction=1.1;
    rejects([&]{config.validate();},"Out-of-range budget fraction was accepted");
    config.apply_preset("accelerated-20261105");config.validate();
    require(config.options.primal_alns && config.options.eliminate_zero_task_columns &&
            config.options.compact_task_heads && config.options.cache_pricing_graph &&
            config.options.persist_recourse_model && config.options.stabilized_pricing &&
            config.options.primal_support_completion && config.options.grouped_local_branching &&
            config.options.cut_pool_deduplication && config.options.cut_pool_management,
            "New preset omitted an acceleration family");
    require(config.metadata().at("parameters").at("pricing_exact_interval")==4,
            "Configuration omitted exact-pricing fallback policy");
    config.options.pricing_smoothing_weight=0;
    rejects([&]{config.validate();},"Zero current-dual smoothing weight was accepted");
    config.apply_preset("accelerated-20261105");config.options.cut_pool_age=0;
    rejects([&]{config.validate();},"Cut pool accepted zero protection age");
    config.apply_preset("accelerated-20261105");config.options.first_leg_consistency=true;
    rejects([&]{config.validate();},"Compaction accepted a different task model");
    config.apply_preset("baseline");
    require(!config.options.eliminate_zero_task_columns && !config.options.compact_task_heads &&
            !config.options.cache_pricing_graph && !config.options.persist_recourse_model &&
            !config.options.stabilized_pricing && !config.options.primal_support_completion &&
            !config.options.grouped_local_branching && !config.options.cut_pool_deduplication &&
            !config.options.cut_pool_management,"Historical preset silently adopted new algorithms");
    const std::vector<std::string> followup={"streamlined_recourse","catalogue_pricing","diverse_pricing",
        "reuse_recourse_rhs","design_certificate_cache","adaptive_pricing","cg_primal_feedback",
        "indexed_support_completion","stabilized_master"};
    for(const char* preset:{"legacy","baseline","full-cplex-20261029","accelerated-20261105"}) {
        config.apply_preset(preset);
        for(const auto& flag:followup)require(!config.metadata().at("flags").at(flag).get<bool>(),
            "Frozen preset silently adopted a follow-up policy");
    }
    config.apply_preset("performance-20261105");config.validate();
    for(const auto& flag:followup)require(config.metadata().at("flags").at(flag).get<bool>(),
        "Performance preset omitted an approved direction");
    config.options.catalogue_pricing=false;
    rejects([&]{config.validate();},"Diversity accepted a disabled catalogue backend");
    config.apply_preset("performance-20261105");config.options.stabilized_pricing=false;
    rejects([&]{config.validate();},"Adaptive pricing accepted disabled smoothing");
    config.apply_preset("performance-20261105");config.options.conditional_demand=true;
    rejects([&]{config.validate();},"RHS reuse accepted conditional coefficients");
    config.apply_preset("performance-20261105");config.options.stabilized_master_level=1;
    rejects([&]{config.validate();},"Proximity level accepted an invalid fraction");
    config.apply_preset("performance-20261105");config.options.design_cache_limit=0;
    rejects([&]{config.validate();},"Design cache accepted zero capacity");
    std::cout<<"{\"passed\":true,\"checks\":"<<checks<<"}\n";
}
