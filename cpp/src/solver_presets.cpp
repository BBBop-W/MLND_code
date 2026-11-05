#include "mlnd/solver_presets.hpp"

#include <cmath>
#include <stdexcept>

namespace mlnd {
namespace {
struct Acceleration {
    const char* name;
    const char* command;
    bool SolverOptions::*member;
    bool legacy_plain;
};
// Canonical names, enable flags and historical --plain membership live here.
// Model choices (demand formulation / first-leg consistency) are not switches
// that --plain may change. Entries without an enable flag keep their old CLI.
const Acceleration accelerations[]={
    {"pareto_cuts","",&SolverOptions::pareto,true},
    {"reuse_dual_lp","",&SolverOptions::reuse_dual_lp,true},
    {"reuse_columns","",&SolverOptions::reuse_columns,true},
    {"courier_savings_bound","",&SolverOptions::courier_savings_bound,true},
    {"initial_design","",&SolverOptions::initial_design,true},
    {"warm_start","",&SolverOptions::warm_start,true},
    {"incumbent_cleanup","",&SolverOptions::incumbent_cleanup,true},
    {"primal_search","",&SolverOptions::primal_search,true},
    {"primal_polish","--primal-polish",&SolverOptions::primal_polish,true},
    {"search_scheduling","--search-scheduling",&SolverOptions::search_scheduling,true},
    {"dual_lp_first","--dual-lp-first",&SolverOptions::dual_lp_first,true},
    {"fractional_path_support","--fractional-path-support",&SolverOptions::fractional_path_support,true},
    {"airline_resource_balance","--airline-resource-balance",&SolverOptions::airline_resource_balance,true},
    {"queue_polymatroid_cuts","--queue-polymatroid-cuts",&SolverOptions::queue_polymatroid_cuts,true},
    {"queue_price_search","--queue-price-search",&SolverOptions::queue_price_search,true},
    {"queue_fractional_separation","--queue-fractional-separation",&SolverOptions::queue_fractional_separation,true},
    {"design_face_cuts","--design-face-cuts",&SolverOptions::design_face_cuts,true},
    {"queue_joint_cuts","--queue-joint-cuts",&SolverOptions::queue_joint_cuts,true},
    {"primal_coupled_neighborhood","--primal-coupled-neighborhood",&SolverOptions::primal_coupled_neighborhood,true},
    {"primal_seed_portfolio","--primal-seed-portfolio",&SolverOptions::primal_seed_portfolio,true},
    {"branch_pricing_correction","--branch-pricing-correction",&SolverOptions::branch_pricing_correction,true},
    {"cg_seed_reuse","--cg-seed-reuse",&SolverOptions::cg_seed_reuse,true},
    {"staged_continuous_root","--staged-continuous-root",&SolverOptions::staged_continuous_root,true},
    {"fast_pricing","--fast-pricing",&SolverOptions::fast_pricing,true},
    {"batch_columns","--batch-columns",&SolverOptions::batch_columns,true},
    {"primal_alns","--primal-alns",&SolverOptions::primal_alns,true},
    {"eliminate_zero_task_columns","--eliminate-zero-task-columns",&SolverOptions::eliminate_zero_task_columns,true},
    {"compact_task_heads","--compact-task-heads",&SolverOptions::compact_task_heads,true},
    {"cache_pricing_graph","--cache-pricing-graph",&SolverOptions::cache_pricing_graph,true},
    {"persist_recourse_model","--persist-recourse-model",&SolverOptions::persist_recourse_model,true},
    {"stabilized_pricing","--stabilized-pricing",&SolverOptions::stabilized_pricing,true},
    {"primal_support_completion","--primal-support-completion",&SolverOptions::primal_support_completion,true},
    {"grouped_local_branching","--grouped-local-branching",&SolverOptions::grouped_local_branching,true},
    {"cut_pool_deduplication","--cut-pool-deduplication",&SolverOptions::cut_pool_deduplication,true},
    {"cut_pool_management","--cut-pool-management",&SolverOptions::cut_pool_management,true},
    {"streamlined_recourse","--streamlined-recourse",&SolverOptions::streamlined_recourse,true},
    {"catalogue_pricing","--catalogue-pricing",&SolverOptions::catalogue_pricing,true},
    {"diverse_pricing","--diverse-pricing",&SolverOptions::diverse_pricing,true},
    {"reuse_recourse_rhs","--reuse-recourse-rhs",&SolverOptions::reuse_recourse_rhs,true},
    {"design_certificate_cache","--design-certificate-cache",&SolverOptions::design_certificate_cache,true},
    {"adaptive_pricing","--adaptive-pricing",&SolverOptions::adaptive_pricing,true},
    {"cg_primal_feedback","--cg-primal-feedback",&SolverOptions::cg_primal_feedback,true},
    {"indexed_support_completion","--indexed-support-completion",&SolverOptions::indexed_support_completion,true},
    {"stabilized_master","--stabilized-master",&SolverOptions::stabilized_master,true},
    {"target_master_refinement","--target-master-refinement",&SolverOptions::target_master_refinement,true},
    {"weak_dual_cuts","--weak-dual-cuts",&SolverOptions::weak_dual_cuts,true},
    {"interior_cuts","--interior-cuts",&SolverOptions::interior_cuts,true},
    {"primal_asset_neighborhood","--primal-asset-neighborhood",&SolverOptions::primal_asset_neighborhood,true},
    {"branch_covering_seeds","",&SolverOptions::branch_covering_seeds,true},
    {"resource_bounds","",&SolverOptions::resource_bounds,true},
    {"capacity_cuts","",&SolverOptions::capacity_cuts,true},
    {"resource_lp_cuts","",&SolverOptions::resource_lp_cuts,true},
    {"adaptive_master","",&SolverOptions::adaptive_master,true},
    {"neighborhood_cuts","",&SolverOptions::neighborhood_cuts,true},
    {"continuous_root","",&SolverOptions::continuous_root,true},
    {"root_projection","",&SolverOptions::root_projection,true},
    {"root_inout","",&SolverOptions::root_inout,true},
    {"auxiliary_valid_cuts","",&SolverOptions::auxiliary_valid_cuts,true},
    {"resource_airline_bounds","",&SolverOptions::resource_airline_bounds,true},
    {"od_airline_bounds","--od-airline-bounds",&SolverOptions::od_airline_bounds,true},
    {"tight_airline_bounds","--tight-airline-bounds",&SolverOptions::tight_airline_bounds,false},
    {"assignment_incident_clique","--assignment-incident-clique",&SolverOptions::assignment_incident_clique,false},
    {"shared_rho_joint","--shared-rho-joint",&SolverOptions::shared_rho_joint,false},
    {"queue_perspective","--queue-perspective",&SolverOptions::queue_perspective,false},
    {"topology_cuts","--topology-cuts",&SolverOptions::topology_cuts,false},
    {"assignment_support_cuts","--assignment-support-cuts",&SolverOptions::assignment_support_cuts,false},
    {"double_support_cuts","--double-support-cuts",&SolverOptions::double_support_cuts,false},
    {"joint_base_cost_cuts","--joint-base-cost-cuts",&SolverOptions::joint_base_cost_cuts,false},
    {"path_support_cuts","--path-support-cuts",&SolverOptions::path_support_cuts,false},
    {"task_savings_bound","--task-savings-bound",&SolverOptions::task_savings_bound,false},
    {"task_resource_cuts","--task-resource-cuts",&SolverOptions::task_resource_cuts,false},
    {"task_airline_bounds","--task-airline-bounds",&SolverOptions::task_airline_bounds,false},
    {"airline_connectivity_cuts","--airline-connectivity-cuts",&SolverOptions::airline_connectivity_cuts,false},
    {"airline_resource_cuts","--airline-resource-cuts",&SolverOptions::airline_resource_cuts,false},
    {"root_cuts","",&SolverOptions::root_cuts,true},
    {"restricted_mip","",&SolverOptions::restricted_mip,true},
    {"local_branching","",&SolverOptions::local_branching,true},
    {"incumbent_cutoff","",&SolverOptions::incumbent_cutoff,true}
};
}

bool SolverConfiguration::acceleration_flag(const std::string& command) {
    for(const auto& flag:accelerations)if(command==flag.command && !command.empty()) {
        options.*(flag.member)=true;return true;
    }
    return false;
}
void SolverConfiguration::set_acceleration(const std::string& name,bool enabled) {
    for(const auto& flag:accelerations)if(name==flag.name) {
        options.*(flag.member)=enabled;return;
    }
    throw std::invalid_argument("Unknown acceleration: "+name);
}
void SolverConfiguration::plain() {
    for(const auto& flag:accelerations)
        if(preset!="legacy" || flag.legacy_plain)options.*(flag.member)=false;
}
void SolverConfiguration::apply_preset(const std::string& name) {
    if(name!="baseline" && name!="baseline-20261029" &&
       name!="full-cplex-20261029" && name!="bounds-20261030" && name!="lb-20261030" && name!="speed-20261030" && name!="alns-20261030" && name!="accelerated-20261105" && name!="performance-20261105" && name!="legacy")
        throw std::invalid_argument("Unknown solver preset: "+name);
    SolverOptions selected;
    // Output destinations, logging and export requests are not solver policy.
    selected.output=options.output;selected.progress=options.progress;
    selected.log=options.log;selected.export_catalogue=options.export_catalogue;
    options=std::move(selected);
    preset=name=="baseline"?"baseline-20261029":name;
    method="full_path";path_column_limit=500000;
    if(preset=="legacy")return;
    for(const auto& flag:accelerations)options.*(flag.member)=false;
    path_column_limit=0;
    options.preprocessed_demand=true;
    options.seconds=600;options.gap=.01;options.cg_max_columns=512;options.max_iterations=1000000;
    for(const char* name:{"shared_rho_joint","queue_perspective","tight_airline_bounds","task_airline_bounds"})
        set_acceleration(name,true);
    if(preset=="full-cplex-20261029")return;
    method="benders_cg";
    // Exact policy of the frozen fractional_queue command, including legacy
    // default-on features. No untested switch is inherited from a later preset.
    for(const char* name:{"pareto_cuts","reuse_dual_lp","reuse_columns","courier_savings_bound",
        "initial_design","warm_start","incumbent_cleanup","primal_search","branch_covering_seeds",
        "resource_bounds","capacity_cuts","resource_lp_cuts","adaptive_master","neighborhood_cuts",
        "continuous_root","root_projection","root_inout","auxiliary_valid_cuts","resource_airline_bounds",
        "root_cuts","restricted_mip","local_branching","incumbent_cutoff","od_airline_bounds","topology_cuts",
        "path_support_cuts","task_savings_bound","task_resource_cuts","airline_connectivity_cuts",
        "airline_resource_cuts","primal_polish","search_scheduling","fractional_path_support",
        "airline_resource_balance","weak_dual_cuts","interior_cuts","queue_polymatroid_cuts",
        "queue_price_search","target_master_refinement","queue_fractional_separation"})
        set_acceleration(name,true);
    if(preset=="bounds-20261030")
        for(const char* name:{"design_face_cuts","queue_joint_cuts",
                             "primal_coupled_neighborhood","primal_seed_portfolio"})
            set_acceleration(name,true);
    if(preset=="lb-20261030" || preset=="speed-20261030" || preset=="alns-20261030" || preset=="accelerated-20261105" || preset=="performance-20261105")
        for(const char* name:{"design_face_cuts","queue_joint_cuts","branch_pricing_correction",
                             "cg_seed_reuse","staged_continuous_root"})
            set_acceleration(name,true);
    if(preset=="speed-20261030" || preset=="alns-20261030" || preset=="accelerated-20261105" || preset=="performance-20261105")
        for(const char* name:{"fast_pricing","batch_columns"})set_acceleration(name,true);
    if(preset=="alns-20261030" || preset=="accelerated-20261105" || preset=="performance-20261105")options.primal_alns=true;
    if(preset=="accelerated-20261105" || preset=="performance-20261105")
        for(const char* name:{"eliminate_zero_task_columns","compact_task_heads","cache_pricing_graph",
            "persist_recourse_model","stabilized_pricing","primal_support_completion","grouped_local_branching",
            "cut_pool_deduplication","cut_pool_management"})set_acceleration(name,true);
    if(preset=="performance-20261105")
        for(const char* name:{"streamlined_recourse","catalogue_pricing","diverse_pricing","reuse_recourse_rhs","design_certificate_cache","adaptive_pricing","cg_primal_feedback","indexed_support_completion","stabilized_master"})set_acceleration(name,true);
}

void SolverConfiguration::validate() const {
    if((options.catalogue_pricing || options.diverse_pricing || options.adaptive_pricing ||
        options.reuse_recourse_rhs || options.design_certificate_cache || options.stabilized_master) && method!="benders_cg")
        throw std::invalid_argument("Follow-up decomposition policies require Benders CG");
    if(options.diverse_pricing && !options.catalogue_pricing)
        throw std::invalid_argument("Diverse pricing requires the complete-catalogue backend");
    if(options.adaptive_pricing && !options.stabilized_pricing)
        throw std::invalid_argument("Adaptive pricing requires stabilized pricing");
    if(options.reuse_recourse_rhs && (!options.preprocessed_demand || options.conditional_demand))
        throw std::invalid_argument("Cross-design RHS reuse requires preprocessed non-conditional demand");
    if((options.cg_primal_feedback || options.indexed_support_completion) && !options.primal_alns)
        throw std::invalid_argument("CG primal feedback and indexed completion require ALNS");
    if(options.indexed_support_completion && !options.primal_support_completion)
        throw std::invalid_argument("Indexed completion requires support completion");
    if(options.design_cache_limit<1 || !std::isfinite(options.stabilized_master_seconds) ||
       options.stabilized_master_seconds<=0 || !std::isfinite(options.stabilized_master_level) ||
       options.stabilized_master_level<=0 || options.stabilized_master_level>=1)
        throw std::invalid_argument("Invalid follow-up acceleration parameter");
    if((options.eliminate_zero_task_columns || options.compact_task_heads) && !options.preprocessed_demand)
        throw std::invalid_argument("Task model compression requires preprocessed demand");
    if(options.compact_task_heads && options.first_leg_consistency)
        throw std::invalid_argument("Compact task heads require aggregate tasks without first-leg consistency");
    if(options.grouped_local_branching && (!options.preprocessed_demand || options.first_leg_consistency))
        throw std::invalid_argument("Grouped local branching requires preprocessed aggregate tasks");
    if((options.cache_pricing_graph || options.persist_recourse_model || options.stabilized_pricing) && method!="benders_cg")
        throw std::invalid_argument("Pricing and persistent recourse require Benders CG");
    if((options.cut_pool_deduplication || options.cut_pool_management) && method=="full_path")
        throw std::invalid_argument("Certified cut pool requires Benders");
    if(options.primal_support_completion && !options.primal_alns)
        throw std::invalid_argument("Primal support completion requires ALNS");
    if(!std::isfinite(options.pricing_smoothing_weight) || options.pricing_smoothing_weight<=0 ||
        options.pricing_smoothing_weight>1 || options.pricing_exact_interval<1 ||
        options.local_branching_head_radius<0 || options.local_branching_capacity_radius<0 ||
        options.cut_pool_active_limit<1 || options.cut_pool_age<1)
        throw std::invalid_argument("Invalid acceleration parameter");
    if(options.primal_alns && (method!="benders_cg" || !options.preprocessed_demand ||
        options.first_leg_consistency || !options.reuse_columns))
        throw std::invalid_argument("Primal ALNS requires preprocessed origin Benders CG with column reuse");
    if((options.fast_pricing || options.batch_columns) && method!="benders_cg")
        throw std::invalid_argument("Pricing and batch column accelerations require Benders CG");
    if((options.branch_pricing_correction || options.cg_seed_reuse || options.staged_continuous_root) &&
       (method!="benders_cg" || !options.preprocessed_demand || options.first_leg_consistency))
        throw std::invalid_argument("LB extensions require preprocessed aggregate Benders CG");
    if(options.staged_continuous_root && (!options.cg_seed_reuse || !options.continuous_root))
        throw std::invalid_argument("--staged-continuous-root requires CG seed reuse and continuous root");
    if(options.fractional_path_support && !options.path_support_cuts)
        throw std::invalid_argument("--fractional-path-support requires --path-support-cuts");
    if(options.airline_resource_balance && !options.airline_resource_cuts)
        throw std::invalid_argument("--airline-resource-balance requires --airline-resource-cuts");
    if(method!="full_path" && method!="benders_cg" && method!="benders_full")
        throw std::invalid_argument("Unsupported method");
    if(options.conditional_demand && method=="full_path")
        throw std::invalid_argument("Conditional demand is an integer-recourse Benders mode; full_path remains lifted");
    if(options.first_leg_consistency)
        throw std::invalid_argument("--first-leg-consistency is incompatible with independent task assignment");
    if(options.path_support_cuts && (method=="full_path" || !options.preprocessed_demand || options.first_leg_consistency))
        throw std::invalid_argument("--path-support-cuts requires Benders with preprocessed aggregate demand and no first-leg consistency");
    if((options.task_savings_bound || options.task_resource_cuts || options.airline_connectivity_cuts ||
        options.airline_resource_cuts || options.queue_polymatroid_cuts) &&
       (method=="full_path" || !options.preprocessed_demand || options.first_leg_consistency))
        throw std::invalid_argument("Task cost cuts require preprocessed aggregate Benders without first-leg consistency");
    if(options.task_airline_bounds && !options.preprocessed_demand)
        throw std::invalid_argument("Task airline bounds require preprocessed demand");
    if(options.queue_price_search && !options.queue_polymatroid_cuts)
        throw std::invalid_argument("--queue-price-search requires --queue-polymatroid-cuts");
    if(options.queue_fractional_separation && !options.queue_price_search)
        throw std::invalid_argument("--queue-fractional-separation requires --queue-price-search");
    if(options.target_master_refinement && method=="full_path")
        throw std::invalid_argument("--target-master-refinement requires Benders");
    if(options.target_master_refinement && !options.search_scheduling)
        throw std::invalid_argument("--target-master-refinement requires --search-scheduling");
    if((options.design_face_cuts || options.queue_joint_cuts || options.primal_coupled_neighborhood ||
        options.primal_seed_portfolio) &&
       (method!="benders_cg" || !options.preprocessed_demand || options.first_leg_consistency))
        throw std::invalid_argument("Bounds extensions require preprocessed aggregate Benders CG");
    if(options.queue_joint_cuts && !options.queue_polymatroid_cuts)
        throw std::invalid_argument("--queue-joint-cuts requires --queue-polymatroid-cuts");
    if((options.primal_coupled_neighborhood || options.primal_seed_portfolio) && !options.restricted_mip)
        throw std::invalid_argument("Primal extensions require restricted MIP search");
    if(options.airline_resource_cut_limit<0 || !std::isfinite(options.airline_resource_seconds) || options.airline_resource_seconds<=0)
        throw std::invalid_argument("Invalid airline resource separation budget");
    if(options.task_resource_cut_limit<0 || !std::isfinite(options.task_resource_seconds) || options.task_resource_seconds<=0)
        throw std::invalid_argument("Invalid task resource separation budget");
    if(!std::isfinite(options.seconds)||options.seconds<=0||!std::isfinite(options.gap)||options.gap<0
       ||options.threads<1||options.max_iterations<1||options.cg_max_columns<1
       ||!std::isfinite(options.cg_duality_tolerance)||options.cg_duality_tolerance<=0
       ||!std::isfinite(options.pricing_tolerance)||options.pricing_tolerance<=0)
        throw std::invalid_argument("Invalid solver budget or CG tolerance");
    if(options.root_cut_rounds<0||options.pareto_frequency<1||options.local_branching_radius<0||options.capacity_cut_limit<0
       ||options.topology_cut_limit<0||!std::isfinite(options.topology_seconds)||options.topology_seconds<=0
       ||options.assignment_support_cut_limit<0||options.assignment_support_max_cuts<1
       ||!std::isfinite(options.assignment_support_seconds)||options.assignment_support_seconds<=0
       ||!std::isfinite(options.assignment_support_floor)||options.assignment_support_floor<=0
       ||options.path_support_cut_limit<0||options.path_support_max_cuts<1
       ||!std::isfinite(options.path_support_seconds)||options.path_support_seconds<=0
       ||!std::isfinite(options.path_support_preprocessing_seconds)||options.path_support_preprocessing_seconds<=0
       ||options.branch_joint_cut_limit<0||options.branch_joint_max_cuts<1
       ||!std::isfinite(options.branch_joint_seconds)||options.branch_joint_seconds<=0
       ||!std::isfinite(options.branch_joint_preprocessing_seconds)||options.branch_joint_preprocessing_seconds<=0
       ||!std::isfinite(options.branch_joint_floor)||options.branch_joint_floor<=0
       ||options.resource_lp_cut_limit<0||!std::isfinite(options.resource_lp_seconds)||options.resource_lp_seconds<=0
       ||options.heuristic_pool_size<1||!std::isfinite(options.neighborhood_cut_seconds)||options.neighborhood_cut_seconds<=0
       ||!std::isfinite(options.master_initial_gap)||options.master_initial_gap<0||options.master_initial_gap>1
       ||!std::isfinite(options.master_slice_seconds)||options.master_slice_seconds<=0
       ||!std::isfinite(options.continuous_root_seconds)||options.continuous_root_seconds<=0
       ||options.continuous_root_max_columns<1
       ||!std::isfinite(options.root_inout_weight)||options.root_inout_weight<=0||options.root_inout_weight>=1
       ||!std::isfinite(options.pareto_call_seconds)||options.pareto_call_seconds<=0)
        throw std::invalid_argument("Invalid strengthening frequency or budget");
    for(double fraction:{options.pareto_budget_fraction,options.root_budget_fraction,options.heuristic_budget_fraction,
                         options.neighborhood_cut_budget_fraction,options.continuous_root_budget_fraction})
        if(!std::isfinite(fraction)||fraction<0||fraction>1)
            throw std::invalid_argument("Budget fractions must be finite and within [0,1]");
}

Json SolverConfiguration::metadata() const {
    Json flags=Json::object();
    for(const auto& flag:accelerations)flags[flag.name]=options.*(flag.member);
    return {{"schema","mlnd-solver-configuration/1"},{"preset",preset},{"method",method},
        {"demand_formulation",options.preprocessed_demand?"preprocessed":options.conditional_demand?"conditional":"lifted"},
        {"first_leg_consistency",options.first_leg_consistency},
        {"seconds",options.seconds},{"gap",options.gap},{"threads",options.threads},{"seed",options.seed},
        {"max_iterations",options.max_iterations},{"cg_max_columns",options.cg_max_columns},
        {"cg_duality_tolerance",options.cg_duality_tolerance},{"pricing_tolerance",options.pricing_tolerance},
        {"path_column_limit_argument",path_column_limit},{"export_catalogue",options.export_catalogue},
        {"flags",std::move(flags)},
        {"parameters",{
            {"design_cache_limit",options.design_cache_limit},
            {"stabilized_master_seconds",options.stabilized_master_seconds},{"stabilized_master_level",options.stabilized_master_level},
            {"pricing_smoothing_weight",options.pricing_smoothing_weight},{"pricing_exact_interval",options.pricing_exact_interval},
            {"local_branching_head_radius",options.local_branching_head_radius},
            {"local_branching_capacity_radius",options.local_branching_capacity_radius},
            {"cut_pool_active_limit",options.cut_pool_active_limit},{"cut_pool_age",options.cut_pool_age},
            {"topology_seconds",options.topology_seconds},{"topology_cut_limit",options.topology_cut_limit},
            {"assignment_support_seconds",options.assignment_support_seconds},
            {"assignment_support_floor",options.assignment_support_floor},
            {"assignment_support_cut_limit",options.assignment_support_cut_limit},
            {"assignment_support_max_cuts",options.assignment_support_max_cuts},
            {"path_support_seconds",options.path_support_seconds},
            {"path_support_preprocessing_seconds",options.path_support_preprocessing_seconds},
            {"path_support_cut_limit",options.path_support_cut_limit},{"path_support_max_cuts",options.path_support_max_cuts},
            {"task_resource_seconds",options.task_resource_seconds},{"task_resource_cut_limit",options.task_resource_cut_limit},
            {"airline_resource_seconds",options.airline_resource_seconds},{"airline_resource_cut_limit",options.airline_resource_cut_limit},
            {"branch_joint_seconds",options.branch_joint_seconds},
            {"branch_joint_preprocessing_seconds",options.branch_joint_preprocessing_seconds},
            {"branch_joint_floor",options.branch_joint_floor},{"branch_joint_cut_limit",options.branch_joint_cut_limit},
            {"branch_joint_max_cuts",options.branch_joint_max_cuts},{"capacity_cut_limit",options.capacity_cut_limit},
            {"heuristic_pool_size",options.heuristic_pool_size},{"neighborhood_cut_seconds",options.neighborhood_cut_seconds},
            {"neighborhood_cut_budget_fraction",options.neighborhood_cut_budget_fraction},
            {"resource_lp_cut_limit",options.resource_lp_cut_limit},{"resource_lp_seconds",options.resource_lp_seconds},
            {"master_initial_gap",options.master_initial_gap},{"master_slice_seconds",options.master_slice_seconds},
            {"continuous_root_max_columns",options.continuous_root_max_columns},
            {"continuous_root_seconds",options.continuous_root_seconds},
            {"continuous_root_budget_fraction",options.continuous_root_budget_fraction},
            {"root_inout_weight",options.root_inout_weight},{"root_cut_rounds",options.root_cut_rounds},
            {"pareto_frequency",options.pareto_frequency},{"heuristic_frequency",options.heuristic_frequency},
            {"local_branching_radius",options.local_branching_radius},
            {"pareto_budget_fraction",options.pareto_budget_fraction},{"pareto_call_seconds",options.pareto_call_seconds},
            {"root_budget_fraction",options.root_budget_fraction},{"heuristic_budget_fraction",options.heuristic_budget_fraction}
        }}};
}
} // namespace mlnd
