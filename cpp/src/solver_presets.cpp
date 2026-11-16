#include "mlnd/solver_presets.hpp"

#include <cmath>
#include <stdexcept>

namespace mlnd {
namespace {
struct Acceleration {
    const char* name;
    bool SolverOptions::*member;
};
// Canonical acceleration names for current-method settings and ablations.
// The fixed-task model is not a user-selectable acceleration.
const Acceleration accelerations[]={
    {"pareto_cuts",&SolverOptions::pareto},
    {"reuse_dual_lp",&SolverOptions::reuse_dual_lp},
    {"reuse_columns",&SolverOptions::reuse_columns},
    {"courier_savings_bound",&SolverOptions::courier_savings_bound},
    {"initial_design",&SolverOptions::initial_design},
    {"warm_start",&SolverOptions::warm_start},
    {"incumbent_cleanup",&SolverOptions::incumbent_cleanup},
    {"primal_search",&SolverOptions::primal_search},
    {"primal_polish",&SolverOptions::primal_polish},
    {"search_scheduling",&SolverOptions::search_scheduling},
    {"dual_lp_first",&SolverOptions::dual_lp_first},
    {"fractional_path_support",&SolverOptions::fractional_path_support},
    {"airline_resource_balance",&SolverOptions::airline_resource_balance},
    {"queue_polymatroid_cuts",&SolverOptions::queue_polymatroid_cuts},
    {"queue_price_search",&SolverOptions::queue_price_search},
    {"queue_fractional_separation",&SolverOptions::queue_fractional_separation},
    {"design_face_cuts",&SolverOptions::design_face_cuts},
    {"queue_joint_cuts",&SolverOptions::queue_joint_cuts},
    {"primal_coupled_neighborhood",&SolverOptions::primal_coupled_neighborhood},
    {"primal_seed_portfolio",&SolverOptions::primal_seed_portfolio},
    {"branch_pricing_correction",&SolverOptions::branch_pricing_correction},
    {"cg_seed_reuse",&SolverOptions::cg_seed_reuse},
    {"staged_continuous_root",&SolverOptions::staged_continuous_root},
    {"fast_pricing",&SolverOptions::fast_pricing},
    {"batch_columns",&SolverOptions::batch_columns},
    {"primal_alns",&SolverOptions::primal_alns},
    {"eliminate_zero_task_columns",&SolverOptions::eliminate_zero_task_columns},
    {"compact_task_heads",&SolverOptions::compact_task_heads},
    {"cache_pricing_graph",&SolverOptions::cache_pricing_graph},
    {"persist_recourse_model",&SolverOptions::persist_recourse_model},
    {"stabilized_pricing",&SolverOptions::stabilized_pricing},
    {"primal_support_completion",&SolverOptions::primal_support_completion},
    {"grouped_local_branching",&SolverOptions::grouped_local_branching},
    {"cut_pool_deduplication",&SolverOptions::cut_pool_deduplication},
    {"cut_pool_management",&SolverOptions::cut_pool_management},
    {"streamlined_recourse",&SolverOptions::streamlined_recourse},
    {"catalogue_pricing",&SolverOptions::catalogue_pricing},
    {"diverse_pricing",&SolverOptions::diverse_pricing},
    {"reuse_recourse_rhs",&SolverOptions::reuse_recourse_rhs},
    {"design_certificate_cache",&SolverOptions::design_certificate_cache},
    {"adaptive_pricing",&SolverOptions::adaptive_pricing},
    {"cg_primal_feedback",&SolverOptions::cg_primal_feedback},
    {"indexed_support_completion",&SolverOptions::indexed_support_completion},
    {"stabilized_master",&SolverOptions::stabilized_master},
    {"target_master_refinement",&SolverOptions::target_master_refinement},
    {"weak_dual_cuts",&SolverOptions::weak_dual_cuts},
    {"interior_cuts",&SolverOptions::interior_cuts},
    {"primal_asset_neighborhood",&SolverOptions::primal_asset_neighborhood},
    {"branch_covering_seeds",&SolverOptions::branch_covering_seeds},
    {"resource_bounds",&SolverOptions::resource_bounds},
    {"capacity_cuts",&SolverOptions::capacity_cuts},
    {"resource_lp_cuts",&SolverOptions::resource_lp_cuts},
    {"adaptive_master",&SolverOptions::adaptive_master},
    {"neighborhood_cuts",&SolverOptions::neighborhood_cuts},
    {"continuous_root",&SolverOptions::continuous_root},
    {"root_projection",&SolverOptions::root_projection},
    {"root_inout",&SolverOptions::root_inout},
    {"auxiliary_valid_cuts",&SolverOptions::auxiliary_valid_cuts},
    {"resource_airline_bounds",&SolverOptions::resource_airline_bounds},
    {"od_airline_bounds",&SolverOptions::od_airline_bounds},
    {"tight_airline_bounds",&SolverOptions::tight_airline_bounds},
    {"assignment_incident_clique",&SolverOptions::assignment_incident_clique},
    {"shared_rho_joint",&SolverOptions::shared_rho_joint},
    {"queue_perspective",&SolverOptions::queue_perspective},
    {"topology_cuts",&SolverOptions::topology_cuts},
    {"path_support_cuts",&SolverOptions::path_support_cuts},
    {"task_savings_bound",&SolverOptions::task_savings_bound},
    {"task_resource_cuts",&SolverOptions::task_resource_cuts},
    {"task_airline_bounds",&SolverOptions::task_airline_bounds},
    {"airline_connectivity_cuts",&SolverOptions::airline_connectivity_cuts},
    {"airline_resource_cuts",&SolverOptions::airline_resource_cuts},
    {"root_cuts",&SolverOptions::root_cuts},
    {"restricted_mip",&SolverOptions::restricted_mip},
    {"local_branching",&SolverOptions::local_branching},
    {"incumbent_cutoff",&SolverOptions::incumbent_cutoff}
};
}

SolverConfiguration::SolverConfiguration() { apply_method("benders_cg"); }
void SolverConfiguration::set_acceleration(const std::string& name,bool enabled) {
    for(const auto& flag:accelerations)if(name==flag.name) {
        options.*(flag.member)=enabled;return;
    }
    throw std::invalid_argument("Unknown acceleration: "+name);
}
void SolverConfiguration::apply_method(const std::string& name) {
    if(name!="benders_cg" && name!="full_path")
        throw std::invalid_argument("--method expects benders_cg or full_path");
    SolverOptions selected;
    selected.output=options.output;selected.progress=options.progress;
    selected.log=options.log;selected.export_catalogue=options.export_catalogue;
    options=std::move(selected);
    method=name;path_column_limit=0;auto_disabled_accelerations=Json::array();
    for(const auto& flag:accelerations)options.*(flag.member)=false;
    options.preprocessed_demand=true;
    options.conditional_demand=false;options.first_leg_consistency=false;
    options.seconds=600;options.gap=.01;options.cg_max_columns=512;options.max_iterations=1000000;
    for(const char* flag:{"shared_rho_joint","queue_perspective","tight_airline_bounds","task_airline_bounds"})
        set_acceleration(flag,true);
    if(method=="full_path")return;
    for(const char* flag:{"pareto_cuts","reuse_dual_lp","reuse_columns","courier_savings_bound",
        "initial_design","warm_start","incumbent_cleanup","primal_search","branch_covering_seeds",
        "resource_bounds","capacity_cuts","resource_lp_cuts","adaptive_master","neighborhood_cuts",
        "continuous_root","root_projection","root_inout","auxiliary_valid_cuts","resource_airline_bounds",
        "root_cuts","restricted_mip","local_branching","incumbent_cutoff","od_airline_bounds","topology_cuts",
        "path_support_cuts","task_savings_bound","task_resource_cuts","airline_connectivity_cuts",
        "airline_resource_cuts","primal_polish","search_scheduling","fractional_path_support",
        "airline_resource_balance","weak_dual_cuts","interior_cuts","queue_polymatroid_cuts",
        "queue_price_search","target_master_refinement","queue_fractional_separation",
        "design_face_cuts","queue_joint_cuts","branch_pricing_correction","cg_seed_reuse","staged_continuous_root",
        "fast_pricing","batch_columns","primal_alns",
        "eliminate_zero_task_columns","compact_task_heads","cache_pricing_graph",
        "persist_recourse_model","stabilized_pricing","primal_support_completion","grouped_local_branching",
        "cut_pool_deduplication","cut_pool_management",
        "streamlined_recourse","catalogue_pricing","diverse_pricing","reuse_recourse_rhs",
        "design_certificate_cache","adaptive_pricing","cg_primal_feedback","indexed_support_completion","stabilized_master"})
        set_acceleration(flag,true);
    // These implementations retain a complete service catalogue or construct
    // an auxiliary LP with one row/variable per physical path. They are not
    // part of implicit CG; streaming task/EPI/design-face cuts remain enabled.
    for(const char* flag:{"catalogue_pricing","cache_pricing_graph",
        "path_support_cuts","fractional_path_support","airline_resource_cuts","airline_resource_balance"}) {
        set_acceleration(flag,false);
        auto_disabled_accelerations.push_back(flag);
    }
}

void SolverConfiguration::adapt_instance(const Input& data) {
    if(data.n_l!=2 || data.max_orders!=2) {
        for(const char* name:{"primal_alns","primal_support_completion","indexed_support_completion","cg_primal_feedback"})
            for(const auto& flag:accelerations)if(name==std::string(flag.name) && options.*(flag.member)) {
                options.*(flag.member)=false;
                auto_disabled_accelerations.push_back(name);
            }
    }
}

void SolverConfiguration::validate() const {
    if(method=="benders_cg" && path_column_limit!=0)
        throw std::invalid_argument("--path-column-limit is a full_path enumeration limit; implicit CG requires complete-domain scans");
    if(!options.preprocessed_demand || options.conditional_demand || options.first_leg_consistency)
        throw std::invalid_argument("Only preprocessed independent-task demand is supported");
    if((options.catalogue_pricing || options.diverse_pricing || options.adaptive_pricing ||
        options.reuse_recourse_rhs || options.design_certificate_cache || options.stabilized_master) && method!="benders_cg")
        throw std::invalid_argument("Follow-up decomposition policies require Benders CG");
    if(options.diverse_pricing && !options.catalogue_pricing && !options.preprocessed_demand)
        throw std::invalid_argument("Diverse pricing requires catalogue or implicit task pricing");
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
        throw std::invalid_argument("Primal ALNS requires preprocessed task Benders CG with column reuse");
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
    if(method!="full_path" && method!="benders_cg")
        throw std::invalid_argument("Unsupported method");
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
       ||options.path_support_cut_limit<0||options.path_support_max_cuts<1
       ||!std::isfinite(options.path_support_seconds)||options.path_support_seconds<=0
       ||!std::isfinite(options.path_support_preprocessing_seconds)||options.path_support_preprocessing_seconds<=0
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
    return {{"schema","mlnd-solver-configuration/1"},{"method",method},
        {"auto_disabled_accelerations",auto_disabled_accelerations},
        {"path_storage",method=="full_path"?"complete_catalogue":"generated_service_pool"},
        {"implicit_memory_policy",method=="benders_cg"},
        {"auto_disable_reason",method=="benders_cg"?Json("catalogue_dependent_auxiliaries_and_instance_restrictions"):Json(nullptr)},
        {"demand_formulation","preprocessed"},
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
            {"path_support_seconds",options.path_support_seconds},
            {"path_support_preprocessing_seconds",options.path_support_preprocessing_seconds},
            {"path_support_cut_limit",options.path_support_cut_limit},{"path_support_max_cuts",options.path_support_max_cuts},
            {"task_resource_seconds",options.task_resource_seconds},{"task_resource_cut_limit",options.task_resource_cut_limit},
            {"airline_resource_seconds",options.airline_resource_seconds},{"airline_resource_cut_limit",options.airline_resource_cut_limit},
            {"capacity_cut_limit",options.capacity_cut_limit},
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
