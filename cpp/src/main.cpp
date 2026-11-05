#include "mlnd/benders.hpp"
#include "mlnd/output.hpp"
#include "mlnd/preprocessed_demand.hpp"
#include "mlnd/solver_presets.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
mlnd::Json read_json(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot read " + path.string());
    mlnd::Json value; stream >> value; return value;
}
}
int main(int argc,char** argv) {
    using namespace mlnd;
    std::filesystem::path input, start_path, fixed_path;
    SolverConfiguration configuration;
    auto& options=configuration.options;
    auto& method=configuration.method;
    auto& path_column_limit=configuration.path_column_limit;
    std::string stage="arguments";
    int k=0;
    bool inspect=false, have_input=false, preprocessing_complete=false;
    double started=0, budget=0, read_seconds=0, preprocessing_seconds=0, stage_started=0;
    Input data;
    PathCollection paths; paths.complete=false; paths.support_complete=false;
    Json metadata=Json::object(), resolved_configuration=Json::object();
    auto result_method=[&]() -> std::string {
        if(fixed_path.empty()) return method;
        if(method=="full_path") return "fixed_design";
        return method=="benders_cg"?"fixed_design_cg":"fixed_design_benders_full";
    };
    auto demand_formulation=[&]() -> std::string {
        return options.preprocessed_demand ? "preprocessed" :
            options.conditional_demand ? "conditional" : "lifted";
    };
    auto compact_result=[](const Json& source) {
        Json compact=Json::object();
        for(auto it=source.begin();it!=source.end();++it)
            if(it.key().empty()||it.key()[0]!='_') compact[it.key()]=it.value();
        return compact;
    };
    auto failed=[&](const std::string& reason,const std::string& error,Json details,int exit_code) {
        std::cerr<<"Error: "<<error<<'\n';
        set_active_run_progress(nullptr); // Output finalization is outside the solve budget.
        if (!options.progress) return exit_code;
        try {
            if(stage=="input_read"&&stage_started>0) read_seconds=wall_time()-stage_started;
            if((stage=="path_enumeration"||stage=="path_support")&&stage_started>0)
                preprocessing_seconds=wall_time()-stage_started;
            Json result=metadata;
            const auto checkpoint=options.output/"checkpoint.json";
            if(std::filesystem::is_regular_file(checkpoint)) result.update(read_json(checkpoint));
            result["solver"]="CPLEX"; result["engine"]="cpp"; result["requested_method"]=method;
            result["method"]=result_method();
            result["termination_reason"]=reason; result["failure_stage"]=stage; result["error"]=error;
            result["failure_details"]=details;
            result["solved"]=false; result["target_gap_met"]=false; result["pricing_complete"]=false;
            result["optimization_completed"]=stage=="output_finalization";
            result["any_model_constructed"]=options.progress->summary().at("solver_models_built").get<std::size_t>()>0;
            result["preprocessing_complete"]=preprocessing_complete;
            result["failed_during_construction"]=stage=="model_record_construction"||stage=="cplex_model_construction"||
                stage=="cplex_column_insertion"||stage=="path_enumeration"||stage=="path_support"||stage=="demand_preprocessing";
            result["construction_complete"]=result.at("any_model_constructed").get<bool>()&&
                !result.at("failed_during_construction").get<bool>();
            result["support_complete"]=paths.support_complete;
            result["catalogue_complete"]=paths.complete;
            result["has_feasible_solution"]=result.contains("_solution") && !result.at("_solution").is_null();
            result["bound_scope"]=fixed_path.empty()?"global":"fixed_design";
            result["validated_incumbent"]=result.at("has_feasible_solution");
            if(!result.contains("objective")) result["objective"]=nullptr;
            if(!result.contains("bound")) result["bound"]=nullptr;
            if(!result.contains("gap")) result["gap"]=nullptr;
            result["input_read_seconds"]=read_seconds;
            result["support_preprocessing_seconds"]=method=="benders_cg"&&!options.preprocessed_demand?preprocessing_seconds:0.0;
            result["enumeration_seconds"]=method=="benders_cg"&&!options.preprocessed_demand?0.0:preprocessing_seconds;
            result["failed_phase_seconds"]=details.contains("pipeline_stage")?Json(nullptr):
                Json(stage_started>0?wall_time()-stage_started:0);
            result["wall_before_output"]=wall_time()-started;
            result["monitoring"]=options.progress->summary();
            if(result.at("objective").is_number()&&result.at("bound").is_number()) {
                const double upper=result.at("objective").get<double>();
                const double latest=result.at("monitoring").at("global_lower_bound").get<double>();
                if(std::isfinite(latest)&&latest<=upper+1e-6*(1+std::abs(upper))) {
                    result["bound"]=std::min(upper,std::max(latest,result.at("bound").get<double>()));
                    result["gap"]=(upper-result.at("bound").get<double>())/std::max(1.0,std::abs(upper));
                }
            }
            if(details.contains("path_sizes")) result["path_sizes_at_stop"]=details.at("path_sizes");
            else result["path_sizes_at_stop"]=result["monitoring"]["last_path_sizes"];
            if(details.contains("model_size")) result["model_size_at_stop"]=details.at("model_size");
            else result["model_size_at_stop"]=result["monitoring"]["last_model_size"];
            // Establish a terminal overview before optional bulky exports. The
            // last atomic checkpoint remains recoverable if finalization fails.
            atomic_json_file(options.output/"result.json",compact_result(result));
            const double output_started=wall_time();
            try {
                if(have_input && result.at("has_feasible_solution").get<bool>()) {
                    // The early baseline predates support enumeration. Expand
                    // its all-zero operational arrays to the current registry
                    // before exporting a later construction failure.
                    if(result.value("incumbent_kind",std::string())=="all_courier_baseline") {
                        const auto baseline=all_courier_result(data,paths);
                        for(const char* key:{"_solution","_segments","_missions","_relocations"})
                            result[key]=baseline.at(key);
                    }
                    synchronize_result_paths(data,paths,result);
                    write_result(options.output,data,paths,result);
                }
            } catch(const std::exception& output_error) { result["output_error"]=output_error.what(); }
            result["output_seconds"]=wall_time()-output_started;
            result["wall"]=wall_time()-started;
            atomic_json_file(options.output/"result.json",compact_result(result));
            options.progress->record({{"event","run_failed"},{"stage",stage},{"termination_reason",reason},
                {"error",error},{"solved",false},{"has_feasible_solution",result.at("has_feasible_solution")}});
            options.progress->status("failed",{{"termination_reason",reason},{"failure_stage",stage},
                {"solved",false},{"result_file","result.json"},{"error",error}});
        } catch(const std::exception& output_error) {
            std::cerr<<"Unable to finalize failure outputs: "<<output_error.what()<<'\n';
        }
        return exit_code;
    };
    try {
        for(int i=1;i<argc;++i) {
            std::string key=argv[i];
            auto argument=[&]() -> std::string {
                if(++i>=argc) throw std::invalid_argument("Missing value for "+key);
                return argv[i];
            };
            if(configuration.acceleration_flag(key))continue;
            if(key=="--preset") configuration.apply_preset(argument());
            else if(key=="--input") input=argument();
            else if(key=="--out") options.output=argument();
            else if(key=="--k") k=std::stoi(argument());
            else if(key=="--method") method=argument();
            else if(key=="--demand-formulation") {
                const auto value=argument();
                if(value!="lifted" && value!="conditional" && value!="preprocessed")
                    throw std::invalid_argument("--demand-formulation expects lifted, conditional, or preprocessed");
                options.conditional_demand=value=="conditional";
                options.preprocessed_demand=value=="preprocessed";
            }
            else if(key=="--first-leg-consistency") options.first_leg_consistency=true;
            else if(key=="--structural-strengthening") {
                options.assignment_incident_clique=options.shared_rho_joint=options.queue_perspective=true;
            }
            else if(key=="--topology-seconds") options.topology_seconds=std::stod(argument());
            else if(key=="--topology-cut-limit") options.topology_cut_limit=std::stoi(argument());
            else if(key=="--assignment-support-seconds") options.assignment_support_seconds=std::stod(argument());
            else if(key=="--assignment-support-floor") options.assignment_support_floor=std::stod(argument());
            else if(key=="--assignment-support-cut-limit") options.assignment_support_cut_limit=std::stoi(argument());
            else if(key=="--assignment-support-max-cuts") options.assignment_support_max_cuts=std::stoi(argument());
            else if(key=="--airline-resource-seconds") options.airline_resource_seconds=std::stod(argument());
            else if(key=="--airline-resource-cut-limit") options.airline_resource_cut_limit=std::stoi(argument());
            else if(key=="--task-resource-seconds") options.task_resource_seconds=std::stod(argument());
            else if(key=="--task-resource-cut-limit") options.task_resource_cut_limit=std::stoi(argument());
            else if(key=="--path-support-seconds") options.path_support_seconds=std::stod(argument());
            else if(key=="--path-support-preprocessing-seconds") options.path_support_preprocessing_seconds=std::stod(argument());
            else if(key=="--path-support-cut-limit") options.path_support_cut_limit=std::stoi(argument());
            else if(key=="--path-support-max-cuts") options.path_support_max_cuts=std::stoi(argument());
            else if(key=="--branch-joint-seconds") options.branch_joint_seconds=std::stod(argument());
            else if(key=="--branch-joint-preprocessing-seconds") options.branch_joint_preprocessing_seconds=std::stod(argument());
            else if(key=="--branch-joint-floor") options.branch_joint_floor=std::stod(argument());
            else if(key=="--branch-joint-cut-limit") options.branch_joint_cut_limit=std::stoi(argument());
            else if(key=="--branch-joint-max-cuts") options.branch_joint_max_cuts=std::stoi(argument());
            else if(key=="--seconds") options.seconds=std::stod(argument());
            else if(key=="--gap") options.gap=std::stod(argument());
            else if(key=="--threads") options.threads=std::stoi(argument());
            else if(key=="--seed") options.seed=std::stoi(argument());
            else if(key=="--max-iterations") options.max_iterations=std::stoi(argument());
            else if(key=="--cg-max-columns") options.cg_max_columns=std::stoi(argument());
            else if(key=="--path-column-limit") {
                const auto value=argument();
                if(value.empty()||value.front()=='-') throw std::invalid_argument("Path column limit must be nonnegative");
                path_column_limit=std::stoull(value);
            }
            else if(key=="--cg-duality-tolerance") options.cg_duality_tolerance=std::stod(argument());
            else if(key=="--pricing-tolerance") options.pricing_tolerance=std::stod(argument());
            else if(key=="--root-cut-rounds") options.root_cut_rounds=std::stoi(argument());
            else if(key=="--root-inout-weight") options.root_inout_weight=std::stod(argument());
            else if(key=="--reuse-dual-lp") {
                const auto value=argument();
                if(value!="true" && value!="false") throw std::invalid_argument("--reuse-dual-lp expects true or false");
                options.reuse_dual_lp=(value=="true");
            }
            else if(key=="--continuous-root-seconds") options.continuous_root_seconds=std::stod(argument());
            else if(key=="--continuous-root-max-columns") options.continuous_root_max_columns=std::stoi(argument());
            else if(key=="--continuous-root-budget-fraction") options.continuous_root_budget_fraction=std::stod(argument());
            else if(key=="--master-initial-gap") options.master_initial_gap=std::stod(argument());
            else if(key=="--master-slice-seconds") options.master_slice_seconds=std::stod(argument());
            else if(key=="--resource-lp-seconds") options.resource_lp_seconds=std::stod(argument());
            else if(key=="--resource-lp-cut-limit") options.resource_lp_cut_limit=std::stoi(argument());
            else if(key=="--heuristic-pool-size") options.heuristic_pool_size=std::stoi(argument());
            else if(key=="--neighborhood-cut-seconds") options.neighborhood_cut_seconds=std::stod(argument());
            else if(key=="--neighborhood-cut-budget-fraction") options.neighborhood_cut_budget_fraction=std::stod(argument());
            else if(key=="--pareto-frequency") options.pareto_frequency=std::stoi(argument());
            else if(key=="--pareto-budget-fraction") options.pareto_budget_fraction=std::stod(argument());
            else if(key=="--pareto-call-seconds") options.pareto_call_seconds=std::stod(argument());
            else if(key=="--heuristic-budget-fraction") options.heuristic_budget_fraction=std::stod(argument());
            else if(key=="--root-budget-fraction") options.root_budget_fraction=std::stod(argument());
            else if(key=="--local-branching-radius") options.local_branching_radius=std::stoi(argument());
            else if(key=="--local-branching-head-radius") options.local_branching_head_radius=std::stoi(argument());
            else if(key=="--local-branching-capacity-radius") options.local_branching_capacity_radius=std::stoi(argument());
            else if(key=="--pricing-smoothing-weight") options.pricing_smoothing_weight=std::stod(argument());
            else if(key=="--pricing-exact-interval") options.pricing_exact_interval=std::stoi(argument());
            else if(key=="--cut-pool-active-limit") options.cut_pool_active_limit=std::stoi(argument());
            else if(key=="--cut-pool-age") options.cut_pool_age=std::stoi(argument());
            else if(key=="--design-cache-limit") options.design_cache_limit=std::stoi(argument());
            else if(key=="--stabilized-master-seconds") options.stabilized_master_seconds=std::stod(argument());
            else if(key=="--stabilized-master-level") options.stabilized_master_level=std::stod(argument());
            else if(key=="--start") start_path=argument();
            else if(key=="--fixed-design") fixed_path=argument();
            else if(key=="--no-pareto") configuration.set_acceleration("pareto_cuts",false);
            else if(key=="--no-reuse-columns") configuration.set_acceleration("reuse_columns",false);
            else if(key=="--plain") configuration.plain();
            else if(key=="--disable") configuration.set_acceleration(argument(),false);
            else if(key=="--export-catalogue") options.export_catalogue=true;
            else if(key=="--log") options.log=true;
            else if(key=="--capacity-cut-limit") options.capacity_cut_limit=std::stoi(argument());
            else if(key=="--inspect") inspect=true;
            else if(key=="--help") {
                std::cout<<"mlnd_cpp [--preset performance-20261105|accelerated-20261105|baseline-20261029|baseline|bounds-20261030|lb-20261030|speed-20261030|alns-20261030|full-cplex-20261029|legacy] "
                           "--input INSTANCE --k 1|2 --out NEW_DIRECTORY "
                           "[--method full_path|benders_cg|benders_full] [--seconds 600] "
                           "[--demand-formulation lifted|conditional|preprocessed] [--first-leg-consistency] "
                           "[--structural-strengthening] [--assignment-incident-clique] [--shared-rho-joint] [--queue-perspective] "
                           "[--tight-airline-bounds] [--topology-cuts] [--topology-seconds 2] [--topology-cut-limit 1000] "
                           "[--airline-connectivity-cuts] [--airline-resource-cuts] [--airline-resource-seconds 2] [--airline-resource-cut-limit 1000] "
                           "[--primal-polish] "
                           "[--design-face-cuts] [--queue-joint-cuts] "
                           "[--primal-coupled-neighborhood] [--primal-seed-portfolio] "
                           "[--branch-pricing-correction] [--cg-seed-reuse] [--staged-continuous-root] "
                           "[--eliminate-zero-task-columns] [--compact-task-heads] "
                           "[--cache-pricing-graph] [--persist-recourse-model] [--stabilized-pricing] "
                           "[--pricing-smoothing-weight .5] [--pricing-exact-interval 4] "
                           "[--primal-support-completion] [--grouped-local-branching] "
                           "[--local-branching-head-radius 4] [--local-branching-capacity-radius 2] "
                           "[--streamlined-recourse] [--catalogue-pricing] [--diverse-pricing] [--reuse-recourse-rhs] [--design-certificate-cache] [--design-cache-limit 16] [--adaptive-pricing] [--cg-primal-feedback] [--indexed-support-completion] [--stabilized-master] [--stabilized-master-seconds 2] [--stabilized-master-level .5] [--cut-pool-deduplication] [--cut-pool-management] [--cut-pool-active-limit 256] [--cut-pool-age 8] "
                           "[--weak-dual-cuts] [--interior-cuts] [--primal-asset-neighborhood] "
                           "[--search-scheduling] [--dual-lp-first] [--fractional-path-support] [--airline-resource-balance] "
                           "[--od-airline-bounds] "
                           "[--assignment-support-cuts] [--assignment-support-seconds .25] "
                           "[--assignment-support-floor 1e-4] [--assignment-support-cut-limit 2000] [--assignment-support-max-cuts 16] "
                           "[--path-support-cuts] [--path-support-seconds .25] [--path-support-preprocessing-seconds 2] "
                           "[--path-support-cut-limit 50000] [--path-support-max-cuts 32] "
                           "[--task-savings-bound] [--task-resource-cuts] [--task-airline-bounds] "
                           "[--double-support-cuts] [--joint-base-cost-cuts] [--branch-joint-seconds .25] "
                           "[--branch-joint-preprocessing-seconds 2] [--branch-joint-floor 1e-4] "
                           "[--branch-joint-cut-limit 2000] [--branch-joint-max-cuts 16] "
                           "[--gap 0.005] [--threads 1] [--seed 43] "
                           "[--start solution.json] [--fixed-design solution.json] "
                           "[--plain] [--disable ACCELERATION] [--no-pareto] "
                           "[--max-iterations 1000] [--cg-max-columns 64] [--path-column-limit 500000 (0=unlimited)] "
                           "[--root-cut-rounds 20] [--root-budget-fraction 0.1] "
                           "[--continuous-root-seconds 120] [--continuous-root-budget-fraction 0.65] "
                           "[--continuous-root-max-columns 512] "
                           "[--root-inout-weight 0.9] [--reuse-dual-lp true|false] "
                           "[--master-initial-gap 0.05] [--master-slice-seconds 30] "
                           "[--resource-lp-seconds 5] [--resource-lp-cut-limit 100] "
                           "[--heuristic-pool-size 4] [--neighborhood-cut-seconds 15] "
                           "[--neighborhood-cut-budget-fraction 0.05] "
                           "[--pareto-frequency 10] [--pareto-budget-fraction 0.05] "
                           "[--pareto-call-seconds 30] [--heuristic-budget-fraction 0.2] "
                           "[--local-branching-radius 8] "
                           "[--capacity-cut-limit 100] "
                           "[--queue-polymatroid-cuts] "
                           "[--queue-price-search] [--target-master-refinement] "
                           "[--queue-fractional-separation] "
                           "[--cg-duality-tolerance 1e-6] [--pricing-tolerance 1e-7] [--export-catalogue] [--log]\n"
                           "Place --preset first; later options override it. Without a preset, legacy defaults above apply.\n"
                           "baseline-20261029: preprocessed benders_cg, 600 s, gap 0.01, complete paths, fractional queue.\n"
                           "full-cplex-20261029: matching preprocessed full_path formulation and shared strengthening.\n"
                           "--export-catalogue writes full_catalogue.json after optimization, outside its budget.\n";
                return 0;
            } else throw std::invalid_argument("Unknown argument: "+key);
        }
        if(input.empty()) throw std::invalid_argument("--input is required");
        configuration.validate();
        resolved_configuration=configuration.metadata();
        started=wall_time(); budget=options.seconds;
        if(!inspect) {
            if(options.output.empty()) throw std::invalid_argument("--out is required");
            if(std::filesystem::exists(options.output/"result.json") || std::filesystem::exists(options.output/"progress.jsonl"))
                throw std::invalid_argument("Output already contains a run; use a new directory");
            metadata={{"solver","CPLEX"},{"engine","cpp"},{"requested_method",method},{"method",result_method()},
                {"preset",configuration.preset},{"resolved_configuration",resolved_configuration},
                {"export_catalogue",options.export_catalogue},{"demand_formulation",demand_formulation()},
                {"first_leg_consistency",options.first_leg_consistency},
                {"structural_strengthening",{{"assignment_incident_clique",options.assignment_incident_clique},
                    {"shared_rho_joint",options.shared_rho_joint},{"queue_perspective",options.queue_perspective}}},
                {"od_airline_bounds",options.od_airline_bounds},
                {"path_support_cuts",options.path_support_cuts},
                {"task_savings_bound",options.task_savings_bound},
                {"task_resource_cuts",options.task_resource_cuts},
                {"task_airline_bounds",options.task_airline_bounds},
                {"airline_connectivity_cuts",options.airline_connectivity_cuts},
                {"airline_resource_cuts",options.airline_resource_cuts},
                {"queue_polymatroid_cuts",options.queue_polymatroid_cuts},
                {"queue_price_search",options.queue_price_search},
                {"queue_fractional_separation",options.queue_fractional_separation},
                {"target_master_refinement",options.target_master_refinement},
                {"design_face_cuts",options.design_face_cuts},{"queue_joint_cuts",options.queue_joint_cuts},
                {"primal_coupled_neighborhood",options.primal_coupled_neighborhood},
                {"primal_seed_portfolio",options.primal_seed_portfolio},
                {"branch_pricing_correction",options.branch_pricing_correction},
                {"cg_seed_reuse",options.cg_seed_reuse},
                {"staged_continuous_root",options.staged_continuous_root},
                {"fast_pricing",options.fast_pricing},{"batch_columns",options.batch_columns},
                {"primal_alns",options.primal_alns},
                {"input",input.string()},
                {"requested_max_orders",k},
                {"budget",budget},{"threads",options.threads},{"seed",options.seed},{"requested_gap",options.gap},
                {"problem_scope",fixed_path.empty()?"global":"fixed_design"},
                {"path_column_limit",path_column_limit==0?Json(nullptr):Json(path_column_limit)},
                {"path_column_limit_argument",path_column_limit},
                {"budget_scope","read_preprocessing_model_construction_and_optimization"},
                {"output_finalization_outside_budget",true}};
            if(std::filesystem::is_regular_file(argv[0])) metadata["binary_sha256"]=sha256_file(argv[0]);
            if(std::filesystem::is_regular_file(input/"manifest.json")) metadata["input_manifest_sha256"]=sha256_file(input/"manifest.json");
            options.progress=std::make_shared<RunProgress>(options.output,started,budget,metadata);
            set_active_run_progress(options.progress);
            options.progress->status("running",{{"stage","input_read"},{"solved",false}});
            options.progress->record({{"event","run_started"},{"stage","input_read"},{"configuration",metadata}});
        }
        stage="input_read"; stage_started=wall_time();
        data=read_input(input,k); have_input=true;
        data.first_leg_consistency=options.first_leg_consistency;
        if(options.preprocessed_demand) {
            if((data.max_orders!=1 && data.max_orders!=2) || data.flow_policy!="aggregate")
                throw std::invalid_argument("Preprocessed demand requires --k 1 or --k 2 and aggregate flow policy");
            data.preprocessed_demand=true;
        }
        metadata["assignment_scope"]=data.assignment_scope();
        if (data.origin_assignment() && (options.assignment_support_cuts || options.double_support_cuts || options.joint_base_cost_cuts))
            throw std::invalid_argument("Origin assignment requires disabling legacy base-pair assignment-support, double-support, and joint-base-cost cuts");
        read_seconds=wall_time()-stage_started;
        Json summary={{"assignment_scope",data.assignment_scope()},{"assignment_variables",data.assignment_count()},{"name",data.name},{"max_orders",data.max_orders},{"n_l",data.n_l},
                      {"n_h",data.n_h},{"n_od",data.n_od},{"input_read_seconds",read_seconds},
                      {"preset",configuration.preset},{"resolved_configuration",resolved_configuration}};
        if(inspect) { std::cout<<summary.dump(2)<<'\n'; return 0; }
        metadata.update({{"max_orders",data.max_orders},{"name",data.name},
            {"n_l",data.n_l},{"n_h",data.n_h},{"n_od",data.n_od},
            {"flow_policy",data.flow_policy},{"effective_delta",data.effective_delta()},{"path_scope",data.route_mode}});
        options.progress->update_metadata(metadata);
        atomic_json_file(options.output/"input_reference.json",{{"instance_path",data.instance_path.string()},
            {"manifest",data.manifest},{"model",data.model_json},{"max_orders",data.max_orders},
            {"assignment_scope",data.assignment_scope()}});
        options.progress->record({{"event","phase_end"},{"stage",stage},{"seconds",read_seconds},{"instance",summary}});
        // This feasible baseline exists before any path enumeration or model
        // construction. A partial construction must never erase it.
        if(fixed_path.empty()) {
            auto baseline=all_courier_result(data,paths);
            baseline["construction_complete"]=false; baseline["support_complete"]=false;
            baseline["catalogue_complete"]=false; baseline["solved"]=false;
            baseline["generated_columns"]={{"service",0},{"empty",0}};
            baseline["generated_columns_scope"]="before_model_construction";
            baseline["feasibility_basis"]="all_courier_construction";
            baseline["max_orders"]=data.max_orders; baseline["name"]=data.name;
            options.progress->checkpoint(baseline,"before_preprocessing");
        }
        // The fixed task rates use minimum physical route costs and common
        // departure support. Enumerate their full domain once; Benders-CG
        // still inserts only selected columns into each recourse model.
        const bool implicit_paths=method=="benders_cg" && !options.preprocessed_demand;
        stage=implicit_paths?"path_support":"path_enumeration"; stage_started=wall_time();
        options.progress->guard(stage);
        options.progress->record({{"event","phase_start"},{"stage",stage}});
        auto path_progress=[&](std::size_t service,std::size_t empty,std::size_t segments) {
            options.progress->heartbeat(stage,{{"path_sizes",{{"segments",segments},{"service",service},{"empty",empty}}}});
        };
        paths=implicit_paths
            ?prepare_path_support(data,options.progress->remaining(),{},path_progress)
            :enumerate_paths(data,path_column_limit==0?std::nullopt:std::optional<std::size_t>(path_column_limit),
                             options.progress->remaining(),{},path_progress);
        if(options.preprocessed_demand) {
            options.progress->guard("demand_preprocessing");
            const double demand_started=wall_time();
            preprocess_generalized_demand(data,paths);
            data.preprocessed_demand_report["seconds"]=wall_time()-demand_started;
            if(data.first_leg_consistency)
                data.first_leg_support_report=first_leg_support(data,paths);
            options.progress->record({{"event","demand_preprocessed"},{"stage","demand_preprocessing"},
                {"seconds",data.preprocessed_demand_report.at("seconds")},
                {"accepted_pairs",data.preprocessed_demand_report.at("accepted_pairs")},
                {"max_conservation_error",data.preprocessed_demand_report.at("max_conservation_error")}});
        }
        if(options.task_airline_bounds) {
            options.progress->guard("task_airline_preprocessing");
            auto prepared=build_task_airline_bounds(data,paths);
            options.task_airline_upper=std::move(prepared.upper);
            options.task_airline_metadata=std::move(prepared.metadata);
            options.progress->record({{"event","task_airline_preprocessed"},
                {"metadata",options.task_airline_metadata}});
        }
        preprocessing_seconds=wall_time()-stage_started;
        preprocessing_complete=true;
        options.progress->record({{"event","phase_end"},{"stage",stage},{"seconds",preprocessing_seconds},
            {"path_sizes",{{"segments",paths.initial_segments.size()},{"service",paths.service_paths.size()},{"empty",paths.empty_paths.size()}}},
            {"support_complete",paths.support_complete},{"catalogue_complete",paths.complete}});
        std::cout<<(implicit_paths?"Implicit path support: K=":"Complete catalogue: K=")
                 <<data.max_orders<<", segments="<<paths.initial_segments.size()
                 <<", registered service="<<paths.service_paths.size()
                 <<", registered empty="<<paths.empty_paths.size()<<std::endl;
        Json start, fixed;
        if(!start_path.empty()) start=read_json(start_path);
        std::vector<double> fixed_design;
        if(!fixed_path.empty()) {
            fixed=read_json(fixed_path);
            fixed_design=solution_design(data,fixed);
            for(double value:fixed_design) if(value!=0&&value!=1)
                throw std::invalid_argument("--fixed-design must contain binary design decisions");
        }
        stage="optimization"; stage_started=wall_time();
        options.progress->guard(stage);
        options.seconds=options.progress->remaining();
        options.column_generation=method!="benders_full";
        Json result=method=="full_path"
            ?solve_full(data,paths,options,fixed_path.empty()?nullptr:&fixed_design,start_path.empty()?nullptr:&start)
            :fixed_path.empty()
                ?solve_benders(data,paths,options,start_path.empty()?nullptr:&start)
                :solve_fixed_design_cg(data,paths,options,fixed_design);
        set_active_run_progress(nullptr);
        stage="output_finalization"; stage_started=wall_time();
        synchronize_result_paths(data,paths,result);
        result["solver"]="CPLEX";
        result["preset"]=configuration.preset;
        result["resolved_configuration"]=resolved_configuration;
        result["export_catalogue"]=options.export_catalogue;
        result["demand_formulation"]=demand_formulation();
        result["tight_airline_bounds"]=options.tight_airline_bounds;
        result["od_airline_bounds"]=options.od_airline_bounds;
        result["task_airline_bounds"]=options.task_airline_bounds;
        result["task_airline_metadata"]=options.task_airline_metadata;
        { IloEnv env; IloCplex probe(env); result["solver_version"]=probe.getVersion(); env.end(); }
        result["engine"]="cpp";
        result["max_orders"]=data.max_orders;
        result["effective_delta"]=data.effective_delta();
        result["flow_policy"]=data.flow_policy;
        result["path_scope"]=paths.scope;
        result["budget"]=budget;
        result["threads"]=options.threads;
        result["seed"]=options.seed;
        result["requested_gap"]=options.gap;
        result["cg_parameters"]={{"max_columns_per_iteration",options.cg_max_columns},
                                  {"relative_duality_tolerance",options.cg_duality_tolerance},
                                  {"absolute_pricing_tolerance",options.pricing_tolerance}};
        result["budget_scope"]="read_preprocessing_model_construction_and_optimization";
        result["output_finalization_outside_budget"]=true;
        result["path_column_limit"]=path_column_limit==0?Json(nullptr):Json(path_column_limit);
        result["path_column_limit_argument"]=path_column_limit;
        result["requested_method"]=method;
        result["method"]=result_method();
        result["bound_scope"]=fixed_path.empty()?"global":"fixed_design";
        result["optimization_completed"]=true;
        result["has_feasible_solution"]=result.contains("_solution")&&!result.at("_solution").is_null();
        result["target_gap_met"]=result.at("gap").is_number()&&result.at("gap").get<double>()<=options.gap;
        result["solved"]=result.at("target_gap_met");
        result["fixed_design_source"]=fixed_path.empty()?Json(nullptr):Json(fixed_path.string());
        result["warm_start_source"]=start_path.empty()?Json(nullptr):Json(start_path.string());
        if(std::filesystem::is_regular_file(argv[0])) result["binary_sha256"]=sha256_file(argv[0]);
        result["input_manifest_sha256"]=sha256_file(input/"manifest.json");
        result["catalogue"]={{"segments",paths.initial_segments.size()},{"service",paths.service_paths.size()},{"empty",paths.empty_paths.size()}};
        result["path_storage"]=implicit_paths?"generated_columns_only":"complete_catalogue";
        result["pricing_domain"]=options.preprocessed_demand && method=="benders_cg"
            ? "complete_preprocessing_catalogue_with_exact_label_pricing"
            : implicit_paths ? "implicit_complete_physical_domain" : "complete_catalogue";
        result["catalogue_complete"]=paths.complete;
        result["support_complete"]=paths.support_complete;
        if(!result.contains("generated_columns")) {
            result["generated_columns"]={{"service",paths.service_paths.size()},{"empty",paths.empty_paths.size()}};
            result["generated_columns_scope"]=method=="full_path"?"full_path_model":"registered_generated_catalogue";
        }
        result["input_read_seconds"]=read_seconds;
        result["support_preprocessing_seconds"]=implicit_paths?preprocessing_seconds:0.0;
        result["enumeration_seconds"]=implicit_paths?0.0:preprocessing_seconds;
        result["wall_before_output"]=wall_time()-started;
        std::filesystem::create_directories(options.output);
        const double output_started=wall_time();
        if(options.export_catalogue)
            result["complete_catalogue_export"]=write_complete_catalogue(options.output/"full_catalogue.json",data,paths);
        write_result(options.output,data,paths,result);
        result["output_seconds"]=wall_time()-output_started;
        result["wall"]=wall_time()-started;
        result["monitoring"]=options.progress->summary();
        atomic_json_file(options.output/"result.json",compact_result(result));
        options.progress->checkpoint(result,"final_result",true);
        options.progress->record({{"event","run_finished"},{"stage","finished"},{"objective",result.at("objective")},
            {"upper_bound",result.at("objective")},{"bound",result.at("bound")},{"lower_bound",result.at("bound")},
            {"gap",result.at("gap")},{"bound_scope",result.at("bound_scope")},{"solved",result.at("solved")},
            {"termination_reason",result.at("termination_reason")},{"validated_incumbent",true}});
        options.progress->status("finished",{{"termination_reason",result.at("termination_reason")},
            {"solved",result.at("solved")},{"result_file","result.json"}});
        summary.update({{"objective",result["objective"]},{"bound",result["bound"]},{"gap",result["gap"]},
                        {"termination_reason",result["termination_reason"]},{"wall",result["wall"]},
                        {"out",options.output.string()}});
        std::cout<<summary.dump(2)<<'\n';
        return 0;
    } catch(const EnumerationStopped& error) {
        preprocessing_seconds=stage_started>0?wall_time()-stage_started:0;
        return failed("construction_"+error.reason,error.what(),
            {{"path_sizes",{{"columns",error.columns},{"segments",error.initial_segments}}}},1);
    } catch(const RunBudgetExceeded& error) {
        Json details=error.sizes;
        details["pipeline_stage"]=stage;
        details["pipeline_stage_elapsed_seconds"]=stage_started>0?wall_time()-stage_started:0;
        stage=error.stage;
        return failed("construction_time_limit",error.what(),details,1);
    } catch(const IloException& error) {
        return failed("cplex_error",error.getMessage(),Json::object(),2);
    } catch(const std::exception& error) {
        return failed("runtime_error",error.what(),Json::object(),1);
    }
}
