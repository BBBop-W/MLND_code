#include "mlnd/benders.hpp"
#include "mlnd/output.hpp"
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
    std::string method="full_path", stage="arguments";
    int k=0;
    SolverOptions options;
    bool inspect=false, have_input=false, preprocessing_complete=false;
    std::size_t path_column_limit=500000;
    double started=0, budget=0, read_seconds=0, preprocessing_seconds=0, stage_started=0;
    Input data;
    PathCollection paths; paths.complete=false; paths.support_complete=false;
    Json metadata=Json::object();
    auto result_method=[&]() -> std::string {
        if(fixed_path.empty()) return method;
        if(method=="full_path") return "fixed_design";
        return method=="benders_cg"?"fixed_design_cg":"fixed_design_benders_full";
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
                stage=="cplex_column_insertion"||stage=="path_enumeration"||stage=="path_support";
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
            result["support_preprocessing_seconds"]=method=="benders_cg"?preprocessing_seconds:0.0;
            result["enumeration_seconds"]=method=="benders_cg"?0.0:preprocessing_seconds;
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
            if(key=="--input") input=argument();
            else if(key=="--out") options.output=argument();
            else if(key=="--k") k=std::stoi(argument());
            else if(key=="--method") method=argument();
            else if(key=="--demand-formulation") {
                const auto value=argument();
                if(value!="lifted" && value!="conditional")
                    throw std::invalid_argument("--demand-formulation expects lifted or conditional");
                options.conditional_demand=value=="conditional";
            }
            else if(key=="--tight-airline-bounds") options.tight_airline_bounds=true;
            else if(key=="--structural-strengthening") {
                options.assignment_incident_clique=options.shared_rho_joint=options.queue_perspective=true;
            }
            else if(key=="--assignment-incident-clique") options.assignment_incident_clique=true;
            else if(key=="--shared-rho-joint") options.shared_rho_joint=true;
            else if(key=="--queue-perspective") options.queue_perspective=true;
            else if(key=="--od-airline-bounds") options.od_airline_bounds=true;
            else if(key=="--topology-cuts") options.topology_cuts=true;
            else if(key=="--topology-seconds") options.topology_seconds=std::stod(argument());
            else if(key=="--topology-cut-limit") options.topology_cut_limit=std::stoi(argument());
            else if(key=="--assignment-support-cuts") options.assignment_support_cuts=true;
            else if(key=="--assignment-support-seconds") options.assignment_support_seconds=std::stod(argument());
            else if(key=="--assignment-support-floor") options.assignment_support_floor=std::stod(argument());
            else if(key=="--assignment-support-cut-limit") options.assignment_support_cut_limit=std::stoi(argument());
            else if(key=="--assignment-support-max-cuts") options.assignment_support_max_cuts=std::stoi(argument());
            else if(key=="--double-support-cuts") options.double_support_cuts=true;
            else if(key=="--joint-base-cost-cuts") options.joint_base_cost_cuts=true;
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
            else if(key=="--start") start_path=argument();
            else if(key=="--fixed-design") fixed_path=argument();
            else if(key=="--no-pareto") options.pareto=false;
            else if(key=="--no-reuse-columns") options.reuse_columns=false;
            else if(key=="--plain") {
                options.reuse_dual_lp=false;
                options.pareto=options.reuse_columns=options.courier_savings_bound=false;
                options.initial_design=options.warm_start=options.incumbent_cleanup=options.primal_search=false;
                options.branch_covering_seeds=options.resource_bounds=false;
                options.capacity_cuts=false;
                options.resource_lp_cuts=options.adaptive_master=false;
                options.neighborhood_cuts=false;
                options.continuous_root=false;
                options.root_projection=options.root_inout=options.auxiliary_valid_cuts=false;
                options.resource_airline_bounds=false;
                options.od_airline_bounds=false;
                options.root_cuts=options.restricted_mip=options.local_branching=options.incumbent_cutoff=false;
            }
            else if(key=="--disable") {
                const auto flag=argument();
                if(flag=="pareto_cuts") options.pareto=false;
                else if(flag=="reuse_dual_lp") options.reuse_dual_lp=false;
                else if(flag=="reuse_columns") options.reuse_columns=false;
                else if(flag=="courier_savings_bound") options.courier_savings_bound=false;
                else if(flag=="initial_design") options.initial_design=false;
                else if(flag=="warm_start") options.warm_start=false;
                else if(flag=="incumbent_cleanup") options.incumbent_cleanup=false;
                else if(flag=="primal_search") options.primal_search=false;
                else if(flag=="branch_covering_seeds") options.branch_covering_seeds=false;
                else if(flag=="resource_bounds") options.resource_bounds=false;
                else if(flag=="capacity_cuts") options.capacity_cuts=false;
                else if(flag=="resource_lp_cuts") options.resource_lp_cuts=false;
                else if(flag=="adaptive_master") options.adaptive_master=false;
                else if(flag=="neighborhood_cuts") options.neighborhood_cuts=false;
                else if(flag=="continuous_root") options.continuous_root=false;
                else if(flag=="root_projection") options.root_projection=false;
                else if(flag=="root_inout") options.root_inout=false;
                else if(flag=="auxiliary_valid_cuts") options.auxiliary_valid_cuts=false;
                else if(flag=="resource_airline_bounds") options.resource_airline_bounds=false;
                else if(flag=="od_airline_bounds") options.od_airline_bounds=false;
                else if(flag=="tight_airline_bounds") options.tight_airline_bounds=false;
                else if(flag=="assignment_incident_clique") options.assignment_incident_clique=false;
                else if(flag=="shared_rho_joint") options.shared_rho_joint=false;
                else if(flag=="queue_perspective") options.queue_perspective=false;
                else if(flag=="topology_cuts") options.topology_cuts=false;
                else if(flag=="assignment_support_cuts") options.assignment_support_cuts=false;
                else if(flag=="double_support_cuts") options.double_support_cuts=false;
                else if(flag=="joint_base_cost_cuts") options.joint_base_cost_cuts=false;
                else if(flag=="root_cuts") options.root_cuts=false;
                else if(flag=="restricted_mip") options.restricted_mip=false;
                else if(flag=="local_branching") options.local_branching=false;
                else if(flag=="incumbent_cutoff") options.incumbent_cutoff=false;
                else throw std::invalid_argument("Unknown acceleration: "+flag);
            }
            else if(key=="--log") options.log=true;
            else if(key=="--capacity-cut-limit") options.capacity_cut_limit=std::stoi(argument());
            else if(key=="--inspect") inspect=true;
            else if(key=="--help") {
                std::cout<<"mlnd_cpp --input INSTANCE --k 1|2 --out NEW_DIRECTORY "
                           "[--method full_path|benders_cg|benders_full] [--seconds 600] "
                           "[--demand-formulation lifted|conditional] "
                           "[--structural-strengthening] [--assignment-incident-clique] [--shared-rho-joint] [--queue-perspective] "
                           "[--tight-airline-bounds] [--topology-cuts] [--topology-seconds 2] [--topology-cut-limit 1000] "
                           "[--od-airline-bounds] "
                           "[--assignment-support-cuts] [--assignment-support-seconds .25] "
                           "[--assignment-support-floor 1e-4] [--assignment-support-cut-limit 2000] [--assignment-support-max-cuts 16] "
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
                           "[--cg-duality-tolerance 1e-6] [--pricing-tolerance 1e-7] [--log]\n";
                return 0;
            } else throw std::invalid_argument("Unknown argument: "+key);
        }
        if(input.empty()) throw std::invalid_argument("--input is required");
        if(method!="full_path" && method!="benders_cg" && method!="benders_full")
            throw std::invalid_argument("Unsupported method");
        if(options.conditional_demand && method=="full_path")
            throw std::invalid_argument("Conditional demand is an integer-recourse Benders mode; full_path remains lifted");
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
        started=wall_time(); budget=options.seconds;
        if(!inspect) {
            if(options.output.empty()) throw std::invalid_argument("--out is required");
            if(std::filesystem::exists(options.output/"result.json") || std::filesystem::exists(options.output/"progress.jsonl"))
                throw std::invalid_argument("Output already contains a run; use a new directory");
            metadata={{"solver","CPLEX"},{"engine","cpp"},{"requested_method",method},{"method",result_method()},
                {"demand_formulation",options.conditional_demand?"conditional":"lifted"},
                {"structural_strengthening",{{"assignment_incident_clique",options.assignment_incident_clique},
                    {"shared_rho_joint",options.shared_rho_joint},{"queue_perspective",options.queue_perspective}}},
                {"od_airline_bounds",options.od_airline_bounds},
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
        read_seconds=wall_time()-stage_started;
        Json summary={{"name",data.name},{"max_orders",data.max_orders},{"n_l",data.n_l},
                      {"n_h",data.n_h},{"n_od",data.n_od},{"input_read_seconds",read_seconds}};
        if(inspect) { std::cout<<summary.dump(2)<<'\n'; return 0; }
        metadata.update({{"max_orders",data.max_orders},{"name",data.name},
            {"n_l",data.n_l},{"n_h",data.n_h},{"n_od",data.n_od},
            {"flow_policy",data.flow_policy},{"effective_delta",data.effective_delta()},{"path_scope",data.route_mode}});
        options.progress->update_metadata(metadata);
        atomic_json_file(options.output/"input_reference.json",{{"instance_path",data.instance_path.string()},
            {"manifest",data.manifest},{"model",data.model_json},{"max_orders",data.max_orders}});
        options.progress->record({{"event","phase_end"},{"stage",stage},{"seconds",read_seconds},{"instance",summary}});
        // This feasible baseline exists before any path enumeration or model
        // construction. A partial construction must never erase it.
        if(fixed_path.empty()) {
            auto baseline=all_courier_result(data,paths);
            baseline["construction_complete"]=false; baseline["support_complete"]=false;
            baseline["catalogue_complete"]=false; baseline["solved"]=false;
            baseline["feasibility_basis"]="all_courier_construction";
            baseline["max_orders"]=data.max_orders; baseline["name"]=data.name;
            options.progress->checkpoint(baseline,"before_preprocessing");
        }
        const bool implicit_paths=method=="benders_cg";
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
        result["demand_formulation"]=options.conditional_demand?"conditional":"lifted";
        result["tight_airline_bounds"]=options.tight_airline_bounds;
        result["od_airline_bounds"]=options.od_airline_bounds;
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
        result["catalogue_complete"]=paths.complete;
        result["support_complete"]=paths.support_complete;
        result["generated_columns"]={{"service",paths.service_paths.size()},{"empty",paths.empty_paths.size()}};
        result["input_read_seconds"]=read_seconds;
        result["support_preprocessing_seconds"]=implicit_paths?preprocessing_seconds:0.0;
        result["enumeration_seconds"]=implicit_paths?0.0:preprocessing_seconds;
        result["wall_before_output"]=wall_time()-started;
        std::filesystem::create_directories(options.output);
        const double output_started=wall_time();
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
