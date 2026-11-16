#include "mlnd/bdd_search.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace mlnd;
void require(bool condition,const std::string& text){if(!condition)throw std::runtime_error(text);}
Json load(const std::filesystem::path& path){std::ifstream in(path);if(!in)throw std::invalid_argument("Cannot read "+path.string());Json j;in>>j;return j;}
struct ActiveProgress {explicit ActiveProgress(const std::shared_ptr<RunProgress>& p){set_active_run_progress(p);}~ActiveProgress(){set_active_run_progress(nullptr);}};
template<class T> Matrix<T> constant(int rows,int cols,T value){Matrix<T> m(rows,cols);std::fill(m.values.begin(),m.values.end(),value);return m;}
Input fixture(int k){Input d;d.n_l=1;d.n_h=2;d.n_od=2;d.n_levels=2;d.max_orders=k;
    d.name="bdd_search_fixture";d.model_version="consecutive";d.route_mode="direct";d.flow_policy="aggregate";d.demand={4,7};
    d.flight_times=constant<double>(3,3,0);d.flight_times.values={0,.05,.07,.05,0,.1,.07,.1,0};
    d.rider_mean=constant<double>(2,2,0);d.rider_mean.values={.08,.12,.1,.06};
    d.first_cost=constant<double>(2,2,1);d.terminal_cost=constant<double>(2,2,.5);
    d.receipt_mean=constant<double>(2,2,.08);d.removal_mean=constant<double>(2,2,.03);
    d.ground_cost={5,5};d.deadline={1,1};d.base_cost={.5};
    d.slot_counts=constant<int>(2,2,0);d.slot_counts.values={10,20,10,20};
    d.slot_cost=constant<double>(2,2,0);d.slot_cost.values={.1,.2,.1,.2};d.slot_epsilon={.05,.05};
    d.link_cost=constant<double>(3,3,0);d.kappa=constant<double>(2,2,0);d.kappa.values={.3,.9,.2,.7};
    d.delta=.04;d.operating_limit=1;d.fleet_cost=1;d.flight_cost=.2;d.waiting_cost=.3;
    d.allowed_links=constant<int>(3,3,1);for(int i=0;i<3;++i)d.allowed_links(i,i)=0;
    d.occupancy_limit=constant<double>(2,2,0);d.occupancy_limit.values={6.1690072893953225,14.072024748341311,6.1690072893953225,14.072024748341311};return d;}
Json self_test(){Json output={{"schema","mlnd-bdd-search-regression/1"},{"cases",Json::array()}};
    for(int k:{1,2}){const auto data=fixture(k);const auto paths=enumerate_paths(data);SolverOptions options;options.gap=0;options.threads=1;
        const auto full=build_formulation(data,paths);NativeSolver reference(full.matrix,options,"bdd_search_test_full");const auto optimum=reference.optimize(20);
        require(optimum.optimal&&optimum.has_solution&&point_residuals(full.matrix,optimum.values).at("passed").get<bool>(),"Full reference failed");
        auto relaxed=full.matrix;for(auto& v:relaxed.variables)v.binary=false;NativeSolver continuous(relaxed,options,"bdd_search_test_continuous");const auto cont=continuous.optimize(20);
        require(cont.optimal&&cont.has_solution&&point_residuals(relaxed,cont.values).at("passed").get<bool>(),"Continuous reference failed");
        BddAnchor anchor;anchor.name="analytic_nonnegative_operating_cost";anchor.design=std::vector<double>(full.layout.count,0);
        anchor.coefficients=anchor.design;anchor.intercept=0;anchor.operating_objective=55;anchor.pricing_complete=true;
        BddFullDomain oracle(data,paths,options);
        Json cases={{"K",k},{"full_objective",optimum.objective},{"continuous_objective",cont.objective},{"methods",Json::array()}};
        for(const std::string method:{"fixed","search"}){BddSearchOptions config;config.method=method;config.started=wall_time();config.deadline=config.started+25;
            config.inner_seconds=3;config.master_seconds=2;config.box_radius=10;config.multipliers_per_root=3;config.max_evaluations=6;
            const auto result=search_bdd_root_bounds(data,paths,oracle,{anchor},cont.bound-1e-6,options,config);
            require(!result.contains("error"),"BDD search error: "+result.value("error",std::string()));
            require(result.at("baseline").at("certified").get<bool>()&&result.at("latest_root").at("certified").get<bool>(),"Root not certified");
            require(result.at("best_lower_bound").get<double>()<=optimum.objective+2e-5,"Root exceeds integer optimum");
            require(result.at("validated_original_integer_upper_bound").get<double>()>=optimum.objective-2e-5,"Primal UB below optimum");
            require(result.at("root_gain_lower_bound").get<double>()>=0,"Negative certified root gain");
            if(method=="fixed")require(result.at("status")=="all_fixed_multipliers_solved"&&result.at("evaluations_count")==1,"Exact fixed multipliers did not stop");
            for(const auto& cut:result.at("cuts")){const auto beta=cut.at("coefficients").get<std::vector<double>>();
                auto design=std::vector<double>(optimum.values.begin(),optimum.values.begin()+full.layout.count);
                const double f=std::inner_product(full.opening_costs.begin(),full.opening_costs.end(),design.begin(),0.0);
                const double value=cut.at("intercept").get<double>()+std::inner_product(beta.begin(),beta.end(),design.begin(),0.0);
                require(value<=optimum.objective-f+2e-5,"BDD cut excludes independently validated integer optimum");}
            cases["methods"].push_back(result);}
        output["cases"].push_back(cases);}
    output["passed"]=true;return output;}
}

int main(int argc,char** argv){using namespace mlnd;try{
    if(argc==3&&std::string(argv[1])=="--self-test"){const auto output=std::filesystem::absolute(argv[2]);require(!std::filesystem::exists(output),"Refusing overwrite");
        const auto result=self_test();atomic_json_file(output,result);std::cout<<"BDD search regression passed\n";return 0;}
    if(argc<5||(argc-5)%2)throw std::invalid_argument("Usage: mlnd_bdd_search_probe INPUT K ANCHORS_JSON OUTPUT_JSON --continuous-bound-file FILE [--method search|fixed] [--total-seconds 600] [--inner-seconds 30] [--master-seconds 3] [--box-radius 50] [--multipliers-per-root 5] [--max-evaluations N] [--tight-airline-bounds true|false]");
    const double started=wall_time();const auto input=std::filesystem::absolute(argv[1]);const int k=std::stoi(argv[2]);
    const auto anchor_path=std::filesystem::absolute(argv[3]),output=std::filesystem::absolute(argv[4]);
    BddSearchOptions config;config.started=started;double total=600;std::filesystem::path floor_path;std::optional<bool> tight_override;
    for(int a=5;a<argc;a+=2){const std::string flag=argv[a],value=argv[a+1];
        if(flag=="--method")config.method=value;else if(flag=="--total-seconds")total=std::stod(value);
        else if(flag=="--inner-seconds")config.inner_seconds=std::stod(value);else if(flag=="--master-seconds")config.master_seconds=std::stod(value);
        else if(flag=="--box-radius")config.box_radius=std::stod(value);else if(flag=="--multipliers-per-root")config.multipliers_per_root=std::stoi(value);
        else if(flag=="--max-evaluations")config.max_evaluations=std::stoi(value);
        else if(flag=="--continuous-bound-file")floor_path=std::filesystem::absolute(value);
        else if(flag=="--tight-airline-bounds"){require(value=="true"||value=="false","Invalid boolean");tight_override=value=="true";}
        else throw std::invalid_argument("Unknown option "+flag);}
    require((k==1||k==2)&&std::isfinite(total)&&total>0&&!floor_path.empty(),"Require K=1/2, positive budget and continuous-bound-file");
    require(config.method=="search"||config.method=="fixed","Method must be search or fixed");
    config.deadline=started+total;const auto diagnostics=std::filesystem::path(output.string()+".diagnostics");
    const auto search_log=std::filesystem::path(output.string()+".search.jsonl");
    require(!std::filesystem::exists(output)&&!std::filesystem::exists(diagnostics)&&!std::filesystem::exists(search_log),"Refusing existing result, diagnostics or iteration log");
    const auto data=read_input(input,k,true);const auto model_hash=sha256_file(input/"model.json");
    const auto source=load(floor_path);
    require(source.value("method",std::string())=="full_continuous_cg"&&source.value("scope",std::string())=="full_path_continuous_relaxation"&&
        source.value("objective_includes_fixed_costs",false)&&source.value("support_complete",false)&&source.value("bound_valid",false),"Continuous file lacks certified full-domain total-cost bound");
    require(source.at("K").get<int>()==k&&source.at("input_manifest")==data.manifest&&source.at("input_model_sha256")==model_hash,"Continuous file input provenance mismatch");
    const double bound=source.at("certified_lower_bound").get<double>();require(std::isfinite(bound),"Nonfinite continuous lower bound");
    const double floor=bound-1e-8*(1+std::abs(bound));
    const auto document=load(anchor_path);const Json anchors_json=document.is_array()?document:(document.contains("anchors")?document.at("anchors"):Json::array({document}));
    require(anchors_json.is_array()&&!anchors_json.empty(),"Expected ordinary anchors");std::vector<BddAnchor> anchors;
    for(const auto& value:anchors_json){require(value.at("K").get<int>()==k&&value.at("input_manifest")==data.manifest&&value.at("input_model_sha256")==model_hash,"Ordinary anchor input provenance mismatch");
        require(value.value("pricing_complete",false)&&value.value("support_complete",false)&&value.value("bound_valid",false),"Ordinary anchor not fully certified");
        require(value.value("method",std::string())=="fractional_recourse_cg"&&
            value.value("scope",std::string())=="fixed_design_recourse"&&!value.value("conditional_demand",false)&&
            !value.value("objective_includes_fixed_costs",true),"Anchor must be a global lifted fractional recourse certificate, not a conditional/local cut");
        require(value.contains("ordinary_cut")&&value.at("ordinary_cut").is_object(),"Anchor lacks ordinary global cut metadata");
        const auto& cut=value.at("ordinary_cut");
        require(cut.value("kind",std::string())=="ordinary"&&cut.value("scope",std::string())=="global_recourse_epigraph"&&
            cut.value("certified",false)&&cut.value("pricing_complete",false)&&cut.at("coefficients")==value.at("coefficients")&&
            cut.at("intercept")==value.at("intercept"),"Anchor ordinary cut is not the certified global recourse row");
        anchors.push_back(bdd_anchor_from_json(value));}
    const bool tight=tight_override.value_or(source.value("tight_airline_bounds",false));
    Json result={{"schema","mlnd-bdd-root-search/1"},{"created_utc",utc_timestamp()},{"input",input.string()},
        {"input_model_sha256",model_hash},{"input_manifest",data.manifest},{"K",k},{"instance",data.name},
        {"anchors_file",anchor_path.string()},{"anchors_sha256",sha256_file(anchor_path)},
        {"continuous_bound_file",floor_path.string()},{"continuous_bound_sha256",sha256_file(floor_path)},
        {"continuous_bound_reference",{{"certified_lower_bound",bound},{"safe_total_cost_floor",floor},{"provenance_verified",true}}},
        {"tight_airline_bounds",tight},{"method",config.method},{"threads",1},{"seed",43},{"total_budget_seconds",total},
        {"budget_scope","read_provenance_full_path_enumeration_full_oracle_and_root_construction_optimization_validation_checkpoint"},
        {"input_read_seconds",wall_time()-started},{"production_algorithm_changed",false},{"status","running"},
        {"incremental_certificate_log",search_log.string()}};
    auto progress=std::make_shared<RunProgress>(diagnostics,started,total,Json{{"method","bdd_root_search"},{"bound_scope","original_integer_problem"}});ActiveProgress active(progress);
    atomic_json_file(output,result);
    try{double phase=wall_time();progress->guard("bdd_search_full_enumeration");const auto paths=enumerate_paths(data,std::nullopt,progress->remaining());
        result["enumeration_seconds"]=wall_time()-phase;result["catalogue_complete"]=paths.complete;
        SolverOptions options;options.seconds=progress->remaining();options.gap=0;options.threads=1;options.seed=43;options.progress=progress;options.tight_airline_bounds=tight;
        phase=wall_time();BddFullDomain oracle(data,paths,options);result["oracle_setup"]=oracle.setup_metadata();result["oracle_setup_seconds"]=wall_time()-phase;
        std::ofstream stream(search_log);
        require(static_cast<bool>(stream),"Cannot create iteration certificate log");
        const std::vector<std::string> histories={"evaluations","cuts","primal_planes","multiplier_steps","root_curve"};
        std::vector<std::size_t> logged(histories.size(),0);double last_snapshot=-1e100;
        config.checkpoint=[&](const Json& search){
            for(std::size_t h=0;h<histories.size();++h){const auto& records=search.at(histories[h]);
                while(logged[h]<records.size()){const auto index=logged[h]++;
                    stream<<Json{{"history",histories[h]},{"index",index},{"record",records[index]}}.dump()<<'\n';}}
            stream.flush();require(static_cast<bool>(stream),"Cannot persist iteration certificate log");
            if(wall_time()-last_snapshot<5&&search.value("status",std::string())=="running")return;
            Json snapshot=Json::object();
            for(const auto* key:{"method","status","baseline","latest_root","best_lower_bound","validated_original_integer_upper_bound","elapsed_seconds","search_wall_seconds","times"})
                if(search.contains(key))snapshot[key]=search[key];
            for(std::size_t h=0;h<histories.size();++h)snapshot[histories[h]+"_count"]=logged[h];
            snapshot["histories_in_incremental_certificate_log"]=true;
            result["search"]=snapshot;result["elapsed_seconds"]=wall_time()-started;atomic_json_file(output,result);last_snapshot=wall_time();};
        result["search"]=search_bdd_root_bounds(data,paths,oracle,anchors,floor,options,config);result["status"]=result["search"]["status"];
        if(result["search"].contains("error"))result["error"]=result["search"]["error"];
    }catch(const EnumerationStopped& error){result["status"]="enumeration_incomplete";result["error"]=error.what();}
     catch(const RunBudgetExceeded& error){result["status"]="construction_time_limit";result["error"]=error.what();}
     catch(const IloException& error){result["status"]="solver_exception";result["error"]=error.getMessage();}
     catch(const std::exception& error){result["status"]="exception";result["error"]=error.what();}
    result["wall_before_output"]=wall_time()-started;result["budget_overrun_seconds"]=std::max(0.0,wall_time()-started-total);
    result["monitoring"]=progress->summary();result["final_output_outside_budget"]=true;atomic_json_file(output,result);
    std::cout<<result.at("status")<<'\n';return result.contains("error")?2:0;
}catch(const IloException& error){std::cerr<<error.getMessage()<<'\n';return 2;}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
