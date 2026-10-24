#include "mlnd/bdd.hpp"
#include "mlnd/benders.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace mlnd;
void require(bool condition, const std::string& text) {
    if (!condition) throw std::runtime_error(text);
}
Json load(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::invalid_argument("Cannot read " + path.string());
    Json value; stream >> value; return value;
}
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}
Json summarize_comparison(const Json& variants) {
    Json comparison = {{"structural_gain_scope", "integer_lagrangian_value_minus_continuous_lagrangian_value"},
        {"numerical_intercept_recovery_is_not_structural_gain", true}, {"modes", Json::array()}};
    const Json* continuous = nullptr;
    for (const auto& record : variants)
        if (record.at("mode") == "continuous") continuous = &record;
    for (const auto& record : variants) {
        if (record.at("mode") == "continuous") continue;
        Json item = {{"mode", record.at("mode")}, {"structural_gain_lower_bound", nullptr},
            {"structural_gain_upper_bound", nullptr}, {"strict_structural_gain_certified", false}};
        if (continuous && !record.at("inner_lower_bound").is_null() &&
            !continuous->at("inner_upper_bound").is_null()) {
            const double lower = record.at("inner_lower_bound").get<double>() -
                continuous->at("inner_upper_bound").get<double>();
            item["structural_gain_lower_bound"] = std::max(0.0, lower);
            item["strict_structural_gain_certified"] = lower > record.at("improvement_tolerance").get<double>();
        }
        if (continuous && !record.at("inner_upper_bound").is_null() &&
            !continuous->at("inner_lower_bound").is_null())
            item["structural_gain_upper_bound"] = record.at("inner_upper_bound").get<double>() -
                continuous->at("inner_lower_bound").get<double>();
        comparison["modes"].push_back(std::move(item));
    }
    return comparison;
}
Json evaluate_anchor(BddFullDomain& oracle, const BddAnchor& anchor, double seconds, double deadline) {
    const double started = wall_time(), anchor_deadline = std::min(deadline, started + seconds);
    Json result = {{"name", anchor.name}, {"design", anchor.design}, {"intercept", anchor.intercept},
        {"coefficients", anchor.coefficients}, {"operating_objective", anchor.operating_objective},
        {"pricing_complete", anchor.pricing_complete}, {"budget_seconds", seconds}, {"variants", Json::array()}};
    const std::array<BddIntegrality,3> modes = {BddIntegrality::Continuous,
        BddIntegrality::FacilityCapacity, BddIntegrality::AllDesign};
    for (std::size_t i = 0; i < modes.size(); ++i) {
        // Split the remaining anchor budget equally among unprocessed modes;
        // quick early solves leave their unused allowance to later modes.
        const double allowance = std::max(0.0, anchor_deadline - wall_time()) / (modes.size() - i);
        result["variants"].push_back(oracle.evaluate(anchor, modes[i], allowance));
    }
    result["comparison"] = summarize_comparison(result.at("variants"));
    result["wall_seconds"] = wall_time() - started;
    result["budget_overrun_seconds"] = std::max(0.0, result.at("wall_seconds").get<double>() - seconds);
    return result;
}
template<class T> Matrix<T> constant(int rows, int cols, T value) {
    Matrix<T> matrix(rows, cols);
    std::fill(matrix.values.begin(), matrix.values.end(), value);
    return matrix;
}
// Same physical fixture as tests/unit/test_solver.py, including frozen Poisson
// occupancy thresholds. This permits the independent Gurobi cross-check.
Input fixture(int k, const std::string& policy) {
    Input d;
    d.n_l=1; d.n_h=2; d.n_od=2; d.n_levels=2; d.max_orders=k;
    d.name="bdd_physical_fixture"; d.model_version="consecutive";
    d.route_mode="direct"; d.flow_policy=policy; d.demand={4,7};
    d.flight_times=constant<double>(3,3,0);
    d.flight_times.values={0,.05,.07,.05,0,.1,.07,.1,0};
    d.rider_mean=constant<double>(2,2,0); d.rider_mean.values={.08,.12,.1,.06};
    d.first_cost=constant<double>(2,2,1); d.terminal_cost=constant<double>(2,2,.5);
    d.receipt_mean=constant<double>(2,2,.08); d.removal_mean=constant<double>(2,2,.03);
    d.ground_cost={5,5}; d.deadline={1,1}; d.base_cost={.5};
    d.slot_counts=constant<int>(2,2,0); d.slot_counts.values={10,20,10,20};
    d.slot_cost=constant<double>(2,2,0); d.slot_cost.values={.1,.2,.1,.2};
    d.slot_epsilon={.05,.05}; d.link_cost=constant<double>(3,3,0);
    d.kappa=constant<double>(2,2,0); d.kappa.values={.3,.9,.2,.7};
    d.delta=.04; d.operating_limit=1; d.fleet_cost=1; d.flight_cost=.2; d.waiting_cost=.3;
    d.allowed_links=constant<int>(3,3,1);
    for (int i=0;i<3;++i) d.allowed_links(i,i)=0;
    d.occupancy_limit=constant<double>(2,2,0);
    d.occupancy_limit.values={6.1690072893953225,14.072024748341311,6.1690072893953225,14.072024748341311};
    return d;
}
std::vector<double> design(const Input& d, int assignments, int level, double scale=1) {
    DesignLayout layout(d);
    std::vector<double> x(layout.count,0);
    for (int i=0;i<d.n_nodes();++i) x[i]=scale;
    for (int i=0;i<d.n_h;++i) x[layout.capacity+i*d.n_levels+level]=scale;
    for (int r=0;r<d.n_od;++r) x[layout.assignment+r]=(assignments&(1<<r))?scale:0;
    for (int i=0;i<d.n_nodes();++i) for (int j=0;j<d.n_nodes();++j)
        x[layout.airline+i*d.n_nodes()+j]=d.allowed_links(i,j)?scale:0;
    return x;
}
Json self_test() {
    Json report={{"schema","mlnd-bdd-regression/1"},{"cases",Json::array()}};
    std::size_t cross_checks=0;
    double worst=0;
    for (const auto& policy : {std::string("aggregate"),std::string("prefix")}) for (int k : {1,2}) {
        auto data=fixture(k,policy);
        const auto paths=enumerate_paths(data);
        SolverOptions options; options.seconds=10; options.threads=1; options.gap=0;
        auto partial=paths; partial.complete=false;
        bool incomplete_rejected=false;
        try { BddFullDomain invalid(data,partial,options); }
        catch (const std::invalid_argument&) { incomplete_rejected=true; }
        require(incomplete_rejected,"BDD accepted a restricted column universe");
        BddFullDomain oracle(data,paths,options);
        Json anchors=Json::array();
        for (double scale : {1.0,.5}) {
            auto pool=prepare_path_support(data);
            auto anchor_json=evaluate_fractional_recourse_cg(data,pool,options,design(data,3,0,scale));
            require(anchor_json.at("pricing_complete").get<bool>(),"Physical BDD fixture anchor lacks a CG certificate");
            anchor_json["name"]=scale==1?"integer_tight_control":"fractional_half_design";
            const auto anchor=bdd_anchor_from_json(anchor_json);
            auto comparison=evaluate_anchor(oracle,anchor,15,wall_time()+15);
            comparison["anchor_cg"]={{"pricing_complete",anchor_json.at("pricing_complete")},
                {"operating_objective",anchor_json.at("operating_objective")}};
            for (const auto& variant : comparison.at("variants")) {
                require(variant.at("valid_global_lower_bound").get<bool>(),"Tiny BDD inner model lacks an eligible global bound");
                require(variant.at("solver_optimal").get<bool>(),"Tiny BDD inner model did not finish");
                if (scale==1) {
                    const double maximum=anchor.operating_objective-dot(anchor.coefficients,anchor.design);
                    require(variant.at("safe_intercept").get<double>()<=maximum+2e-6*(1+std::abs(maximum)),
                        "BDD falsely strengthened beyond an integer tight anchor");
                }
                if (variant.at("mode")!="all_design")
                    require(variant.at("validated_original_integer_upper_bound").is_null(),
                        "A non-integer BDD primal leaked into the original integer UB");
            }
            if (scale==1) for (const auto& gain : comparison.at("comparison").at("modes"))
                require(gain.at("structural_gain_upper_bound").get<double>()<=
                    5e-5*(1+std::abs(anchor.operating_objective)),"Integer tight anchor has nonzero structural BDD gain");
            const auto zero=oracle.evaluate(anchor,BddIntegrality::AllDesign,0);
            require(!zero.at("accepted").get<bool>() && !zero.at("valid_global_lower_bound").get<bool>() &&
                zero.at("inner_lower_bound").is_null(),"A zero-budget BDD call invented a lower bound");
            auto invalid=anchor; invalid.pricing_complete=false;
            bool uncertified_rejected=false;
            try { oracle.evaluate(invalid,BddIntegrality::Continuous,1); }
            catch (const std::invalid_argument&) { uncertified_rejected=true; }
            require(uncertified_rejected,"Uncertified ordinary anchor was accepted");
            anchors.push_back(std::move(comparison));
        }
        // An independent objective-identity control: beta=-f makes q-beta*z
        // exactly the original full objective. This catches accidental f-beta.
        const auto original=build_formulation(data,paths);
        NativeSolver full(original.matrix,options,"bdd_objective_identity_reference");
        const auto solved=full.optimize(10);
        require(solved.optimal && solved.has_solution &&
            point_residuals(original.matrix,solved.values).at("passed").get<bool>(),"BDD identity reference failed");
        BddAnchor identity;
        identity.name="negative_fixed_cost_objective_identity";
        identity.design=design(data,3,0); identity.pricing_complete=true;
        identity.operating_objective=55; identity.intercept=0;
        identity.coefficients=original.opening_costs;
        for (auto& coefficient:identity.coefficients) coefficient=-coefficient;
        const auto identity_result=oracle.evaluate(identity,BddIntegrality::AllDesign,10);
        require(identity_result.at("valid_global_lower_bound").get<bool>() &&
            std::abs(identity_result.at("inner_upper_bound").get<double>()-solved.objective)<2e-5,
            "BDD objective retained the original fixed costs");
        std::vector<std::vector<double>> designs;
        for (int assignments=0;assignments<4;++assignments) for (int level=0;level<2;++level)
            designs.push_back(design(data,assignments,level));
        auto closed=design(data,3,0); closed[original.layout.airline+1]=0; designs.push_back(closed);
        designs.push_back(std::vector<double>(original.layout.count,0));
        Json recourse=Json::array();
        for (const auto& x:designs) {
            const auto fixed=build_formulation(data,paths,&x);
            NativeSolver reference(fixed.matrix,options,"bdd_cross_design_reference");
            const auto value=reference.optimize(10);
            require(value.has_solution && value.optimal &&
                point_residuals(fixed.matrix,value.values).at("passed").get<bool>(),"Cross-design reference failed");
            recourse.push_back({{"design",x},{"operating_objective",value.objective}});
            for (const auto& anchor:anchors) for (const auto& variant:anchor.at("variants")) {
                const auto coefficients=variant.at("coefficients").get<std::vector<double>>();
                const double cut=variant.at("safe_intercept").get<double>()+dot(coefficients,x);
                worst=std::max(worst,cut-value.objective); ++cross_checks;
                require(cut<=value.objective+3e-6*(1+std::abs(value.objective)),
                    "Full-domain BDD cut excludes an independently solved strategy");
            }
        }
        std::vector<BddAnchor> ordinary;
        Json evaluations=Json::array();
        for (const auto& anchor:anchors) {
            ordinary.push_back(bdd_anchor_from_json(anchor));
            for (const auto& value:anchor.at("variants")) evaluations.push_back(value);
        }
        const auto root=compare_bdd_root_bounds(data,paths,ordinary,evaluations,options,10);
        require(root.at("baseline").at("certified").get<bool>() &&
            root.at("all_accepted_cuts").at("certified").get<bool>(),"BDD root comparison did not certify its LPs");
        require(root.at("all_accepted_cuts").at("lower_bound").get<double>()<=solved.objective+2e-5,
            "BDD root bound exceeds the original integer optimum");
        require(root.at("all_accepted_cuts").at("root_objective_change").get<double>()>=-2e-5,
            "Adding accepted BDD cuts reduced the same compact master bound");
        // Derive a floor from the same complete physical matrix, keeping f*z.
        // Its primal is a continuous reference only, never an integer UB.
        auto continuous_matrix=original.matrix;
        for (auto& variable:continuous_matrix.variables) variable.binary=false;
        NativeSolver continuous_reference(continuous_matrix,options,"bdd_floor_reference");
        const auto continuous=continuous_reference.optimize(10);
        require(continuous.optimal && continuous.has_solution &&
            point_residuals(continuous_matrix,continuous.values).at("passed").get<bool>(),
            "BDD total-cost floor reference did not certify");
        const double total_floor=continuous.bound-1e-7*(1+std::abs(continuous.bound));
        const auto floored_root=compare_bdd_root_bounds(data,paths,ordinary,evaluations,options,10,total_floor);
        require(floored_root.at("baseline").at("certified").get<bool>() &&
            floored_root.at("all_accepted_cuts").at("certified").get<bool>(),
            "Floored BDD root comparison did not certify its LPs");
        require(floored_root.at("baseline").at("objective").get<double>()>=total_floor-2e-5 &&
            floored_root.at("all_accepted_cuts").at("lower_bound").get<double>()<=solved.objective+2e-5,
            "BDD total-cost floor has the wrong objective scope or violates the integer optimum");
        require(floored_root.at("all_accepted_cuts").at("root_objective_change").get<double>()>=-2e-5,
            "Accepted BDD cuts reduced the floored master bound");
        report["cases"].push_back({{"K",k},{"policy",policy},{"setup",oracle.setup_metadata()},
            {"anchors",anchors},{"cross_design_recourse",recourse},{"incomplete_catalogue_rejected",true},
            {"zero_budget_bound_rejected",true},{"uncertified_anchor_rejected",true},
            {"integer_tight_anchor_zero_structural_gain",true},
            {"objective_identity_control",identity_result},{"full_reference_objective",solved.objective},
            {"root_master_comparison",root},{"root_master_comparison_with_continuous_floor",floored_root},
            {"continuous_floor_reference",{{"objective",continuous.objective},{"bound",continuous.bound},
                {"safe_total_cost_floor",total_floor},{"is_integer_upper_bound",false}}}});
    }
    report["cross_design_cut_checks"]=cross_checks;
    report["max_cross_design_violation"]=worst;
    report["passed"]=true;
    return report;
}
struct ActiveProgress {
    explicit ActiveProgress(const std::shared_ptr<RunProgress>& value) {set_active_run_progress(value);}
    ~ActiveProgress() {set_active_run_progress(nullptr);}
};
}

int main(int argc,char** argv) {
    using namespace mlnd;
    try {
        if (argc==3 && std::string(argv[1])=="--self-test") {
            const auto output=std::filesystem::absolute(argv[2]);
            require(!std::filesystem::exists(output),"Refusing to overwrite an existing regression result");
            const auto result=self_test(); atomic_json_file(output,result);
            std::cout<<"BDD cases="<<result.at("cases").size()<<" cross-design checks="
                <<result.at("cross_design_cut_checks")<<" passed\n";
            return 0;
        }
        if (argc<5 || (argc-5)%2)
            throw std::invalid_argument("Usage: mlnd_bdd_probe INPUT K ANCHORS_JSON OUTPUT_JSON [--seconds-per-anchor 60] [--master-comparison-seconds 15] [--total-seconds N] [--continuous-bound-file FILE] | --self-test OUTPUT_JSON");
        const double started=wall_time();
        const auto input=std::filesystem::absolute(argv[1]);
        const int k=std::stoi(argv[2]);
        const auto anchor_path=std::filesystem::absolute(argv[3]);
        const auto output=std::filesystem::absolute(argv[4]);
        double per_anchor=60,master_seconds=15,total=-1;
        std::optional<std::filesystem::path> continuous_bound_file;
        bool total_supplied=false;
        for (int arg=5;arg<argc;arg+=2) {
            const std::string flag=argv[arg];
            if (flag=="--seconds-per-anchor") per_anchor=std::stod(argv[arg+1]);
            else if (flag=="--master-comparison-seconds") master_seconds=std::stod(argv[arg+1]);
            else if (flag=="--total-seconds") {total=std::stod(argv[arg+1]); total_supplied=true;}
            else if (flag=="--continuous-bound-file") continuous_bound_file=std::filesystem::absolute(argv[arg+1]);
            else throw std::invalid_argument("Unknown option: "+flag);
        }
        const auto document=load(anchor_path);
        const Json anchors=document.is_array()?document:document.at("anchors");
        require(anchors.is_array() && !anchors.empty(),"Expected a nonempty anchors array");
        require((k==1||k==2) && std::isfinite(per_anchor) && per_anchor>0 && per_anchor<=60,
            "Require K in {1,2}, and 0 < seconds-per-anchor <= 60");
        require(std::isfinite(master_seconds) && master_seconds>=0,"Invalid master comparison budget");
        if (!total_supplied) total=60+per_anchor*anchors.size()+master_seconds;
        require(std::isfinite(total) && total>0,"Invalid total BDD budget");
        const auto diagnostics=std::filesystem::path(output.string()+".diagnostics");
        require(!std::filesystem::exists(output) && !std::filesystem::exists(diagnostics),
            "Refusing to overwrite an existing BDD result or diagnostics directory");
        const auto data=read_input(input,k,true);
        std::optional<double> continuous_floor;
        Json continuous_source=nullptr;
        if (continuous_bound_file) {
            const auto source=load(*continuous_bound_file);
            require(source.value("method",std::string())=="full_continuous_cg" &&
                source.value("scope",std::string())=="full_path_continuous_relaxation" &&
                source.value("objective_includes_fixed_costs",false) && source.value("support_complete",false) &&
                source.value("bound_valid",false),"Continuous bound file lacks a certified full-domain total-cost bound");
            require(source.at("K").get<int>()==k && source.at("input_manifest")==data.manifest &&
                source.at("input_model_sha256")==sha256_file(input/"model.json"),
                "Continuous bound provenance differs from BDD input manifest/model hash/K");
            const double bound=source.at("certified_lower_bound").get<double>();
            require(std::isfinite(bound),"Continuous bound is not finite");
            // Match the production integration's additional floating-point shift.
            continuous_floor=bound-1e-8*(1+std::abs(bound));
            continuous_source={{"file",continuous_bound_file->string()},
                {"sha256",sha256_file(*continuous_bound_file)}, {"certified_lower_bound",bound},
                {"safe_total_cost_floor",*continuous_floor},{"provenance_verified",true},
                {"checks",{"input_manifest","input_model_sha256","K","full_domain_total_cost_bound"}}};
        }
        Json result={{"schema","mlnd-bdd-cut-quality/1"},{"created_utc",utc_timestamp()},
            {"input",input.string()},{"input_model_sha256",sha256_file(input/"model.json")},
            {"input_manifest",data.manifest},{"anchors_file",anchor_path.string()},
            {"anchors_sha256",sha256_file(anchor_path)},{"K",k},{"instance",data.name},
            {"flow_policy",data.flow_policy},{"route_mode",data.route_mode},
            {"threads",1},{"seconds_per_anchor",per_anchor},{"total_budget_seconds",total},
            {"master_comparison_budget_seconds",master_seconds},
            {"master_comparison_budget_reserved_before_anchor_solves",true},
            {"continuous_bound_reference",continuous_source},
            {"requested_anchor_count",anchors.size()},
            {"budget_scope","input_read_shared_full_enumeration_shared_matrix_and_solver_construction_all_anchor_modes"},
            {"input_read_seconds",wall_time()-started},{"production_algorithm_changed",false},
            {"complete_domain_lower_bounds_only",true},{"shared_preprocessing",Json::object()},
            {"anchors",Json::array()},{"status","running"}};
        auto progress=std::make_shared<RunProgress>(diagnostics,started,total,
            Json{{"method","bdd_cut_quality"},{"bound_scope","lagrangian_inner_model"}});
        ActiveProgress active(progress);
        std::string stage="full_path_enumeration";
        double phase=wall_time();
        try {
            progress->guard(stage);
            const auto paths=enumerate_paths(data,std::nullopt,progress->remaining());
            result["shared_preprocessing"]["enumeration_seconds"]=wall_time()-phase;
            result["shared_preprocessing"]["catalogue_complete"]=paths.complete;
            stage="shared_model_construction"; phase=wall_time();
            SolverOptions options; options.seconds=progress->remaining(); options.threads=1; options.gap=0; options.progress=progress;
            BddFullDomain oracle(data,paths,options);
            result["shared_preprocessing"]["model_setup"]=oracle.setup_metadata();
            result["shared_preprocessing"]["model_setup_wall_seconds"]=wall_time()-phase;
            atomic_json_file(output,result);
            std::vector<BddAnchor> ordinary;
            Json evaluations=Json::array();
            for (std::size_t i=0;i<anchors.size();++i) {
                stage="anchor_"+std::to_string(i); phase=wall_time();
                Json comparison;
                try {
                    auto anchor=bdd_anchor_from_json(anchors[i]);
                    if (anchor.name.empty()) anchor.name="anchor_"+std::to_string(i);
                    const double available=std::max(0.0,progress->remaining()-master_seconds);
                    comparison=evaluate_anchor(oracle,anchor,std::min(per_anchor,available),
                        started+total-master_seconds);
                    ordinary.push_back(anchor);
                    for (const auto& value:comparison.at("variants")) evaluations.push_back(value);
                } catch (const std::exception& error) {
                    comparison={{"index",i},{"status","anchor_rejected"},{"error",error.what()}};
                }
                result["anchors"].push_back(std::move(comparison));
                atomic_json_file(output,result);
            }
            stage="root_master_comparison"; phase=wall_time();
            const double comparison_deadline=wall_time()+std::min(master_seconds,progress->remaining());
            result["root_master_comparison"]=compare_bdd_root_bounds(data,paths,ordinary,evaluations,
                options,std::max(0.0,comparison_deadline-wall_time())/(continuous_floor?2:1));
            if (continuous_floor)
                result["root_master_comparison_with_continuous_floor"]=compare_bdd_root_bounds(data,paths,
                    ordinary,evaluations,options,std::max(0.0,comparison_deadline-wall_time()),continuous_floor);
            result["validated_anchor_count"]=ordinary.size();
            result["rejected_anchor_count"]=anchors.size()-ordinary.size();
            result["accepted_bdd_cut_count"]=std::count_if(evaluations.begin(),evaluations.end(),
                [](const Json& value) {return value.value("accepted",false);});
            result["status"]=ordinary.size()==anchors.size()?"completed_diagnostic":"completed_with_rejected_anchors";
        } catch (const EnumerationStopped& error) {
            result["status"]="enumeration_incomplete"; result["error"]=error.what();
            result["partial_paths"]={{"columns",error.columns},{"initial_segments",error.initial_segments}};
        } catch (const RunBudgetExceeded& error) {
            result["status"]="construction_time_limit"; result["error"]=error.what();
            result["budget_interruption_stage"]=error.stage;
        } catch (const IloException& error) {
            result["status"]="solver_exception"; result["error"]=error.getMessage();
        } catch (const std::exception& error) {
            result["status"]="exception"; result["error"]=error.what();
        }
        if (result.contains("error")) {
            result["interrupted_stage"]=stage; result["interrupted_stage_seconds"]=wall_time()-phase;
        }
        result["wall_before_output"]=wall_time()-started;
        result["budget_overrun_seconds"]=std::max(0.0,wall_time()-started-total);
        result["monitoring"]=progress->summary();
        result["output_finalization_outside_method_budget"]=true;
        atomic_json_file(output,result);
        std::cout<<result.at("status")<<" anchors="<<result.at("anchors").size()<<'\n';
        return result.contains("error")?2:0;
    } catch (const IloException& error) {std::cerr<<error.getMessage()<<'\n'; return 2;}
      catch (const std::exception& error) {std::cerr<<error.what()<<'\n'; return 2;}
}
