#include "mlnd/benders.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace mlnd;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class T> Matrix<T> matrix(int rows, int columns, T value) {
    Matrix<T> result(rows, columns);
    std::fill(result.values.begin(), result.values.end(), value);
    return result;
}
Input fixture(int k) {
    Input d;
    d.n_l=1; d.n_h=2; d.n_od=2; d.n_levels=2; d.max_orders=k;
    d.name="continuous_cg_test"; d.model_version="consecutive";
    d.route_mode="direct"; d.flow_policy="aggregate";
    d.demand={4,7}; d.ground_cost={5,5}; d.deadline={1,1}; d.base_cost={.5};
    d.flight_times=matrix<double>(3,3,0);
    d.flight_times.values={0,.05,.07,.05,0,.1,.07,.1,0};
    d.rider_mean=matrix<double>(2,2,0); d.rider_mean.values={.08,.12,.1,.06};
    d.first_cost=matrix<double>(2,2,1); d.terminal_cost=matrix<double>(2,2,.5);
    d.receipt_mean=matrix<double>(2,2,.08); d.removal_mean=matrix<double>(2,2,.03);
    d.slot_counts=matrix<int>(2,2,0); d.slot_counts.values={10,20,10,20};
    d.slot_cost=matrix<double>(2,2,0); d.slot_cost.values={.1,.2,.1,.2};
    d.slot_epsilon={.05,.05}; d.link_cost=matrix<double>(3,3,0);
    d.kappa=matrix<double>(2,2,0); d.kappa.values={.3,.9,.2,.7};
    d.delta=.04; d.operating_limit=1; d.fleet_cost=1; d.flight_cost=.2; d.waiting_cost=.3;
    d.allowed_links=matrix<int>(3,3,1);
    for (int i=0;i<3;++i) d.allowed_links(i,i)=0;
    d.occupancy_limit=matrix<double>(2,2,0);
    d.occupancy_limit.values={6.1690072893953225,14.072024748341311,
                            6.1690072893953225,14.072024748341311};
    return d;
}
double reference(const Input& d, const PathCollection& paths, const SolverOptions& options,
                 const std::vector<double>* design=nullptr) {
    auto model=build_formulation(d,paths,design,nullptr,nullptr,true);
    for (auto& variable:model.matrix.variables) variable.binary=false;
    NativeSolver solver(model.matrix,options,"continuous_probe_reference");
    const auto solved=solver.optimize(options.seconds);
    require(solved.optimal && solved.has_solution,"Tiny full continuous reference was not optimal");
    require(point_residuals(model.matrix,solved.values).at("passed").get<bool>(),
            "Tiny full continuous reference failed original residuals");
    return objective_value(model.matrix,solved.values);
}
std::vector<double> interior_anchor_core(const Input& d) {
    // Match the established full-support strategic core in benders.cpp.
    // This is a diagnostic anchor choice, not a modification of that model.
    DesignLayout layout(d);
    std::vector<double> core(layout.count,0);
    std::fill(core.begin()+layout.facility,core.begin()+layout.capacity,.5);
    std::fill(core.begin()+layout.capacity,core.begin()+layout.assignment,.5/d.n_levels);
    for (int r=0;r<d.n_od;++r) for (int l=0;l<d.n_l;++l) for (int b=0;b<d.assignment_destinations();++b)
        core[layout.assignment_index(r,l,b)]=d.origin_assignment()?.25/(d.n_l+1):(l==b?.5:.25)/(d.n_l*d.n_l+1);
    for (int u=0;u<d.n_nodes();++u) for (int v=0;v<d.n_nodes();++v)
        core[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v)?(u==v?.25:.125):0;
    require(point_residuals(design_model(d,false).matrix,core).at("passed").get<bool>(),
            "Diagnostic core violates the strategic model");
    return core;
}
Json tiny_tests() {
    Json records=Json::array();
    for (int k:{1,2}) {
        const auto d=fixture(k);
        const auto catalogue=enumerate_paths(d);
        SolverOptions options;
        options.reuse_dual_lp=false;
        options.seconds=20; options.threads=1; options.cg_max_columns=1;
        options.pareto=false; options.branch_covering_seeds=false;
        const double full=reference(d,catalogue,options), tolerance=5e-6*(1+std::abs(full));
        // A complete catalogue starts with an empty RMP in CG validation mode.
        // This forces actual incremental pricing and nonterminal bound repair.
        auto pool=catalogue;
        const auto result=solve_continuous_cg(d,pool,options);
        require(result.at("bound_valid") && result.at("pricing_complete") && result.at("primal_valid"),
                "Tiny free-design CG failed to obtain both primal and full-domain dual certificates");
        const double lower=result.at("certified_lower_bound"), objective=result.at("continuous_objective");
        require(lower<=full+tolerance && std::abs(objective-full)<=tolerance && objective-lower<=tolerance,
                "Free-design CG disagrees with the identical full continuous formulation");
        require(result.at("cg_iterations").get<int>()>1,"Empty RMP did not exercise incremental free-design CG");
        require(result.at("objective_includes_fixed_costs") && !result.at("is_integer_solution").get<bool>() &&
                !result.contains("_solution") && !result.contains("upper_bound"),
                "Continuous CG leaked an integer incumbent or lost strategic costs");
        int nonterminal_bounds=0;
        for (const auto& iteration:result.at("cg_trace")) if (iteration.contains("certified_full_lower_bound")) {
            require(iteration.at("certified_full_lower_bound").get<double>()<=full+tolerance,
                    "A nonterminal pricing correction produced an invalid full-domain lower bound");
            if (iteration.at("termination_reason")=="columns_added") ++nonterminal_bounds;
        }
        require(nonterminal_bounds>0,"No full-domain lower bound was retained before CG convergence");
        auto warm_pool=catalogue;
        auto warm_options=options; warm_options.reuse_dual_lp=true;
        const auto warm=solve_continuous_cg(d,warm_pool,warm_options);
        require(warm.at("bound_valid") && warm.at("pricing_complete") && warm.at("primal_valid"),
                "Persistent-dual CG failed to certify the full path domain");
        require(warm.at("certified_lower_bound").get<double>()<=full+tolerance &&
                std::abs(warm.at("continuous_objective").get<double>()-full)<=tolerance &&
                warm.at("continuous_objective").get<double>()-warm.at("certified_lower_bound").get<double>()<=tolerance,
                "Persistent-dual CG changed the full continuous optimum");
        int warm_bases=0;
        for (const auto& iteration:warm.at("cg_trace")) {
            if (iteration.contains("certified_full_lower_bound"))
                require(iteration.at("certified_full_lower_bound").get<double>()<=full+tolerance,
                    "Persistent dual leaked an invalid nonterminal lower bound");
            for (const auto& attempt:iteration.at("dual_attempts"))
                if (attempt.contains("dual_lp_workspace") && attempt.at("dual_lp_workspace").at("basis_submitted").get<bool>())
                    ++warm_bases;
        }
        require(warm_bases>0,"Incremental CG did not exercise persistent dual basis reuse");
        const auto design=result.at("design").get<std::vector<double>>();
        require(point_residuals(design_model(d,false).matrix,design).at("passed").get<bool>(),
                "Free-design CG returned an invalid strategic point");
        const auto core=interior_anchor_core(d);
        auto mixed=design;
        for (std::size_t j=0;j<mixed.size();++j) mixed[j]=.99*design[j]+.01*core[j];
        require(point_residuals(design_model(d,false).matrix,mixed).at("passed").get<bool>(),
                "Interior diagnostic anchor is not strategically feasible");
        auto fractional_pool=prepare_path_support(d);
        const auto fractional=evaluate_fractional_recourse_cg(d,fractional_pool,options,design);
        require(fractional.at("pricing_complete") && fractional.at("certified") && fractional.at("primal_valid"),
                "Fractional recourse failed to certify the continuous design");
        require(!fractional.at("objective_includes_fixed_costs").get<bool>() && !fractional.contains("_solution"),
                "Fractional recourse was packaged as an integer/full-cost solution");
        const double recourse=reference(d,catalogue,options,&design);
        require(std::abs(fractional.at("operating_objective").get<double>()-recourse)<=tolerance,
                "Fractional CG changed the fixed-design operational objective");
        const auto coefficients=fractional.at("coefficients").get<std::vector<double>>();
        require(coefficients.size()==design.size(),"Fractional cut lost strategic coefficients");
        std::vector<double> closed(design.size(),0), opened(design.size(),0);
        DesignLayout layout(d);
        std::fill(opened.begin(),opened.begin()+d.n_nodes(),1);
        for (int h=0;h<d.n_h;++h) opened[layout.capacity+h*d.n_levels+d.n_levels-1]=1;
        for (int r=0;r<d.n_od;++r) opened[layout.assignment+r]=1;
        for (int u=0;u<d.n_nodes();++u) for (int v=0;v<d.n_nodes();++v)
            opened[layout.airline+u*d.n_nodes()+v]=d.allowed_links(u,v);
        for (const auto& alternative:{closed,opened,design}) {
            const double exact=reference(d,catalogue,options,&alternative);
            const double cut=fractional.at("intercept").get<double>()+
                std::inner_product(coefficients.begin(),coefficients.end(),alternative.begin(),0.0);
            require(cut<=exact+tolerance,"Fractional anchor cut is not valid across strategic designs");
        }
        auto integer_pool=prepare_path_support(d);
        const auto integer_anchor=evaluate_fractional_recourse_cg(d,integer_pool,options,closed);
        require(integer_anchor.at("pricing_complete") && integer_anchor.at("design")==Json(closed),
                "Fractional recourse API rejected a legal integer design");
        auto implicit=prepare_path_support(d);
        const auto implicit_result=solve_continuous_cg(d,implicit,options);
        require(implicit_result.at("pricing_complete") && !implicit.complete &&
                std::abs(implicit_result.at("continuous_objective").get<double>()-full)<=tolerance,
                "Implicit continuous CG required a full physical catalogue or changed its optimum");
        options.seconds=0;
        const auto stopped=solve_continuous_cg(d,implicit,options);
        require(!stopped.at("bound_valid").get<bool>() && !stopped.at("pricing_complete").get<bool>() &&
                stopped.at("certified_lower_bound").is_null() && !stopped.at("primal_valid").get<bool>(),
                "Zero-budget CG fabricated a primal or lower-bound certificate");
        records.push_back({{"K",k},{"full_continuous",full},{"cg_lower_bound",lower},
            {"cg_continuous_objective",objective},{"cg_iterations",result.at("cg_iterations")},
            {"nonterminal_bounds",nonterminal_bounds},{"fractional_anchor",fractional},
            {"free_cg",result},{"implicit_cg",implicit_result},{"persistent_dual_cg",warm},
            {"persistent_dual_basis_submissions",warm_bases},{"passed",true}});
    }
    return {{"tests",records},{"passed",true}};
}
}

int main(int argc,char** argv) {
    using namespace mlnd;
    try {
        if (argc==1) { std::cout<<tiny_tests().dump(2)<<'\n'; return 0; }
        if (argc<5 || (argc-5)%2!=0)
            throw std::invalid_argument("Usage: mlnd_continuous_probe [INPUT K SECONDS OUTPUT_JSON [--anchor DESIGN_JSON|zero] [--cg-max-columns N] [--anchor-interior-weight EPS] [--reuse-dual-lp true|false] [--tight-airline-bounds true|false] [--assignment-incident-clique true|false] [--shared-rho-joint true|false] [--queue-perspective true|false]]");
        bool anchor_requested=false;
        std::string anchor_source;
        int batch=64;
        bool reuse_dual_lp=false, tight_airline_bounds=false;
        bool assignment_incident_clique=false, shared_rho_joint=false, queue_perspective=false;
        double interior_weight=0;
        for (int i=5;i<argc;i+=2) {
            const std::string flag=argv[i];
            if (flag=="--anchor" && !anchor_requested) { anchor_requested=true; anchor_source=argv[i+1]; }
            else if (flag=="--cg-max-columns") batch=std::stoi(argv[i+1]);
            else if (flag=="--anchor-interior-weight") interior_weight=std::stod(argv[i+1]);
            else if (flag=="--reuse-dual-lp" && (std::string(argv[i+1])=="true" || std::string(argv[i+1])=="false"))
                reuse_dual_lp=std::string(argv[i+1])=="true";
            else if (flag=="--tight-airline-bounds" && (std::string(argv[i+1])=="true" || std::string(argv[i+1])=="false"))
                tight_airline_bounds=std::string(argv[i+1])=="true";
            else if (flag=="--assignment-incident-clique" && (std::string(argv[i+1])=="true" || std::string(argv[i+1])=="false"))
                assignment_incident_clique=std::string(argv[i+1])=="true";
            else if (flag=="--shared-rho-joint" && (std::string(argv[i+1])=="true" || std::string(argv[i+1])=="false"))
                shared_rho_joint=std::string(argv[i+1])=="true";
            else if (flag=="--queue-perspective" && (std::string(argv[i+1])=="true" || std::string(argv[i+1])=="false"))
                queue_perspective=std::string(argv[i+1])=="true";
            else throw std::invalid_argument("Unknown or repeated probe option");
        }
        if (batch<1) throw std::invalid_argument("CG column batch must be positive");
        if (!std::isfinite(interior_weight) || interior_weight<0 || interior_weight>1 ||
            (interior_weight>0 && !anchor_requested))
            throw std::invalid_argument("Interior weight requires an anchor and must lie in [0,1]");
        const std::filesystem::path input=std::filesystem::absolute(argv[1]);
        const int k=std::stoi(argv[2]); const double seconds=std::stod(argv[3]);
        const std::filesystem::path output=std::filesystem::absolute(argv[4]);
        auto diagnostics=output; diagnostics+=".diagnostics";
        if ((k!=1 && k!=2) || !std::isfinite(seconds) || seconds<=0 ||
            std::filesystem::exists(output) || std::filesystem::exists(diagnostics))
            throw std::invalid_argument("Require K=1/2, positive budget and unused output paths");
        const auto d=read_input(input,k,true);
        std::vector<double> anchor;
        Json anchor_construction=nullptr;
        if (anchor_requested) {
            if (anchor_source=="zero") anchor.assign(DesignLayout(d).count,0);
            else {
                std::ifstream stream(anchor_source);
                if (!stream) throw std::invalid_argument("Could not read anchor JSON");
                Json source; stream>>source;
                if (d.origin_assignment() && (!source.is_object() || source.value("assignment_scope",std::string()) != "origin"))
                    throw std::invalid_argument("K=1 anchor must declare assignment_scope=origin");
                anchor=(source.is_array() ? source : source.at("design")).get<std::vector<double>>();
            }
            if (interior_weight>0) {
                const auto core=interior_anchor_core(d);
                if (anchor.size()!=core.size()) throw std::invalid_argument("Anchor has the wrong strategic dimension");
                anchor_construction={{"method","convex_combination_with_established_strategic_core"},
                    {"interior_weight",interior_weight},{"original_design",anchor},{"core_design",core},
                    {"is_original_relaxation_optimum",false}};
                for (std::size_t j=0;j<anchor.size();++j)
                    anchor[j]=(1-interior_weight)*anchor[j]+interior_weight*core[j];
            }
        }
        const double started=wall_time();
        auto progress=std::make_shared<RunProgress>(diagnostics,started,seconds,
            Json{{"method",anchor_requested ? "fractional_recourse_cg" : "full_continuous_cg"},
                 {"integer_optimization_requested",false}});
        set_active_run_progress(progress);
        auto paths=prepare_path_support(d,progress->remaining());
        const double preprocessing=wall_time()-started;
        SolverOptions options; options.seconds=progress->remaining(); options.threads=1; options.progress=progress;
        options.cg_max_columns=batch;
        options.reuse_dual_lp=reuse_dual_lp;
        options.tight_airline_bounds=tight_airline_bounds;
        options.assignment_incident_clique=assignment_incident_clique;
        options.shared_rho_joint=shared_rho_joint;
        options.queue_perspective=queue_perspective;
        Json result=anchor_requested ? evaluate_fractional_recourse_cg(d,paths,options,anchor)
                           : solve_continuous_cg(d,paths,options);
        set_active_run_progress(nullptr);
        result["input"]=input.string(); result["K"]=k; result["input_manifest"]=d.manifest;
        result["reuse_dual_lp"]=reuse_dual_lp;
        result["tight_airline_bounds"]=tight_airline_bounds;
        result["input_model_sha256"]=sha256_file(input/"model.json");
        if (anchor_requested) result["anchor_source"]=anchor_source;
        if (!anchor_construction.is_null()) result["anchor_construction"]=anchor_construction;
        result["preprocessing_seconds"]=preprocessing; result["end_to_end_seconds"]=wall_time()-started;
        result["end_to_end_budget_seconds"]=seconds;
        atomic_json_file(output,result);
        std::cout<<Json{{"output",output.string()},{"termination_reason",result.at("termination_reason")},
            {"bound_valid",result.at("bound_valid")},{"certified_lower_bound",result.at("certified_lower_bound")},
            {"continuous_objective",result.at("continuous_objective")},{"pricing_complete",result.at("pricing_complete")}}.dump(2)<<'\n';
        return 0;
    } catch (const std::exception& error) {
        set_active_run_progress(nullptr); std::cerr<<error.what()<<'\n'; return 1;
    }
}
