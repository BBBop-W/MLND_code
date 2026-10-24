#include "mlnd/formulation.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace mlnd;

void recovery_fixture() {
    const auto fixed=PrimalPolishRestriction::FixedConeCoordinates;
    const auto box=PrimalPolishRestriction::MonotoneConeBox;
    LinearModel m;
    const int u=m.variable("u",0,2),v=m.variable("v",0,1),t=m.variable("one",1,1);
    const int y=m.variable("y",0,5,1);
    m.cones.push_back({u,v,t,1});
    m.row("u_fixed",'E',1,{{u,1}});m.row("v_fixed",'E',1,{{v,1}});
    m.row("required_y",'G',3,{{y,1}});
    SolverOptions options;options.threads=1;
    std::vector<double> point{1,1,1,0};
    const auto repaired=polish_conic_primal(m,options,5,point,fixed);
    if(!repaired.at("applied").get<bool>() || !point_residuals(m,point).at("passed").get<bool>() ||
       std::abs(point[y]-3)>1e-8 || !repaired.at("lower_bound").is_null() ||
       repaired.at("original_model_optimality_certified").get<bool>())
        throw std::runtime_error("Primal recovery fixture failed or leaked an LP lower bound");
    // The unrestricted conic model is feasible. Fixing this particular cone
    // anchor makes only its repair LP infeasible; it must leave the point and
    // original feasibility conclusion unchanged.
    point={1,.5,1,0};const auto before=point;
    const auto rejected=polish_conic_primal(m,options,5,point,fixed);
    if(rejected.at("applied").get<bool>() || point!=before || !rejected.at("lower_bound").is_null())
        throw std::runtime_error("Failed restricted recovery was not atomic");
    const auto no_budget=polish_conic_primal(m,options,0,point);
    if(no_budget.at("applied").get<bool>() || point!=before || no_budget.at("reason")!="no_budget")
        throw std::runtime_error("No-budget primal recovery was not atomic");

    // The box contains its fixed corner but permits v to rise, corresponding
    // to lower utilization in the inventory cone. Fixed-coordinate recovery
    // is infeasible here even though the enlarged inner LP is feasible.
    LinearModel flexible;
    flexible.variable("u",0,3,1);flexible.variable("v",0,1);flexible.variable("one",1,1);
    flexible.cones.push_back({0,1,2,1});flexible.row("v_required",'E',.75,{{1,1}});
    point={2,.5,1};const auto flexible_before=point;
    const auto fixed_failed=polish_conic_primal(flexible,options,5,point,fixed);
    if(fixed_failed.at("applied").get<bool>() || point!=flexible_before)
        throw std::runtime_error("Fixed-corner contrast did not fail atomically");
    const auto box_passed=polish_conic_primal(flexible,options,5,point,box);
    if(!box_passed.at("applied").get<bool>() || !point_residuals(flexible,point).at("passed").get<bool>() ||
       std::abs(point[0]-2)>1e-8 || std::abs(point[1]-.75)>1e-8 ||
       !box_passed.at("lower_bound").is_null())
        throw std::runtime_error("Monotone cone box did not enlarge the fixed-corner restriction safely");
    point=flexible_before;
    const auto adaptive=polish_conic_primal(flexible,options,5,point);
    if(!adaptive.at("applied").get<bool>() || adaptive.at("adaptive_attempts").size()!=2 ||
       adaptive.at("adaptive_attempts").at(0).at("status")!="Infeasible" ||
       adaptive.at("method")!="monotone_cone_box_lp_restriction" ||
       !point_residuals(flexible,point).at("passed").get<bool>())
        throw std::runtime_error("Adaptive recovery did not preserve the fixed-to-box fallback contract");
    if(adaptive.at("adaptive_attempts").at(1).at("allocated_seconds").get<double>()>
       5-adaptive.at("adaptive_attempts").at(0).at("seconds").get<double>()+1e-9)
        throw std::runtime_error("Adaptive fallback reset its budget instead of using the remainder");
    point={1,1,1,0};
    const auto preserved=polish_conic_primal(m,options,5,point);
    if(!preserved.at("applied").get<bool>() || preserved.at("adaptive_attempts").size()!=1 ||
       preserved.at("method")!="fixed_cone_coordinates_lp_restriction")
        throw std::runtime_error("Adaptive recovery changed a successful fixed-coordinate path");

    // t is a squared coordinate in the first cone and a resource in the
    // second. It must remain fixed at 1 while the other resource is raised.
    LinearModel shared;
    for(const auto* name:{"u","v","t","w","s"})shared.variable(name,0,5,1);
    shared.cones={{0,1,2,1},{2,3,4,1}};
    point={1,1,1,.25,1};
    const auto shared_result=polish_conic_primal(shared,options,5,point,box);
    if(!shared_result.at("applied").get<bool>() || !point_residuals(shared,point).at("passed").get<bool>() ||
       std::abs(point[2]-1)>1e-8 || point[3]<1-1e-8)
        throw std::runtime_error("Shared squared/resource coordinates were not handled safely");

    LinearModel signed_one;
    signed_one.variable("u",0,2,1);signed_one.variable("v",0,2,1);signed_one.variable("t",-1,-1);
    signed_one.cones={{0,1,2,1}};point={0,0,-1};
    const auto signed_result=polish_conic_primal(signed_one,options,5,point,box);
    if(!signed_result.at("applied").get<bool>() || !point_residuals(signed_one,point).at("passed").get<bool>())
        throw std::runtime_error("Signed squared coordinate/zero corner was not repaired safely");
    // A product can also be positive with two negative resources. The box
    // method supports only the nonnegative branch and must reject this case;
    // the fixed-coordinate comparison remains available and valid.
    LinearModel negative;
    negative.variable("u",-2,-.5);negative.variable("v",-2,-.5);negative.variable("t",1,1);
    negative.cones={{0,1,2,1}};point={-1,-1,1};const auto negative_before=point;
    const auto negative_result=polish_conic_primal(negative,options,5,point,box);
    if(negative_result.at("applied").get<bool>() || point!=negative_before)
        throw std::runtime_error("Negative resource branch was falsely certified as monotone");
    if(!polish_conic_primal(negative,options,5,point,fixed).at("applied").get<bool>())
        throw std::runtime_error("Fixed-coordinate fallback was lost on a negative product branch");

    // New path variables can be zero-extended, but a changed bound or a new
    // cone invalidating that vector must not be silently accepted.
    LinearModel extension=flexible;
    extension.variable("new_flow",0,10,2);
    const std::vector<double> previous{2,.75,1};
    std::vector<double> output{99};
    const auto extended=extend_primal_candidate(extension,previous,output);
    if(!extended.at("passed").get<bool>() || output!=std::vector<double>({2,.75,1,0}) ||
       !extended.at("lower_bound").is_null())
        throw std::runtime_error("Zero-column candidate extension failed");
    const auto extended_before=output;
    extension.variables.back().lower=1;
    if(extend_primal_candidate(extension,previous,output).at("applied").get<bool>() || output!=extended_before)
        throw std::runtime_error("Candidate extension ignored a positive new lower bound");
    extension.variables.back().lower=0;extension.cones.push_back({0,3,2,1});
    if(extend_primal_candidate(extension,previous,output).at("applied").get<bool>() || output!=extended_before)
        throw std::runtime_error("Candidate extension ignored a new cone violation");
}

int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--fixtures") {
            recovery_fixture();
            std::cout<<Json({{"passed",true},{"success_original_matrix_validated",true},
                {"infeasible_restriction_atomic",true},{"no_budget_atomic",true},
                {"box_strictly_contains_fixed_corner",true},{"shared_cone_coordinates",true},
                {"adaptive_preserves_fixed_success_and_shares_budget",true},
                {"signed_coordinate_and_negative_branch",true},{"zero_extension_fully_validated",true},
                {"original_lower_bound_never_reported",true}}).dump(2)<<'\n';
            return 0;
        }
        if((argc==7 || argc==8) && std::string(argv[1])=="--repair") {
            const double started=wall_time(),seconds=std::stod(argv[5]);
            const std::string mode=argv[6],prefix=argv[4];
            if(mode!="box" && mode!="fixed" && mode!="adaptive")throw std::invalid_argument("Repair mode must be box, fixed or adaptive");
            const auto read=[](const char* path) {std::ifstream stream(path);Json data;stream>>data;return data;};
            auto data=read(argv[2]);auto point=read(argv[3]).get<std::vector<double>>();
            LinearModel model;model.constant=data.at("constant");model.minimize=data.at("minimize");
            for(const auto& v:data.at("variables"))
                model.variable(v.at(0),v.at(1),v.at(2),v.at(3),v.at(4));
            for(const auto& r:data.at("rows"))
                model.row(r.at(0),r.at(1).get<std::string>().at(0),r.at(2),r.at(3).get<Coefficients>());
            for(const auto& c:data.at("cones"))model.cones.push_back({c.at(0),c.at(1),c.at(2),c.at(3)});
            std::string validated_matrix=argv[2];
            Json report={{"mode",mode},{"source_matrix",argv[2]},{"source_matrix_sha256",sha256_file(argv[2])},
                {"source_point",argv[3]},{"source_point_sha256",sha256_file(argv[3])},
                {"budget_seconds",seconds},{"threads",1},{"lower_bound",nullptr}};
            if(argc==8) {
                const auto source=read(argv[7]);
                if (data.value("assignment_scope",std::string())=="origin" &&
                    (!source.is_object() || source.value("assignment_scope",std::string())!="origin"))
                    throw std::invalid_argument("K=1 design sidecar must declare assignment_scope=origin");
                const auto design=(source.is_array()?source:source.at("design")).get<std::vector<double>>();
                if(design.size()>point.size())throw std::invalid_argument("Design is larger than candidate");
                for(std::size_t j=0;j<design.size();++j) {
                    model.variables[j].cost=0;data["variables"][j][3]=0;point[j]=design[j];
                    model.row("diagnostic_fix_"+std::to_string(j),'E',design[j],{{static_cast<int>(j),1}});
                    data["rows"].push_back({model.rows.back().name,"E",design[j],model.rows.back().coefficients});
                }
                data["scope"]="fixed_design_full_path_continuous_diagnostic";
                validated_matrix=prefix+".matrix.json";
                std::ofstream matrix_stream(validated_matrix);matrix_stream<<data.dump()<<'\n';
                report["design_source"]=argv[7];report["design_sha256"]=sha256_file(argv[7]);
            }
            report["validated_matrix"]=validated_matrix;report["scope"]=data.at("scope");
            SolverOptions options;options.threads=1;options.seconds=seconds;
            report["repair"]=polish_conic_primal(model,options,std::max(0.0,seconds-(wall_time()-started)),point,
                mode=="box"?PrimalPolishRestriction::MonotoneConeBox:mode=="fixed"?
                    PrimalPolishRestriction::FixedConeCoordinates:PrimalPolishRestriction::AdaptiveFixedThenBox);
            report["total_seconds"]=wall_time()-started;
            if(report["repair"].at("applied").get<bool>()) {
                std::ofstream values(prefix+".values.json");values<<Json(point).dump()<<'\n';
            }
            std::ofstream output(prefix+".json");output<<report.dump(2)<<'\n';
            std::cout<<report.dump(2)<<'\n';return 0;
        }
        if (argc<5 || argc>6)
            throw std::invalid_argument("Usage: mlnd_continuous_matrix_probe INSTANCE K OUTPUT_PREFIX SECONDS [default|equilibrated|seeded|seeded-equilibrated]");
        const double started=wall_time(), budget=std::stod(argv[4]);
        recovery_fixture();
        const std::string profile=argc==6?argv[5]:"default";
        const bool seeded=profile=="seeded" || profile=="seeded-equilibrated";
        const bool equilibrated=profile=="equilibrated" || profile=="seeded-equilibrated";
        if (profile!="default" && !seeded && !equilibrated) throw std::invalid_argument("Unknown profile");
        const auto d=read_input(argv[1],std::stoi(argv[2]));
        auto paths=seeded?prepare_path_support(d):enumerate_paths(d);
        if(seeded) {
            greedy_seed_paths(d,paths,1,started+budget);
            branch_covering_seed_paths(d,paths,nullptr,std::min(started+budget,wall_time()+std::min(5.0,.05*budget)));
        }
        auto f=build_formulation(d,paths);
        for (auto& variable:f.matrix.variables) variable.binary=false;
        Json matrix={{"schema","mlnd-linear-model/1"},{"K",d.max_orders},{"assignment_scope",d.assignment_scope()},
            {"input_manifest",d.manifest},{"scope",seeded?"restricted_seed_rmp_continuous_relaxation":"full_path_continuous_relaxation"},
            {"constant",f.matrix.constant},{"minimize",f.matrix.minimize},
            {"variables",Json::array()},{"rows",Json::array()},{"cones",Json::array()}};
        for (const auto& v:f.matrix.variables) matrix["variables"].push_back({v.name,v.lower,v.upper,v.cost,v.binary});
        for (const auto& r:f.matrix.rows) matrix["rows"].push_back({r.name,std::string(1,r.sense),r.rhs,r.coefficients});
        for (const auto& c:f.matrix.cones) matrix["cones"].push_back({c.up,c.down,c.one,c.factor});
        const std::string prefix=argv[3],matrix_path=prefix+".matrix.json";
        {std::ofstream stream(matrix_path); if(!stream)throw std::runtime_error("Cannot write matrix");stream<<matrix.dump()<<'\n';}
        LinearModel solver_matrix=f.matrix;
        std::vector<double> scales(f.matrix.variables.size(),1),row_scales(f.matrix.rows.size(),1);
        double objective_scale=1;
        if(equilibrated) {
            // Exact diagonal transformation x=D*z and positive row scaling.
            // Keep cone coordinates unscaled; map every solution back before
            // applying the unchanged original-matrix tolerances.
            for(std::size_t j=0;j<scales.size();++j) {
                const auto& v=f.matrix.variables[j];
                scales[j]=std::max(1.0,std::min(v.upper,1000.0));
            }
            for(const auto& c:f.matrix.cones) scales[c.up]=scales[c.down]=scales[c.one]=1;
            objective_scale=std::inner_product(d.demand.begin(),d.demand.end(),d.ground_cost.begin(),0.0);
            for(std::size_t j=0;j<scales.size();++j) {
                auto& v=solver_matrix.variables[j];
                v.lower/=scales[j];v.upper/=scales[j];v.cost*=scales[j]/objective_scale;
            }
            solver_matrix.constant/=objective_scale;
            for(std::size_t i=0;i<solver_matrix.rows.size();++i) {
                auto& r=solver_matrix.rows[i];double magnitude=std::abs(r.rhs);
                for(auto& [j,c]:r.coefficients) {c*=scales[j];magnitude=std::max(magnitude,std::abs(c));}
                row_scales[i]=magnitude>0?1/magnitude:1;
                r.rhs*=row_scales[i];for(auto& [j,c]:r.coefficients)c*=row_scales[i];
            }
        }
        SolverOptions options;options.threads=1;options.seconds=budget;
        NativeSolver solver(solver_matrix,options,"continuous_matrix_reference");
        if(equilibrated) {
            solver.cplex.setParam(IloCplex::Param::Emphasis::Numerical,true);
            solver.cplex.setParam(IloCplex::Param::Barrier::QCPConvergeTol,1e-10);
        }
        auto out=solver.optimize(std::max(0.0,budget-(wall_time()-started)));
        Json report={{"K",d.max_orders},{"solver","CPLEX"},{"threads",1},{"profile",profile},
            {"matrix_sha256",sha256_file(matrix_path)},{"matrix_path",matrix_path},
            {"scope",matrix.at("scope")},{"service_columns",paths.service_paths.size()},{"empty_columns",paths.empty_paths.size()},
            {"variables",f.matrix.variables.size()},{"rows",f.matrix.rows.size()},{"cones",f.matrix.cones.size()},
            {"status",out.status},{"status_code",out.cplex_status},{"solver_optimal",out.optimal},
            {"solve_return_has_solution",out.has_solution},{"solver_seconds",out.seconds},
            {"candidate_available",out.candidate_available},{"candidate_extraction_error",out.candidate_extraction_error},
            {"budget_seconds",budget},{"dual_certified_lower_bound",nullptr},
            {"recovery_fixture_passed",true},
            {"solver_reported_objective",out.has_solution?Json(out.objective*objective_scale):Json(nullptr)}};
        // NumBest may retain a point even when solve() returns false. Inspect
        // it diagnostically, but do not promote its objective to a lower bound.
        const bool extracted=out.candidate_available;
        if(extracted) {
            for(std::size_t j=0;j<scales.size();++j)out.values[j]*=scales[j];
            report["objective"]=objective_value(f.matrix,out.values);
            report["residuals"]=point_residuals(f.matrix,out.values);
            report["validated_primal_upper_bound"]=report["residuals"]["passed"].get<bool>()?report["objective"]:Json(nullptr);
            {std::ofstream stream(prefix+".cplex.values.json");stream<<Json(out.values).dump()<<'\n';}
            const auto before_no_budget=out.values;
            const auto no_budget=polish_conic_primal(f.matrix,options,0,out.values);
            if(no_budget.at("applied").get<bool>() || out.values!=before_no_budget)
                throw std::runtime_error("No-budget primal recovery modified its input");
            report["recovery_no_budget_atomicity_checked"]=true;
            if(!report["residuals"]["passed"].get<bool>() && budget-(wall_time()-started)>1) {
                report["primal_repair"]=polish_conic_primal(f.matrix,options,
                    std::max(0.0,budget-(wall_time()-started)),out.values);
                if(report["primal_repair"].at("applied").get<bool>()) {
                    std::ofstream stream(prefix+".cplex_lp_repaired.values.json");stream<<Json(out.values).dump()<<'\n';
                }
            }
        }
        report["total_seconds"]=wall_time()-started;
        {std::ofstream stream(prefix+".cplex.json");stream<<report.dump(2)<<'\n';}
        std::cout<<report.dump(2)<<'\n';
        return 0;
    }catch(const IloException& e){std::cerr<<e<<'\n';return 1;}
     catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
