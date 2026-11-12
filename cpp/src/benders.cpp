#include "mlnd/benders.hpp"
#include "mlnd/benders_control.hpp"
#include "mlnd/certified_cut_pool.hpp"
#include "mlnd/continuous_root_control.hpp"
#include "mlnd/pricing.hpp"
#include "mlnd/path_identity.hpp"
#include "mlnd/adaptive_pricing.hpp"
#include "mlnd/stabilized_master.hpp"
#include <optional>
#include "mlnd/output.hpp"
#include "mlnd/strengthening.hpp"
#include "mlnd/topology_cuts.hpp"
#include "mlnd/assignment_support_cuts.hpp"
#include "mlnd/path_support_cuts.hpp"
#include "mlnd/airline_resources.hpp"
#include "mlnd/queue_polymatroid.hpp"
#include "mlnd/queue_fractional.hpp"
#include "mlnd/queue_fractional_control.hpp"
#include "mlnd/task_savings.hpp"
#include "mlnd/branch_joint_cuts.hpp"
#include "mlnd/heuristics.hpp"
#include "mlnd/primal_portfolio.hpp"
#include "mlnd/primal_alns.hpp"
#include "mlnd/primal_schedule.hpp"
#include "mlnd/design_face_cuts.hpp"
#include "mlnd/dual_workspace.hpp"
#include "mlnd/conditional_cut.hpp"
#include "mlnd/preprocessed_demand.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace mlnd {
namespace {
struct Cut {
    bool valid = false;
    double intercept = 0, correction = 0, stationarity_error = 0;
    double raw_anchor = 0, raw_core = 0;
    std::vector<double> coefficients, prices, missing_service, missing_empty;
    bool optimal = false;
    std::string status, method = "explicit_conic_dual";
    Json diagnostics = Json::object();
    Json condition = nullptr;
    QueueNativePrices queue_prices;
};
struct Subproblem {
    Json solution, pareto = {{"attempted", false}, {"accepted", false}, {"reason", "disabled"}};
    Json trace = Json::array();
    Cut ordinary, selected, auxiliary;
    std::vector<int> service, empty;
    std::optional<PrimalCgFeedback> primal_feedback;
    Json timings = {{"build", 0.0}, {"primal", 0.0}, {"dual", 0.0}, {"pricing", 0.0},
                    {"add_columns", 0.0}, {"pareto", 0.0}, {"read_solution", 0.0},
                    {"validation", 0.0}, {"certificate", 0.0}};
    int iterations = 0;
    double wall = 0;
    std::string reason = "time_limit";
    double continuous_objective = INFINITY, certified_lower_bound = -INFINITY;
    std::vector<double> continuous_design;
    Json continuous_residuals = nullptr, lower_bound_certificate = nullptr, failure = nullptr;
    Json continuous_operating_point = nullptr;
    bool full_bound_valid = false;
    bool recourse_optimal = false;
    bool conditional_demand = false;
    Json model_size = nullptr;
};
// A single fixed-design continuation state, owned by one solve_benders call.
// It contains no retained dual or certificate: every slice solves the RMP
// again and reads its current native multipliers before pricing/certification.
struct RecourseModelState {
    std::unique_ptr<Formulation> formulation;
    std::unique_ptr<NativeSolver> solver;
    const Input* input=nullptr;
    const PathCollection* paths=nullptr;
    const RunProgress* progress=nullptr;
    std::string domain_identity,options_identity;
    std::vector<double> design,price_center;
    std::size_t pricing_round=0;
    AdaptivePricingControl pricing_control;
    std::vector<int> recent_service,recent_empty;
    double previous_pricing_seconds=0;
    int primal_retries_for_columns=0,active_primal_profile=0;
    bool primal_polish_attempted_for_columns=false;
    std::string primal_retry_reason;
    std::vector<std::pair<int,std::string>> service_payloads,empty_payloads;
};
std::string seed_path_key(const ServicePath&);
std::string seed_path_key(const EmptyPath&);
bool recourse_columns_unchanged(const RecourseModelState& state,const PathCollection& paths) {
    for(const auto& [id,key]:state.service_payloads)
        if(id<0 || static_cast<std::size_t>(id)>=paths.service_paths.size() || seed_path_key(paths.service_paths[id])!=key)return false;
    for(const auto& [id,key]:state.empty_payloads)
        if(id<0 || static_cast<std::size_t>(id)>=paths.empty_paths.size() || seed_path_key(paths.empty_paths[id])!=key)return false;
    return true;
}
struct SubproblemStateCache {
    std::unique_ptr<RecourseModelState> state;
    std::size_t hits=0,builds=0,rhs_updates=0;
    std::string last_rebuild_reason="unused";
    Json metadata() const {
        return {{"hits",hits},{"builds",builds},{"rhs_updates",rhs_updates},{"live_models",state?1:0},
                {"last_rebuild_reason",last_rebuild_reason}};
    }
};
std::string recourse_options_identity(const SolverOptions& o) {
    // Matrix structure, native solver parameters and price-center policy.
    // Budgets, Pareto anchors and callbacks do not change the primal model.
    return Json{{"conditional",o.conditional_demand},{"tight_airline",o.tight_airline_bounds},
        {"assignment_clique",o.assignment_incident_clique},{"shared_rho",o.shared_rho_joint},
        {"queue_perspective",o.queue_perspective},{"airline_upper",o.task_airline_upper},
        {"zero_tasks",o.eliminate_zero_task_columns},{"compact_heads",o.compact_task_heads},
        {"threads",o.threads},{"seed",o.seed},{"gap",o.gap},{"log",o.log},
        {"stabilized_pricing",o.stabilized_pricing},{"smoothing_weight",o.pricing_smoothing_weight},
        {"exact_interval",o.pricing_exact_interval},{"adaptive_pricing",o.adaptive_pricing},
        {"catalogue_pricing",o.catalogue_pricing},{"diverse_pricing",o.diverse_pricing},
        {"cg_max_columns",o.cg_max_columns},{"reuse_rhs",o.reuse_recourse_rhs}}.dump();
}
void add_time(Json& times, const char* name, double begin) {
    times[name] = times.value(name, 0.0) + wall_time() - begin;
}
double remaining(double deadline) { return std::max(0.0, deadline - wall_time()); }
double total_demand(const Input& d) { return std::accumulate(d.demand.begin(), d.demand.end(), 0.0); }
double courier_cost(const Input& d) {
    return std::inner_product(d.demand.begin(), d.demand.end(), d.ground_cost.begin(), 0.0);
}
double value(const Cut& c, const std::vector<double>& design) {
    return c.intercept + std::inner_product(c.coefficients.begin(), c.coefficients.end(), design.begin(), 0.0);
}
double gap(double upper, double lower) { return std::max(0.0, upper - lower) / std::max(1.0, std::abs(upper)); }
// Invocation-local memoization. Input/initial support and existing physical
// path payloads are immutable throughout solve_benders; generated IDs append.
// No cache is shared between solves or different mathematical model versions.
struct SparseRecoursePoint {
    Json envelope=Json::object(),point=Json::object();
    std::map<std::string,std::vector<std::pair<std::size_t,double>>> sparse;
    explicit SparseRecoursePoint(const Json& result) {
        for(auto it=result.begin();it!=result.end();++it)
            if(it.key().empty() || it.key()[0]!='_')envelope[it.key()]=it.value();
        for(auto it=result.at("_solution").begin();it!=result.at("_solution").end();++it) {
            if(it.key()=="flow" || it.key()=="empty_flow" || it.key()=="head" || it.key()=="eta") {
                auto& entries=sparse[it.key()];
                for(std::size_t i=0;i<it.value().size();++i) {
                    const double x=it.value().at(i).get<double>();
                    if(!std::isfinite(x))throw std::invalid_argument("Nonfinite cached primal point");
                    if(x!=0)entries.emplace_back(i,x);
                }
            } else point[it.key()]=it.value();
        }
    }
    Json restore(const Input& d,const PathCollection& paths) const {
        Json result=envelope;result["_solution"]=point;
        for(const auto& [name,entries]:sparse) {
            const std::size_t count=name=="flow"?paths.service_paths.size():name=="empty_flow"?paths.empty_paths.size():paths.initial_segments.size();
            std::vector<double> values(count,0);
            for(auto [i,x]:entries)values.at(i)=x;
            result["_solution"][name]=std::move(values);
        }
        synchronize_result_paths(d,paths,result);
        return result;
    }
};
struct DesignCertificateMemo {
    struct Entry {
        Cut lower;
        double upper=INFINITY;
        bool optimal=false;
        std::unique_ptr<SparseRecoursePoint> point;
        std::vector<int> service,empty;
        std::size_t age=0;
    };
    std::map<std::vector<double>,Entry> entries;
    std::size_t capacity=16,clock=0,hits=0,stores=0,misses=0,unresolved=0,evictions=0;
    explicit DesignCertificateMemo(std::size_t limit):capacity(std::max<std::size_t>(1,limit)){}
    bool remember(const Input& d,const PathCollection&,const std::vector<double>& design,const Subproblem& sub) {
        if(!sub.ordinary.valid || sub.conditional_demand || !sub.ordinary.condition.is_null() ||
           sub.solution.is_null() || !sub.solution.value("native_model_validated",false) ||
           solution_design(d,sub.solution)!=design)return false;
        const double lo=value(sub.ordinary,design),up=sub.solution.at("continuous_cost").get<double>();
        if(!std::isfinite(lo) || !std::isfinite(up) || lo>up+2e-7*(1+std::abs(up)))return false;
        auto& entry=entries[design];
        if(!entry.lower.valid || lo>value(entry.lower,design)) {
            // Store only the affine certificate. Copying then clearing the
            // row/catalogue multipliers would retain their O(paths) capacity.
            const auto& source=sub.ordinary;Cut certificate;
            certificate.valid=source.valid;certificate.intercept=source.intercept;
            certificate.correction=source.correction;certificate.stationarity_error=source.stationarity_error;
            certificate.raw_anchor=source.raw_anchor;certificate.raw_core=source.raw_core;
            certificate.coefficients=source.coefficients;certificate.optimal=source.optimal;
            certificate.status=source.status;certificate.method=source.method;
            certificate.diagnostics=source.diagnostics;certificate.condition=source.condition;
            entry.lower=std::move(certificate);
        }
        if(up<entry.upper) {
            entry.upper=up;entry.point=std::make_unique<SparseRecoursePoint>(sub.solution);
            entry.service=sub.service;entry.empty=sub.empty;
        }
        entry.optimal=entry.optimal || sub.recourse_optimal;
        entry.age=++clock;++stores;
        if(entries.size()>capacity) {
            auto oldest=std::min_element(entries.begin(),entries.end(),[](const auto& a,const auto& b){return a.second.age<b.second.age;});
            entries.erase(oldest);++evictions;
        }
        return true;
    }
    // A hit either supplies a violated VALID cut or a known feasible epigraph.
    // An interval still straddling theta must be refined, never silently skipped.
    std::optional<Subproblem> reuse(const Input& d,const PathCollection& paths,
            const std::vector<double>& design,double theta,double tolerance) {
        const double started=wall_time();
        auto it=entries.find(design);
        if(it==entries.end()){++misses;return std::nullopt;}
        auto& entry=it->second;entry.age=++clock;
        const double lo=value(entry.lower,design);
        const double eps=tolerance*(1+std::max(std::abs(entry.upper),std::abs(theta)));
        const bool separate=theta<lo-eps;
        const double close_eps=.5*tolerance*(1+std::abs(entry.upper));
        const bool closed=entry.optimal && entry.upper-lo<=close_eps && entry.upper-lo>=-close_eps;
        if(!separate && theta<entry.upper && !closed){++unresolved;return std::nullopt;}
        Subproblem sub;sub.ordinary=entry.lower;sub.selected=entry.lower;
        sub.solution=entry.point->restore(d,paths);sub.service=entry.service;sub.empty=entry.empty;
        sub.recourse_optimal=closed;
        sub.reason=separate?"cached_design_separation":"cached_design_epigraph";
        sub.wall=wall_time()-started;sub.timings["design_certificate_restore"]=sub.wall;
        ++hits;return sub;
    }
    Json metadata() const {
        return {{"entries",entries.size()},{"capacity",capacity},{"stores",stores},{"hits",hits},
            {"misses",misses},{"unresolved",unresolved},{"evictions",evictions},
            {"scope","one_solve_immutable_domain"},{"global_lower_bound_from_point",false}};
    }
};

bool structural_enabled(const SolverOptions& o) {
    return o.assignment_incident_clique || o.shared_rho_joint || o.queue_perspective || !o.task_airline_upper.empty();
}
const char* continuous_scope(const SolverOptions& o) {
    return structural_enabled(o) ? "structurally_strengthened_continuous_relaxation" : "full_path_continuous_relaxation";
}
Json structural_metadata(const SolverOptions& o) {
    return {{"assignment_incident_clique",o.assignment_incident_clique},
        {"shared_rho_joint",o.shared_rho_joint},{"queue_perspective",o.queue_perspective},
        {"task_airline_bounds",!o.task_airline_upper.empty()},
        {"integer_feasible_set_preserved",true},{"master_added_variables",0},
        {"continuous_relaxation",continuous_scope(o)},
        {"integer_recourse_effect","redundant_on_fixed_integer_design"}};
}

// This numeric model check is performed before accepting a native incumbent;
// the repository's physical reconstruction remains an independent final check.
bool feasible_point(const LinearModel& model, const std::vector<double>& values) {
    if (values.size() != model.variables.size()) return false;
    for (std::size_t j = 0; j < values.size(); ++j) {
        const auto& v = model.variables[j];
        if (!std::isfinite(values[j])) return false;
        double tol = 2e-7 * (1 + std::abs(values[j]));
        if (values[j] < v.lower - tol || values[j] > v.upper + tol) return false;
        if (v.binary && std::abs(values[j] - std::round(values[j])) > 1e-7) return false;
    }
    for (const auto& row : model.rows) {
        double lhs = 0, scale = 1 + std::abs(row.rhs);
        for (auto [j, coefficient] : row.coefficients) {
            lhs += coefficient * values[j];
            scale += std::abs(coefficient * values[j]);
        }
        double residual = lhs - row.rhs;
        if (row.sense == 'L') residual = std::max(0.0, residual);
        else if (row.sense == 'G') residual = std::max(0.0, -residual);
        else residual = std::abs(residual);
        if (residual > 2e-7 * scale) return false;
    }
    for (const auto& cone : model.cones) {
        double lhs = values[cone.one] * values[cone.one];
        double rhs = cone.factor * values[cone.up] * values[cone.down];
        if (lhs > rhs + 2e-7 * (1 + std::abs(lhs) + std::abs(rhs))) return false;
    }
    return true;
}

Json feasibility_diagnostics(const LinearModel& model,const std::vector<double>& values) {
    if (values.size()!=model.variables.size()) return {{"passed",false},{"reason","dimension"}};
    if (!std::all_of(values.begin(),values.end(),[](double value){return std::isfinite(value);}))
        return {{"passed",false},{"reason","nonfinite"}};
    auto report=point_residuals(model,values);
    report["passed"]=feasible_point(model,values);
    double worst=0; Json detail=nullptr;
    auto check=[&](double residual,double tolerance,const std::string& kind,const std::string& name) {
        if (residual/tolerance>worst) {
            worst=residual/tolerance;
            detail={{"kind",kind},{"name",name},{"violation",residual},{"tolerance",tolerance}};
        }
    };
    for (std::size_t j=0;j<values.size();++j) {
        const auto& v=model.variables[j];
        check(std::max({0.0,v.lower-values[j],values[j]-v.upper}),2e-7*(1+std::abs(values[j])),"bound",v.name);
        if (v.binary) check(std::abs(values[j]-std::round(values[j])),1e-7,"integrality",v.name);
    }
    for (const auto& row:model.rows) {
        double lhs=0,scale=1+std::abs(row.rhs);
        for (auto [j,c]:row.coefficients) { lhs+=c*values[j]; scale+=std::abs(c*values[j]); }
        double residual=lhs-row.rhs;
        residual=row.sense=='L'?std::max(0.0,residual):row.sense=='G'?std::max(0.0,-residual):std::abs(residual);
        check(residual,2e-7*scale,"linear",row.name);
    }
    for (const auto& cone:model.cones) {
        const double lhs=values[cone.one]*values[cone.one],rhs=cone.factor*values[cone.up]*values[cone.down];
        check(std::max(0.0,lhs-rhs),2e-7*(1+std::abs(lhs)+std::abs(rhs)),"cone",model.variables[cone.one].name);
    }
    report["worst_tolerance_ratio"]=worst;
    report["worst_constraint"]=detail;
    return report;
}

void validate_design(const Input& d, const std::vector<double>& design, bool fractional = false) {
    const auto constraints = design_model(d, !fractional);
    if (design.size() != static_cast<std::size_t>(constraints.layout.count) ||
        !std::all_of(design.begin(), design.end(), [fractional](double x) {
            return std::isfinite(x) && (fractional ? x >= 0 && x <= 1 : x == 0.0 || x == 1.0); }))
        throw std::invalid_argument("Benders fixed design must be exactly binary");
    if (!feasible_point(constraints.matrix, design))
        throw std::invalid_argument("Benders fixed design violates facility, capacity, assignment or airline constraints");
}

// Read the dual of the SOCP already solved by CPLEX.  Projection and bounded
// stationarity corrections below are algebraic; they never solve another model.
Cut read_cplex_dual(const Input& d, const PathCollection& paths, const Formulation& f,
                    NativeSolver& solver, double primal_objective,
                    const SolverOptions& options) {
    const double started=wall_time();
    Cut cut;
    cut.method="native_cplex_dual";
    auto native=solver.read_dual(f.matrix);
    cut.diagnostics=native.diagnostics;
    cut.diagnostics["auxiliary_optimization_calls"]=0;
    const auto& model=f.matrix;
    const std::size_t n=model.variables.size(),m=model.rows.size();
    if(!native.available || native.row_prices.size()!=m || native.bound_prices.size()!=n ||
       native.cone_vectors.size()!=model.cones.size()) {
        cut.status="NativeDualUnavailable";
        cut.diagnostics["error"]=native.error;
        cut.diagnostics["seconds"]=wall_time()-started;
        return cut;
    }
    std::vector<bool> fixed(n,false),fix_row(m,false),omitted(m,false);
    std::vector<double> anchor;
    for(std::size_t k=0;k<f.design_fix_rows.size();++k) {
        const int j=f.design_vars[k],i=f.design_fix_rows[k];
        fixed[j]=true;fix_row[i]=true;
        anchor.push_back(model.rows[i].rhs);
    }
    cut.prices=native.row_prices;
    for(std::size_t i=0;i<m;++i) {
        // Retain empty operating rows: omitted path columns can touch them.
        bool operating=false;
        for(auto [j,a]:model.rows[i].coefficients) if(a!=0 && !fixed[j]) {operating=true;break;}
        omitted[i]=fix_row[i] || (!f.design_fix_rows.empty() &&
            i<static_cast<std::size_t>(f.design_fix_rows.front()) && !operating);
        if(omitted[i])cut.prices[i]=0;
        else if(model.rows[i].sense=='L')cut.prices[i]=std::min(0.0,cut.prices[i]);
        else if(model.rows[i].sense=='G')cut.prices[i]=std::max(0.0,cut.prices[i]);
    }
    if(options.queue_price_search)
        cut.queue_prices=native_queue_resource_prices(d,f,cut.prices);
    std::vector<double> residual(n),cone_values(n,0);
    for(std::size_t k=0;k<model.cones.size();++k) {
        const auto& c=model.cones[k];
        double up=std::max(0.0,native.cone_vectors[k][0]);
        double down=std::max(0.0,native.cone_vectors[k][1]);
        const double one=native.cone_vectors[k][2],factor=4.0/c.factor;
        if(factor*up*down<one*one) {
            if(up>=down && up>0)down=std::nextafter(one*one/(factor*up),INFINITY);
            else if(down>0)up=std::nextafter(one*one/(factor*down),INFINITY);
            else up=down=std::nextafter(std::abs(one)/std::sqrt(factor),INFINITY);
        }
        cone_values[c.up]+=up;cone_values[c.down]+=down;cone_values[c.one]+=one;
    }
    double rhs=0,bounds=0,absolute_terms=0;
    for(std::size_t j=0;j<n;++j) {
        const auto& v=model.variables[j];
        if(v.lower<0 || !std::isfinite(v.upper) || v.upper>=IloInfinity)
            throw std::runtime_error("Native dual certificate requires finite nonnegative recourse bounds");
        // Original bounds on fixed design coordinates are redundant.  Their
        // prices and fix multipliers are replaced by the affine coefficient.
        const double price=fixed[j]?0:native.bound_prices[j];
        const double lower=std::max(0.0,price),upper=std::max(0.0,-price);
        bounds+=v.lower*lower-v.upper*upper;
        absolute_terms+=std::abs(v.lower*lower)+std::abs(v.upper*upper);
        residual[j]=lower-upper+cone_values[j]-v.cost;
    }
    for(std::size_t i=0;i<m;++i) if(!omitted[i]) {
        rhs+=model.rows[i].rhs*cut.prices[i];
        absolute_terms+=std::abs(model.rows[i].rhs*cut.prices[i]);
        for(auto [j,a]:model.rows[i].coefficients)residual[j]+=a*cut.prices[i];
    }
    double coefficient_norm=0;
    for(std::size_t k=0;k<f.design_fix_rows.size();++k) {
        const int j=f.design_vars[k],i=f.design_fix_rows[k];
        const double slope=-residual[j];
        const double rounded=slope==0?0:std::nextafter(slope,-INFINITY);
        cut.coefficients.push_back(rounded);cut.prices[i]=rounded;
        coefficient_norm+=std::abs(rounded);residual[j]+=rounded;
    }
    double residual_correction=0;
    for(std::size_t j=0;j<n;++j) {
        const double violation=std::max(0.0,residual[j]);
        cut.stationarity_error=std::max(cut.stationarity_error,violation);
        residual_correction+=model.variables[j].upper*violation;
    }
    const double base_margin=1e-8*(1+courier_cost(d));
    const double rounding_margin=128*std::numeric_limits<double>::epsilon()*
        (1+absolute_terms+coefficient_norm);
    cut.correction=residual_correction+base_margin+rounding_margin;
    cut.intercept=model.constant+rhs+bounds;
    cut.raw_anchor=value(cut,anchor);cut.raw_core=cut.raw_anchor;
    cut.intercept-=cut.correction;
    cut.missing_service.assign(paths.service_paths.size(),0);
    cut.missing_empty.assign(paths.empty_paths.size(),0);
    cut.valid=std::isfinite(cut.intercept) && std::isfinite(cut.correction) &&
        std::all_of(cut.coefficients.begin(),cut.coefficients.end(),[](double x){return std::isfinite(x);});
    const double tolerance=options.cg_duality_tolerance*(1+std::abs(primal_objective));
    const double certificate_gap=primal_objective-value(cut,anchor);
    cut.optimal=cut.valid && std::isfinite(certificate_gap) &&
        certificate_gap>=-tolerance && certificate_gap<=tolerance;
    cut.status=cut.optimal?"NativeDualCertified":cut.valid?"NativeDualGapFailed":"NativeDualInvalid";
    cut.diagnostics["correction_components"]={{"bound_weighted_residual",residual_correction},
        {"base_margin",base_margin},{"rounding_margin",rounding_margin},
        {"coefficient_norm",coefficient_norm},{"rhs_value",rhs},{"bound_terms",bounds}};
    cut.diagnostics["fixed_design_dual_elimination"]={{"enabled",!anchor.empty()},
        {"eliminated_coordinates",anchor.size()},{"free_design_unchanged",anchor.empty()}};
    cut.diagnostics["original_rmp_duality_gap"]=std::isfinite(certificate_gap)?Json(certificate_gap):Json(nullptr);
    cut.diagnostics["original_rmp_duality_tolerance"]=tolerance;
    cut.diagnostics["original_rmp_gap_passed"]=cut.optimal;
    cut.diagnostics["seconds"]=wall_time()-started;
    return cut;
}

Cut solve_dual(const Input& d, const PathCollection& paths, const Formulation& f,
               const SolverOptions& options, double deadline,
               const std::vector<double>* core = nullptr,
               double anchor_value = 0, double face_tolerance = 0,
               const std::vector<int>& separated_service = {},
               const std::vector<int>& separated_empty = {}, int numerical_profile = 0,
               const std::vector<double>* primal_anchor = nullptr,
               DualRayWorkspace* ray_workspace = nullptr) {
    const bool minimum_norm_recovery=numerical_profile==5;
    if(minimum_norm_recovery)numerical_profile=4;
    Cut result;
    const double dual_started = wall_time();
    result.diagnostics = {{"allocated_seconds", remaining(deadline)},
        {"linear_model_build_seconds", 0.0}, {"solver_build_seconds", 0.0},
        {"optimize_seconds", 0.0}, {"certificate_seconds", 0.0}};
    auto finish_dual = [&](const char* reason) {
        result.diagnostics["termination_reason"] = reason;
        result.diagnostics["seconds"] = wall_time() - dual_started;
        result.diagnostics["valid"] = result.valid;
        result.diagnostics["optimal"] = result.optimal;
        result.diagnostics["status"] = result.status;
        return std::move(result);
    };
    if (numerical_profile == 4) result.method = "cone_ray_lp_recovery";
    if (remaining(deadline) <= 0) return finish_dual("deadline_before_model_build");
    const auto& primal = f.matrix;
    const int n = static_cast<int>(primal.variables.size());
    const int m = static_cast<int>(primal.rows.size());
    LinearModel dual;
    dual.minimize = false;
    dual.constant = primal.constant;
    std::vector<int> pi(m, -1), upper(n, -1), lower(n, -1);
    // Eliminate fixed design coordinates from the dual, including their
    // redundant bound prices and stationarity rows.  Substitution leaves the
    // recourse problem unchanged and avoids unbounded dual representations
    // (notably arbitrary negative fix prices at a zero design coordinate).
    if (core && core->size()!=f.design_fix_rows.size())
        throw std::runtime_error("Invalid Benders core dimensions");
    std::vector<int> fixed_position(n, -1), fixed_row(m, -1);
    std::vector<double> fixed_anchor(f.design_fix_rows.size()), fixed_objective(f.design_fix_rows.size());
    for (std::size_t k=0;k<f.design_fix_rows.size();++k) {
        const int row=f.design_fix_rows[k], j=f.design_vars[k];
        if (primal.rows[row].coefficients.size()!=1 || primal.rows[row].coefficients.front()!=std::make_pair(j,1.0))
            throw std::runtime_error("Unexpected fixed-design row");
        fixed_position[j]=static_cast<int>(k);fixed_row[row]=static_cast<int>(k);
        fixed_anchor[k]=primal.rows[row].rhs;
        fixed_objective[k]=core?core->at(k):fixed_anchor[k];
        dual.constant+=primal.variables[j].cost*fixed_objective[k];
    }
    std::vector<Coefficients> stationarity(n);
    std::vector<double> core_rhs(m);
    std::vector<double> anchor_rhs(m);
    for (int i=0;i<m;++i) {
        core_rhs[i]=anchor_rhs[i]=primal.rows[i].rhs;
        for(auto [j,c]:primal.rows[i].coefficients) if(fixed_position[j]>=0) {
            core_rhs[i]-=c*fixed_objective[fixed_position[j]];
            anchor_rhs[i]-=c*fixed_anchor[fixed_position[j]];
        }
    }
    for (int i = 0; i < m; ++i) {
        if(fixed_row[i]>=0) continue;
        // A row containing only fixed coordinates is a strategic-domain
        // constraint.  At an admissible anchor/core its dual contribution is
        // nonpositive, so the zero price is an exact optimal representative.
        // Keep the row's original position in result.prices for path pricing.
        bool operating_row=false;
        for(auto [j,c]:primal.rows[i].coefficients)
            if(c!=0 && fixed_position[j]<0) { operating_row=true; break; }
        // Only the original strategic prefix is permanent. An operating
        // row with no current RMP coefficient can receive a missing path
        // column and must retain its price for ordinary/Pareto pricing.
        if(!f.design_fix_rows.empty() && i<f.design_fix_rows.front() && !operating_row) continue;
        const auto& row = primal.rows[i];
        pi[i] = dual.variable("pi_" + std::to_string(i), row.sense == 'G' ? 0 : -IloInfinity,
                              row.sense == 'L' ? 0 : IloInfinity, core_rhs[i]);
        for (auto [j, coefficient] : row.coefficients) if(fixed_position[j]<0) stationarity[j].emplace_back(pi[i], coefficient);
    }
    for (int j = 0; j < n; ++j) {
        if(fixed_position[j]>=0) continue;
        const auto& v = primal.variables[j];
        if (v.lower < 0 || !std::isfinite(v.upper) || v.upper >= IloInfinity)
            throw std::runtime_error("Conic certificate requires finite nonnegative recourse bounds");
        upper[j] = dual.variable("upper_" + std::to_string(j), 0, IloInfinity, -v.upper);
        stationarity[j].emplace_back(upper[j], -1);
        if (v.lower > 0) {
            lower[j] = dual.variable("lower_" + std::to_string(j), 0, IloInfinity, v.lower);
            stationarity[j].emplace_back(lower[j], 1);
        }
    }
    std::vector<std::array<int, 3>> cones;
    for (std::size_t k = 0; k < primal.cones.size(); ++k) {
        const auto& cone = primal.cones[k];
        int up = dual.variable("cone_u_" + std::to_string(k), 0, IloInfinity, fixed_position[cone.up]>=0?-fixed_objective[fixed_position[cone.up]]:0);
        int down = dual.variable("cone_d_" + std::to_string(k), 0, IloInfinity, fixed_position[cone.down]>=0?-fixed_objective[fixed_position[cone.down]]:0);
        int one = dual.variable("cone_o_" + std::to_string(k), -IloInfinity, IloInfinity, fixed_position[cone.one]>=0?-fixed_objective[fixed_position[cone.one]]:0);
        if (primal_anchor && numerical_profile == 4) {
            // Restrict (never relax) the conic dual to the complementary ray
            // suggested by the optimal primal queue point; optimize its
            // nonnegative scale in the LP. Only the unchanged primal/dual gap
            // check can certify this restricted dual solution.
            const double primal_up = primal_anchor->at(cone.up);
            const double primal_down = primal_anchor->at(cone.down);
            if (!(primal_up > 0 && primal_down > 0)) return finish_dual("invalid_cone_recovery_anchor");
            const double a = std::sqrt(cone.factor * primal_down / primal_up);
            const double b = std::nextafter(std::max(std::sqrt(cone.factor * primal_up / primal_down), cone.factor / a), INFINITY);
            const double c = -2 * (primal_anchor->at(cone.one) < 0 ? -1 : 1);
            int scale = dual.variable("cone_ray_scale_" + std::to_string(k), 0, IloInfinity);
            dual.row("cone_ray_up_" + std::to_string(k), 'E', 0, {{up, 1}, {scale, -a}});
            dual.row("cone_ray_down_" + std::to_string(k), 'E', 0, {{down, 1}, {scale, -b}});
            dual.row("cone_ray_one_" + std::to_string(k), 'E', 0, {{one, 1}, {scale, -c}});
        } else dual.cones.push_back({up, down, one, 4.0 / cone.factor});
        stationarity[cone.up].emplace_back(up, 1);
        stationarity[cone.down].emplace_back(down, 1);
        stationarity[cone.one].emplace_back(one, 1);
        cones.push_back({up, down, one});
    }
    for (int j = 0; j < n; ++j) if(fixed_position[j]<0)
        dual.row("stationarity_" + std::to_string(j), 'L', primal.variables[j].cost, std::move(stationarity[j]));
    struct Missing { bool empty; int index, variable; Coefficients column; double cost, upper; };
    std::vector<Missing> missing;
    const double total = total_demand(d);
    if (core) {
        auto add_missing = [&](bool empty, int index, Coefficients column, double cost, double bound) {
            if(bound==0)return; // The eliminated physical path is fixed to zero.
            int u = dual.variable("missing_" + std::to_string(empty) + "_" + std::to_string(index), 0, IloInfinity, -bound);
            Coefficients terms{{u, -1}};
            for (auto [row, coefficient] : column) {
                if (pi[row]<0 && coefficient!=0) throw std::runtime_error("Missing path touches fixed design row");
                if(pi[row]>=0) terms.emplace_back(pi[row], coefficient);
            }
            dual.row("missing_stationarity_" + std::to_string(empty) + "_" + std::to_string(index), 'L', cost, std::move(terms));
            missing.push_back({empty, index, u, std::move(column), cost,bound});
        };
        // The auxiliary dual is separated by the same exact label oracle as
        // ordinary CG. Only previously separated columns are materialized;
        // there is no all-missing-catalogue shortcut in Pareto optimization.
        for (int i : separated_service)
            add_missing(false, i, path_column(f, paths.service_paths.at(i)), paths.service_paths.at(i).cost,
                        service_path_upper_bound(f,paths.service_paths.at(i)));
        for (int i : separated_empty)
            add_missing(true, i, path_column(f, paths.empty_paths.at(i)), paths.empty_paths.at(i).cost,total);
        Coefficients face;
        for (int i = 0; i < m; ++i) if (pi[i]>=0 && anchor_rhs[i]!=0) face.emplace_back(pi[i],anchor_rhs[i]);
        for (int j = 0; j < n; ++j) {
            if(upper[j]>=0) face.emplace_back(upper[j], -primal.variables[j].upper);
            if (lower[j] >= 0) face.emplace_back(lower[j], primal.variables[j].lower);
        }
        for (const auto& entry : missing) face.emplace_back(entry.variable, -entry.upper);
        double face_constant=primal.constant;
        for(std::size_t k=0;k<fixed_anchor.size();++k) face_constant+=primal.variables[f.design_vars[k]].cost*fixed_anchor[k];
        for(std::size_t k=0;k<cones.size();++k) {
            const auto& c=primal.cones[k];const std::array<int,3> js{c.up,c.down,c.one};
            for(int q=0;q<3;++q) if(fixed_position[js[q]]>=0) face.emplace_back(cones[k][q],-fixed_anchor[fixed_position[js[q]]]);
        }
        dual.row("optimal_face", 'G', anchor_value - face_tolerance - face_constant, std::move(face));
    }
    result.diagnostics["fixed_design_dual_elimination"]={{"enabled",!f.design_fix_rows.empty()},
        {"eliminated_coordinates",f.design_fix_rows.size()},
        {"omitted_row_prices",std::count(pi.begin(),pi.end(),-1)},
        {"free_design_unchanged",f.design_fix_rows.empty()}};
    std::vector<double> original_dual_cost;
    const double original_dual_constant=dual.constant;
    for(const auto& v:dual.variables)original_dual_cost.push_back(v.cost);
    if(minimum_norm_recovery) {
        // This auxiliary LP selects a small dual representative on a near
        // optimal face. Its optimum is NOT an original-RMP certificate: the
        // original primal/dual residual and gap checks still decide that.
        // The cone-ray restriction is already a subset of the conic dual.
        if(core || !primal_anchor)throw std::runtime_error("Minimum-norm recovery requires an ordinary primal anchor");
        const double q=objective_value(primal,*primal_anchor);
        const double tolerance=options.cg_duality_tolerance*(1+std::abs(q));
        Coefficients optimal_face;
        for(std::size_t j=0;j<original_dual_cost.size();++j)if(original_dual_cost[j]!=0)optimal_face.emplace_back(static_cast<int>(j),original_dual_cost[j]);
        dual.row("minimum_norm_anchor_face",'G',q-.1*tolerance-dual.constant,std::move(optimal_face));
        dual.minimize=true;dual.constant=0;
        const int original_n=static_cast<int>(dual.variables.size());
        for(int j=0;j<original_n;++j) {
            dual.variables[j].cost=0;
            if(dual.variables[j].lower>=0)dual.variables[j].cost=1;
            else if(dual.variables[j].upper<=0)dual.variables[j].cost=-1;
            else {
                int z=dual.variable("norm_abs_"+std::to_string(j),0,IloInfinity,1);
                dual.row("norm_plus_"+std::to_string(j),'L',0,{{j,1},{z,-1}});
                dual.row("norm_minus_"+std::to_string(j),'L',0,{{j,-1},{z,-1}});
            }
        }
        result.method="minimum_norm_cone_ray_lp";
        result.diagnostics["minimum_norm_recovery"]=true;
        result.diagnostics["minimum_norm_target_primal"]=q;
        result.diagnostics["norm_face_tolerance"]=.1*tolerance;
    }
    result.diagnostics["linear_model_build_seconds"] = wall_time() - dual_started;
    result.diagnostics["model_size"] = {{"variables", dual.variables.size()},
        {"linear_rows", dual.rows.size()}, {"cones", dual.cones.size()}};
    if (remaining(deadline) <= 0) return finish_dual("deadline_during_linear_model_build");
    const double solver_build_started = wall_time();
    const bool reuse_ray = options.reuse_dual_lp && ray_workspace && numerical_profile == 4 && !core && !minimum_norm_recovery;
    std::unique_ptr<NativeSolver> cold_solver;
    Json workspace_diagnostics;
    NativeSolver& solver = reuse_ray ? ray_workspace->synchronize(dual,options,workspace_diagnostics)
        : *(cold_solver=std::make_unique<NativeSolver>(dual, options, core ? "pareto_dual" : "benders_dual"));
    if (numerical_profile > 0) {
        solver.cplex.setParam(IloCplex::Param::Emphasis::Numerical, true);
        solver.cplex.setParam(IloCplex::Param::Barrier::QCPConvergeTol, 1e-12);
        solver.cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility, 1e-9);
        solver.cplex.setParam(IloCplex::Param::Simplex::Tolerances::Optimality, 1e-9);
        solver.cplex.setParam(IloCplex::Param::Read::Scale, numerical_profile == 1 ? 1 : numerical_profile == 2 ? -1 : 0);
        if (numerical_profile == 3) solver.cplex.setParam(IloCplex::Param::Preprocessing::Presolve, false);
        if (numerical_profile == 4) solver.cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Dual);
    }
    result.diagnostics["solver_build_seconds"] = wall_time() - solver_build_started;
    if (reuse_ray) result.diagnostics["dual_lp_workspace"] = workspace_diagnostics;
    if (remaining(deadline) <= 0) return finish_dual("deadline_during_solver_build");
    result.diagnostics["solver_time_limit_seconds"] = remaining(deadline);
    const double optimize_started = wall_time();
    auto solved = reuse_ray ? ray_workspace->optimize(remaining(deadline),workspace_diagnostics)
                           : solver.optimize(remaining(deadline));
    if (reuse_ray) result.diagnostics["dual_lp_workspace"] = workspace_diagnostics;
    result.diagnostics["optimize_seconds"] = wall_time() - optimize_started;
    result.diagnostics["solver_status_code"] = solved.cplex_status;
    result.diagnostics["has_solution"] = solved.has_solution;
    result.optimal = solved.optimal;
    result.status = numerical_profile == 4 ? "RecoveredConeRayLP_" + solved.status : solved.status;
    if (!solved.has_solution)
        return finish_dual(solved.status == "AbortTimeLim" ? "solver_time_limit" : "solver_no_solution");
    if (solved.values.size() != dual.variables.size()) return finish_dual("solver_values_dimension");
    if (!std::all_of(solved.values.begin(), solved.values.end(), [](double x) { return std::isfinite(x); }))
        return finish_dual("solver_values_nonfinite");
    if (reuse_ray) {
        const double validation_started=wall_time();
        result.diagnostics["dual_lp_source_residuals"]=feasibility_diagnostics(dual,solved.values);
        result.diagnostics["dual_lp_source_objective"]=objective_value(dual,solved.values);
        const bool source_valid=feasible_point(dual,solved.values);
        result.diagnostics["dual_lp_source_validation_seconds"]=wall_time()-validation_started;
        if (!source_valid)
            return finish_dual("persistent_dual_lp_source_validation_failed");
    }
    const double certificate_started = wall_time();
    result.prices.resize(m);
    for (int i = 0; i < m; ++i) {
        if(pi[i]<0) { result.prices[i]=0; continue; }
        double x = solved.values[pi[i]];
        if (primal.rows[i].sense == 'L') x = std::min(0.0, x);
        if (primal.rows[i].sense == 'G') x = std::max(0.0, x);
        result.prices[i] = x;
    }
    std::vector<double> cone_values(n, 0), upper_prices(n), lower_prices(n, 0), residual(n, 0);
    for (std::size_t k = 0; k < cones.size(); ++k) {
        const auto& source = primal.cones[k];
        auto indices = cones[k];
        double up = std::max(0.0, solved.values[indices[0]]);
        double down = std::max(0.0, solved.values[indices[1]]);
        double one = solved.values[indices[2]];
        const double f = 4.0 / source.factor;
        if (f * up * down < one * one) {
            if (up >= down && up > 0) down = std::nextafter(one * one / (f * up), INFINITY);
            else if (down > 0) up = std::nextafter(one * one / (f * down), INFINITY);
            else up = down = std::nextafter(std::abs(one) / std::sqrt(f), INFINITY);
        }
        cone_values[source.up] += up;
        cone_values[source.down] += down;
        cone_values[source.one] += one;
    }
    double bound_terms = 0;
    for (int j = 0; j < n; ++j) {
        upper_prices[j] = upper[j]<0?0:std::max(0.0, solved.values[upper[j]]);
        if (lower[j] >= 0) lower_prices[j] = std::max(0.0, solved.values[lower[j]]);
        bound_terms += -primal.variables[j].upper * upper_prices[j] + primal.variables[j].lower * lower_prices[j];
        residual[j] = -upper_prices[j] + lower_prices[j] + cone_values[j] - primal.variables[j].cost;
    }
    for (int i = 0; i < m; ++i)
        for (auto [j, coefficient] : primal.rows[i].coefficients) residual[j] += coefficient * result.prices[i];
    for (std::size_t k=0;k<f.design_fix_rows.size();++k) {
        const int j=f.design_vars[k],row=f.design_fix_rows[k];
        // Reconstruct the affine recourse coefficient c_x-A_x^T*pi-z_x.
        // A downward rounding preserves the certificate for nonnegative x.
        const double coefficient=-residual[j];
        result.prices[row]=coefficient==0?0:std::nextafter(coefficient,-INFINITY);
        residual[j]+=result.prices[row];
    }
    for (int j = 0; j < n; ++j) {
        double violation = std::max(0.0, residual[j]);
        result.stationarity_error = std::max(result.stationarity_error, violation);
        result.correction += primal.variables[j].upper * violation;
    }
    result.missing_service.resize(paths.service_paths.size(), 0);
    result.missing_empty.resize(paths.empty_paths.size(), 0);
    for (const auto& entry : missing) {
        double price = std::max(0.0, solved.values[entry.variable]);
        double violation = -entry.cost;
        for (auto [row, coefficient] : entry.column) violation += coefficient * result.prices[row];
        price = std::max(price, violation);
        if (price > 0) price = std::nextafter(price, INFINITY);
        (entry.empty ? result.missing_empty : result.missing_service)[entry.index] = price;
        bound_terms -= entry.upper * price;
    }
    double rhs_value = 0;
    result.raw_anchor=primal.constant;
    result.raw_core=original_dual_constant;
    for(std::size_t j=0;j<original_dual_cost.size();++j)result.raw_core+=original_dual_cost[j]*solved.values[j];
    for(std::size_t k=0;k<fixed_anchor.size();++k) result.raw_anchor+=primal.variables[f.design_vars[k]].cost*fixed_anchor[k];
    for(int i=0;i<m;++i) if(pi[i]>=0) {
        rhs_value+=primal.rows[i].rhs*result.prices[i];
        result.raw_anchor+=anchor_rhs[i]*solved.values[pi[i]];
    }
    for(int j=0;j<n;++j) if(upper[j]>=0) {
        result.raw_anchor-=primal.variables[j].upper*solved.values[upper[j]];
        if(lower[j]>=0)result.raw_anchor+=primal.variables[j].lower*solved.values[lower[j]];
    }
    for(std::size_t k=0;k<cones.size();++k) {
        const auto& c=primal.cones[k];const std::array<int,3> js{c.up,c.down,c.one};
        for(int q=0;q<3;++q) if(fixed_position[js[q]]>=0) result.raw_anchor-=fixed_anchor[fixed_position[js[q]]]*solved.values[cones[k][q]];
    }
    for(const auto& entry:missing) result.raw_anchor-=entry.upper*solved.values[entry.variable];
    result.intercept=primal.constant+rhs_value+bound_terms;
    result.coefficients.resize(f.design_fix_rows.size());
    double coefficient_norm=0;
    for(std::size_t k=0;k<f.design_fix_rows.size();++k) {
        result.coefficients[k]=result.prices[f.design_fix_rows[k]];
        coefficient_norm+=std::abs(result.coefficients[k]);
    }
    result.correction += 1e-8 * (1 + courier_cost(d)) + 128 * std::numeric_limits<double>::epsilon()
                         * (1 + std::abs(rhs_value) + std::abs(bound_terms) + coefficient_norm);
    double residual_correction=0;
    for(int j=0;j<n;++j)
        residual_correction+=primal.variables[j].upper*std::max(0.0,residual[j]);
    result.diagnostics["correction_components"]={{"bound_weighted_residual",residual_correction},
        {"base_margin",1e-8*(1+courier_cost(d))},
        {"rounding_margin",128*std::numeric_limits<double>::epsilon()*
            (1+std::abs(rhs_value)+std::abs(bound_terms)+coefficient_norm)},
        {"coefficient_norm",coefficient_norm},{"rhs_value",rhs_value},{"bound_terms",bound_terms}};
    result.intercept -= result.correction;
    result.valid = std::isfinite(result.intercept) && std::isfinite(result.correction)
                   && std::all_of(result.coefficients.begin(), result.coefficients.end(), [](double x) { return std::isfinite(x); });
    if(minimum_norm_recovery && result.valid) {
        const double q=objective_value(primal,*primal_anchor);
        const double tolerance=options.cg_duality_tolerance*(1+std::abs(q));
        const double anchor=value(result,fixed_anchor);
        const double certified_gap=q-anchor;
        const bool gap_passed=std::isfinite(certified_gap) &&
            certified_gap>=-tolerance && certified_gap<=tolerance;
        result.diagnostics["minimum_norm_solver_optimal"]=solved.optimal;
        result.diagnostics["original_rmp_duality_gap"]=certified_gap;
        result.diagnostics["original_rmp_duality_tolerance"]=tolerance;
        result.diagnostics["original_rmp_gap_passed"]=gap_passed;
        result.optimal=solved.optimal && gap_passed;
        result.status="MinimumNormConeRay_"+solved.status;
    }
    result.diagnostics["certificate_seconds"] = wall_time() - certificate_started;
    return finish_dual(!result.valid ? "invalid_dual_certificate" :
                       result.optimal ? "dual_optimal" : "dual_not_optimal");
}

// The final reduced-cost residual is converted into a valid full-model dual
// correction by the label oracle. Its bound includes every omitted column,
// including labels discarded by a proved dominance relation.
bool correct_missing_columns(Cut& cut, const LabelPricingResult& pricing) {
    if (!cut.valid || !pricing.complete || pricing.discovery_only ||
        !std::isfinite(pricing.minimum) || !std::isfinite(pricing.min_service) ||
        !std::isfinite(pricing.min_empty) || !std::isfinite(pricing.correction) || pricing.correction < 0) return false;
    const double corrected_intercept = cut.intercept - pricing.correction;
    const double corrected_total = cut.correction + pricing.correction;
    if (!std::isfinite(corrected_intercept) || !std::isfinite(corrected_total)) return false;
    cut.intercept = corrected_intercept;
    cut.correction = corrected_total;
    return true;
}
bool certify(Cut& cut, const LabelPricingResult& pricing, double tolerance) {
    if (!cut.optimal || pricing.minimum < -tolerance || pricing.negative_paths != 0 ||
        !pricing.service.empty() || !pricing.empty.empty()) return false;
    return correct_missing_columns(cut, pricing);
}

Json pricing_record(const LabelPricingResult& pricing) {
    return {{"complete", pricing.complete}, {"minimum_reduced_cost", pricing.minimum},
            {"discovery_only",pricing.discovery_only},
            {"minimum_service_reduced_cost", pricing.min_service}, {"minimum_empty_reduced_cost", pricing.min_empty},
            {"dual_correction", pricing.correction},
            {"fast_pricing_used",pricing.fast_pricing_used},
            {"catalogue_pricing_used",pricing.catalogue_pricing_used},{"catalogue_paths_scanned",pricing.catalogue_paths_scanned},
            {"diverse_pricing_used",pricing.diverse_pricing_used},{"diversity_candidates",pricing.diversity_candidates},
            {"diversity_groups",pricing.diversity_groups},{"branch_time_overlaps_pricing",pricing.catalogue_pricing_used},
            {"catalogue_path_reuses",pricing.catalogue_path_reuses},
            {"physical_path_builds",pricing.physical_path_builds},
            {"old_service_correction",pricing.old_service_correction},
            {"branch_service_correction",pricing.branch_service_correction},
            {"chosen_service_correction",pricing.chosen_service_correction},
            {"empty_correction",pricing.empty_correction},
            {"branch_correction_requested",pricing.branch_correction_requested},
            {"branch_correction_complete",pricing.branch_correction_complete},
            {"branch_correction_used",pricing.branch_correction_used},
            {"branch_correction_status",pricing.branch_correction_status},
            {"branch_correction_scope",pricing.branch_correction_scope},
            {"branch_seconds",pricing.branch_seconds},
            {"branch_paths_scanned",pricing.branch_paths_scanned},{"branch_groups",pricing.branch_groups},
            {"generated_service", pricing.service.size()}, {"generated_empty", pricing.empty.size()},
            {"labels_created", pricing.labels_created}, {"labels_extended", pricing.labels_extended},
            {"labels_dominated", pricing.labels_dominated}, {"terminal_paths", pricing.terminal_paths},
            {"omitted_paths", pricing.omitted_paths}, {"negative_paths", pricing.negative_paths},
            {"resource_pruned", pricing.resource_pruned}, {"excluded_paths", pricing.excluded_paths},
            {"dominance", pricing.dominance}};
}

Subproblem subproblem(const Input& d, PathCollection& paths, const std::vector<double>& design,
                      const SolverOptions& options, double deadline, const std::vector<int>& initial_p,
                      const std::vector<int>& initial_e, const std::vector<double>* core,
                      const std::function<void(const Json&)>& retain_incumbent = {},
                      bool fractional = false, bool free_design = false,
                      bool robust_continuous_recovery = false,
                      int face_head = -1, int face_state = -1,
                      double face_no_gain_upper = std::numeric_limits<double>::quiet_NaN(),
                      const Json* retained_primal = nullptr,
                      LabelPricingGraphCache* graph_cache = nullptr,
                      SubproblemStateCache* state_cache = nullptr) {
    double begin = wall_time(), phase = begin;
    Subproblem result;
    std::vector<std::pair<int,double>> feedback_negative;
    if (!free_design) validate_design(d, design, fractional);
    if (free_design && (!fractional || core))
        throw std::invalid_argument("Free-design CG requires continuous variables and no Pareto anchor");
    if (face_head>=0 && (!free_design || face_head>=d.assignment_entities() || face_state<0 || face_state>d.n_l || d.n_l<=0))
        throw std::invalid_argument("Design-face CG requires one valid task state and free designs");
    const char* free_scope=face_head>=0?"complete_domain_head_design_face":continuous_scope(options);
    const bool continuous_recovery = free_design || robust_continuous_recovery;
    if (continuous_recovery && (!fractional || core))
        throw std::invalid_argument("Continuous recovery requires a continuous RMP and no Pareto anchor");
    const bool conditional = options.conditional_demand && !fractional && !free_design;
    result.conditional_demand = conditional;
    // All fractional/free-design strengthening continues to use the original
    // lifted relaxation. An integer-recourse core must stay on its alpha face.
    std::vector<double> face_core;
    if (conditional && core) {
        face_core = *core;
        const DesignLayout layout(d);
        for (int j=layout.assignment;j<layout.airline;++j) face_core[j]=design[j];
        for (int r=0;r<d.assignment_entities();++r) for (int l=0;l<d.n_l;++l) for (int m=0;m<d.assignment_destinations();++m)
            if (design[layout.assignment_index(r,l,m)]==1) {
                face_core[layout.facility+l]=1;
                if (!d.origin_assignment()) face_core[layout.facility+m]=1;
            }
        validate_design(d,face_core,true);
        core = &face_core;
    }
    const bool persist=state_cache && (options.persist_recourse_model || options.reuse_recourse_rhs) && options.column_generation &&
                       !fractional && !free_design && face_head<0;
    const double graph_started=wall_time();
    if(options.progress) options.progress->record({{"event","phase_start"},{"stage","label_graph_construction"},
        {"fractional_design",fractional}});
    LabelPricingOracle oracle(d,paths,deadline,options.cache_pricing_graph?graph_cache:nullptr,
                              options.catalogue_pricing,persist);
    if(options.progress) options.progress->record({{"event","phase_end"},{"stage","label_graph_construction"},
        {"seconds",wall_time()-graph_started},{"fractional_design",fractional},{"cache_hit",oracle.graph_reused()}});
    std::unique_ptr<RecourseModelState> state;
    bool model_reused=false,rhs_reused=false,reset_profile=false;
    if(persist) {
        const std::string& domain=oracle.domain_identity();
        const std::string identity=recourse_options_identity(options);
        auto& old=state_cache->state;
        std::string reason="first_model";
        if(old) {
            if(old->input!=&d || old->paths!=&paths || old->domain_identity!=domain)reason="input_or_support_changed";
            else if(old->design!=design && !(options.reuse_recourse_rhs && d.preprocessed_demand && !conditional))reason="design_changed";
            else if(old->options_identity!=identity || old->progress!=options.progress.get())reason="options_changed";
            else if(!recourse_columns_unchanged(*old,paths))reason="registered_columns_changed";
            else {
                rhs_reused=old->design!=design;state=std::move(old);model_reused=true;++state_cache->hits;
                if(rhs_reused)++state_cache->rhs_updates;
            }
        }
        if(!state) {
            old.reset();
            state=std::make_unique<RecourseModelState>();
            state->input=&d;state->paths=&paths;state->progress=options.progress.get();
            state->domain_identity=domain;state->options_identity=identity;state->design=design;
            state_cache->last_rebuild_reason=reason;
            ++state_cache->builds;
        }
    } else state=std::make_unique<RecourseModelState>();
    if(!model_reused)state->formulation=std::make_unique<Formulation>(build_formulation(d, paths,
        free_design ? nullptr : &design,options.column_generation ? &initial_p : nullptr,
        options.column_generation ? &initial_e : nullptr, fractional, conditional, options.tight_airline_bounds,
        formulation_strengthening(options)));
    auto& f=*state->formulation;
    if(rhs_reused) {
        // Only the ordinary preprocessed fixed-design rows depend on this x.
        // Update both representations, then discard every point/price/profile.
        for(std::size_t j=0;j<design.size();++j) {
            auto& row=f.matrix.rows.at(f.design_fix_rows.at(j));row.rhs=design[j];
            state->solver->set_rhs(f.design_fix_rows[j],row.sense,row.rhs);
        }
        reset_profile=state->active_primal_profile!=0;
        state->design=design;state->price_center.clear();state->pricing_round=0;
        state->pricing_control=AdaptivePricingControl{};state->recent_service.clear();state->recent_empty.clear();
        state->previous_pricing_seconds=0;
        state->active_primal_profile=0;state->primal_retries_for_columns=0;
        state->primal_polish_attempted_for_columns=false;state->primal_retry_reason.clear();
    }
    if (free_design) for (auto& variable : f.matrix.variables) variable.binary = false;
    if (face_head>=0) for(int l=0;l<d.n_l;++l)
        // Keep these ordinary equality rows in the native dual. Registering
        // them as all-design fixes would incorrectly eliminate free costs.
        f.matrix.row("head_design_face_"+std::to_string(l),'E',face_state==l+1?1.0:0.0,
                     {{f.layout.assignment_index(face_head,l),1}});
    // Relative complete recourse: for every admissible fixed binary design,
    // courier=demand, rho=head=flow=fleet=idle=0 and cone variables equal
    // their opening indicator (one in the baseline formulation) is
    // feasible. Thus the empty-column RMP needs neither artificials nor Phase I.
    std::vector<double> baseline(f.matrix.variables.size(), 0);
    for (std::size_t j = 0; j < baseline.size(); ++j) baseline[j] = f.matrix.variables[j].lower;
    for (std::size_t j = 0; j < design.size(); ++j) baseline[f.design_vars[j]] = design[j];
    if (face_head>=0 && face_state>0) {
        baseline[f.layout.assignment_index(face_head,face_state-1)]=1;
        baseline[f.layout.facility+face_state-1]=1;
    }
    for (int r = 0; r < d.n_od; ++r) baseline[f.courier[r]] = d.demand[r];
    for (std::size_t l=0; l<f.matrix.cones.size(); ++l) {
        const auto& cone=f.matrix.cones[l];
        const double indicator=f.strengthening.queue_perspective ? baseline[f.layout.facility+l] : 1;
        baseline[cone.up] = baseline[cone.down] = baseline[cone.one] = indicator;
    }
    if (!feasible_point(f.matrix, baseline)) throw std::runtime_error("All-courier RMP feasibility construction failed");
    if (!fractional) {
        result.solution = solution_json(d, paths, f, baseline);
        result.solution["native_model_validated"] = true;
        // Only an in-memory point of this unchanged fixed design is carried
        // between CG slices. Its physical domain is unchanged; no disk import
        // or restricted lower bound is involved.
        if(retained_primal && !retained_primal->is_null()) {
            if(!retained_primal->value("native_model_validated",false) ||
               solution_design(d,*retained_primal)!=design)
                throw std::invalid_argument("CG resume primal has a different fixed design");
            if(retained_primal->at("objective").get<double>()<result.solution.at("objective").get<double>())
                result.solution=*retained_primal;
        }
    }
    const char* primal_role = face_head>=0 ? "design_face_rmp" : free_design ? "continuous_rmp" : fractional ? "root_rmp" : "integer_rmp";
    auto& solver=state->solver;
    if(!model_reused || reset_profile)solver=std::make_unique<NativeSolver>(f.matrix, options, primal_role);
    if(model_reused) {
        // An external seed can have grown while this design was pending.
        // Appending its absent columns preserves the fixed-design matrix.
        const auto before=f.matrix.variables.size();
        add_columns(f,*solver,d,paths,initial_p,initial_e,options.batch_columns);
        if(f.matrix.variables.size()!=before) {
            if(state->active_primal_profile!=0)
                solver=std::make_unique<NativeSolver>(f.matrix,options,"integer_rmp_resume_profile_reset");
            state->active_primal_profile=0;state->primal_retries_for_columns=0;
            state->primal_polish_attempted_for_columns=false;state->primal_retry_reason.clear();
        }
    }
    add_time(result.timings, "build", phase);
    LabelPricingOptions pricing_options;
    pricing_options.deadline = deadline;
    pricing_options.tolerance = options.pricing_tolerance;
    pricing_options.max_columns = options.cg_max_columns;
    pricing_options.branch_correction = options.branch_pricing_correction;
    pricing_options.fast_pricing = options.fast_pricing;
    pricing_options.catalogue_pricing=options.catalogue_pricing;
    pricing_options.diverse_pricing=options.diverse_pricing;
    state->pricing_control.initialize(options.pricing_smoothing_weight,options.pricing_exact_interval,options.cg_max_columns);
    auto& primal_retries_for_columns=state->primal_retries_for_columns;
    auto& active_primal_profile=state->active_primal_profile;
    auto& primal_polish_attempted_for_columns=state->primal_polish_attempted_for_columns;
    auto& primal_retry_reason=state->primal_retry_reason;
    if(options.stabilized_pricing && (!std::isfinite(options.pricing_smoothing_weight) ||
        options.pricing_smoothing_weight<=0 || options.pricing_smoothing_weight>1 || options.pricing_exact_interval<1))
        throw std::invalid_argument("Invalid stabilized-pricing parameters");
    auto primal_qcp_tolerance = [](int profile) {
        return profile == 0 ? 1e-9 : profile == 1 ? 1e-10 : profile == 2 ? 1e-11 : 1e-12;
    };
    auto tighter_primal = [&](int profile) {
        auto recovery=std::make_unique<NativeSolver>(f.matrix,options,
            free_design ? "continuous_rmp_recovery" : fractional ? "root_rmp_recovery" : "integer_rmp_recovery");
        recovery->cplex.setParam(IloCplex::Param::Emphasis::Numerical,true);
        recovery->cplex.setParam(IloCplex::Param::Barrier::QCPConvergeTol,primal_qcp_tolerance(profile));
        recovery->cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility,1e-9);
        recovery->cplex.setParam(IloCplex::Param::Simplex::Tolerances::Optimality,1e-9);
        recovery->cplex.setParam(IloCplex::Param::Read::Scale,profile==1?1:profile==2?-1:0);
        if (profile==3) recovery->cplex.setParam(IloCplex::Param::Preprocessing::Presolve,false);
        return recovery;
    };
    try {
    while (remaining(deadline) > 0) {
        ++result.iterations;
        Json iteration = {{"iteration", result.iterations}, {"service_columns", f.service_indices.size()},
                          {"recourse_model_reused",model_reused},{"cross_design_rhs_reused",rhs_reused},{"pricing_graph_reused",oracle.graph_reused()},
                          {"demand_formulation",d.preprocessed_demand?"preprocessed":conditional?"conditional":"lifted"},
                          {"structural_strengthening",f.strengthening_metadata},
                          {"empty_columns", f.empty_indices.size()}, {"certified", false},
                          {"model_size",{{"variables",f.matrix.variables.size()},{"linear_rows",f.matrix.rows.size()},
                              {"cones",f.matrix.cones.size()}}}};
        auto finish = [&](const std::string& status) {
            iteration["termination_reason"] = status;
            iteration["elapsed_seconds"] = options.progress ? options.progress->elapsed() : wall_time()-begin;
            iteration["subproblem_elapsed_seconds"] = wall_time()-begin;
            if (options.progress) {
                Json event=iteration;
                event["event"]="cg_iteration";
                event["stage"]=primal_role;
                event["bound_scope"]=free_design ? free_scope : "fixed_design_recourse";
                options.progress->record(std::move(event));
            }
            result.trace.push_back(iteration);
            result.reason = status;
        };
        phase = wall_time();
        const bool certificate_retry = !primal_retry_reason.empty();
        if (certificate_retry) iteration["primal_retry_reason"] = primal_retry_reason;
        primal_retry_reason.clear();
        auto solved = solver->optimize(certificate_retry ? std::min(20.0,.25*remaining(deadline)) : remaining(deadline));
        iteration["primal_attempts"]=Json::array();
        std::optional<PointAssessment> point_assessment;
        const std::vector<double>* assessed_values=nullptr;
        auto invalidate_assessment=[&](){point_assessment.reset();assessed_values=nullptr;};
        auto assessment=[&](const SolveOutcome& point)->const PointAssessment& {
            if(!point_assessment || assessed_values!=&point.values) {
                point_assessment=assess_point(f.matrix,point.values);assessed_values=&point.values;
            }
            return *point_assessment;
        };
        auto point_feasible=[&](const SolveOutcome& point) {
            return options.streamlined_recourse?assessment(point).feasible:feasible_point(f.matrix,point.values);
        };
        auto point_diagnostics=[&](const SolveOutcome& point) {
            return options.streamlined_recourse?assessment(point).diagnostics():feasibility_diagnostics(f.matrix,point.values);
        };
        auto record_primal=[&](int profile,const SolveOutcome& point) {
            Json record={{"numerical_profile",profile},{"status",point.status},{"optimal",point.optimal},
                         {"qcp_convergence_tolerance",primal_qcp_tolerance(profile)},
                         {"has_solution",point.has_solution},{"candidate_available",point.candidate_available},
                         {"candidate_extraction_error",point.candidate_extraction_error.empty() ? Json(nullptr) : Json(point.candidate_extraction_error)},
                         {"seconds",point.seconds},{"objective",point.has_solution?Json(point.objective):Json(nullptr)}};
            if (point.has_solution || point.candidate_available) {
                record["residuals"]=point_diagnostics(point);
                record["candidate_objective"]=point.objective;
            }
            iteration["primal_attempts"].push_back(std::move(record));
        };
        record_primal(active_primal_profile,solved);
        iteration["original_rmp_solver_status"] = solved.status;
        iteration["original_rmp_solver_optimal"] = solved.optimal;
        iteration["original_rmp_solver_has_solution"] = solved.has_solution;
        bool polished_primal = false;
        auto recover_candidate = [&](SolveOutcome& point) {
            if (!continuous_recovery || (!point.has_solution && !point.candidate_available)) return false;
            if (!point.has_solution && point_feasible(point)) {
                point.has_solution = true;
                point.optimal = false;
                return true;
            }
            if (primal_polish_attempted_for_columns || point_feasible(point) ||
                remaining(deadline) <= 0) return false;
            primal_polish_attempted_for_columns = true;
            auto polished = point.values;
            iteration["primal_polish"] = polish_conic_primal(f.matrix, options,
                std::min(20.0, .25 * remaining(deadline)), polished);
            if (iteration.at("primal_polish").at("applied").get<bool>() && feasible_point(f.matrix, polished)) {
                point.values = std::move(polished);invalidate_assessment();
                point.objective = objective_value(f.matrix, point.values);
                // An optimal restricted LP supplies only an original-model
                // feasible point. Its QCP optimality must be established below
                // with the independent original conic dual, never inherited.
                point.has_solution = true;
                point.optimal = false;
                polished_primal = true;
                return true;
            }
            return false;
        };
        bool recovered_primal = recover_candidate(solved);
        // Sparse initial RMPs can be highly degenerate. Retry the unchanged
        // model after fresh extraction when CPLEX's returned point fails our
        // original residual tolerances; a solver status alone is insufficient.
        while (primal_retries_for_columns < 3) {
            if (recovered_primal) break;
            if (solved.has_solution && solved.optimal && point_feasible(solved)) break;
            if (remaining(deadline)<=0) break;
            const int profile = ++primal_retries_for_columns;
            auto recovery=tighter_primal(profile);
            auto candidate=recovery->optimize(std::min(20.0,.25*remaining(deadline)));
            // A rejected local candidate may reuse the same stack address on
            // the next recovery attempt, but it is a new optimizer point.
            invalidate_assessment();
            record_primal(profile,candidate);
            const bool candidate_recovered = recover_candidate(candidate);
            if (candidate.has_solution && point_feasible(candidate)) {
                solved=std::move(candidate); invalidate_assessment(); solver=std::move(recovery);
                active_primal_profile=profile;
                recovered_primal=candidate_recovered;
            }
        }
        iteration["same_column_primal_retries"] = primal_retries_for_columns;
        iteration["rmp_seconds"] = wall_time() - phase;
        iteration["rmp_status"] = solved.status;
        iteration["rmp_optimal"] = solved.optimal;
        iteration["rmp_objective"] = solved.has_solution ? Json(solved.objective) : Json(nullptr);
        add_time(result.timings, "primal", phase);
        if (!solved.has_solution) { finish("recourse_" + solved.status); break; }
        phase = wall_time();
        bool feasible = point_feasible(solved);
        iteration["rmp_feasible"] = feasible;
        iteration["rmp_residuals"]=point_diagnostics(solved);
        const double recomputed_objective = objective_value(f.matrix, solved.values);
        const bool objective_matches = std::isfinite(recomputed_objective) &&
            std::abs(recomputed_objective - solved.objective) <= 2e-7 * (1 + std::abs(recomputed_objective));
        iteration["rmp_recomputed_objective"] = recomputed_objective;
        iteration["rmp_objective_matches"] = objective_matches;
        add_time(result.timings, "validation", phase);
        if(options.adaptive_pricing && feasible) {
            std::size_t positive=0;
            for(int id:state->recent_service)if(f.flow.at(id)>=0 && solved.values[f.flow[id]]>1e-9)++positive;
            for(int id:state->recent_empty)if(f.empty_flow.at(id)>=0 && solved.values[f.empty_flow[id]]>1e-9)++positive;
            state->pricing_control.participation(state->recent_service.size()+state->recent_empty.size(),positive,
                iteration.at("rmp_seconds").get<double>(),state->previous_pricing_seconds);
            state->recent_service.clear();state->recent_empty.clear();
        }
        if (!feasible) { finish("primal_validation_failed"); break; }
        if (!objective_matches) { finish("primal_objective_validation_failed"); break; }
        phase = wall_time();
        Json trial = nullptr;
        if (!fractional) {
            trial = solution_json(d, paths, f, solved.values);
            trial["native_model_validated"] = true;
        }
        add_time(result.timings, "read_solution", phase);
        auto accept_primal = [&]() {
            if (fractional && solved.objective < result.continuous_objective) {
                result.continuous_objective = solved.objective;
                result.continuous_residuals = iteration.at("rmp_residuals");
                result.continuous_design.clear();
                for (int j : f.design_vars)
                    result.continuous_design.push_back(std::clamp(solved.values[j], 0.0, 1.0));
                result.continuous_operating_point={{"rho",Json::array()},{"idle",Json::array()},
                    {"v",Json::array()},{"v_order",d.preprocessed_demand?"task_then_departure":d.origin_assignment()?"r_then_departure":"r_then_departure_then_return"},
                    {"assignment_scope",d.assignment_scope()},
                    {"fleet",solved.values[f.fleet]}};
                for (int j:f.rho) result.continuous_operating_point["rho"].push_back(solved.values[j]);
                for (int j:f.idle) result.continuous_operating_point["idle"].push_back(solved.values[j]);
                for (std::size_t j=0;j<f.matrix.variables.size();++j)
                    if (f.matrix.variables[j].name.rfind("v_",0)==0)
                        result.continuous_operating_point["v"].push_back(solved.values[j]);
            }
            if (trial.is_null()) return;
            const double accept_started = wall_time();
            if (trial["objective"].get<double>() < result.solution["objective"].get<double>())
                result.solution = std::move(trial);
            // Promote the verified original-model point before any optional
            // dual, pricing, model-update or output work can exhaust its budget.
            if (retain_incumbent) retain_incumbent(result.solution);
            if (options.progress) options.progress->checkpoint(result.solution,"integer_rmp_feasible");
            add_time(result.timings, "read_solution", accept_started);
        };
        // Any validated RMP feasible point is also feasible in this complete
        // continuous face: missing columns are simply zero. If its objective
        // is already below the certified integer floor, this state cannot
        // strengthen max(global_floor, face_LB). It supplies no new LB.
        const double face_upper_margin=2e-7*(1+std::abs(recomputed_objective));
        const double face_upper=std::max(recomputed_objective,solved.objective)+face_upper_margin;
        if(face_head>=0 && options.cg_seed_reuse && std::isfinite(face_no_gain_upper) &&
           std::isfinite(face_upper) && face_upper<=face_no_gain_upper) {
            accept_primal();
            iteration["validated_face_upper"]=face_upper;
            iteration["face_upper_numerical_margin"]=face_upper_margin;
            iteration["caller_global_lower"]=face_no_gain_upper;
            iteration["direct_global_bound_update"]=false;
            finish("face_cannot_strengthen_global_floor");break;
        }
        // A feasible time-limited RMP remains a primal incumbent, but its
        // multipliers cannot be used as the optimal RMP prices required by CG.
        if (!solved.optimal && !recovered_primal) { accept_primal(); finish("rmp_not_optimal_" + solved.status); break; }
        const double q = solved.objective;
        const double tight_tolerance = options.cg_duality_tolerance * (1 + std::abs(q));
        phase = wall_time();
        Cut dual=read_cplex_dual(d,paths,f,*solver,q,options);
        const double candidate_gap=dual.valid?q-value(dual,design):INFINITY;
        const bool primal_dual_contradiction=dual.valid && std::isfinite(candidate_gap) &&
            candidate_gap < -tight_tolerance;
        iteration["dual_profile_order"]=Json::array();
        iteration["dual_attempts"]=Json::array({{{"numerical_profile",-1},{"method",dual.method},
            {"status",dual.status},{"optimal",dual.optimal},{"valid",dual.valid},
            {"seconds",wall_time()-phase},{"auxiliary_optimization_calls",0},
            {"corrected_gap",std::isfinite(candidate_gap)?Json(candidate_gap):Json(nullptr)},
            {"correction",dual.correction},{"stationarity_error",dual.stationarity_error},
            {"diagnostics",dual.diagnostics}}});
        iteration["dual_seconds"] = wall_time() - phase;
        iteration["dual_status"] = dual.status;
        iteration["dual_certificate_method"] = dual.method;
        iteration["dual_optimal"] = dual.optimal;
        iteration["dual_diagnostics"] = dual.diagnostics;
        add_time(result.timings, "dual", phase);
        if (primal_dual_contradiction) {
            iteration["primal_rejected_due_to_duality"] = true;
            iteration["rmp_duality_gap"] = q - value(dual, design);
            iteration["duality_tolerance"] = tight_tolerance;
            if (primal_retries_for_columns < 3 && remaining(deadline) > 0) {
                active_primal_profile = ++primal_retries_for_columns;
                primal_retry_reason = "valid_dual_exceeds_primal_objective";
                iteration["next_primal_numerical_profile"] = active_primal_profile;
                finish("primal_reoptimization_required");
                const double rebuild_started = wall_time();
                solver = tighter_primal(active_primal_profile);
                add_time(result.timings, "build", rebuild_started);
                continue;
            }
            finish(remaining(deadline) <= 0 ? "time_limit" : "primal_dual_contradiction_retry_limit");
            break;
        }
        accept_primal();
        if((!dual.valid || (!dual.optimal && !options.weak_dual_cuts)) && primal_retries_for_columns<3 && remaining(deadline)>0) {
            // Retry the ORIGINAL SOCP. There is no separately optimized dual
            // or certificate-LP fallback, even when native extraction fails.
            active_primal_profile=++primal_retries_for_columns;
            primal_retry_reason=dual.valid?"native_dual_gap_failed":"native_dual_unavailable";
            iteration["next_primal_numerical_profile"]=active_primal_profile;
            finish("native_dual_requires_primal_reoptimization");
            const double rebuild_started=wall_time();
            solver=tighter_primal(active_primal_profile);
            add_time(result.timings,"build",rebuild_started);
            continue;
        }
        if (!dual.valid) { finish(remaining(deadline) <= 0 ? "time_limit" : "native_dual_unavailable"); break; }
        const double rmp_gap = q - value(dual, design);
        iteration["dual_objective"] = value(dual, design);
        iteration["raw_dual_objective"] = dual.raw_anchor;
        iteration["dual_stationarity_error"] = dual.stationarity_error;
        iteration["dual_numerical_correction"] = dual.correction;
        iteration["rmp_duality_gap"] = rmp_gap;
        iteration["duality_tolerance"] = tight_tolerance;
        if (!dual.optimal && !options.weak_dual_cuts) { finish("dual_not_optimal_" + dual.status); break; }
        if (rmp_gap < -tight_tolerance || (rmp_gap > tight_tolerance && !options.weak_dual_cuts)) { finish("rmp_duality_gap_failed"); break; }
        iteration["rmp_optimal"] = dual.optimal;
        iteration["rmp_optimality_certificate"] = {{"method","original_conic_primal_dual_gap"},
            {"primal_source",polished_primal ? "cone_restriction_lp_primal_only" :
                             recovered_primal ? "validated_numerical_solver_candidate" : "original_qcp"},
            {"primal_feasible",true},{"dual_feasible",true},{"dual_certificate_tight",dual.optimal},{"auxiliary_dual_optimization",false},
            {"gap",rmp_gap},{"tolerance",tight_tolerance}};
        phase = wall_time();
        LabelPricingResult pricing;
        bool exact_pricing=true;
        if(options.progress) options.progress->record({{"event","phase_start"},{"stage","ordinary_label_pricing"},
            {"iteration",result.iterations},{"fractional_design",fractional}});
        if (options.column_generation) {
            ++state->pricing_round;
            const bool has_center=state->price_center.size()==dual.prices.size();
            const double smoothing=options.adaptive_pricing?state->pricing_control.weight:options.pricing_smoothing_weight;
            const bool due=options.adaptive_pricing?state->pricing_control.exact_due(remaining(deadline)):
                (state->pricing_round-1)%static_cast<std::size_t>(options.pricing_exact_interval)==0;
            const bool discover=options.stabilized_pricing && has_center && !due;
            if(options.adaptive_pricing)pricing_options.max_columns=state->pricing_control.batch;
            if(options.stabilized_pricing) {
                if(!has_center)state->price_center=dual.prices;
                else for(std::size_t j=0;j<dual.prices.size();++j)
                    state->price_center[j]=(1-smoothing)*state->price_center[j]+smoothing*dual.prices[j];
            }
            if(discover) {
                auto discovery_options=pricing_options;
                discovery_options.branch_correction=false;
                const auto discovery=oracle.price(f,state->price_center,{}, {},discovery_options);
                pricing=reprice_discovered_columns(f,paths,dual.prices,discovery,options.pricing_tolerance);
                iteration["stabilized_discovery"]={{"pricing_round",state->pricing_round},
                    {"smoothing_weight",smoothing},{"search",pricing_record(discovery)},
                    {"original_dual_accepted",pricing.service.size()+pricing.empty.size()},
                    {"certificate_eligible",false}};
                exact_pricing=pricing.service.empty() && pricing.empty.empty();
                iteration["stabilized_discovery"]["exact_fallback"]=exact_pricing;
                if(options.adaptive_pricing)state->pricing_control.discovery(discovery.service.size()+discovery.empty.size(),
                    pricing.service.size()+pricing.empty.size());
            }
            if(exact_pricing) {
                const double exact_started=wall_time();
                pricing=oracle.price(f,dual.prices,{}, {},pricing_options);
                if(options.adaptive_pricing)state->pricing_control.exact(wall_time()-exact_started);
            }
            if(options.adaptive_pricing)iteration["adaptive_pricing"]={{"current_price_weight",smoothing},
                {"next_price_weight",state->pricing_control.weight},{"exact_interval",state->pricing_control.interval},
                {"batch_limit",pricing_options.max_columns},{"original_dual_pass",exact_pricing}};
            if(options.cg_primal_feedback && !fractional && !pricing.service.empty()) {
                feedback_negative.clear();
                for(int id:pricing.service) {
                    const auto& p=paths.service_paths.at(id);double rc=p.cost;
                    for_each_path_coefficient(f,p,[&](int row,double coefficient){rc-=coefficient*dual.prices.at(row);});
                    if(std::isfinite(rc) && rc < -options.pricing_tolerance)feedback_negative.emplace_back(id,rc);
                    if(feedback_negative.size()==64)break;
                }
            }
        } else pricing.complete = paths.complete; // Full recourse already contains every column.
        iteration["pricing_original_dual_complete_pass"]=exact_pricing && pricing.complete;
        iteration["pricing_seconds"] = wall_time() - phase;
        state->previous_pricing_seconds=iteration.at("pricing_seconds").get<double>();
        iteration["pricing"] = pricing_record(pricing);
        if(options.progress) options.progress->record({{"event","phase_end"},{"stage","ordinary_label_pricing"},
            {"seconds",iteration.at("pricing_seconds")},{"iteration",result.iterations},
            {"fractional_design",fractional},{"pricing",iteration.at("pricing")}});
        add_time(result.timings, "pricing", phase);
        if (exact_pricing && !pricing.complete) { finish("pricing_incomplete"); break; }
        phase = wall_time();
        if (exact_pricing && options.weak_dual_cuts && !free_design) {
            Cut full_certificate=dual;
            if (correct_missing_columns(full_certificate,pricing)) {
                full_certificate.optimal=false;
                full_certificate.diagnostics["full_domain_certificate"]={{"source_iteration",result.iterations},
                    {"pricing",pricing_record(pricing)},{"anchor_tight",false}};
                const double lower=value(full_certificate,design);
                if (lower>q+tight_tolerance) {
                    add_time(result.timings,"certificate",phase);
                    finish("full_bound_primal_dual_contradiction"); break;
                }
                iteration["full_domain_bound_valid"]=true;
                iteration["certified_recourse_lower_bound"]=lower;
                if (!result.ordinary.valid || lower>value(result.ordinary,design)) {
                    result.ordinary=full_certificate;
                    result.selected=full_certificate;
                }
            }
        }
        if (exact_pricing && free_design) {
            // Even a nonterminal complete pricing pass gives a full-domain
            // bound: stationarity is repaired for every omitted service flow
            // using sum(flow)<=total demand, and for bounded empty flows.
            // It is not the restricted primal objective and need not be tight.
            Cut full_certificate = dual;
            if (correct_missing_columns(full_certificate, pricing)) {
                const double lower = full_certificate.intercept;
                const double primal_tolerance = options.cg_duality_tolerance *
                    (1 + std::abs(result.continuous_objective));
                if (std::isfinite(result.continuous_objective) &&
                    lower > result.continuous_objective + primal_tolerance) {
                    iteration["rejected_full_lower_bound"] = lower;
                    add_time(result.timings, "certificate", phase);
                    finish("full_bound_primal_dual_contradiction"); break;
                }
                iteration["certified_full_lower_bound"] = lower;
                iteration["full_lower_bound_correction"] = full_certificate.correction;
                if (!result.full_bound_valid || lower > result.certified_lower_bound) {
                    result.full_bound_valid = true;
                    result.certified_lower_bound = lower;
                    result.lower_bound_certificate = {{"method","native_cplex_dual_with_complete_label_correction"},
                        {"iteration",result.iterations},{"rmp_dual_objective",dual.intercept},
                        {"pricing_correction",pricing.correction},{"total_correction",full_certificate.correction},
                        {"stationarity_error",full_certificate.stationarity_error},
                        {"pricing",pricing_record(pricing)}};
                    if (options.progress) options.progress->record({{"event","continuous_cg_bound"},
                        {"bound_scope",free_scope},{"lower_bound",lower},
                        {"bound_valid",true},{"integer_incumbent",false},
                        {"certificate",result.lower_bound_certificate}});
                }
            }
        }
        if(options.adaptive_pricing && exact_pricing) {
            if(free_design && result.full_bound_valid)state->pricing_control.certificate(result.certified_lower_bound);
            else if(!free_design && result.ordinary.valid)state->pricing_control.certificate(value(result.ordinary,design));
        }
        bool complete = exact_pricing && certify(dual, pricing, options.pricing_tolerance);
        add_time(result.timings, "certificate", phase);
        if (complete) {
            const double final_gap = q - value(dual, design);
            iteration["full_dual_objective"] = value(dual, design);
            iteration["full_duality_gap"] = final_gap;
            if (final_gap < -tight_tolerance || final_gap > tight_tolerance) {
                finish("full_duality_gap_failed"); break;
            }
            result.ordinary = dual;
            result.selected = dual;
            result.recourse_optimal = true;
            result.ordinary.diagnostics["full_domain_certificate"]={{"source_iteration",result.iterations},
                {"pricing",pricing_record(pricing)},{"anchor_tight",true}};
            result.selected=result.ordinary;
            iteration["certified"] = true;
            finish("pricing_complete");
            if (core && options.pareto) {
                phase = wall_time();
                const double face = options.cg_duality_tolerance * (1 + std::abs(q));
                const double verify = 2e-7 * (1 + std::abs(q));
                double ordinary_gap = q - value(dual, design);
                result.pareto = {{"attempted", true}, {"accepted", false}, {"reason", "ordinary_anchor_not_tight"},
                                 {"ordinary_anchor_gap", ordinary_gap}, {"face_tolerance", face}, {"verification_tolerance", verify},
                                 {"separation", "exact_label_pricing"}, {"iterations", Json::array()},
                                 {"auxiliary_solves", Json::array()}, {"requested_seconds", options.pareto_call_seconds},
                                 {"allocated_seconds", 0.0}, {"deadline_remaining_seconds", remaining(deadline)},
                                 {"pricing_seconds", 0.0}, {"certificate_seconds", 0.0}, {"timeout", false}};
                if (ordinary_gap >= -verify && ordinary_gap <= face + verify && remaining(deadline) > 0) {
                    // The outer scheduler owns the total and per-call Pareto
                    // budgets. Do not silently shrink its explicit allowance.
                    const double auxiliary_started = wall_time();
                    const double auxiliary_deadline = std::min(deadline,
                        auxiliary_started + std::max(0.0, options.pareto_call_seconds));
                    const double auxiliary_budget = auxiliary_deadline - auxiliary_started;
                    result.pareto["allocated_seconds"] = auxiliary_budget;
                    result.pareto["reason"] = "auxiliary_budget_exhausted_before_build";
                    std::vector<int> separated_p, separated_e;
                    try {
                    while (remaining(auxiliary_deadline) > 0) {
                        Cut auxiliary = solve_dual(d, paths, f, options, auxiliary_deadline, core, q, face,
                                                   separated_p, separated_e);
                        result.pareto["auxiliary_solves"].push_back(auxiliary.diagnostics);
                        result.pareto["auxiliary_status"] = auxiliary.status;
                        result.pareto["reason"] = "auxiliary_" + auxiliary.diagnostics.value("termination_reason", std::string("unavailable"));
                        result.pareto["timeout"] = auxiliary.status == "AbortTimeLim" || remaining(auxiliary_deadline) <= 0;
                        if (!auxiliary.valid || !auxiliary.optimal) break;
                        auto auxiliary_options = pricing_options;
                        auxiliary_options.deadline = auxiliary_deadline;
                        auxiliary_options.excluded_service = separated_p;
                        auxiliary_options.excluded_empty = separated_e;
                        LabelPricingResult check;
                        const double auxiliary_pricing_started=wall_time();
                        if(options.progress) options.progress->record({{"event","phase_start"},{"stage","pareto_label_pricing"},
                            {"iteration",result.pareto.at("iterations").size()+1}});
                        if (options.column_generation)
                            check = oracle.price(f, auxiliary.prices, auxiliary.missing_service,
                                                 auxiliary.missing_empty, auxiliary_options);
                        else check.complete = paths.complete;
                        if(options.progress) options.progress->record({{"event","phase_end"},{"stage","pareto_label_pricing"},
                            {"seconds",wall_time()-auxiliary_pricing_started},
                            {"iteration",result.pareto.at("iterations").size()+1},{"pricing",pricing_record(check)}});
                        result.pareto["pricing_seconds"] = result.pareto.at("pricing_seconds").get<double>()
                            + wall_time() - auxiliary_pricing_started;
                        result.pareto["iterations"].push_back(pricing_record(check));
                        result.pareto["pricing_complete"] = check.complete;
                        result.pareto["reason"] = "pricing_incomplete";
                        if (!check.complete) {
                            result.pareto["timeout"] = remaining(auxiliary_deadline) <= 0;
                            break;
                        }
                        if (!check.service.empty() || !check.empty.empty()) {
                            separated_p.insert(separated_p.end(), check.service.begin(), check.service.end());
                            separated_e.insert(separated_e.end(), check.empty.begin(), check.empty.end());
                            result.pareto["reason"] = "auxiliary_separation_incomplete";
                            continue;
                        }
                        result.pareto["reason"] = "pricing_violation";
                        const double certificate_started = wall_time();
                        if (certify(auxiliary, check, options.pricing_tolerance)) {
                            // Global weak-duality validity is distinct from
                            // tightness on the MW face and core optimality.
                            // Keep a conservatively corrected valid cut even
                            // when it cannot be labelled a Pareto-style cut.
                            double anchor_gap = q - value(auxiliary, design);
                            double gain = value(auxiliary, *core) - value(dual, *core);
                            double core_loss = auxiliary.raw_core - value(auxiliary, *core);
                            const bool weak_duality_consistent = std::isfinite(anchor_gap) && anchor_gap >= -verify;
                            result.pareto["global_cut_valid"] = weak_duality_consistent;
                            if (weak_duality_consistent) result.auxiliary = auxiliary;
                            result.pareto["anchor_gap"] = anchor_gap;
                            result.pareto["core_gain"] = gain;
                            result.pareto["core_optimality_loss"] = core_loss;
                            result.pareto["anchor_check"] = anchor_gap >= -verify && anchor_gap <= face + verify;
                            result.pareto["core_gain_check"] = gain >= -verify;
                            result.pareto["core_optimality_check"] = std::abs(core_loss) <= verify;
                            result.pareto["reason"] = weak_duality_consistent
                                ? "certificate_check_failed" : "auxiliary_primal_dual_contradiction";
                            if (anchor_gap >= -verify && anchor_gap <= face + verify && gain >= -verify && std::abs(core_loss) <= verify) {
                                result.selected = std::move(auxiliary);
                                result.pareto["accepted"] = true;
                                result.pareto["reason"] = "optimal_face";
                            }
                        }
                        result.pareto["certificate_seconds"] = result.pareto.at("certificate_seconds").get<double>()
                            + wall_time() - certificate_started;
                        break;
                    }
                    if (remaining(auxiliary_deadline) <= 0 &&
                        (result.pareto.at("reason") == "auxiliary_separation_incomplete" ||
                         result.pareto.at("reason") == "auxiliary_budget_exhausted_before_build")) {
                        result.pareto["timeout"] = true;
                        result.pareto["reason"] = result.pareto.at("auxiliary_solves").empty()
                            ? "auxiliary_budget_exhausted_before_build" : "auxiliary_separation_time_limit";
                    }
                    } catch (const IloException& exception) {
                        result.pareto["reason"] = "auxiliary_solver_error";
                        result.pareto["error"] = exception.getMessage();
                    } catch (const std::exception& exception) {
                        result.pareto["reason"] = "auxiliary_error";
                        result.pareto["error"] = exception.what();
                    }
                } else if (ordinary_gap >= -verify && ordinary_gap <= face + verify) {
                    result.pareto["reason"] = "deadline_before_auxiliary";
                    result.pareto["timeout"] = true;
                }
                result.pareto["seconds"] = wall_time() - phase;
                result.pareto["dual_seconds"] = 0.0;
                for (const auto& auxiliary : result.pareto.at("auxiliary_solves"))
                    result.pareto["dual_seconds"] = result.pareto.at("dual_seconds").get<double>() + auxiliary.at("seconds").get<double>();
                if (options.progress) options.progress->record({{"event", "pareto_diagnostic"}, {"stage", "pareto"},
                    {"fractional_design", fractional}, {"diagnostics", result.pareto}});
                add_time(result.timings, "pareto", phase);
            }
            break;
        }
        phase = wall_time();
        if (pricing.service.empty() && pricing.empty.empty()) {
            finish(options.weak_dual_cuts && (result.ordinary.valid || result.full_bound_valid)
                ? "weak_dual_bound" : "pricing_stalled"); break;
        }
        add_columns(f, *solver, d, paths, pricing.service, pricing.empty, options.batch_columns);
        if(options.adaptive_pricing) {state->recent_service=pricing.service;state->recent_empty=pricing.empty;}
        if (active_primal_profile != 0) {
            // Recovery is local to the unchanged RMP. New columns can change
            // its conditioning, so never retain a restrictive recovery profile
            // for every subsequent CG solve. Fresh extraction restores all
            // NativeSolver defaults and discards the old numerical state.
            const double reset_started = wall_time();
            const int previous_profile = active_primal_profile;
            solver = std::make_unique<NativeSolver>(f.matrix, options,
                free_design ? "continuous_rmp_profile_reset" : fractional ? "root_rmp_profile_reset" : "integer_rmp_profile_reset");
            active_primal_profile = 0;
            iteration["primal_profile_reset"] = {{"from_profile",previous_profile},{"to_profile",0},
                {"reason","column_set_changed"},{"fresh_extraction",true},
                {"qcp_convergence_tolerance",primal_qcp_tolerance(0)},
                {"seconds",wall_time()-reset_started}};
        }
        primal_retries_for_columns = 0;
        primal_polish_attempted_for_columns = false;
        iteration["add_columns_seconds"] = wall_time() - phase;
        add_time(result.timings, "add_columns", phase);
        finish("columns_added");
    }
    } catch (const RunBudgetExceeded& error) {
        if (!continuous_recovery) throw;
        result.reason = "construction_time_limit";
        result.failure = {{"stage",error.stage},{"sizes",error.sizes},{"message",error.what()}};
    } catch (const IloException& error) {
        if (!continuous_recovery) throw;
        result.reason = "solver_exception";
        result.failure = {{"message",error.getMessage()}};
    } catch (const std::exception& error) {
        if (!continuous_recovery || face_head>=0) throw;
        result.reason = "exception";
        result.failure = {{"message",error.what()}};
    }
    if (result.reason == "columns_added" && remaining(deadline) <= 0) result.reason = "time_limit";
    result.service = f.service_indices;
    result.empty = f.empty_indices;
    std::size_t nonzeros=0;
    for (const auto& row:f.matrix.rows) nonzeros+=row.coefficients.size();
    result.model_size={{"variables",f.matrix.variables.size()},{"linear_rows",f.matrix.rows.size()},
                       {"cones",f.matrix.cones.size()},{"nonzeros",nonzeros}};
    if (conditional) {
        const double conditioning_started=wall_time();
        auto gate=[&](Cut& cut) {
            if (!cut.valid) return;
            const auto converted=condition_assignment_cut(cut.intercept,cut.coefficients,design,
                                                          f.layout.assignment,f.layout.airline,0.0);
            cut.intercept=converted.intercept;
            cut.coefficients=converted.coefficients;
            cut.correction+=converted.additional_correction;
            cut.condition={{"cut_kind","conditional_dual"},
                {"cut_scope","fixed_assignment_all_facility_designs"},
                {"assignment_fingerprint",converted.assignment_key},{"gate_M",converted.gate_m},
                {"affine_upper_bound",converted.affine_upper_bound},
                {"gate_rounding_margin",converted.gate_rounding_margin},
                {"conditioning_correction",converted.additional_correction},
                {"anchor_before",converted.anchor_value_before},{"anchor_after",converted.anchor_value_after}};
        };
        gate(result.ordinary); gate(result.selected); gate(result.auxiliary);
        result.pareto["scope"]="fixed_assignment_face";
        add_time(result.timings,"conditional_cut",conditioning_started);
    }
    // Pricing, including auxiliary Pareto separation, may append paths after
    // the last primal incumbent was packaged. Those new columns have zero
    // flow in that incumbent and must not alter its physical solution.
    if (!result.solution.is_null()) synchronize_result_paths(d, paths, result.solution);
    if(options.cg_primal_feedback && !fractional && !result.solution.is_null()) {
        const double feedback_started=wall_time();
        std::vector<std::pair<int,double>> positive;
        const auto& flows=result.solution.at("_solution").at("flow");
        for(int id:f.service_indices)if(flows.at(id).get<double>()>0) {
            positive.emplace_back(id,flows.at(id).get<double>());if(positive.size()==64)break;
        }
        result.primal_feedback=make_primal_cg_feedback(d,paths,design,feedback_negative,positive,64);
        add_time(result.timings,"primal_feedback",feedback_started);
    }
    if(persist) {
        const double snapshot_started=wall_time();
        auto append_payloads=[](auto& saved,const auto& ids,const auto& catalogue) {
            std::set<int> known;for(const auto& entry:saved)known.insert(entry.first);
            for(int id:ids)if(known.insert(id).second)saved.emplace_back(id,seed_path_key(catalogue.at(id)));
        };
        append_payloads(state->service_payloads,f.service_indices,paths.service_paths);
        append_payloads(state->empty_payloads,f.empty_indices,paths.empty_paths);
        state_cache->state=std::move(state);
        add_time(result.timings,"state_cache_snapshot",snapshot_started);
    }
    result.wall = wall_time() - begin;
    return result;
}

std::vector<double> initial_core(const Input& d) {
    DesignLayout layout(d);
    std::vector<double> core(layout.count, 0);
    std::fill(core.begin() + layout.facility, core.begin() + layout.capacity, 0.5);
    std::fill(core.begin() + layout.capacity, core.begin() + layout.assignment, 0.5 / d.n_levels);
    for (int r = 0; r < d.assignment_entities(); ++r)
        for (int l = 0; l < d.n_l; ++l)
            for (int m = 0; m < d.assignment_destinations(); ++m)
                core[layout.assignment_index(r,l,m)] = d.preprocessed_demand && d.task_rate(r)<=0 ? 0 : d.origin_assignment() ? .25 / (d.n_l+1) :
                    (l == m ? .5 : .25) / (d.n_l * d.n_l + 1);
    for (int u = 0; u < d.n_nodes(); ++u)
        for (int v = 0; v < d.n_nodes(); ++v)
            core[layout.airline + u * d.n_nodes() + v] = d.allowed_links(u, v) ? (u == v ? .25 : .125) : 0;
    return core;
}

std::vector<double> initial_design(const Input& d, const PathCollection& paths) {
    DesignLayout layout(d);
    std::vector<double> design(layout.count, 0);
    std::vector<double> reposition(d.n_l * d.n_l, INFINITY), gains(d.assignment_count(), 0);
    for (const auto& p : paths.empty_paths)
        reposition[p.b[0] * d.n_l + p.b[1]] = std::min(reposition[p.b[0] * d.n_l + p.b[1]], p.cost + d.fleet_cost * p.time);
    for (const auto& p : paths.service_paths) {
        if (!d.preprocessed_demand && p.mode != "single") continue;
        double cost = p.cost + d.fleet_cost * p.time + (p.b[0] == p.b[1] ? 0 : reposition[p.b[1] * d.n_l + p.b[0]]);
        const int entity=d.preprocessed_demand?d.task_index(p.first_od,p.second_od):p.first_od;
        int index = d.assignment_offset(entity,p.b[0],p.b[1]);
        const double ground=d.ground_cost[p.first_od]+(d.preprocessed_demand && p.second_od>=0?d.ground_cost[p.second_od]:0);
        gains[index] = std::max(gains[index], ground - cost);
    }
    if (d.preprocessed_demand) {
        // Each fixed-demand task is independently assigned. Reconstruction's
        // common-origin filter does not constrain execution of sibling tasks.
        std::fill(gains.begin(),gains.end(),0);
        for(const auto& p:paths.service_paths) {
            const int q=d.task_index(p.first_od,p.second_od);
            if(d.task_rate(q)<=0)continue;
            const double cost=p.cost+d.fleet_cost*p.time+
                (p.b[0]==p.b[1]?0:reposition[p.b[1]*d.n_l+p.b[0]]);
            const double ground=d.ground_cost[p.first_od]+(p.second_od<0?0:d.ground_cost[p.second_od]);
            const int index=d.assignment_offset(q,p.b[0]);
            gains[index]=std::max(gains[index],ground-cost);
        }
    }
    std::vector<int> order(gains.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        int ar = a / d.assignment_pool_count(), br = b / d.assignment_pool_count();
        double sa = gains[a] * (d.preprocessed_demand?d.task_rate(ar):d.demand[ar]),
               sb = gains[b] * (d.preprocessed_demand?d.task_rate(br):d.demand[br]);
        return std::make_tuple(-sa, a) < std::make_tuple(-sb, b);
    });
    std::vector<bool> assigned(d.assignment_entities(), false);
    // This is a primal proposal, not a restriction on the global design
    // domain. Each assigned first order must have a single route and every
    // positive-kappa continuation to the proposed assigned pool. Otherwise
    // an absent compulsory branch can force the whole group's head to zero.
    std::set<std::array<int,4>> branches;
    std::map<int,std::set<int>> prefix_branches;
    std::map<std::array<int,3>,std::vector<int>> prefixes;
    for (const auto& p:paths.service_paths) {
        branches.insert({p.b[0],p.b[1],p.first_od,p.second_od});
        prefix_branches[p.segment_id].insert(p.second_od);
    }
    for (const auto& s:paths.initial_segments)
        prefixes[{s.b[0],s.b[1],s.first_od}].push_back(s.id);
    std::vector<std::vector<int>> pools(d.assignment_pool_count());
    auto supported = [&](int l,int b,int first,const std::vector<int>& pool) {
        if(d.preprocessed_demand) {
            // A candidate origin needs only this task's own physical witness.
            return gains[d.assignment_offset(first,l)]>0;
        }
        if (d.effective_delta()<=0) return true;
        if (d.flow_policy=="aggregate") {
            if (!branches.count({l,b,first,-1})) return false;
            for (int second:pool)
                if (d.kappa(first,second)>0 && !branches.count({l,b,first,second})) return false;
            return true;
        }
        for (int segment:prefixes[{l,b,first}]) {
            const auto& available=prefix_branches[segment];
            if (!available.count(-1)) continue;
            bool complete=true;
            for (int second:pool) if (d.kappa(first,second)>0 && !available.count(second)) { complete=false; break; }
            if (complete) return true;
        }
        return false;
    };
    for (int index : order) {
        int r = index / d.assignment_pool_count();
        if (assigned[r] || gains[index] <= 0 || (d.preprocessed_demand?d.task_rate(r):d.demand[r]) <= 0) continue;
        const int pair=index%d.assignment_pool_count(),l=pair/d.assignment_destinations(),
            b=d.origin_assignment() ? 0 : pair%d.n_l;
        auto trial=pools[pair]; trial.push_back(r);
        bool compatible=true;
        for (int first:trial) if (!supported(l,b,first,trial)) { compatible=false; break; }
        if (!compatible) continue;
        // Missing generated branches only reject this heuristic proposal;
        // exact CG and the global master retain every physical possibility.
        pools[pair]=std::move(trial);
        design[layout.assignment + index] = 1; assigned[r] = true;
    }
    if (std::any_of(assigned.begin(), assigned.end(), [](bool x) { return x; })) {
        std::fill(design.begin() + layout.facility, design.begin() + layout.capacity, 1);
        for (int i = 0; i < d.n_h; ++i) {
            int best = 0;
            for (int s = 1; s < d.n_levels; ++s) if (d.slot_counts(i, s) > d.slot_counts(i, best)) best = s;
            design[layout.capacity + i * d.n_levels + best] = 1;
        }
        for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
            design[layout.airline + u * d.n_nodes() + v] = d.allowed_links(u, v);
    }
    return design;
}

// Asset cleanup preserves every flow, assignment and queue variable. An
// arbitrarily small nonzero flow still protects all its visited assets.
Json cleanup_incumbent(const Input& d, const PathCollection& paths, const Json& original) {
    Json candidate = original;
    auto& sol = candidate["_solution"];
    std::set<int> used_nodes;
    std::set<std::pair<int, int>> used_links;
    std::vector<double> occupancy(d.n_h, 0);
    auto protect = [&](const auto& catalogue, const char* field) {
        for (std::size_t index = 0; index < catalogue.size(); ++index) {
            if (sol[field][index].template get<double>() == 0) continue;
            for (const auto& leg : catalogue[index].legs) {
                for (int node : leg) used_nodes.insert(node);
                for (std::size_t j = 1; j < leg.size(); ++j) used_links.emplace(leg[j - 1], leg[j]);
            }
        }
    };
    protect(paths.service_paths, "flow");
    protect(paths.empty_paths, "empty_flow");
    for (std::size_t index = 0; index < paths.service_paths.size(); ++index)
        for (auto [i, time] : paths.service_paths[index].slot_times)
            occupancy[i] += sol["flow"][index].get<double>() * time;
    for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
        if (!used_links.count({u, v})) sol["airline"][u][v] = 0.0;
    for (int i = 0; i < d.n_h; ++i) {
        if (!used_nodes.count(d.n_l + i) && occupancy[i] == 0) {
            sol["facility"][d.n_l + i] = 0.0;
            for (int s = 0; s < d.n_levels; ++s) sol["capacity_level"][i][s] = 0.0;
        } else if (sol["facility"][d.n_l + i].get<double>() > .5) {
            int level = -1;
            double old_cost = 0;
            for (int s = 0; s < d.n_levels; ++s) {
                old_cost += d.slot_cost(i, s) * sol["capacity_level"][i][s].get<double>();
                if (d.occupancy_limit(i, s) >= occupancy[i] &&
                    (level < 0 || std::make_tuple(d.slot_cost(i, s), d.slot_counts(i, s), s) <
                                  std::make_tuple(d.slot_cost(i, level), d.slot_counts(i, level), level))) level = s;
            }
            if (level >= 0 && d.slot_cost(i, level) <= old_cost) {
                for (int s = 0; s < d.n_levels; ++s) sol["capacity_level"][i][s] = s == level ? 1.0 : 0.0;
            }
        }
    }
    for (int l = 0; l < d.n_l; ++l) {
        bool assigned = false;
        for (int r = 0; r < d.assignment_entities(); ++r) for (int m = 0; m < d.assignment_destinations(); ++m)
            assigned = assigned || assignment_value(d,sol["assignment"],r,l,m) != 0 ||
                (!d.origin_assignment() && assignment_value(d,sol["assignment"],r,m,l) != 0);
        if (!used_nodes.count(l) && !assigned && sol["rho"][l].get<double>() == 0 && sol["idle"][l].get<double>() == 0)
            sol["facility"][l] = 0.0;
    }
    double bases = 0, lockers = 0, links = 0;
    for (int l = 0; l < d.n_l; ++l) bases += d.base_cost[l] * sol["facility"][l].get<double>();
    for (int i = 0; i < d.n_h; ++i) for (int s = 0; s < d.n_levels; ++s)
        lockers += d.slot_cost(i, s) * sol["capacity_level"][i][s].get<double>();
    for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
        links += d.link_cost(u, v) * sol["airline"][u][v].get<double>();
    const double opening = bases + lockers + links;
    const double objective = opening + candidate["continuous_cost"].get<double>();
    if (!std::isfinite(objective) || objective > original["objective"].get<double>() + 1e-8) return original;
    // These are exactly the constraints affected by changing x/z/y. All other
    // rows and cones retain their previously checked values and coefficients.
    try { validate_design(d, solution_design(d, candidate)); }
    catch (const std::invalid_argument&) { return original; }
    for (int l = 0; l < d.n_l; ++l)
        if (sol["rho"][l].get<double>() > sol["facility"][l].get<double>() + 1e-8) return original;
    for (int i = 0; i < d.n_h; ++i) {
        double limit = 0;
        for (int s = 0; s < d.n_levels; ++s) limit += d.occupancy_limit(i, s) * sol["capacity_level"][i][s].get<double>();
        if (occupancy[i] > limit + 2e-7 * (1 + std::abs(occupancy[i]) + std::abs(limit))) return original;
    }
    for (auto [u, v] : used_links) if (sol["airline"][u][v].get<double>() == 0) return original;
    candidate["opening_cost"] = opening;
    candidate["objective"] = objective;
    candidate["metrics"]["costs_per_hour"]["bases"] = bases;
    candidate["metrics"]["costs_per_hour"]["lockers"] = lockers;
    candidate["metrics"]["costs_per_hour"]["links"] = links;
    candidate["cleanup"] = {{"fixed_flows", true}, {"affected_constraints_checked", true},
                             {"saving", original["objective"].get<double>() - objective}};
    return candidate;
}

std::vector<double> neighbor_design(const Input& d, const PathCollection& paths, const Json& incumbent) {
    const DesignLayout layout(d);
    auto design = solution_design(d, incumbent);
    std::vector<double> reposition(d.n_l * d.n_l, INFINITY), gains(d.assignment_count(), 0);
    for (const auto& p : paths.empty_paths)
        reposition[p.b[0] * d.n_l + p.b[1]] = std::min(reposition[p.b[0] * d.n_l + p.b[1]], p.cost + d.fleet_cost * p.time);
    for (const auto& p : paths.service_paths) {
        if (!d.preprocessed_demand && p.mode != "single") continue;
        double cost = p.cost + d.fleet_cost * p.time + (p.b[0] == p.b[1] ? 0 : reposition[p.b[1] * d.n_l + p.b[0]]);
        const int entity=d.preprocessed_demand?d.task_index(p.first_od,p.second_od):p.first_od;
        int index = d.assignment_offset(entity,p.b[0],p.b[1]);
        const double ground=d.ground_cost[p.first_od]+(d.preprocessed_demand && p.second_od>=0?d.ground_cost[p.second_od]:0);
        gains[index] = std::max(gains[index], ground - cost);
    }
    std::vector<int> order(gains.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        int ar = a / d.assignment_pool_count(), br = b / d.assignment_pool_count();
        return std::make_tuple(-(d.preprocessed_demand?d.task_rate(ar):d.demand[ar]), -gains[a], a) <
               std::make_tuple(-(d.preprocessed_demand?d.task_rate(br):d.demand[br]), -gains[b], b);
    });
    for (int index : order) {
        int r = index / d.assignment_pool_count();
        bool assigned = false;
        for (int a = 0; a < d.n_l; ++a) for (int b = 0; b < d.assignment_destinations(); ++b)
            assigned = assigned || design[layout.assignment_index(r,a,b)] != 0;
        if (assigned || gains[index] <= 0 || (d.preprocessed_demand?d.task_rate(r):d.demand[r]) <= 0) continue;
        design[layout.assignment + index] = 1;
        std::fill(design.begin() + layout.facility, design.begin() + layout.capacity, 1);
        std::fill(design.begin() + layout.capacity, design.begin() + layout.assignment, 0);
        for (int i = 0; i < d.n_h; ++i) {
            int best = 0;
            for (int s = 1; s < d.n_levels; ++s) if (d.slot_counts(i, s) > d.slot_counts(i, best)) best = s;
            design[layout.capacity + i * d.n_levels + best] = 1;
        }
        for (int u = 0; u < d.n_nodes(); ++u) for (int v = 0; v < d.n_nodes(); ++v)
            design[layout.airline + u * d.n_nodes() + v] = d.allowed_links(u, v);
        return design;
    }
    return {};
}

Json add_master_bounds(const Input& d, const PathCollection& paths, Formulation& master,
                       int theta, const SolverOptions& options) {
    return add_master_strengthening(d, paths, master, theta, options.resource_bounds);
}

struct InitialColumns {
    std::vector<int> service, empty;
    Json metadata;
};
std::string seed_input_key(const Input& d) {
    // Compare actual mathematical input, including manually constructed
    // fixtures whose manifest/model_json is empty. Reporting coordinates and
    // provenance do not define a different physical column universe.
    return Json::array({d.n_l,d.n_h,d.n_od,d.n_levels,d.max_orders,d.delta,d.operating_limit,
        d.fleet_cost,d.flight_cost,d.waiting_cost,d.route_mode,d.flow_policy,d.model_version,
        d.preprocessed_demand,d.assignment_scope(),d.first_leg_consistency,d.has_candidate_routes,
        d.demand,d.ground_cost,d.deadline,d.base_cost,d.slot_epsilon,
        d.flight_times.values,d.rider_mean.values,d.first_cost.values,d.terminal_cost.values,
        d.receipt_mean.values,d.home_receipt_mean.values,d.removal_mean.values,d.slot_cost.values,d.link_cost.values,
        d.kappa.values,d.occupancy_limit.values,d.slot_counts.values,d.allowed_links.values,
        d.candidate_routes,d.single_rates,d.head_rates,d.pair_rates.values,d.common_origin_support.values}).dump();
}
std::string seed_path_key(const ServicePath& p) {return exact_path_identity(p);}
std::string seed_path_key(const EmptyPath& p) {return exact_path_identity(p);}
InitialColumns initial_columns(const Input& d, PathCollection& paths,
                               const SolverOptions& options, double deadline,
                               const CgColumnSeeds* reused = nullptr) {
    const double started=wall_time();
    InitialColumns seeds;
    Json coverage = {{"enabled",options.branch_covering_seeds}};
    if(options.cg_seed_reuse && options.column_generation && reused) {
        reused->validate(d,paths);
        seeds.service=reused->service();seeds.empty=reused->empty();
        if(!seeds.service.empty() || !seeds.empty.empty()) {
            seeds.metadata={{"method","reused_cg_columns"},{"service",seeds.service.size()},
                {"empty",seeds.empty.size()},{"seconds",wall_time()-started},
                {"feasibility","all_courier_independent_of_seed_quality"},
                {"physical_columns_only",true},{"transferred_bound",false}};
            return seeds;
        }
    }
    if (options.column_generation && !paths.complete) {
        greedy_seed_paths(d, paths, 1, deadline);
        if (options.branch_covering_seeds) {
            const auto report = branch_covering_seed_paths(d, paths, nullptr,
                std::min(deadline, wall_time()+std::min(5.0,.05*options.seconds)));
            coverage.update({{"service_added",report.service_added},{"empty_added",report.empty_added},
                {"groups_requested",report.groups_requested},{"groups_completed",report.groups_completed},
                {"required_branches",report.required_branches},{"covered_branches",report.covered_branches},
                {"physically_missing_branches",report.physically_missing_branches},
                {"groups_without_complete_prefix",report.groups_without_complete_prefix},{"stopped",report.stopped}});
        }
        seeds.service.resize(paths.service_paths.size());
        seeds.empty.resize(paths.empty_paths.size());
        std::iota(seeds.service.begin(),seeds.service.end(),0);
        std::iota(seeds.empty.begin(),seeds.empty.end(),0);
    }
    seeds.metadata={{"method", !options.column_generation ? "all_columns" :
                              (paths.complete ? "empty_rmp_catalogue_validation" :
                               options.branch_covering_seeds ? "greedy_with_branch_coverage" : "greedy_physical_cost")},
                    {"service",seeds.service.size()},{"empty",seeds.empty.size()},
                    {"seconds",wall_time()-started},
                    {"feasibility", "all_courier_independent_of_seed_quality"}, {"branch_coverage",coverage}};
    return seeds;
}

Json continuous_cg_result(const Input& d, PathCollection& paths, const SolverOptions& options,
                          const std::vector<double>* fixed_design,
                          int face_head = -1, int face_state = -1,
                          CgColumnSeeds* reused = nullptr,
                          double no_gain_upper = std::numeric_limits<double>::quiet_NaN()) {
    if (!paths.support_complete) throw std::invalid_argument("Continuous CG requires complete physical support");
    if (d.origin_assignment() && (options.assignment_support_cuts || options.double_support_cuts || options.joint_base_cost_cuts))
        throw std::invalid_argument("K=1 origin assignment does not support base-pair assignment-support, double-support, or joint-base-cost cuts; disable these optional cuts");
    if (!options.column_generation && !paths.complete)
        throw std::invalid_argument("Full continuous recourse requires a complete path catalogue");
    if (!std::isfinite(options.seconds) || options.seconds < 0 || options.cg_max_columns < 1 ||
        !std::isfinite(options.cg_duality_tolerance) || options.cg_duality_tolerance <= 0 ||
        !std::isfinite(options.pricing_tolerance) || options.pricing_tolerance < 0)
        throw std::invalid_argument("Invalid continuous CG budget or certification tolerance");
    if (fixed_design) validate_design(d, *fixed_design, true);
    const double started = wall_time();
    if(options.cg_seed_reuse && reused)reused->validate(d,paths);
    double deadline = started + options.seconds;
    if (options.progress) deadline = std::min(deadline, started + options.progress->remaining());
    Json result = {{"method",fixed_design ? "fractional_recourse_cg" : "full_continuous_cg"},
        {"solver","CPLEX"},{"tight_airline_bounds",options.tight_airline_bounds},
        {"scope",fixed_design ? "fixed_design_recourse" : continuous_scope(options)},
        {"structural_strengthening",structural_metadata(options)},
        {"objective_includes_fixed_costs",!fixed_design},{"is_integer_solution",false},
        {"certified_lower_bound",nullptr},{"bound_valid",false},{"continuous_objective",nullptr},
        {"primal_valid",false},
        {"operating_objective",nullptr},{"design",fixed_design ? Json(*fixed_design) : Json::array()},
        {"certified",false},{"pricing_complete",false},{"recourse_optimal",false},{"intercept",nullptr},
        {"coefficients",Json::array()},{"ordinary_cut",nullptr},{"cg_trace",Json::array()},
        {"timings",Json::object()},{"termination_reason","time_limit"},
        {"model_construction_complete",false},{"support_complete",true},
        {"catalogue_complete",paths.complete},{"budget_seconds",options.seconds},
        {"budget_scope","seed_construction_model_build_optimization_pricing_validation"},
        {"pricing_method",options.column_generation ? (options.catalogue_pricing && paths.complete?"exact_catalogue_scan":"exact_resource_labeling") : "all_columns"},
        {"cg_max_columns",options.cg_max_columns},
        {"continuous_numerical_recovery",true},
        {"cg_duality_tolerance",options.cg_duality_tolerance},{"pricing_tolerance",options.pricing_tolerance}};
    if(face_head>=0) {
        result["method"]="head_design_face_cg";
        result["scope"]="complete_domain_head_design_face";
        result["head"]=face_head;result["state"]=face_state;
        result["assignment_scope"]=d.assignment_scope();
        result["other_designs_free"]=true;
        result["direct_global_bound_update"]=false;
        result["bases"]=d.n_l;result["state_count"]=d.n_l+1;
        result["face_fix_rows"]=d.n_l;
        result["validity_scope"]="all_original_integer_designs_in_declared_head_state";
    }
    InitialColumns seeds;
    double initialization_seconds = 0;
    try {
        if (remaining(deadline) > 0) {
            const double initialization_started = wall_time();
            seeds = initial_columns(d, paths, options, deadline,reused);
            initialization_seconds = wall_time() - initialization_started;
            result["initial_columns"] = seeds.metadata;
            if (remaining(deadline) > 0) {
                const std::vector<double> no_design;
                auto sub = subproblem(d, paths, fixed_design ? *fixed_design : no_design, options,
                    deadline, seeds.service, seeds.empty, nullptr, {}, true, fixed_design == nullptr, true,
                    face_head,face_state,no_gain_upper);
                result["model_construction_complete"] = true;
                result["termination_reason"] = sub.reason;
                if(face_head>=0)result["face_no_gain_stopped"]=sub.reason=="face_cannot_strengthen_global_floor";
                result["cg_trace"] = std::move(sub.trace);
                result["cg_iterations"] = sub.iterations;
                result["service_columns"] = sub.service.size();
                result["empty_columns"] = sub.empty.size();
                result["_service_indices"] = sub.service;
                result["_empty_indices"] = sub.empty;
                if(options.cg_seed_reuse && reused)reused->include(d,paths,sub.service,sub.empty);
                result["timings"] = sub.timings;
                result["primal_residuals"] = sub.continuous_residuals;
                result["operating_point"] = sub.continuous_operating_point;
                result["model_size"] = sub.model_size;
                if (std::isfinite(sub.continuous_objective)) {
                    result["continuous_objective"] = sub.continuous_objective;
                    result["primal_valid"] = true;
                    if (fixed_design) result["operating_objective"] = sub.continuous_objective;
                    else result["design"] = sub.continuous_design;
                }
                result["pricing_complete"] = sub.ordinary.valid;
                result["certified"] = sub.ordinary.valid;
                result["recourse_optimal"] = sub.recourse_optimal;
                result["failure"] = sub.failure;
                if (!fixed_design && sub.full_bound_valid) {
                    result["bound_valid"] = true;
                    result["certified_lower_bound"] = sub.certified_lower_bound;
                    result["lower_bound_certificate"] = sub.lower_bound_certificate;
                }
                if (fixed_design && sub.ordinary.valid) {
                    result["bound_valid"] = true;
                    result["certified_lower_bound"] = value(sub.ordinary, *fixed_design);
                    result["intercept"] = sub.ordinary.intercept;
                    result["coefficients"] = sub.ordinary.coefficients;
                    result["ordinary_cut"] = {{"kind","ordinary"},{"intercept",sub.ordinary.intercept},
                        {"coefficients",sub.ordinary.coefficients},{"design",*fixed_design},
                        {"operating_objective",result.at("operating_objective")},
                        {"correction",sub.ordinary.correction},{"stationarity_error",sub.ordinary.stationarity_error},
                        {"pricing_complete",true},{"certified",true},
                        {"scope","global_recourse_epigraph"}};
                }
            }
        }
    } catch (const RunBudgetExceeded& error) {
        result["termination_reason"] = "construction_time_limit";
        result["failure"] = {{"stage",error.stage},{"sizes",error.sizes},{"message",error.what()}};
    } catch (const EnumerationStopped& error) {
        result["termination_reason"] = error.reason;
        result["failure"] = {{"stage","initial_columns"},{"message",error.what()}};
    } catch (const IloException& error) {
        result["termination_reason"] = "solver_exception";
        result["failure"] = {{"message",error.getMessage()}};
    }
    result["timings"]["greedy_initial_columns"] = initialization_seconds;
    result["algorithm_wall"] = wall_time() - started;
    double accounted = 0;
    for (const auto& seconds : result.at("timings")) accounted += seconds.get<double>();
    result["timings"]["other_and_interrupted"] = std::max(0.0, result.at("algorithm_wall").get<double>() - accounted);
    result["registered_service_paths"] = paths.service_paths.size();
    result["registered_empty_paths"] = paths.empty_paths.size();
    result["generated_service_paths"] = d.preprocessed_demand ? result.value("service_columns",std::size_t(0)) : paths.service_paths.size();
    result["generated_empty_paths"] = d.preprocessed_demand ? result.value("empty_columns",std::size_t(0)) : paths.empty_paths.size();
    result["generated_columns_scope"] = d.preprocessed_demand ? "continuous_recourse_rmp" : "registered_generated_catalogue";
    result["catalogue_complete"] = paths.complete;
    if (options.progress) options.progress->record({{"event","continuous_cg_finished"},
        {"bound_scope",result.at("scope")},{"bound_valid",result.at("bound_valid")},
        {"certified_lower_bound",result.at("certified_lower_bound")},
        {"continuous_objective",result.at("continuous_objective")},
        {"pricing_complete",result.at("pricing_complete")},{"termination_reason",result.at("termination_reason")},
        {"integer_incumbent",false}});
    return result;
}
} // namespace

void CgColumnSeeds::validate(const Input& d,const PathCollection& paths) const {
    if(!input_)return;
    if(input_!=&d || paths_!=&paths || input_key_!=seed_input_key(d) || scope_!=paths.scope ||
       !paths.support_complete || paths.service_paths.size()<service_count_ || paths.empty_paths.size()<empty_count_)
        throw std::invalid_argument("CG columns belong to a different or changed input/catalogue");
    for(std::size_t i=0;i<service_.size();++i)
        if(seed_path_key(paths.service_paths.at(service_[i]))!=service_keys_[i])
            throw std::invalid_argument("CG service-column identity changed");
    for(std::size_t i=0;i<empty_.size();++i)
        if(seed_path_key(paths.empty_paths.at(empty_[i]))!=empty_keys_[i])
            throw std::invalid_argument("CG empty-column identity changed");
}
void CgColumnSeeds::include(const Input& d,const PathCollection& paths,
                          const std::vector<int>& service,const std::vector<int>& empty) {
    validate(d,paths);
    if(!paths.support_complete)throw std::invalid_argument("CG column reuse requires complete physical support");
    for(int id:service)if(id<0 || static_cast<std::size_t>(id)>=paths.service_paths.size())
        throw std::invalid_argument("CG service seed index is outside its catalogue");
    for(int id:empty)if(id<0 || static_cast<std::size_t>(id)>=paths.empty_paths.size())
        throw std::invalid_argument("CG empty seed index is outside its catalogue");
    auto merge=[](std::vector<int>& ids,std::vector<std::string>& keys,
                  std::vector<int> incoming,const auto& catalogue) {
        std::sort(incoming.begin(),incoming.end());incoming.erase(std::unique(incoming.begin(),incoming.end()),incoming.end());
        std::vector<int> joined;std::vector<std::string> payloads;
        joined.reserve(ids.size()+incoming.size());payloads.reserve(ids.size()+incoming.size());
        std::size_t i=0,j=0;
        while(i<ids.size() || j<incoming.size()) {
            if(i<ids.size() && (j==incoming.size() || ids[i]<=incoming[j])) {
                if(j<incoming.size() && ids[i]==incoming[j])++j;
                joined.push_back(ids[i]);payloads.push_back(std::move(keys[i]));++i;
            } else {
                const int id=incoming[j++];joined.push_back(id);payloads.push_back(seed_path_key(catalogue.at(id)));
            }
        }
        ids=std::move(joined);keys=std::move(payloads);
    };
    merge(service_,service_keys_,service,paths.service_paths);
    merge(empty_,empty_keys_,empty,paths.empty_paths);
    input_=&d;paths_=&paths;input_key_=seed_input_key(d);scope_=paths.scope;
    service_count_=paths.service_paths.size();empty_count_=paths.empty_paths.size();
}

Json solve_continuous_cg(const Input& d, PathCollection& paths, const SolverOptions& options,
                         CgColumnSeeds* seeds) {
    return continuous_cg_result(d, paths, options, nullptr,-1,-1,seeds);
}

Json solve_design_face_cg(const Input& d, PathCollection& paths, const SolverOptions& options,
                          int head, int state,CgColumnSeeds* seeds,double no_gain_upper) {
    if(!d.preprocessed_demand || !d.origin_assignment() || d.n_l<=0 || d.max_orders!=2 ||
       d.flow_policy!="aggregate" || d.first_leg_consistency || d.route_mode!="direct" ||
       d.has_candidate_routes || !paths.complete || !paths.support_complete ||
       head<0 || head>=d.assignment_entities() || state<0 || state>d.n_l)
        throw std::invalid_argument("Design faces require complete direct K2 preprocessed origin tasks");
    SolverOptions face_options=options;
    face_options.column_generation=true;face_options.pareto=false;face_options.conditional_demand=false;
    // A nontight native dual is useful only after the existing full-domain
    // correction. No auxiliary dual recovery LP is introduced here.
    face_options.weak_dual_cuts=true;
    auto result=continuous_cg_result(d,paths,face_options,nullptr,head,state,seeds,no_gain_upper);
    if(result.value("termination_reason",std::string()).find("contradiction")!=std::string::npos)
        throw std::logic_error("Design-face primal/dual contradiction: "+
                               result.at("termination_reason").get<std::string>());
    return result;
}

Json evaluate_fractional_recourse_cg(const Input& d, PathCollection& paths, const SolverOptions& options,
                                    const std::vector<double>& design) {
    return continuous_cg_result(d, paths, options, &design);
}

Json solve_fixed_design_cg(const Input& d, PathCollection& paths, const SolverOptions& options,
                          const std::vector<double>& design, CgColumnSeeds* reused) {
    if (!paths.support_complete) throw std::runtime_error("Fixed-design CG requires complete initial-segment support");
    if (!options.column_generation && !paths.complete) throw std::runtime_error("Full recourse requires a complete path catalogue");
    if(options.task_resource_cut_limit<0 || !std::isfinite(options.task_resource_seconds) || options.task_resource_seconds<=0)
        throw std::invalid_argument("Invalid task resource separation budget");
    if(options.task_airline_bounds && options.task_airline_upper.empty())
        throw std::invalid_argument("Prepare task airline bounds from the complete catalogue before Benders");
    const double started = wall_time();
    const auto seeds=initial_columns(d,paths,options,started+options.seconds,reused);
    auto sub = subproblem(d, paths, design, options, started + options.seconds, seeds.service, seeds.empty, nullptr);
    if(options.cg_seed_reuse && reused)reused->include(d,paths,sub.service,sub.empty);
    if (sub.solution.is_null()) throw std::runtime_error("Fixed-design CG has no validated feasible solution");
    Json result = sub.solution;
    const double opening = result["opening_cost"].get<double>();
    const double upper = result["objective"].get<double>();
    const double continuous_bound = sub.ordinary.valid ? std::max(0.0, value(sub.ordinary, design)) : 0;
    const double lower = std::min(upper, opening + continuous_bound);
    result["method"] = options.column_generation ? "fixed_design_cg" : "fixed_design_benders_full";
    result["solver"] = "CPLEX";
    result["incumbent_kind"] = "validated_fixed_design";
    result["bound"] = lower;
    result["continuous_bound"] = continuous_bound;
    result["gap"] = gap(upper, lower);
    result["termination_reason"] = sub.reason;
    result["pricing_complete"] = sub.ordinary.valid;
    result["bound_valid"] = sub.ordinary.valid;
    result["recourse_optimal"] = sub.recourse_optimal;
    result["construction_complete"] = true;
    result["catalogue_complete"] = paths.complete;
    result["support_complete"] = paths.support_complete;
    result["initial_columns"] = seeds.metadata;
    result["initial_service_columns"] = seeds.service.size();
    result["initial_empty_columns"] = seeds.empty.size();
    result["cg_iterations"] = sub.iterations;
    result["cg_trace"] = sub.trace;
    result["service_columns"] = sub.service.size();
    result["empty_columns"] = sub.empty.size();
    result["generated_columns"] = {{"service",sub.service.size()},{"empty",sub.empty.size()}};
    result["generated_columns_scope"] = "fixed_design_recourse_rmp";
    result["pricing_method"] = options.column_generation ? (options.catalogue_pricing && paths.complete?"exact_catalogue_scan":"exact_resource_labeling") : "all_columns";
    result["cg_max_columns"] = options.cg_max_columns;
    result["pricing_tolerance"] = options.pricing_tolerance;
    result["cg_duality_tolerance"] = options.cg_duality_tolerance;
    result["phase_one"] = "unnecessary_validated_all_courier_complete_recourse";
    result["timings"] = sub.timings;
    result["timings"]["greedy_initial_columns"] = seeds.metadata.at("seconds");
    result["algorithm_wall"] = wall_time() - started;
    result["demand_formulation"] = d.preprocessed_demand ? "preprocessed" : options.conditional_demand ? "conditional" : "lifted";
    result["structural_strengthening"] = structural_metadata(options);
    result["subproblem_model_size"] = sub.model_size;
    result["subproblem_model_size_peak"] = sub.model_size;
    if (sub.ordinary.valid)
        result["benders_cuts"] = Json::array({{{"kind", "ordinary"}, {"intercept", sub.ordinary.intercept},
            {"coefficients", sub.ordinary.coefficients}, {"correction", sub.ordinary.correction},
            {"stationarity_error", sub.ordinary.stationarity_error}, {"pricing_complete", true},
            {"catalogue_complete", paths.complete}, {"support_complete", paths.support_complete},
            {"condition",sub.ordinary.condition},{"recourse_optimal",sub.recourse_optimal}}});
    else result["benders_cuts"] = Json::array();
    if (options.progress) {
        options.progress->record({{"event","global_bounds"},{"stage","fixed_design_cg_finished"},
            {"objective",upper},{"upper_bound",upper},{"bound",lower},{"lower_bound",lower},
            {"gap",result.at("gap")},{"bound_scope","fixed_design"},{"validated_incumbent",true},
            {"pricing_complete",sub.ordinary.valid}});
        options.progress->checkpoint(result,"fixed_design_cg_finished",true);
    }
    return result;
}

Json solve_benders(const Input& d, PathCollection& paths, const SolverOptions& options,
                   const Json* start_solution) {
    if (options.heuristic_frequency < 1 || options.pareto_frequency < 1 || options.root_cut_rounds < 0)
        throw std::invalid_argument("Invalid Benders strengthening frequency");
    if (!paths.support_complete) throw std::runtime_error("Benders requires complete initial-segment support");
    if(options.primal_alns && (d.n_l!=2 || d.max_orders!=2 || !d.preprocessed_demand ||
       !d.origin_assignment() || d.flow_policy!="aggregate" || d.first_leg_consistency ||
       !paths.complete || !options.reuse_columns))
        throw std::invalid_argument("Primal ALNS requires complete L2 K2 preprocessed origin aggregate columns");
    if(options.staged_continuous_root && (!options.cg_seed_reuse || !options.continuous_root))
        throw std::invalid_argument("Staged continuous root requires root CG and column seed reuse");
    if(options.fractional_path_support && !options.path_support_cuts)
        throw std::invalid_argument("Fractional path support requires path support cuts");
    if(options.airline_resource_balance && !options.airline_resource_cuts)
        throw std::invalid_argument("Airline balance requires airline resource cuts");
    if(options.queue_fractional_separation && (!options.queue_price_search || !options.queue_polymatroid_cuts))
        throw std::invalid_argument("Fractional queue separation requires queue price search and queue cuts");
    if (options.path_support_cuts && (!d.preprocessed_demand || !d.origin_assignment() ||
        d.flow_policy != "aggregate" || d.first_leg_consistency || options.conditional_demand))
        throw std::invalid_argument("Path support dominance requires fixed generalized demand, origin assignment and aggregate flow");
    if (options.path_support_cut_limit < 0 || options.path_support_max_cuts < 1 ||
        !std::isfinite(options.path_support_seconds) || options.path_support_seconds <= 0 ||
        !std::isfinite(options.path_support_preprocessing_seconds) || options.path_support_preprocessing_seconds <= 0)
        throw std::invalid_argument("Invalid path support separation budget");
    if (options.topology_cut_limit < 0 || !std::isfinite(options.topology_seconds) || options.topology_seconds <= 0)
        throw std::invalid_argument("Invalid topology separation budget");
    if(options.task_resource_cut_limit<0 || !std::isfinite(options.task_resource_seconds) || options.task_resource_seconds<=0)
        throw std::invalid_argument("Invalid task resource separation budget");
    if(options.task_airline_bounds && options.task_airline_upper.empty())
        throw std::invalid_argument("Prepare task airline bounds from the complete catalogue before Benders");
    if (options.assignment_support_cut_limit < 0 || options.assignment_support_max_cuts < 1 ||
        !std::isfinite(options.assignment_support_seconds) || options.assignment_support_seconds <= 0 ||
        !std::isfinite(options.assignment_support_floor) || options.assignment_support_floor <= 0)
        throw std::invalid_argument("Invalid assignment-support separation settings");
    if (options.branch_joint_cut_limit < 0 || options.branch_joint_max_cuts < 1 ||
        !std::isfinite(options.branch_joint_seconds) || options.branch_joint_seconds <= 0 ||
        !std::isfinite(options.branch_joint_preprocessing_seconds) || options.branch_joint_preprocessing_seconds <= 0 ||
        !std::isfinite(options.branch_joint_floor) || options.branch_joint_floor <= 0)
        throw std::invalid_argument("Invalid branch/joint separation settings");
    if (!options.column_generation && !paths.complete) throw std::runtime_error("Full recourse requires a complete path catalogue");
    if (start_solution) validate_design(d, solution_design(d, *start_solution));
    const double started = wall_time(), deadline = started + options.seconds;
    const auto seeds=initial_columns(d,paths,options,deadline);
    Json best = all_courier_result(d, paths);
    best["incumbent_kind"] = "all_courier_baseline";
    double upper = best["objective"].get<double>(), lower = 0;
    // Receives only same-model candidates whose native primal and full cost
    // have already been verified. Never import a checkpoint or a fractional
    // candidate here; saving must not need a new model after the deadline.
    auto retain_incumbent = [&](const Json& candidate,const char* kind) {
        if(candidate.is_null())return false;
        const double objective=candidate.at("objective").get<double>();
        if(!std::isfinite(objective))throw std::runtime_error("Nonfinite validated incumbent objective");
        if(objective>=upper)return false;
        if(!candidate.value("native_model_validated",false) || !candidate.contains("_solution") ||
           candidate.value("assignment_scope",std::string())!=d.assignment_scope() ||
           candidate.value("flow_policy",std::string())!=d.flow_policy ||
           (candidate.value("demand_formulation",std::string())=="preprocessed")!=d.preprocessed_demand)
            throw std::runtime_error("Incumbent promotion requires a verified same-model integer solution");
        if(objective<lower-1e-7*(1+std::abs(objective)))
            throw std::runtime_error("Native primal contradicts certified Benders lower bound");
        best=candidate;upper=objective;best["incumbent_kind"]=kind;
        return true;
    };
    const auto retain_recourse = [&](const Json& candidate) {retain_incumbent(candidate,"optimized");};
    Formulation master = design_model(d, true, formulation_strengthening(options));
    int theta = master.matrix.variable("operating_cost", 0, courier_cost(d), 1);
    Json bound_metadata = options.courier_savings_bound ? add_master_bounds(d, paths, master, theta, options)
                            : Json{{"enabled", false}, {"reason", "disabled"}};
    TaskSavingsData task_data;
    Json task_savings_metadata={{"enabled",false}};
    if(options.task_savings_bound || options.task_resource_cuts) {
        task_data=task_savings_data(d,paths);
        if(options.task_savings_bound)
            task_savings_metadata=add_task_savings_bound(task_data,master,theta);
    }
    Json airline_connectivity_metadata={{"enabled",false}};
    if(options.airline_connectivity_cuts) {
        const auto connectivity=build_airline_connectivity(d,paths);
        airline_connectivity_metadata=connectivity.metadata;
        for(const auto& row:connectivity.rows) master.matrix.rows.push_back(row);
        airline_connectivity_metadata["rows"]=Json::array();
        for(const auto& row:connectivity.rows)
            airline_connectivity_metadata["rows"].push_back({{"name",row.name},
                {"sense",std::string(1,row.sense)},{"rhs",row.rhs},{"coefficients",row.coefficients}});
    }
    int cutoff_row = -1;
    if (options.incumbent_cutoff) {
        Coefficients objective{{theta, 1}};
        for (int j : master.design_vars)
            if (master.matrix.variables[j].cost != 0)
                objective.emplace_back(j, master.matrix.variables[j].cost);
        cutoff_row = master.matrix.row("incumbent_objective_cutoff", 'L',
                                      upper + 1e-8 * (1 + std::abs(upper)), std::move(objective));
    }
    const std::size_t initial_master_rows = master.matrix.rows.size();
    NativeSolver solver(master.matrix, options, "benders_master");
    CertifiedCutPool certified_cut_pool;
    auto set_cut_rhs=[&](int row,char sense,double rhs){solver.set_rhs(row,sense,rhs);};
    LabelPricingGraphCache pricing_graph_cache;
    SubproblemStateCache recourse_state_cache;
    DesignCertificateMemo design_memo(options.design_cache_limit);
    Json design_memo_records=Json::array();
    Json stabilized_master_records=Json::array();
    double stabilized_master_seconds=0;
    const double master_build = wall_time() - started - seeds.metadata.at("seconds").get<double>();
    std::vector<double> core = initial_core(d), original_core = core;
    std::set<int> pool_p(seeds.service.begin(),seeds.service.end()), pool_e(seeds.empty.begin(),seeds.empty.end());
    std::set<int> used_p=pool_p,used_e=pool_e;
    CgColumnSeeds continuous_columns;
    ContinuousRootProgress continuous_progress;
    double last_continuous_stage=started;
    const double continuous_total_limit=std::min(options.continuous_root_seconds,
        options.continuous_root_budget_fraction*options.seconds);
    Json continuous_stage_records=Json::array();
    int cuts = 0, ordinary_count = 0, pareto_count = 0, auxiliary_count = 0, pareto_attempts = 0, cg_iterations = 0;
    int conditional_cuts = 0, conditional_subproblems = 0;
    std::set<std::string> assignment_patterns;
    std::set<std::size_t> conditional_row_indices, topology_row_indices, assignment_support_row_indices;
    std::set<std::string> topology_keys;
    Json topology_records = Json::array(), topology_attempts = Json::array();
    int topology_skipped = 0, topology_certificates = 0;
    double topology_seconds = 0;
    AssignmentSupportCache assignment_support_cache;
    std::set<std::string> assignment_support_keys;
    Json assignment_support_records = Json::array(), assignment_support_attempts = Json::array();
    int assignment_support_skipped = 0, assignment_support_certificates = 0;
    std::size_t assignment_support_numerical_skips = 0;
    double assignment_support_seconds = 0;
    PathSupportCache path_support_cache;
    std::set<std::string> path_support_keys;
    Json path_support_records = Json::array(), path_support_attempts = Json::array();
    std::size_t path_support_static = 0, path_support_dynamic = 0;
    std::size_t path_support_checks = 0, path_support_blocked = 0, path_support_skipped = 0;
    double path_support_seconds = 0;
    double fractional_support_seconds=0, fractional_refresh_seconds=0;
    int fractional_support_cuts=0, support_rejections_since_refresh=0;
    Json fractional_support_attempts=Json::array(), fractional_support_records=Json::array();
    const bool branch_joint_enabled = (options.double_support_cuts || options.joint_base_cost_cuts)
        && options.branch_joint_cut_limit > 0;
    BranchJointCache branch_joint_cache;
    std::set<std::size_t> branch_joint_row_indices;
    std::set<std::string> branch_joint_keys;
    Json branch_joint_records = Json::array(), branch_joint_attempts = Json::array();
    int branch_joint_skipped = 0, double_support_count = 0, joint_base_count = 0;
    std::size_t branch_joint_certificates = 0, branch_joint_numerical_skips = 0;
    double branch_joint_seconds = 0;
    Json subproblem_size=nullptr, subproblem_peak=nullptr;
    int cleanup_improvements = 0, primal_search_attempts = 0;
    int ordinary_cg_iterations = 0, heuristic_cg_iterations = 0, master_starts = 0;
    int root_count = 0, restricted_attempts = 0, restricted_improvements = 0;
    int scheduled_restricted_calls = 0, local_branching_attempts = 0, ordinary_restricted_attempts = 0;
    int pareto_failures = 0, next_pareto_call = 0;
    int pareto_budget_failures = 0, master_stalled_passes = 0;
    int root_master_solves = 0, root_cg_calls = 0, integer_master_solves = 0, integer_cg_calls = 0;
    double heuristic_seconds = 0, restricted_seconds = 0, pareto_seconds = 0;
    PrimalSchedule primal_schedule(options.progress?options.progress->budget_seconds():options.seconds,started);
    std::unique_ptr<PrimalAlnsSearch> primal_alns;
    std::optional<PrimalCgFeedback> pending_feedback;
    Json cg_feedback_records=Json::array();
    Json primal_schedule_records=Json::array(), elite_records=Json::array();
    int primal_yields=0, elite_calls=0;
    double cleanup_savings = 0;
    std::vector<double> last_neighbor_design;
    std::vector<double> last_master_start;
    std::set<std::vector<double>> integer_designs;
    int repeated_integer_designs = 0, all_courier_candidates = 0, bound_stalled_passes = 0;
    int supported_cg_passes = 0, closed_without_progress = 0, resource_restart_streak = 0;
    int supported_candidates = 0, resource_budget_skips = 0, forced_cg_passes = 0;
    double last_restricted_search = started, integer_resource_seconds = 0;
    Json search_schedule_records = Json::array();
    std::vector<double> pending_design;
    Json pending_primal=nullptr;
    double pending_theta = 0;
    std::string active_candidate, pending_candidate;
    Json iterations = Json::array(), heuristic_records = Json::array(), cut_records = Json::array();
    Json primal_design_polish_records = Json::array();
    Json interior_cut_records = Json::array();
    double interior_cut_seconds = 0;
    int next_interior_call = 5;
    Json master_records = Json::array(), resource_records = Json::array(), resource_attempts = Json::array();
    Json task_resource_records=Json::array(),task_resource_attempts=Json::array();
    std::size_t task_resource_skipped=0;
    Json airline_resource_records=Json::array(),airline_resource_attempts=Json::array();
    Json airline_resource_metadata={{"enabled",false}};
    std::size_t airline_resource_skipped=0;
    Json queue_records=Json::array(),queue_attempts=Json::array();
    Json queue_metadata={{"enabled",false}};
    Json queue_price_records=Json::array(), target_master_records=Json::array();
    double queue_price_seconds=0, target_master_seconds=0;
    double last_target_master=started-30;
    QueueNativePrices deferred_queue_prices;
    std::string deferred_queue_candidate;
    std::size_t deferred_queue_serial=0, consumed_queue_serial=0;
    double queue_fractional_lp_seconds=0, queue_fractional_wall=0, last_queue_fractional=started-5;
    Json queue_fractional_records=Json::array();
    Json design_face_records=Json::array(), primal_portfolio_records=Json::array();
    Json coupled_neighborhood_records=Json::array();
    double design_face_seconds=0, joint_queue_seconds=0, last_design_face=started-60;
    int design_face_count=0, joint_queue_count=0;
    std::set<int> face_heads;
    std::set<std::string> face_keys;
    std::vector<double> latest_master_point;
    auto attach_queue_records = [&](Json& result) {
        result["primal_alns_summary"]=primal_alns ? primal_alns->summary() : Json{{"enabled",false}};
        result["primal_alns_records"]=primal_alns ? primal_alns->records() : Json::array();
        result["primal_alns_schedule"]={{"enabled",options.primal_alns},{"budget",primal_schedule.limit()},
            {"seconds",primal_schedule.spent()},{"budget_scope","end_to_end_configured_seconds"},
            {"budget_overrun_seconds",std::max(0.0,primal_schedule.spent()-primal_schedule.limit())},
            {"local_deadline_enforcement","solver_limit_and_construction_validation_boundaries"},{"startup_seconds",primal_schedule.used(PrimalPhase::startup)},
            {"regular_seconds",primal_schedule.used(PrimalPhase::regular)},
            {"elite_seconds",primal_schedule.used(PrimalPhase::elite)},{"cg_yields",primal_yields},
            {"records",primal_schedule_records},{"elite_records",elite_records}};
        result["continuous_root_stage_records"]=continuous_stage_records;
        result["continuous_root_schedule"]={{"enabled",options.staged_continuous_root},
            {"calls",continuous_stage_records.size()},{"seconds",continuous_progress.spent},
            {"total_allowance",continuous_total_limit},{"closed",continuous_progress.closed}};
        result["queue_polymatroid_cut_records"]=queue_records;
        result["queue_polymatroid_attempts"]=queue_attempts;
        result["queue_polymatroid_metadata"]=queue_metadata;
        result["queue_price_search_records"]=queue_price_records;
        result["target_master_refinement_records"]=target_master_records;
        result["queue_fractional_records"]=queue_fractional_records;
        result["design_face_cut_records"]=design_face_records;
        result["primal_seed_portfolio_records"]=primal_portfolio_records;
        result["primal_coupled_neighborhood_records"]=coupled_neighborhood_records;
    };
    Json root_records = Json::array();
    Json continuous_root = {{"enabled", options.continuous_root}, {"bound_valid", false},
                            {"termination_reason", "disabled"}};
    int continuous_bound_rows = 0;
    bool continuous_root_closed = false;
    Json neighborhood_candidates = Json::array();
    std::set<std::vector<double>> neighborhood_seen;
    int neighborhood_cg_calls = 0, neighborhood_cut_count = 0;
    double neighborhood_seconds = 0;
    double root_lower = 0, root_upper = upper;
    std::string root_reason = "disabled";
    Json timing = {{"master_build", master_build}, {"master_optimize", 0.0}, {"master_update", 0.0},
                   {"greedy_initial_columns", seeds.metadata.at("seconds")},
                   {"subproblem_wall", 0.0}, {"subproblem", Json::object()}};
    path_support_cache.metadata = {{"enabled",false}};
    if (options.path_support_cuts && remaining(deadline) > 0) {
        const double phase = wall_time();
        path_support_cache = build_path_support_cache(d, paths,
            std::min(deadline, phase + options.path_support_preprocessing_seconds));
        path_support_cache.metadata["enabled"] = true;
        for (auto cut : path_support_cache.static_cuts) {
            if (!cut.valid || !path_support_keys.insert(cut.key).second) continue;
            cut.row.name += "_"+std::to_string(cut.head)+"_"+std::to_string(cut.origin);
            master.matrix.rows.push_back(cut.row); solver.add_row(cut.row);
            Json record = cut.metadata;
            record.update({{"stage","preprocessing"},{"accepted",true},{"key",cut.key},
                {"row_name",cut.row.name},{"sense",std::string(1,cut.row.sense)},
                {"rhs",cut.row.rhs},{"coefficients",cut.row.coefficients}});
            path_support_records.push_back(std::move(record)); ++path_support_static;
        }
        add_time(timing,"path_support_preprocessing",phase);
    }
    auto path_support_summary = [&]() -> Json {
        return {{"enabled",options.path_support_cuts},{"complete_domain",path_support_cache.complete_domain},
            {"static_filters",path_support_static},{"dynamic_cuts",path_support_dynamic},
            {"cuts",path_support_records.size()},{"attempts",path_support_attempts.size()},
            {"checks",path_support_checks},{"blocked_groups",path_support_blocked},
            {"skipped_integer_subproblems",path_support_skipped},{"seconds",path_support_seconds},
            {"preprocessing_seconds",path_support_cache.seconds},{"master_added_variables",0},
            {"preserves_all_design_encodings",false},
            {"validity_scope","optimality_preserving_canonical_assignments"}};
    };
    std::unique_ptr<FractionalPathSupportSeparator> fractional_support_separator;
    if(options.fractional_path_support && path_support_cache.complete_domain)
        fractional_support_separator=std::make_unique<FractionalPathSupportSeparator>(d,path_support_cache,options);
    auto separate_fractional_support = [&](const std::vector<double>& point,const char* stage,double stage_deadline) {
        if(!fractional_support_separator || remaining(stage_deadline)<=0 ||
            path_support_dynamic>=static_cast<std::size_t>(options.path_support_cut_limit))return false;
        std::vector<double> design(master.layout.count);
        for(int j=0;j<master.layout.count;++j)design[j]=std::clamp(point[master.design_vars[j]],0.0,1.0);
        const auto limit=std::min<std::size_t>(options.path_support_max_cuts,
            options.path_support_cut_limit-path_support_dynamic);
        auto separated=fractional_support_separator->separate(design,
            std::min(stage_deadline,wall_time()+1.0),limit);
        path_support_seconds+=separated.seconds;fractional_support_seconds+=separated.seconds;
        int added=0;
        for(auto& cut:separated.cuts) {
            if(!cut.valid || !cut.violated || !path_support_keys.insert(cut.key).second)continue;
            cut.row.name+="_"+std::to_string(path_support_dynamic);
            master.matrix.rows.push_back(cut.row);solver.add_row(cut.row);
            Json record=cut.metadata;
            record.update({{"stage",stage},{"candidate_id",active_candidate},{"accepted",true},
                {"key",cut.key},{"violation",cut.violation},{"row_name",cut.row.name},
                {"sense",std::string(1,cut.row.sense)},{"rhs",cut.row.rhs},{"coefficients",cut.row.coefficients}});
            fractional_support_records.push_back(record);
            path_support_records.push_back(std::move(record));
            ++path_support_dynamic;++fractional_support_cuts;++added;
        }
        Json attempt=separated.metadata;
        attempt.update({{"stage",stage},{"candidate_id",active_candidate},{"added",added}});
        fractional_support_attempts.push_back(attempt);
        if(options.progress){attempt["event"]="fractional_path_support_separation";options.progress->record(attempt);}
        return added>0;
    };
    assignment_support_cache.metadata = {{"enabled", false}};
    if (((options.assignment_support_cuts && options.assignment_support_cut_limit > 0) || branch_joint_enabled)
        && remaining(deadline) > 0) {
        const double phase = wall_time();
        assignment_support_cache = build_assignment_support_cache(d, paths,
            std::min(deadline, phase + options.assignment_support_seconds));
        assignment_support_cache.metadata["enabled"] = true;
        add_time(timing, "assignment_support_preprocessing", phase);
    }
    branch_joint_cache.metadata = {{"enabled", false}};
    if (branch_joint_enabled && remaining(deadline) > 0) {
        const double phase = wall_time();
        branch_joint_cache = build_branch_joint_cache(d, paths, assignment_support_cache,
            std::min(deadline, phase + options.branch_joint_preprocessing_seconds), options.branch_joint_floor);
        branch_joint_cache.metadata["enabled"] = true;
        add_time(timing, "branch_joint_preprocessing", phase);
    }
    auto branch_joint_summary = [&]() -> Json {
        return {{"enabled",branch_joint_enabled},{"double_support_enabled",options.double_support_cuts},
            {"joint_base_cost_enabled",options.joint_base_cost_cuts},{"cuts",branch_joint_records.size()},
            {"double_cuts",double_support_count},{"joint_cuts",joint_base_count},
            {"attempts",branch_joint_attempts.size()},{"skipped_integer_subproblems",branch_joint_skipped},
            {"certificates",branch_joint_certificates},{"numerical_skips",branch_joint_numerical_skips},
            {"seconds",branch_joint_seconds},{"preprocessing_seconds",branch_joint_cache.seconds},
            {"master_added_variables",0},{"validity_scope","binary_designs_branch_support_and_distinct_origin_costs"}};
    };
    auto observe_global = [&](const char* stage, bool checkpoint = false) {
        if (!options.progress) return;
        options.progress->record({{"event","global_bounds"},{"stage",stage},{"objective",upper},{"upper_bound",upper},
            {"bound",lower},{"lower_bound",lower},{"gap",gap(upper,lower)},{"bound_scope","global"},
            {"validated_incumbent",true},{"certified_cuts",cuts},
            {"path_sizes",{{"segments",paths.initial_segments.size()},{"service",paths.service_paths.size()},
                {"empty",paths.empty_paths.size()}}}});
        if (checkpoint && options.progress->checkpoint_needed(upper)) {
            Json snapshot=best;
            synchronize_result_paths(d,paths,snapshot);
            snapshot["bound"]=lower; snapshot["gap"]=gap(upper,lower);
            snapshot["tight_airline_bounds"]=options.tight_airline_bounds;
            snapshot["od_airline_bounds"]=options.od_airline_bounds;
            snapshot["topology_summary"]={{"enabled",options.topology_cuts},{"cuts",topology_records.size()},
                {"attempts",topology_attempts.size()},{"skipped_integer_subproblems",topology_skipped},
                {"zero_service_certificates",topology_certificates},{"seconds",topology_seconds}};
            snapshot["assignment_support_summary"]={{"enabled",options.assignment_support_cuts},
                {"cuts",assignment_support_records.size()},{"attempts",assignment_support_attempts.size()},
                {"skipped_integer_subproblems",assignment_support_skipped},
                {"certificates",assignment_support_certificates},{"numerical_skips",assignment_support_numerical_skips},
                {"seconds",assignment_support_seconds},{"preprocessing_seconds",assignment_support_cache.seconds}};
            snapshot["construction_complete"]=true;
            snapshot["path_support_summary"]=path_support_summary();
            snapshot["airline_connectivity_metadata"]=airline_connectivity_metadata;
            snapshot["airline_resource_cut_records"]=airline_resource_records;
            attach_queue_records(snapshot);
            snapshot["queue_polymatroid_summary"]={{"enabled",options.queue_polymatroid_cuts},
                {"cuts",queue_records.size()},{"attempts",queue_attempts.size()},{"master_added_variables",0}};
            snapshot["airline_resource_attempts"]=airline_resource_attempts;
            snapshot["airline_resource_metadata"]=airline_resource_metadata;
            snapshot["airline_resource_summary"]={{"enabled",options.airline_resource_cuts},
                {"cuts",airline_resource_records.size()},{"attempts",airline_resource_attempts.size()},
                {"skipped_integer_subproblems",airline_resource_skipped},{"master_added_variables",0}};
            snapshot["branch_joint_summary"]=branch_joint_summary();
            snapshot["support_complete"]=paths.support_complete;
            snapshot["catalogue_complete"]=paths.complete;
            snapshot["generated_columns"]={{"service",d.preprocessed_demand?used_p.size():paths.service_paths.size()},
                {"empty",d.preprocessed_demand?used_e.size():paths.empty_paths.size()}};
            snapshot["generated_columns_scope"]=d.preprocessed_demand?
                "completed_recourse_column_union_including_continuous_root":"registered_generated_catalogue";
            snapshot["reusable_columns"]={{"service",pool_p.size()},{"empty",pool_e.size()}};
            options.progress->checkpoint(snapshot,stage);
        }
    };
    observe_global("benders_initial_baseline",true);
    CapacitySavingsNetwork capacity_network;
    const bool capacity_enabled = options.capacity_cuts && options.capacity_cut_limit > 0;
    Json capacity_records = Json::array();
    std::set<std::string> capacity_partitions;
    if (capacity_enabled) {
        const double phase = wall_time();
        capacity_network = capacity_savings_network(d);
        add_time(timing,"capacity_cut_build",phase);
    }
    const bool resource_enabled = options.resource_lp_cuts && options.resource_lp_cut_limit > 0;
    const bool od_strengthened_root = resource_enabled && options.od_airline_bounds;
    const bool integer_strengthened_root = options.design_face_cuts || options.queue_joint_cuts ||
        od_strengthened_root || path_support_static > 0 || options.fractional_path_support ||
        options.task_savings_bound || options.task_resource_cuts ||
        options.airline_connectivity_cuts || options.airline_resource_cuts || options.queue_polymatroid_cuts;
    const char* root_scope = (path_support_static > 0 || options.airline_connectivity_cuts || options.fractional_path_support) ? "optimality_preserving_support_root" :
        integer_strengthened_root ? "integer_valid_strengthened_root" : continuous_scope(options);
    std::unique_ptr<ResourceSavingsRelaxation> resource_relaxation;
    std::unique_ptr<TaskResourceSavingsRelaxation> task_resource_relaxation;
    std::unique_ptr<AirlineResourceRelaxation> airline_resource_relaxation;
    std::unique_ptr<QueuePolymatroidCuts> queue_separator;
    std::set<std::string> queue_keys;
    if(options.queue_polymatroid_cuts && remaining(deadline)>0) {
        const double phase=wall_time();
        queue_separator=std::make_unique<QueuePolymatroidCuts>(d,paths,options,
            (options.task_savings_bound || options.task_resource_cuts) ? &task_data : nullptr);
        queue_metadata=queue_separator->metadata();
        add_time(timing,"queue_polymatroid_build",phase);
    }
    std::set<std::string> airline_resource_keys;
    if(options.airline_resource_cuts && options.airline_resource_cut_limit>0 && remaining(deadline)>0) {
        const double phase=wall_time();
        airline_resource_relaxation=std::make_unique<AirlineResourceRelaxation>(d,paths,options,options.airline_resource_balance);
        airline_resource_metadata=airline_resource_relaxation->metadata();
        add_time(timing,"airline_resource_build",phase);
    }
    std::set<std::string> task_resource_keys;
    if(options.task_resource_cuts && options.task_resource_cut_limit>0 && remaining(deadline)>0) {
        const double phase=wall_time();
        task_resource_relaxation=std::make_unique<TaskResourceSavingsRelaxation>(d,task_data,options);
        add_time(timing,"task_resource_build",phase);
    }
    std::set<std::string> resource_keys;
    if (resource_enabled && remaining(deadline) > 0) {
        const double phase = wall_time();
        resource_relaxation = capacity_enabled
            ? std::make_unique<ResourceSavingsRelaxation>(resource_savings_model(d, capacity_network, options.resource_airline_bounds, options.tight_airline_bounds, options.od_airline_bounds), options)
            : std::make_unique<ResourceSavingsRelaxation>(d, options);
        add_time(timing, "resource_lp_build", phase);
    }
    auto separate_capacity = [&](const std::vector<double>& point,const char* stage) {
        if (!capacity_enabled || static_cast<int>(capacity_records.size())>=options.capacity_cut_limit ||
            remaining(deadline)<=0) return false;
        const double phase = wall_time();
        std::vector<double> design(master.layout.count);
        for (int j=0;j<master.layout.count;++j) design[j]=std::clamp(point[master.design_vars[j]],0.0,1.0);
        auto cut=capacity_network.separate(design,point[theta],theta);
        const bool insert=cut.violated && capacity_partitions.insert(cut.partition_key).second;
        if (insert) {
            cut.row.name += "_"+std::to_string(capacity_records.size());
            master.matrix.rows.push_back(cut.row); solver.add_row(cut.row);
            capacity_records.push_back({{"stage",stage},{"candidate_id",active_candidate},{"violation",cut.violation},
                {"cut_capacity",cut.cut_capacity},{"maxflow_diagnostic",cut.maxflow},
                {"rhs",cut.row.rhs},{"coefficients",cut.row.coefficients},
                {"partition",cut.partition_key},{"certificate","original_crossing_affine_capacities"}});
        }
        add_time(timing,"capacity_cut_separation",phase);
        return insert;
    };
    auto separate_resource = [&](const std::vector<double>& point, const char* stage, double stage_deadline) {
        if (!resource_relaxation || static_cast<int>(resource_records.size()) >= options.resource_lp_cut_limit ||
            remaining(stage_deadline) <= 0) return false;
        const double phase = wall_time();
        std::vector<double> design(master.layout.count);
        for (int j = 0; j < master.layout.count; ++j)
            design[j] = std::clamp(point[master.design_vars[j]], 0.0, 1.0);
        auto cut = resource_relaxation->separate(design, point[theta], theta,
            std::min(options.resource_lp_seconds, remaining(stage_deadline)));
        const bool insert = cut.valid && cut.violated && resource_keys.insert(cut.key).second;
        Json record = {{"stage", stage}, {"candidate_id", active_candidate}, {"accepted", insert},
            {"valid", cut.valid}, {"violated", cut.violated}, {"optimal", cut.optimal},
            {"fallback", cut.fallback}, {"status", cut.status}, {"seconds", cut.seconds},
            {"solver_seconds", cut.solver_seconds}, {"violation", cut.violation},
            {"dual_savings_upper_bound", cut.dual_savings},
            {"max_dual_violation_before", cut.max_dual_violation_before},
            {"max_dual_violation_after", cut.max_dual_violation_after},
            {"validity_scope",options.od_airline_bounds ? "integer_design_convex_hull" : "original_continuous_relaxation"},
            {"certificate", "full_occurrence_domain_dual_feasibility"}};
        const std::size_t first_od = resource_relaxation->metadata().value("legacy_airline_resource_rows", cut.airline_dual.size());
        std::size_t active_od = 0;
        if (options.od_airline_bounds)
            for (std::size_t a = first_od; a < cut.airline_dual.size(); ++a)
                if (cut.airline_dual[a] > 1e-10) ++active_od;
        record["active_od_airline_prices"] = active_od;
        resource_attempts.push_back(record);
        if (insert) {
            cut.row.name += "_" + std::to_string(resource_records.size());
            master.matrix.rows.push_back(cut.row); solver.add_row(cut.row);
            record["rhs"] = cut.row.rhs; record["coefficients"] = cut.row.coefficients;
            record["demand_dual"] = cut.demand_dual; record["locker_dual"] = cut.locker_dual;
            record["airline_dual"] = cut.airline_dual;
            resource_records.push_back(record);
        }
        if (options.progress) {
            record["event"] = "resource_lp_separation";
            options.progress->record(std::move(record));
        }
        add_time(timing, "resource_lp_separation", phase);
        return insert;
    };
    auto separate_task_resource = [&](const std::vector<double>& point,const char* stage,double stage_deadline) {
        if(!task_resource_relaxation || static_cast<int>(task_resource_records.size())>=options.task_resource_cut_limit ||
           remaining(stage_deadline)<=0) return false;
        const double phase=wall_time();
        std::vector<double> design(master.layout.count);
        for(int j=0;j<master.layout.count;++j) design[j]=std::clamp(point[master.design_vars[j]],0.0,1.0);
        auto cut=task_resource_relaxation->separate(design,point[theta],theta,
            std::min(options.task_resource_seconds,remaining(stage_deadline)));
        const bool insert=cut.valid && cut.violated && task_resource_keys.insert(cut.key).second;
        Json record={{"stage",stage},{"candidate_id",active_candidate},{"accepted",insert},
            {"valid",cut.valid},{"violated",cut.violated},{"optimal",cut.optimal},{"fallback",cut.fallback},
            {"status",cut.status},{"seconds",cut.seconds},{"solver_seconds",cut.solver_seconds},
            {"violation",cut.violation},{"dual_savings_upper_bound",cut.dual_savings},
            {"max_dual_violation_before",cut.max_dual_violation_before},
            {"max_dual_violation_after",cut.max_dual_violation_after},
            {"validity_scope","all_original_integer_designs"},
            {"certificate","full_task_resource_dual_feasibility"}};
        task_resource_attempts.push_back(record);
        if(insert) {
            cut.row.name+="_"+std::to_string(task_resource_records.size());
            master.matrix.rows.push_back(cut.row);solver.add_row(cut.row);
            record["rhs"]=cut.row.rhs;record["coefficients"]=cut.row.coefficients;
            record["dual_prices"]=cut.dual_prices;
            task_resource_records.push_back(record);
        }
        if(options.progress) {record["event"]="task_resource_separation";options.progress->record(record);}
        add_time(timing,"task_resource_separation",phase);
        return insert;
    };
    auto separate_airline_resource = [&](const std::vector<double>& point,const char* stage,double stage_deadline) {
        if(!airline_resource_relaxation || static_cast<int>(airline_resource_records.size())>=options.airline_resource_cut_limit ||
           remaining(stage_deadline)<=0) return false;
        const double phase=wall_time();
        std::vector<double> design(master.layout.count);
        for(int j=0;j<master.layout.count;++j) design[j]=std::clamp(point[master.design_vars[j]],0.0,1.0);
        auto cut=airline_resource_relaxation->separate(design,point[theta],theta,
            std::min(options.airline_resource_seconds,remaining(stage_deadline)));
        double profile_seconds=0;
        if(queue_separator && cut.valid && remaining(stage_deadline)>0) {
            const double profile_started=wall_time();
            queue_separator->add_profile(airline_resource_relaxation->queue_prices(cut.dual_prices),
                std::min(1.0,remaining(stage_deadline)));
            queue_metadata=queue_separator->metadata();
            profile_seconds=wall_time()-profile_started;
            timing["queue_polymatroid_profiles"]=timing.value("queue_polymatroid_profiles",0.0)+profile_seconds;
        }
        const bool insert=cut.valid && cut.violated && airline_resource_keys.insert(cut.key).second;
        Json record={{"stage",stage},{"candidate_id",active_candidate},{"accepted",insert},
            {"valid",cut.valid},{"violated",cut.violated},{"optimal",cut.optimal},{"fallback",cut.fallback},
            {"status",cut.status},{"seconds",cut.seconds},{"solver_seconds",cut.solver_seconds},
            {"violation",cut.violation},{"dual_savings_upper_bound",cut.dual_savings},
            {"max_dual_violation_before",cut.max_dual_violation_before},
            {"max_dual_violation_after",cut.max_dual_violation_after},
            {"validity_scope","all_original_integer_designs"},
            {"certificate","full_airline_resource_dual_feasibility"}};
        airline_resource_attempts.push_back(record);
        if(insert) {
            cut.row.name+="_"+std::to_string(airline_resource_records.size());
            master.matrix.rows.push_back(cut.row);solver.add_row(cut.row);
            record["rhs"]=cut.row.rhs;record["coefficients"]=cut.row.coefficients;
            record["dual_prices"]=cut.dual_prices;
            airline_resource_records.push_back(record);
        }
        if(options.progress) {record["event"]="airline_resource_separation";options.progress->record(record);}
        timing["airline_resource_separation"]=timing.value("airline_resource_separation",0.0)+wall_time()-phase-profile_seconds;
        return insert;
    };
    std::string reason = "iteration_limit";
    auto queue_attempt_record = [](const TaskSavingsCut& cut, Json context) {
        context.update({{"valid",cut.valid},{"violated",cut.violated},{"status",cut.status},
            {"violation",cut.violation},{"seconds",cut.seconds},
            {"validity_scope","all_original_integer_designs"}});
        return context;
    };
    auto describe_queue_row = [](Json& record,const TaskSavingsCut& cut,
                                 const Json& design,const Json& separation_theta) {
        record.update({{"rhs",cut.row.rhs},{"coefficients",cut.row.coefficients},
            {"certificate",cut.certificate},{"design",design},{"theta",separation_theta}});
    };
    auto separate_face = [&](const std::vector<double>& point,const char* stage,double stage_deadline) {
        if(!options.design_face_cuts || !d.preprocessed_demand || !d.origin_assignment() ||
           d.n_l<=0 || d.max_orders!=2 || !pending_design.empty() ||
           point.size()!=master.matrix.variables.size() || wall_time()-last_design_face<60)return false;
        const double allowance=std::min({12.0,.08*options.seconds-design_face_seconds,
                                         .2*remaining(stage_deadline)});
        if(allowance<=.05 || remaining(stage_deadline)<1)return false;
        int head=-1;double score=-1;
        for(int r=0;r<d.assignment_entities();++r) {
            if(face_heads.count(r) || d.task_rate(r)<=0)continue;
            double fractional=0,other_mass=0,largest=0;
            for(int l=0;l<d.n_l;++l) {
                const double a=std::clamp(point[master.layout.assignment_index(r,l)],0.0,1.0);
                fractional+=a*(1-a);
                if(a>largest){other_mass+=largest;largest=a;}else other_mass+=a;
            }
            // Sum outside the largest base equals min(alpha_1,alpha_2) for L2.
            const double candidate=fractional+other_mass+1e-9*d.task_rate(r);
            if(candidate>score){score=candidate;head=r;}
        }
        if(head<0)return false;
        std::vector<double> design(master.layout.count);
        for(int j=0;j<master.layout.count;++j)design[j]=std::clamp(point[master.design_vars[j]],0.0,1.0);
        const double phase=wall_time();
        if(options.cg_seed_reuse)
            continuous_columns.include(d,paths,std::vector<int>(pool_p.begin(),pool_p.end()),
                                       std::vector<int>(pool_e.begin(),pool_e.end()));
        auto separated=separate_design_face(d,paths,options,head,design,point[theta],theta,lower,
            std::max(0.0,allowance-(wall_time()-phase)),options.cg_seed_reuse?&continuous_columns:nullptr);
        if(options.cg_seed_reuse) {
            used_p.insert(continuous_columns.service().begin(),continuous_columns.service().end());
            used_e.insert(continuous_columns.empty().begin(),continuous_columns.empty().end());
            if(options.reuse_columns) {
                pool_p.insert(continuous_columns.service().begin(),continuous_columns.service().end());
                pool_e.insert(continuous_columns.empty().begin(),continuous_columns.empty().end());
            }
        }
        const double elapsed=wall_time()-phase;
        design_face_seconds+=elapsed;last_design_face=wall_time();face_heads.insert(head);
        auto& cut=separated.cut;
        const bool insert=cut.valid && cut.violated &&
            cut.certificate.value("improved_faces",0)>0 && face_keys.insert(cut.key).second;
        Json record={{"stage",stage},{"head",head},{"accepted",insert},{"seconds",elapsed},
            {"budget_seconds",allowance},{"status",cut.status},{"violation",cut.violation},
            {"diagnostics",separated.diagnostics},{"faces",separated.faces},
            {"direct_global_bound_update",false},{"certificate",cut.certificate}};
        if(insert) {
            cut.row.name+="_"+std::to_string(design_face_count++);
            master.matrix.rows.push_back(cut.row);solver.add_row(cut.row);
            record["rhs"]=cut.row.rhs;record["coefficients"]=cut.row.coefficients;
        }
        design_face_records.push_back(record);timing["design_face_separation"]=design_face_seconds;
        if(options.progress){record["event"]="design_face_separation";options.progress->record(record);}
        return insert;
    };
    auto separate_queue = [&](const std::vector<double>& point,const char* stage,double stage_deadline,
                              const QueueNativePrices* native_prices=nullptr) {
        if(!queue_separator || queue_records.size()>=1000 || remaining(stage_deadline)<=0) return false;
        const double allowance=std::min({.5,remaining(stage_deadline),.02*options.seconds-queue_price_seconds});
        if(native_prices && (!options.queue_price_search || !native_prices->valid || allowance<=0)) return false;
        const double phase=wall_time();
        std::vector<double> design(master.layout.count);
        for(int j=0;j<master.layout.count;++j)
            design[j]=std::clamp(point[master.design_vars[j]],0.0,1.0);
        TaskSavingsCut cut;
        Json search_diagnostics, joint_diagnostics=Json::object();
        if(native_prices) {
            auto search=queue_separator->search_prices(native_prices->prices,design,point[theta],theta,allowance);
            cut=std::move(search.cut);search_diagnostics=std::move(search.diagnostics);
        } else {
            cut=queue_separator->separate(design,point[theta],theta);
            const double joint_allowance=std::min({.05,.01*options.seconds-joint_queue_seconds,
                                                   remaining(stage_deadline)});
            if(options.queue_joint_cuts && joint_allowance>0) {
                const double joint_started=wall_time();
                TaskSavingsCut joint;
                try {joint=queue_separator->separate_joint(design,point[theta],theta,joint_allowance);}
                catch(const std::exception& error) {joint_diagnostics["error"]=error.what();}
                joint_queue_seconds+=wall_time()-joint_started;
                joint_diagnostics.update({{"status",joint.status},{"valid",joint.valid},
                    {"violation",joint.violation},{"seconds",wall_time()-joint_started}});
                if(joint.valid && joint.violated && !queue_keys.count(joint.key) &&
                   (!cut.valid || queue_keys.count(cut.key) || joint.violation>cut.violation))cut=std::move(joint);
            }
        }
        queue_metadata=queue_separator->metadata();
        const bool insert=cut.valid && cut.violated && queue_keys.insert(cut.key).second;
        Json record=queue_attempt_record(cut,
            {{"stage",stage},{"candidate_id",active_candidate},{"accepted",insert}});
        if(options.queue_joint_cuts)record["joint"]=joint_diagnostics;
        queue_attempts.push_back(record);
        if(insert) {
            if(cut.certificate.value("kind",std::string())=="queue_joint")++joint_queue_count;
            cut.row.name+="_"+std::to_string(queue_records.size());
            master.matrix.rows.push_back(cut.row);solver.add_row(cut.row);
            describe_queue_row(record,cut,design,point[theta]);
            queue_records.push_back(record);
        }
        if(options.progress) {record["event"]="queue_polymatroid_separation";options.progress->record(record);}
        if(native_prices) {
            const double elapsed=wall_time()-phase;
            queue_price_seconds+=elapsed;
            queue_price_records.push_back({{"stage",stage},{"candidate_id",active_candidate},
                {"donor_status",native_prices->status},{"diagnostics",search_diagnostics},
                {"seconds",elapsed},{"time_budget",allowance},{"accepted",insert}});
            timing["queue_price_search"]=queue_price_seconds;
            if(options.progress) {
                Json search_record=queue_price_records.back();search_record["event"]="queue_price_search";
                options.progress->record(std::move(search_record));
            }
        } else add_time(timing,"queue_polymatroid_separation",phase);
        return insert;
    };
    auto refresh_fractional_queue = [&](const char* stage,double stage_deadline) {
        if(!options.queue_fractional_separation || !queue_separator ||
           !deferred_queue_prices.valid || queue_records.size()+2>1000)return false;
        const auto allowance=queue_fractional_budget(remaining(stage_deadline),options.seconds,
            queue_fractional_lp_seconds,queue_price_seconds,wall_time()-last_queue_fractional,
            deferred_queue_serial>consumed_queue_serial,!pending_design.empty());
        if(!allowance)return false;
        const double phase=wall_time(), lower_before=lower;
        const double previous_finished=last_queue_fractional-started;
        const std::size_t call=queue_fractional_records.size();
        const std::string candidate="fractional:"+std::to_string(call+1);
        auto separated=separate_fractional_queue(master.matrix,master.layout,theta,*queue_separator,
            deferred_queue_prices,queue_keys,options,allowance->seconds,
            allowance->lp_seconds,allowance->price_seconds);
        for(const auto& round:separated.diagnostics.at("rounds"))
            if(round.value("optimal",false) && round.value("feasible",false) &&
               round.at("actual_design").is_array() && round.at("theta").is_number()) {
                const auto design=round.at("actual_design").get<std::vector<double>>();
                if(design.size()==static_cast<std::size_t>(master.layout.count)) {
                    latest_master_point.assign(master.matrix.variables.size(),0.0);
                    for(int j=0;j<master.layout.count;++j)latest_master_point[master.design_vars[j]]=design[j];
                    latest_master_point[theta]=round.at("theta").get<double>();
                }
            }
        consumed_queue_serial=deferred_queue_serial;
        std::size_t next_cut=0;
        for(std::size_t round=0;round<separated.diagnostics.at("rounds").size();++round) {
            if(!separated.diagnostics.at("rounds")[round].value("accepted",false))continue;
            auto& cut=separated.cuts.at(next_cut++);
            if(cut.certificate.value("kind",std::string())=="queue_joint")++joint_queue_count;
            if(!cut.valid || !queue_keys.insert(cut.key).second)
                throw std::runtime_error("Fractional queue row synchronization failed");
            cut.row.name+="_"+std::to_string(queue_records.size());
            master.matrix.rows.push_back(cut.row);solver.add_row(cut.row);
            Json record=queue_attempt_record(cut,
                {{"stage",stage},{"candidate_id",candidate},{"donor_candidate_id",deferred_queue_candidate},
                 {"fractional_call",call},{"fractional_round",round},{"accepted",true}});
            queue_attempts.push_back(record);
            describe_queue_row(record,cut,cut.certificate.at("separation_design"),
                cut.certificate.at("separation_theta"));
            queue_records.push_back(record);
            if(options.progress) {record["event"]="queue_polymatroid_separation";options.progress->record(record);}
        }
        if(next_cut!=separated.cuts.size())
            throw std::runtime_error("Fractional queue round/cut accounting failed");
        // Every row used by the temporary LP is now present in the real master.
        // Its solved relaxation objective is a bound; a cut's point value is not.
        if(std::isfinite(separated.certified_lower)) {
            if(separated.certified_lower>upper+1e-7*(1+std::abs(upper)))
                throw std::runtime_error("Fractional queue master contradicts incumbent");
            lower=std::max(lower,std::min(upper,separated.certified_lower));
            if(lower>lower_before+1e-7*(1+std::abs(lower)))bound_stalled_passes=0;
        }
        for(std::size_t i=0;i<separated.price_records.size();++i) {
            Json record=separated.price_records[i];
            record.update({{"stage",stage},{"candidate_id",candidate},{"donor_candidate_id",deferred_queue_candidate},
                {"donor_status",deferred_queue_prices.status},{"fractional_call",call},{"local_price_index",i}});
            queue_price_records.push_back(record);
            if(options.progress) {record["event"]="queue_price_search";options.progress->record(record);}
        }
        queue_metadata=queue_separator->metadata();
        const double elapsed=wall_time()-phase;
        const double lp_elapsed=std::max(0.0,elapsed-separated.price_seconds);
        queue_fractional_lp_seconds+=lp_elapsed;
        queue_price_seconds+=separated.price_seconds;
        queue_fractional_wall+=elapsed;
        last_queue_fractional=wall_time();
        timing["queue_fractional_master"]=queue_fractional_lp_seconds;
        timing["queue_price_search"]=queue_price_seconds;
        Json record={{"stage",stage},{"candidate_id",candidate},{"donor_candidate_id",deferred_queue_candidate},
            {"donor_serial",deferred_queue_serial},{"started_elapsed_seconds",phase-started},
            {"previous_call_elapsed_seconds",previous_finished},{"global_lower_before",lower_before},
            {"global_lower_after",lower},{"global_upper",upper},{"time_budget",allowance->seconds},
            {"lp_budget",allowance->lp_seconds},{"price_budget",allowance->price_seconds},
            {"lp_seconds",lp_elapsed},{"price_seconds",separated.price_seconds},{"seconds",elapsed},
            {"certified_lower",std::isfinite(separated.certified_lower)?Json(separated.certified_lower):Json(nullptr)},
            {"diagnostics",separated.diagnostics},{"bound_scope","optimality_preserving_master_relaxation"}};
        queue_fractional_records.push_back(record);
        if(options.progress) {record["event"]="queue_fractional_separation";options.progress->record(record);}
        observe_global("queue_fractional_master_finished");
        return !separated.cuts.empty();
    };
    Json budget_interruption = nullptr;
    bool pricing_complete = false;
    Json last_subproblem_stage = nullptr, last_subproblem_termination = nullptr;
    auto keep_cut = [&](const Cut& cut, const char* kind) {
        if (!cut.valid) return;
        RowRecord row;
        row.name = (cut.condition.is_null()?"":"conditional_") + std::string(kind) + "_cut_" + std::to_string(cuts);
        row.sense = 'G'; row.rhs = cut.intercept; row.coefficients.emplace_back(theta, 1);
        for (std::size_t j = 0; j < cut.coefficients.size(); ++j)
            if (cut.coefficients[j] != 0) row.coefficients.emplace_back(master.design_vars[j], -cut.coefficients[j]);
        const bool managed=cut.condition.is_null() && (options.cut_pool_deduplication || options.cut_pool_management);
        if(managed && options.cut_pool_deduplication && certified_cut_pool.duplicate(row))return;
        if (!cut.condition.is_null()) conditional_row_indices.insert(master.matrix.rows.size());
        master.matrix.rows.push_back(row);
        solver.add_row(row);
        if(managed)certified_cut_pool.remember(row,static_cast<int>(master.matrix.rows.size())-1,master.matrix);
        cut_records.push_back({{"kind", kind}, {"candidate_id",active_candidate},{"intercept", cut.intercept},
                               {"coefficients", cut.coefficients}, {"correction", cut.correction},
                               {"stationarity_error", cut.stationarity_error}, {"pricing_complete", true},
                               {"catalogue_complete", paths.complete}, {"support_complete", paths.support_complete},
                               {"condition",cut.condition},{"anchor_tight",cut.optimal},
                               {"certificate",cut.diagnostics.value("full_domain_certificate",Json(nullptr))}});
        if (!cut.condition.is_null()) {
            ++conditional_cuts;
            assignment_patterns.insert(cut.condition.at("assignment_fingerprint").get<std::string>());
        }
        ++cuts;
        if (std::string(kind) == "pareto") ++pareto_count;
        else if (std::string(kind) == "auxiliary_valid") ++auxiliary_count;
        else { ++ordinary_count; if (std::string(kind) == "root") ++root_count; }
    };
    auto update_cutoff = [&]() {
        if (cutoff_row < 0) return;
        double rhs = upper + 1e-8 * (1 + std::abs(upper));
        master.matrix.rows[cutoff_row].rhs = rhs;
        solver.set_rhs(cutoff_row, 'L', rhs);
    };
    auto account = [&](Subproblem& sub, bool heuristic = false, const char* stage = "benders") {
        if (std::string(stage)!="root_relaxation" && std::string(stage)!="interior_cut" && !sub.model_size.is_null()) {
            subproblem_size=sub.model_size;
            if (subproblem_peak.is_null()) subproblem_peak=sub.model_size;
            else for (const char* key:{"variables","linear_rows","cones","nonzeros"})
                subproblem_peak[key]=std::max(subproblem_peak.at(key).get<std::size_t>(),sub.model_size.at(key).get<std::size_t>());
        }
        if (sub.conditional_demand) ++conditional_subproblems;
        timing["subproblem_wall"] = timing["subproblem_wall"].get<double>() + sub.wall;
        if (std::string(stage) == "root_relaxation")
            timing["root_projection_cg"] = timing.value("root_projection_cg", 0.0) + sub.wall;
        for (auto it = sub.timings.begin(); it != sub.timings.end(); ++it)
            timing["subproblem"][it.key()] = timing["subproblem"].value(it.key(), 0.0) + it.value().get<double>();
        cg_iterations += sub.iterations;
        (heuristic ? heuristic_cg_iterations : ordinary_cg_iterations) += sub.iterations;
        used_p.insert(sub.service.begin(),sub.service.end());
        used_e.insert(sub.empty.begin(),sub.empty.end());
        if (options.reuse_columns) {
            pool_p.insert(sub.service.begin(), sub.service.end());
            pool_e.insert(sub.empty.begin(), sub.empty.end());
        }
        retain_incumbent(sub.solution,"optimized");
        if (options.incumbent_cleanup && !sub.solution.is_null()) {
            double phase = wall_time();
            Json cleaned = cleanup_incumbent(d, paths, sub.solution);
            double saving = sub.solution["objective"].get<double>() - cleaned["objective"].get<double>();
            if (saving > 1e-8) { ++cleanup_improvements; cleanup_savings += saving; }
            sub.solution = std::move(cleaned);
            add_time(timing, "incumbent_cleanup", phase);
        }
        retain_incumbent(sub.solution,"optimized");
        double phase = wall_time();
        keep_cut(sub.ordinary, std::string(stage) == "root_relaxation" ? "root" : "ordinary");
        const bool retain_auxiliary = options.auxiliary_valid_cuts && sub.auxiliary.valid &&
            sub.pareto.value("core_gain", 0.0) > 1e-7 * (1 + courier_cost(d));
        if (sub.pareto.value("attempted", false)) {
            ++pareto_attempts;
            pareto_seconds += sub.timings.at("pareto").get<double>();
            pareto_failures = (sub.pareto.value("accepted", false) || retain_auxiliary)
                ? 0 : pareto_failures + 1;
        }
        if (sub.pareto.value("accepted", false)) keep_cut(sub.selected, "pareto");
        else if (retain_auxiliary) {
            keep_cut(sub.auxiliary, "auxiliary_valid");
            sub.pareto["retained_as_valid_cut"] = true;
        }
        update_cutoff();
        add_time(timing, "master_update", phase);
        pricing_complete = sub.ordinary.valid;
        last_subproblem_stage = stage;
        last_subproblem_termination = sub.reason;
        if(options.queue_fractional_separation && sub.ordinary.queue_prices.valid &&
           (std::string(stage)=="root_relaxation" || std::string(stage)=="benders")) {
            deferred_queue_prices=sub.ordinary.queue_prices;
            deferred_queue_candidate=active_candidate;
            ++deferred_queue_serial;
            if(options.progress) options.progress->record({{"event","queue_native_price_cached"},
                {"candidate_id",active_candidate},{"donor_serial",deferred_queue_serial},{"stage",stage}});
        }
        observe_global(stage,true);
    };
    auto separate_neighborhood = [&]() {
        const double available = options.seconds * options.neighborhood_cut_budget_fraction - neighborhood_seconds;
        if (!options.neighborhood_cuts || neighborhood_candidates.empty() || available <= .01 || remaining(deadline) <= .01)
            return;
        // Choose a design whose known feasible operating cost is most
        // underestimated by the CURRENT global affine cuts. This is only a
        // candidate score, never a recourse certificate or global bound.
        int chosen = 0;
        double largest_score = -INFINITY, predicted_theta = 0;
        for (std::size_t i = 0; i < neighborhood_candidates.size(); ++i) {
            auto x = neighborhood_candidates[i].at("design").get<std::vector<double>>();
            double floor = 0, fixed = 0;
            for (int j = 0; j < master.layout.count; ++j) fixed += master.matrix.variables[j].cost * x[j];
            for (const auto& row : master.matrix.rows) {
                if (row.sense != 'G') continue;
                double coefficient = 0, other = 0;
                for (auto [j, a] : row.coefficients) {
                    if (j == theta) coefficient += a;
                    else other += a * x[j];
                }
                if (coefficient > 0) floor = std::max(floor, (row.rhs - other) / coefficient);
            }
            const double score = neighborhood_candidates[i].at("objective").get<double>() - fixed - floor;
            if (score > largest_score) { chosen = static_cast<int>(i); largest_score = score; predicted_theta = floor; }
        }
        const Json candidate = neighborhood_candidates[chosen];
        neighborhood_candidates.erase(neighborhood_candidates.begin() + chosen);
        const auto design = candidate.at("design").get<std::vector<double>>();
        const std::string previous_candidate = active_candidate;
        active_candidate = "neighborhood:" + std::to_string(++neighborhood_cg_calls);
        const int cuts_before = cuts;
        std::vector<int> selected_p(pool_p.begin(), pool_p.end()), selected_e(pool_e.begin(), pool_e.end());
        const double allowance = std::min({available, options.neighborhood_cut_seconds, .1 * remaining(deadline)});
        SolverOptions neighborhood_options=options;
        auto sub = subproblem(d, paths, design, neighborhood_options, std::min(deadline, wall_time() + allowance),
                              selected_p, selected_e, nullptr,retain_recourse,
                              false,false,false,-1,-1,std::numeric_limits<double>::quiet_NaN(),nullptr,&pricing_graph_cache);
        neighborhood_seconds += sub.wall;
        account(sub, true, "neighborhood_cut");
        neighborhood_cut_count += cuts - cuts_before;
        heuristic_records.push_back({{"stage", "neighborhood_cut"}, {"candidate_id", active_candidate},
            {"candidate_source", candidate.value("source", "restricted_mip_pool")},
            {"candidate_score", largest_score}, {"master_operating_floor", predicted_theta},
            {"time_budget", allowance}, {"wall", sub.wall}, {"benders_cuts_added", cuts - cuts_before},
            {"pricing_complete", sub.ordinary.valid}, {"termination_reason", sub.reason},
            {"cg_iterations", sub.iterations}, {"cg_trace", sub.trace},
            {"regional_bound_used", false}});
        if (options.progress) {
            Json record = heuristic_records.back(); record["event"] = "neighborhood_cut_evaluation";
            options.progress->record(std::move(record));
        }
        active_candidate = previous_candidate;
    };
    auto restricted_search = [&](const char* stage, int radius) {
        const double available = options.seconds * options.heuristic_budget_fraction - heuristic_seconds;
        if (options.primal_alns || !options.restricted_mip || available <= .01 || remaining(deadline) <= .01) return;
        const double before = upper, phase = wall_time();
        RestrictedSearchCharge charge(heuristic_seconds,restricted_seconds,last_restricted_search,
                                      phase,[](){return wall_time();});
        const bool polish = options.primal_polish;
        const auto mode=restricted_search_mode(ordinary_restricted_attempts,radius,polish,
            options.primal_coupled_neighborhood,options.primal_asset_neighborhood,best.value("served",0.0)>0);
        const bool coupled=mode.coupled, fixed_assignment=mode.fixed_assignment,
            fixed_infrastructure=mode.fixed_infrastructure;
        // Spend the same total heuristic budget in useful joint asset/flow
        // searches, rather than repeatedly restarting a full-catalogue MIP.
        const double call_limit = polish ? (fixed_assignment ? 60.0 : 30.0) : 15.0;
        const double allowance = polish
            ? std::min({available, call_limit, remaining(deadline)})
            : std::min({available, call_limit, .08 * options.seconds, remaining(deadline)});
        std::vector<int> selected_p, selected_e;
        if (polish) {
            selected_p.assign(pool_p.begin(), pool_p.end());
            selected_e.assign(pool_e.begin(), pool_e.end());
        }
        ++restricted_attempts;
        if(radius>=0)++local_branching_attempts;
        else ++ordinary_restricted_attempts;
        Json attempt;
        if(coupled) {
            std::vector<double> guide;
            if(latest_master_point.size()==master.matrix.variables.size()) {
                guide.resize(master.layout.count);
                for(int j=0;j<master.layout.count;++j)guide[j]=latest_master_point[master.design_vars[j]];
            }
            attempt=solve_primal_coupled_neighborhood(d,paths,options,phase+allowance,best,
                guide.empty()?nullptr:&guide,coupled_neighborhood_records.size());
        } else attempt=solve_restricted_heuristic(d, paths, options, phase + allowance, &best, radius,
            polish ? &selected_p : nullptr, polish ? &selected_e : nullptr, fixed_assignment,fixed_infrastructure);
        const double restricted_solver_seconds=wall_time()-phase;
        const char* incumbent_kind=coupled ? "primal_coupled_neighborhood" : radius>=0 ? "local_branching" :
            polish ? "primal_design_polish" : "restricted_column_mip";
        bool incumbent_improved=false;
        if(attempt.value("validated",false) && attempt.contains("solution"))
            incumbent_improved=retain_incumbent(attempt.at("solution"),incumbent_kind);
        if(incumbent_improved)++restricted_improvements;
        if (options.neighborhood_cuts && attempt.contains("design_pool")) {
            for (const auto& candidate : attempt.at("design_pool")) {
                auto x = candidate.at("design").get<std::vector<double>>();
                if (!candidate.value("validated", false) || neighborhood_seen.count(x)) continue;
                validate_design(d, x);
                if (neighborhood_candidates.size() >= static_cast<std::size_t>(2 * options.heuristic_pool_size)) break;
                neighborhood_seen.insert(x);
                neighborhood_candidates.push_back(candidate);
            }
        }
        if (attempt.value("validated", false) && attempt.contains("solution") && !attempt["solution"].is_null()) {
            Json candidate = attempt["solution"];
            if (options.incumbent_cleanup) candidate = cleanup_incumbent(d, paths, candidate);
            if(retain_incumbent(candidate,incumbent_kind) && !incumbent_improved)++restricted_improvements;
        }
        attempt.erase("solution");
        attempt["stage"] = stage;
        attempt["upper_before"] = before; attempt["upper_after"] = upper;
        attempt["primal_design_polish"] = polish && radius<0;
        attempt["primal_coupled_neighborhood"] = coupled;
        attempt["local_branching"] = radius>=0;
        attempt["local_branching_radius"] = radius;
        attempt["allocated_seconds"] = allowance;
        attempt["restricted_solver_seconds"] = restricted_solver_seconds;
        const std::size_t coupled_record=coupled_neighborhood_records.size();
        if(coupled) {
            coupled_neighborhood_records.push_back(attempt);
        }
        const std::size_t polish_record=primal_design_polish_records.size();
        if (polish && radius<0) {
            Json record = {{"stage",stage},{"fixed_assignment",fixed_assignment},
                {"fixed_infrastructure",fixed_infrastructure},
                {"allocated_seconds",allowance},{"seconds",0.0},
                {"elapsed_seconds",options.progress ? options.progress->elapsed() : wall_time()-started},
                {"upper_before",before},{"upper_after",upper},
                {"validated",attempt.value("validated",false)},
                {"status",attempt.value("status",std::string())},
                {"catalogue_service_paths",paths.service_paths.size()},
                {"requested_service_columns",selected_p.size()},
                {"requested_empty_columns",selected_e.size()},
                {"global_lower_bound_available",false}};
            for (const char* key : {"model_size", "column_selection", "timings", "node_count", "source"})
                if (attempt.contains(key)) record[key] = attempt.at(key);
            primal_design_polish_records.push_back(std::move(record));
        }
        attempt["cg_trace"] = Json::array();
        heuristic_records.push_back(std::move(attempt));
        update_cutoff();
        observe_global(stage,true);
        const double used=charge.finish();
        heuristic_records.back()["seconds"]=used;
        heuristic_records.back()["time_scope"]="complete_restricted_search_including_postprocessing";
        if(coupled) {
            coupled_neighborhood_records[coupled_record]["seconds"]=used;
            coupled_neighborhood_records[coupled_record]["time_scope"]=heuristic_records.back()["time_scope"];
        }
        if(polish && radius<0)primal_design_polish_records[polish_record]["seconds"]=used;
    };
    auto run_continuous_stage = [&](const char* stage,double allowance) {
        if(allowance<=0 || remaining(deadline)<=0)return;
        const std::string previous_candidate=active_candidate;
        active_candidate=stage;
        const double phase=wall_time();
        bool charged=false;
        auto charge=[&]() {
            if(charged)return;
            charged=true;
            // The per-stage record and cumulative allowance must use the
            // same timestamp, including the first (and possibly only) stage.
            last_continuous_stage=wall_time();
            const double elapsed=last_continuous_stage-phase;
            continuous_progress.spent+=elapsed;
            continuous_progress.initialized=true;
            timing["continuous_root"]=timing.value("continuous_root",0.0)+elapsed;
        };
        try {
        SolverOptions continuous_options=options;
        continuous_options.cg_max_columns=options.continuous_root_max_columns;
        continuous_options.pareto=false;
        if(options.cg_seed_reuse)
            continuous_columns.include(d,paths,std::vector<int>(pool_p.begin(),pool_p.end()),
                                       std::vector<int>(pool_e.begin(),pool_e.end()));
        continuous_options.seconds=std::min(remaining(deadline),
            std::max(0.0,allowance-(wall_time()-phase)));
        if(options.progress)options.progress->record({{"event","phase_start"},
            {"stage",stage},{"allocated_seconds",allowance}});
        Json stage_result;
        try {
            stage_result=solve_continuous_cg(d,paths,continuous_options,
                options.cg_seed_reuse?&continuous_columns:nullptr);
        } catch(const RunBudgetExceeded& error) {
            stage_result={{"bound_valid",false},{"termination_reason","time_limit"},
                {"budget_interruption",{{"stage",error.stage},{"error",error.what()}}}};
        }
        stage_result["enabled"]=true;
        stage_result["stage"]=stage;
        stage_result["allocated_seconds"]=allowance;
        // A preprocessed catalogue contains all physical paths, not just the
        // columns generated by the continuous RMP. Reuse its actual indices.
        const auto root_service=stage_result.value("_service_indices",std::vector<int>{});
        const auto root_empty=stage_result.value("_empty_indices",std::vector<int>{});
        used_p.insert(root_service.begin(),root_service.end());
        used_e.insert(root_empty.begin(),root_empty.end());
        if(d.preprocessed_demand) {
            if(options.reuse_columns) {
                pool_p.insert(root_service.begin(),root_service.end());
                pool_e.insert(root_empty.begin(),root_empty.end());
            }
        } else {
            for(std::size_t j=0;j<paths.service_paths.size();++j)pool_p.insert(static_cast<int>(j));
            for(std::size_t j=0;j<paths.empty_paths.size();++j)pool_e.insert(static_cast<int>(j));
        }
        stage_result["reused_service_columns"]=d.preprocessed_demand ? (options.reuse_columns?root_service.size():0) : paths.service_paths.size();
        stage_result["reused_empty_columns"]=d.preprocessed_demand ? (options.reuse_columns?root_empty.size():0) : paths.empty_paths.size();
        stage_result["reuse_scope"]=d.preprocessed_demand?"continuous_root_rmp_columns":"registered_generated_catalogue";
        stage_result.erase("_service_indices");
        stage_result.erase("_empty_indices");
        if (stage_result.value("bound_valid", false)) {
            const double bound = stage_result.at("certified_lower_bound").get<double>();
            // Q includes only operational cost; this is a TOTAL-cost lower
            // bound and must retain F(x) on the left hand side.
            const double safe_bound = bound - 1e-8 * (1 + std::abs(bound));
            if (!std::isfinite(bound) || safe_bound > upper)
                throw std::runtime_error("Certified continuous bound contradicts integer incumbent");
            stage_result["retained_global_lower_bound"] = safe_bound;
            stage_result["master_floor_added"] = false;
            if (safe_bound > lower) {
                // The complete-domain dual certificate remains valid even if
                // the global budget expires while updating the master row.
                lower = safe_bound;
                root_lower = std::max(root_lower, safe_bound);
                Coefficients lhs{{theta, 1}};
                for (int j : master.design_vars)
                    if (master.matrix.variables[j].cost != 0)
                        lhs.emplace_back(j, master.matrix.variables[j].cost);
                master.matrix.row(options.staged_continuous_root ?
                    "continuous_global_objective_floor_"+std::to_string(continuous_bound_rows) :
                    "continuous_global_objective_floor", 'G', safe_bound, std::move(lhs));
                ++continuous_bound_rows;
                stage_result["master_floor_recorded"] = true;
                try {
                    solver.add_row(master.matrix.rows.back());
                    stage_result["master_floor_added"] = true;
                } catch(const RunBudgetExceeded& error) {
                    stage_result["master_floor_budget_interruption"]=error.what();
                }
                stage_result["applied_total_cost_lower_bound"] = safe_bound;
            }
        }
        const bool stage_closed=stage_result.value("bound_valid",false) &&
            stage_result.value("primal_valid",false) &&
            stage_result.at("continuous_objective").get<double>()-
                stage_result.at("certified_lower_bound").get<double>() <=
                2*options.cg_duality_tolerance*(1+std::abs(stage_result.at("continuous_objective").get<double>()));
        continuous_progress.closed=continuous_progress.closed || stage_closed;
        continuous_root_closed=continuous_progress.closed;
        charge();
        stage_result["stage_wall_seconds"]=last_continuous_stage-phase;
        stage_result["cumulative_stage_seconds"]=continuous_progress.spent;
        // Keep a complete matching certificate/trace from the strongest stage.
        if(!continuous_root.value("bound_valid",false) ||
            (stage_result.value("bound_valid",false) && stage_result.at("certified_lower_bound").get<double>() >
                continuous_root.at("certified_lower_bound").get<double>()))
            continuous_root=stage_result;
        continuous_stage_records.push_back(stage_result);
        if(options.progress)options.progress->record({{"event","continuous_root_result"},
            {"stage",stage},{"bound_scope",continuous_scope(options)},{"diagnostics",stage_result}});
        observe_global("continuous_root_finished",true);
        active_candidate=previous_candidate;
        } catch(...) {charge();active_candidate=previous_candidate;throw;}
    };
    auto submit_alns = [&](const Json& candidate) {
        if(retain_incumbent(candidate,"primal_alns")) {
            // Persist the validated point before elite selection or model work.
            if(options.progress) {
                Json snapshot=best;snapshot["bound"]=lower;snapshot["gap"]=gap(upper,lower);
                options.progress->checkpoint(snapshot,"primal_alns_improvement",true);
            }
        }
    };
    auto merge_alns_columns = [&]() {
        if(!primal_alns)return;
        for(int i:primal_alns->service_columns()){pool_p.insert(i);used_p.insert(i);}
        for(int i:primal_alns->empty_columns()){pool_e.insert(i);used_e.insert(i);}
    };
    auto run_primal_window = [&](bool startup) {
        if(!options.primal_alns)return;
        const auto phase_kind=startup ? PrimalPhase::startup : PrimalPhase::regular;
        const double phase=wall_time();
        const double allowance=primal_schedule.allowance(phase_kind,phase,remaining(deadline));
        if(allowance<=.5)return;
        const double upper_before=upper;
        auto charge=[&]() {
            const double used=wall_time()-phase;
            primal_schedule.charge(phase_kind,used,wall_time());
            if(primal_alns)primal_alns->synchronize_used_seconds(primal_schedule.spent());
            heuristic_seconds+=used;restricted_seconds+=used;
            primal_schedule_records.push_back({{"stage",startup?"startup":"regular"},
                {"allocated_seconds",allowance},{"seconds",used},{"elapsed_seconds",wall_time()-started},
                {"pending_design",!pending_design.empty()},{"upper_before",upper_before},{"upper_after",upper},
                {"direct_global_bound_update",false}});
        };
        try {
            if(startup && start_solution) {
                // Validate a supplied point in the original restricted model;
                // no external point enters the trusted submission path directly.
                const std::vector<int> empty_selection;
                auto checked=solve_restricted_heuristic(d,paths,options,phase+std::min(2.0,allowance),
                    start_solution,-1,&empty_selection,&empty_selection);
                if(checked.value("validated",false))submit_alns(checked.at("solution"));
            }
            if(!primal_alns)primal_alns=std::make_unique<PrimalAlnsSearch>(d,paths,options,best,primal_schedule.limit());
            else primal_alns->observe_validated(best);
            if(pending_feedback && options.cg_primal_feedback) {
                cg_feedback_records.push_back(primal_alns->observe_cg_columns(*pending_feedback,std::min(deadline,phase+allowance)));
                pending_feedback.reset();
            }
            if(startup)primal_alns->startup(phase+allowance,submit_alns);
            else {
                std::vector<double> guide;
                if(latest_master_point.size()==master.matrix.variables.size()) {
                    guide.resize(master.layout.count);
                    for(int j=0;j<master.layout.count;++j)guide[j]=latest_master_point[master.design_vars[j]];
                }
                primal_alns->step(phase+allowance,guide.empty()?nullptr:&guide,submit_alns);
            }
            merge_alns_columns();update_cutoff();
            charge();
        } catch(...) {charge();throw;}
        observe_global(startup?"primal_alns_startup_finished":"primal_alns_finished",true);
    };
    auto run_elite_window = [&]() {
        if(!primal_alns || elite_calls>=2)return false;
        const double phase=wall_time();
        const double allowance=primal_schedule.allowance(PrimalPhase::elite,phase,remaining(deadline));
        if(allowance<=.5)return false;
        auto elite=primal_alns->take_elite_refinement();
        if(!elite)return false;
        ++elite_calls;
        const auto old_candidate=active_candidate;
        active_candidate="alns_elite:"+std::to_string(elite_calls);
        auto charge=[&]() {
            const double used=wall_time()-phase;
            primal_schedule.charge(PrimalPhase::elite,used,wall_time());heuristic_seconds+=used;
            primal_alns->synchronize_used_seconds(primal_schedule.spent());
            active_candidate=old_candidate;
        };
        try {
            auto design=solution_design(d,*elite);
            std::vector<int> selected_p(pool_p.begin(),pool_p.end()),selected_e(pool_e.begin(),pool_e.end());
            auto elite_options=options;elite_options.pareto=false;
            const double before=upper;
            auto sub=subproblem(d,paths,design,elite_options,phase+allowance,selected_p,selected_e,nullptr,
                submit_alns,false,false,false,-1,-1,std::numeric_limits<double>::quiet_NaN(),&*elite,&pricing_graph_cache);
            // Only ordinary full-domain certified cuts pass through account().
            // No restricted MIP bound or fixed-design bound updates global LB.
            account(sub,true,"alns_elite");
            if(!sub.solution.is_null())primal_alns->observe_validated(sub.solution);
            elite_records.push_back({{"allocated_seconds",allowance},{"seconds",wall_time()-phase},
                {"initial_service_columns",selected_p.size()},{"service_columns",sub.service.size()},
                {"retained_primal",true},{"upper_before",before},{"upper_after",upper},
                {"pricing_complete",sub.ordinary.valid},{"recourse_optimal",sub.recourse_optimal},
                {"termination_reason",sub.reason},{"direct_global_bound_update",false}});
            charge();return true;
        } catch(...) {charge();throw;}
    };
    try {
    run_primal_window(true);
    if(!options.primal_alns && options.primal_seed_portfolio && remaining(deadline)>0 &&
       options.seconds*options.heuristic_budget_fraction>heuristic_seconds) {
        const double phase=wall_time(), before=upper;
        const double allowance=std::min({18.0,.03*options.seconds,remaining(deadline),
            options.seconds*options.heuristic_budget_fraction-heuristic_seconds});
        auto portfolio=solve_primal_seed_portfolio(d,paths,options,phase+allowance,&best);
        if(portfolio.value("validated",false) && portfolio.contains("solution") && !portfolio["solution"].is_null()) {
            Json candidate=portfolio["solution"];
            retain_incumbent(candidate,"primal_seed_portfolio");
            if(options.incumbent_cleanup)candidate=cleanup_incumbent(d,paths,candidate);
            retain_incumbent(candidate,"primal_seed_portfolio");
        }
        const double used=wall_time()-phase;
        heuristic_seconds+=used;restricted_seconds+=used;
        portfolio.erase("solution");
        portfolio.update({{"stage","before_continuous_root"},{"upper_before",before},{"upper_after",upper},
            {"seconds",used},{"allocated_seconds",allowance},{"direct_global_bound_update",false}});
        primal_portfolio_records.push_back(portfolio);
        update_cutoff();observe_global("primal_seed_portfolio_finished",true);
    }
    if(options.continuous_root && remaining(deadline)>0 && continuous_total_limit>0) {
        const double allowance=options.staged_continuous_root ?
            continuous_root_slice_budget(continuous_total_limit,remaining(deadline),continuous_progress) :
            std::min(continuous_total_limit,remaining(deadline));
        run_continuous_stage("continuous_root",allowance);
    }
    if (!options.primal_alns && (options.initial_design || start_solution) && remaining(deadline) > 0 &&
        options.seconds * options.heuristic_budget_fraction > heuristic_seconds) {
        std::vector<double> design = start_solution ? solution_design(d, *start_solution) :
            ((restricted_improvements || (options.primal_seed_portfolio && best.value("served",0.0)>0))
                ? solution_design(d, best) : initial_design(d, paths));
        double allowance = std::min({30.0,.05*options.seconds,
                                    options.seconds*options.heuristic_budget_fraction-heuristic_seconds});
        double heuristic_deadline = std::min(deadline, wall_time() + allowance);
        active_candidate = start_solution ? "supplied_design" : "initial_design";
        SolverOptions initial_options=options;
        auto sub = subproblem(d, paths, design, initial_options, heuristic_deadline, seeds.service, seeds.empty, nullptr,retain_recourse,
            false,false,false,-1,-1,std::numeric_limits<double>::quiet_NaN(),nullptr,&pricing_graph_cache);
        heuristic_seconds += sub.wall;
        account(sub, true, start_solution ? "supplied_design" : "initial_design");
        heuristic_records.push_back({{"stage", start_solution ? "supplied_design" : "initial_design"},
                                     {"candidate_id",active_candidate},
                                     {"termination_reason", sub.reason}, {"pricing_complete", sub.ordinary.valid},
                                     {"wall", sub.wall}, {"cg_iterations", sub.iterations}, {"cg_trace", sub.trace}});
    }
    // Give the joint MIP a useful operational warm start, instead of asking
    // it to discover all discrete design and continuation choices from zero.
    restricted_search("initial_restricted_mip", -1);
    separate_neighborhood();
    // The continuous master and fixed-fractional recourse share exactly the
    // integer model's matrix. Fractional recourse is only for lower cuts; its
    // primal point is never rounded or admitted as an integer incumbent.
    const double root_deadline = std::min(deadline, wall_time() + options.seconds * options.root_budget_fraction);
    root_upper = upper;
    if (!integer_strengthened_root && continuous_root.value("primal_valid", false)) {
        const double continuous_value = continuous_root.at("continuous_objective").get<double>();
        if (!std::isfinite(continuous_value) ||
            continuous_value < root_lower - 2 * options.cg_duality_tolerance * (1 + std::abs(continuous_value)))
            throw std::runtime_error("Validated continuous primal contradicts continuous lower bound");
        root_upper = std::min(root_upper, continuous_value);
    }
    continuous_root_closed = continuous_root.value("bound_valid", false) &&
        continuous_root.value("primal_valid", false) &&
        continuous_root.at("continuous_objective").get<double>() -
            continuous_root.at("certified_lower_bound").get<double>() <=
            2 * options.cg_duality_tolerance *
                (1 + std::abs(continuous_root.at("continuous_objective").get<double>()));
    std::vector<double> projection_core = original_core;
    if (options.root_projection && continuous_root.value("primal_valid", false)) {
        const auto x = continuous_root.at("design").get<std::vector<double>>();
        validate_design(d, x, true);
        for (std::size_t j = 0; j < x.size(); ++j)
            projection_core[j] = .99 * x[j] + .01 * original_core[j];
    }
    auto record_root = [&](Json record) {
        record["lower_bound"] = root_lower;
        record["continuous_upper_bound"] = integer_strengthened_root ? Json(nullptr) : Json(root_upper);
        record["continuous_gap"] = integer_strengthened_root ? Json(nullptr) : Json(gap(root_upper, root_lower));
        record["bound_scope"] = root_scope;
        if (integer_strengthened_root) {
            record["integer_feasible_upper_bound"] = upper;
            record["gap_to_integer_incumbent"] = gap(upper, root_lower);
        }
        root_records.push_back(record);
        if (options.progress) {
            record["event"] = "root_separation_record";
            options.progress->record(std::move(record));
        }
    };
    if (options.root_cuts || capacity_enabled || resource_enabled || task_resource_relaxation || airline_resource_relaxation || queue_separator) root_reason = "round_limit";
    if (continuous_root_closed && !integer_strengthened_root) root_reason = "continuous_gap_certified";
    if ((!continuous_root_closed || options.root_projection || integer_strengthened_root) &&
        (options.root_cuts || capacity_enabled || resource_enabled || task_resource_relaxation || airline_resource_relaxation || queue_separator)) for (int round = 0; round < options.root_cut_rounds; ++round) {
        if (remaining(root_deadline) <= 0) { root_reason = "time_limit"; break; }
        const double phase = wall_time();
        LinearModel relaxation = master.matrix;
        // Conditional, topology and assignment-support cuts use binary decisions, not the chosen
        // lifted continuous feasible set. Keep the root's primal and dual
        // diagnostics on that same continuous relaxation by excluding exactly
        // these append-only row indices from its temporary copy. They remain
        // in the integer master and retain their full-domain integer validity.
        // OD resource cuts are deliberately retained when requested: this
        // then defines a DIFFERENT integer-valid root relaxation, whose LB
        // must never be compared with the original continuous primal as UB.
        // Static path-support filters are retained as an optimality-preserving
        // restriction of assignment encodings; root_scope labels that separately.
        if (!conditional_row_indices.empty() || !topology_row_indices.empty() || !assignment_support_row_indices.empty()
            || !branch_joint_row_indices.empty()) {
            std::vector<RowRecord> root_rows;
            root_rows.reserve(relaxation.rows.size()-conditional_row_indices.size()-topology_row_indices.size()
                -assignment_support_row_indices.size()-branch_joint_row_indices.size());
            for (std::size_t j=0;j<relaxation.rows.size();++j)
                if (!conditional_row_indices.count(j) && !topology_row_indices.count(j) && !assignment_support_row_indices.count(j)
                    && !branch_joint_row_indices.count(j))
                    root_rows.push_back(std::move(relaxation.rows[j]));
            relaxation.rows=std::move(root_rows);
        }
        for (auto& variable : relaxation.variables) variable.binary = false;
        NativeSolver root_solver(relaxation, options, "root_master");
        ++root_master_solves;
        auto root = root_solver.optimize(remaining(root_deadline));
        add_time(timing, "root_master", phase);
        if (!root.optimal || !root.has_solution || !feasible_point(relaxation, root.values)) {
            root_reason = "master_not_certified_" + root.status; break;
        }
        if (root.objective > upper + 1e-7 * (1 + std::abs(upper)))
            throw std::runtime_error("Root relaxation contradicts incumbent");
        lower = std::max(lower, std::min(upper, root.objective - 1e-8 * (1 + std::abs(root.objective))));
        root_lower = lower;
        observe_global("root_master_finished");
        active_candidate = "root:" + std::to_string(round + 1);
        latest_master_point=root.values;
        // A support row does not consume the round's recourse opportunity.
        // The current point can still generate a globally valid recourse
        // cut; the new support row affects the LB only after reoptimization.
        const bool fractional_support_added=separate_fractional_support(root.values,"root_relaxation",root_deadline);
        const bool capacity_added = separate_capacity(root.values,"root_relaxation");
        const bool resource_added = separate_resource(root.values,"root_relaxation",root_deadline);
        const bool task_resource_added=separate_task_resource(root.values,"root_relaxation",root_deadline);
        const bool airline_resource_added=separate_airline_resource(root.values,"root_relaxation",root_deadline);
        const bool queue_added=separate_queue(root.values,"root_relaxation",root_deadline);
        const bool fractional_queue_added=refresh_fractional_queue("fractional_queue_root",root_deadline);
        const bool face_added=separate_face(latest_master_point,"root_relaxation",root_deadline);
        if(options.queue_fractional_separation)root_lower=lower;
        Json root_record = {{"round", round + 1}, {"candidate_id", active_candidate},
            {"conditional_rows_excluded",conditional_row_indices.size()},
            {"topology_rows_excluded",topology_row_indices.size()},
            {"assignment_support_rows_excluded",assignment_support_row_indices.size()},
            {"branch_joint_rows_excluded",branch_joint_row_indices.size()},
            {"master_objective", root.objective}, {"lower_bound", root_lower},
            {"fractional_support_added",fractional_support_added},
            {"theta", root.values[theta]}, {"capacity_cuts_added", capacity_added ? 1 : 0},
            {"resource_lp_cuts_added", resource_added ? 1 : 0},
            {"task_resource_cuts_added",task_resource_added ? 1 : 0},
            {"airline_resource_cuts_added",airline_resource_added ? 1 : 0},
            {"queue_polymatroid_cuts_added",queue_added ? 1 : 0},
            {"queue_fractional_cuts_added",fractional_queue_added},{"design_face_cuts_added",face_added},
            {"pricing_complete", false}};
        // A capacity cut and a certified recourse cut can strengthen the same
        // fractional candidate. Capacity separation must not consume a CG round.
        if (!options.root_cuts) {
            record_root(root_record);
            if (fractional_support_added || capacity_added || resource_added || task_resource_added || airline_resource_added || queue_added || fractional_queue_added || face_added) continue;
            root_reason = "resource_separation_complete_cg_disabled";
            break;
        }
        std::vector<double> design(master.layout.count);
        for (int j=0;j<master.layout.count;++j) design[j]=std::clamp(root.values[master.design_vars[j]],0.0,1.0);
        root_record["design"] = design;
        root_record["evaluations"] = Json::array();
        root_record["benders_cuts_added"] = 0;
        root_record["cg_seconds"] = 0.0;
        const bool use_inout = options.root_projection && options.root_inout;
        bool point_closed = false, separated = false, any_certified = false;
        std::string last_reason = "not_evaluated";
        // In-out selects a legal fractional point. If that cut does not
        // separate the CURRENT master point, certify the original point as
        // a fallback before claiming pointwise consistency.
        for (int pass = 0; pass < (use_inout ? 2 : 1); ++pass) {
            if (remaining(root_deadline) <= 0) break;
            const bool mixed = use_inout && pass == 0;
            std::vector<double> anchor = design;
            if (mixed) for (std::size_t j = 0; j < anchor.size(); ++j)
                anchor[j] = options.root_inout_weight * design[j]
                    + (1 - options.root_inout_weight) * projection_core[j];
            std::vector<int> selected_p(pool_p.begin(),pool_p.end()), selected_e(pool_e.begin(),pool_e.end());
            const int cuts_before = cuts;
            ++root_cg_calls;
            auto sub = subproblem(d,paths,anchor,options,root_deadline,selected_p,selected_e,nullptr,{},
                                  true,false,options.root_projection,-1,-1,std::numeric_limits<double>::quiet_NaN(),nullptr,&pricing_graph_cache);
            account(sub, false, "root_relaxation");
            if(!options.queue_fractional_separation)
                separate_queue(root.values,"root_native_prices",root_deadline,&sub.ordinary.queue_prices);
            last_reason = sub.reason;
            Json evaluation = {{"anchor_kind",mixed ? "inout" : "master_point"}, {"design",anchor},
                {"pricing_complete",sub.ordinary.valid},{"termination_reason",sub.reason},
                {"benders_cuts_added",cuts-cuts_before},{"cg_seconds",sub.wall}};
            root_record["benders_cuts_added"] = root_record["benders_cuts_added"].get<int>() + cuts-cuts_before;
            root_record["cg_seconds"] = root_record["cg_seconds"].get<double>() + sub.wall;
            if (sub.ordinary.valid) {
                any_certified = true;
                const double q = sub.continuous_objective;
                double total = q;
                for (int j = 0; j < master.layout.count; ++j)
                    total += master.matrix.variables[j].cost * anchor[j];
                if (!std::isfinite(total) || (!integer_strengthened_root &&
                    total < root_lower - 2 * options.cg_duality_tolerance * (1 + std::abs(total))))
                    throw std::runtime_error("Continuous recourse incumbent contradicts root lower bound");
                if (!integer_strengthened_root) root_upper = std::min(root_upper, total);
                const double violation = value(sub.ordinary,design) - root.values[theta];
                const double tolerance = 2 * options.cg_duality_tolerance * (1 + std::abs(q));
                separated = violation > tolerance;
                point_closed = !mixed && q - root.values[theta] <= tolerance;
                evaluation["operating_objective"] = q;
                evaluation["continuous_feasible_objective"] = total;
                evaluation["original_candidate_cut_violation"] = violation;
                evaluation["original_candidate_primal_gap"] = mixed ? Json(nullptr) : Json(q-root.values[theta]);
                evaluation["ordinary_cut"] = {{"intercept",sub.ordinary.intercept},{"coefficients",sub.ordinary.coefficients}};
                if (!mixed) {
                    root_record["ordinary_cut"] = evaluation["ordinary_cut"];
                    root_record["operating_objective"] = q;
                    root_record["recourse_underestimation"] = q-root.values[theta];
                    root_record["continuous_feasible_objective"] = total;
                }
                for (std::size_t j = 0; j < anchor.size(); ++j)
                    projection_core[j] = .50 * projection_core[j] + .49 * anchor[j] + .01 * original_core[j];
            }
            root_record["evaluations"].push_back(evaluation);
            heuristic_records.push_back({{"stage","root_relaxation"},{"round",round+1},{"bound",lower},
                {"candidate_id",active_candidate},{"anchor_kind",mixed ? "inout" : "master_point"},
                {"capacity_cuts_added",pass==0 && capacity_added ? 1 : 0},
                {"resource_lp_cuts_added",pass==0 && resource_added ? 1 : 0},
                {"benders_cuts_added",cuts-cuts_before},{"termination_reason",sub.reason},
                {"pricing_complete",sub.ordinary.valid},{"cg_iterations",sub.iterations},
                {"cg_trace",sub.trace},{"wall",sub.wall}});
            if (separated || point_closed) break;
        }
        root_record["pricing_complete"] = any_certified;
        root_record["subproblem_termination"] = last_reason;
        root_record["candidate_epigraph_certified"] = !integer_strengthened_root && point_closed;
        root_record["original_candidate_recourse_consistent"] = point_closed;
        root_record["candidate_separated"] = separated;
        root_record["continuous_upper_bound"] = root_upper;
        root_record["continuous_gap"] = gap(root_upper,root_lower);
        record_root(std::move(root_record));
        if (!any_certified) { root_reason = "cg_not_certified_" + last_reason; break; }
        if (options.root_projection && point_closed && !fractional_support_added && !capacity_added && !resource_added && !task_resource_added && !airline_resource_added && !queue_added && !fractional_queue_added && !face_added) {
            root_reason = integer_strengthened_root ? "original_candidate_recourse_consistent" : "candidate_epigraph_certified";
            break;
        }
        if (!integer_strengthened_root && !options.root_projection && root_upper-root_lower <=
            2*options.cg_duality_tolerance*(1+std::abs(root_upper))) {
            root_reason = "continuous_gap_certified"; break;
        }
        root_reason = "round_limit";
        if (gap(upper,lower)<=options.gap) { root_reason = "global_gap_limit"; break; }
    }
    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        if (remaining(deadline) <= 0) { reason = "time_limit"; break; }
        // Pending integer designs do not block an upper-bound window. Elite
        // refinement is rare; all other due windows run one ALNS repair.
        if(options.primal_alns) {
            const bool elite_due=primal_schedule.used(PrimalPhase::regular)>=12.0;
            if(!elite_due || !run_elite_window())run_primal_window(false);
            if(gap(upper,lower)<=options.gap){reason="gap_limit";break;}
        }
        if(options.queue_fractional_separation) {
            refresh_fractional_queue("fractional_queue_integer",deadline);
            if(gap(upper,lower)<=options.gap) {reason="gap_limit";break;}
        }
        separate_face(latest_master_point,"integer_search",deadline);
        if (options.search_scheduling && options.restricted_mip && !options.primal_alns) {
            const RestrictedSearchWindow window{options.seconds,remaining(deadline),
                options.seconds*options.heuristic_budget_fraction-heuristic_seconds,
                wall_time()-last_restricted_search,scheduled_restricted_calls,
                options.local_branching_radius,!pending_design.empty(),options.local_branching};
            if(const auto radius=scheduled_restricted_radius(window)) {
                const int previous_attempts=restricted_attempts;
                restricted_search(*radius>=0?"scheduled_local_branching":"scheduled_column_mip",*radius);
                if(restricted_attempts>previous_attempts)++scheduled_restricted_calls;
                observe_global("scheduled_primal_finished");
                if (gap(upper,lower)<=options.gap) { reason="gap_limit"; break; }
            }
        }
        if(options.staged_continuous_root) {
            continuous_progress.pending=!pending_design.empty();
            continuous_progress.integer_passes=static_cast<int>(iterations.size());
            continuous_progress.since_last=wall_time()-last_continuous_stage;
            const double allowance=continuous_root_slice_budget(continuous_total_limit,
                remaining(deadline),continuous_progress);
            if(allowance>0)run_continuous_stage("continuous_root_resume",allowance);
            if(gap(upper,lower)<=options.gap){reason="gap_limit";break;}
        }
        const double lower_before_iteration = lower;
        double phase = wall_time();
        if (options.warm_start && pending_design.empty()) {
            auto design_start = solution_design(d, best);
            std::vector<int> start_indices = master.design_vars;
            std::vector<double> start_values = design_start;
            start_indices.push_back(theta); start_values.push_back(best["continuous_cost"].get<double>());
            if (start_values != last_master_start) {
                solver.add_start(start_values, start_indices);
                last_master_start = std::move(start_values);
                ++master_starts;
            }
        }
        add_time(timing, "master_update", phase);
        phase = wall_time();
        SolveOutcome solved;
        if (pending_design.empty()) {
            ++integer_master_solves;
            const auto control = options.search_scheduling
                ? scheduled_master_budget(options.adaptive_master,remaining(deadline),
                    options.master_initial_gap,options.master_slice_seconds,gap(upper,lower),
                    options.gap,bound_stalled_passes,supported_cg_passes,closed_without_progress)
                : master_solve_budget(options.adaptive_master, remaining(deadline),
                    options.master_initial_gap, options.master_slice_seconds, gap(upper, lower),
                    options.gap, master_stalled_passes);
            solver.cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, control.relative_gap);
            solved = solver.optimize(control.seconds);
            Json record = {{"solve", integer_master_solves}, {"seconds", solved.seconds},
                {"time_budget", control.seconds}, {"relative_gap_target", control.relative_gap},
                {"exact_phase", control.exact_phase}, {"stalled_passes", master_stalled_passes},
                {"status", solved.status}, {"optimal", solved.optimal}, {"has_solution", solved.has_solution},
                {"candidate_objective", solved.has_solution ? Json(solved.objective) : Json(nullptr)},
                {"certified_bound", std::isfinite(solved.bound) ? Json(solved.bound) : Json(nullptr)},
                {"global_lower_before", lower}, {"global_upper", upper}, {"nodes", solved.nodes}};
            record.update({{"search_scheduling",options.search_scheduling},
                {"bound_stalled_passes",bound_stalled_passes},{"supported_cg_passes",supported_cg_passes},
                {"closed_without_progress",closed_without_progress}});
            master_records.push_back(record);
            if (options.progress) { record["event"] = "master_control"; options.progress->record(record); }
        }
        add_time(timing, "master_optimize", phase);
        if (pending_design.empty() && std::isfinite(solved.bound)) {
            if (solved.bound > upper + 1e-7 * (1 + std::abs(upper))) { reason = "numerical_bound_inconsistency"; break; }
            lower = std::max(lower, std::min(upper, solved.bound));
        }
        if(options.target_master_refinement && pending_design.empty() && solved.has_solution &&
           feasible_point(master.matrix,solved.values)) {
            const bool ready=supported_cg_passes>=3 && bound_stalled_passes>=3 &&
                wall_time()-last_target_master>=30;
            const auto target=target_master_budget(solved.objective,upper,lower,options.gap,
                remaining(deadline),.02*options.seconds-target_master_seconds,ready);
            if(target) {
                const double previous_tolerance=master_records.back().at("relative_gap_target");
                const double tolerance=std::min(previous_tolerance,target->budget.relative_gap);
                const double refine_started=wall_time();
                solver.cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap,tolerance);
                auto refined=solver.optimize(target->budget.seconds);
                solver.cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap,previous_tolerance);
                const double elapsed=wall_time()-refine_started;
                const double previous_refinement=last_target_master-started;
                target_master_seconds+=elapsed;last_target_master=wall_time();
                timing["target_master_refinement"]=target_master_seconds;
                Json record={{"master_solve",integer_master_solves},{"time_budget",target->budget.seconds},
                    {"started_elapsed_seconds",refine_started-started},
                    {"previous_refinement_elapsed_seconds",previous_refinement},
                    {"supported_cg_passes",supported_cg_passes},{"bound_stalled_passes",bound_stalled_passes},
                    {"seconds",elapsed},{"relative_gap_target",tolerance},
                    {"previous_relative_gap_target",previous_tolerance},{"target_bound",target->target_bound},
                    {"bound_deficit",target->bound_deficit},{"candidate_margin",target->candidate_margin},
                    {"status",refined.status},{"has_solution",refined.has_solution},
                    {"candidate_objective",refined.has_solution?Json(refined.objective):Json(nullptr)},
                    {"certified_bound",std::isfinite(refined.bound)?Json(refined.bound):Json(nullptr)},
                    {"global_lower_before",lower},{"global_upper",upper}};
                const bool inconsistent=std::isfinite(refined.bound) &&
                    refined.bound>upper+1e-7*(1+std::abs(upper));
                if(!inconsistent && std::isfinite(refined.bound))
                    lower=std::max(lower,std::min(upper,refined.bound));
                record["global_lower_after"]=lower;
                target_master_records.push_back(record);
                if(options.progress) {record["event"]="target_master_refinement";options.progress->record(record);}
                if(inconsistent) {reason="numerical_bound_inconsistency";break;}
                // Keep a validated candidate and the best certified bound separately.
                const double certified=std::max(solved.bound,refined.bound);
                if(refined.has_solution && refined.objective<solved.objective &&
                   feasible_point(master.matrix,refined.values)) solved=std::move(refined);
                solved.bound=certified;
            }
        }
        if (options.search_scheduling && pending_design.empty()) {
            bound_stalled_passes=updated_bound_stalls(lower_before_iteration,lower,bound_stalled_passes);
            master_records.back()["global_lower_after"]=lower;
            master_records.back()["bound_stalled_passes_after"]=bound_stalled_passes;
        }
        bool stabilized_candidate=false;
        if(options.stabilized_master && pending_design.empty() && solved.has_solution &&
           integer_master_solves%2==0 && feasible_point(master.matrix,solved.values)) {
            const double allowance=std::min({options.stabilized_master_seconds,
                .05*options.seconds-stabilized_master_seconds,remaining(deadline)});
            if(allowance>.01) {
                const double certified_global=solved.bound;
                auto trial=select_stabilized_design(d,master,theta,solution_design(d,best),lower,upper,
                    options,std::min(deadline,wall_time()+allowance));
                stabilized_master_seconds+=trial.seconds;
                timing["stabilized_master"]=stabilized_master_seconds;
                trial.metadata["global_lower_before"]=lower;trial.metadata["global_lower_after"]=lower;
                trial.metadata["global_master_solve"]=integer_master_solves;
                trial.metadata["seconds"]=trial.seconds;
                trial.metadata["used_candidate"]=trial.candidate.has_solution;
                stabilized_master_records.push_back(trial.metadata);
                if(trial.candidate.has_solution) {
                    solved=std::move(trial.candidate);
                    // This remains the bound of the independent GLOBAL solve,
                    // never the objective/bound of the proximity problem.
                    solved.bound=certified_global;stabilized_candidate=true;
                }
            }
        }
        if(options.cut_pool_management && pending_design.empty() && solved.has_solution) {
            const auto restored=certified_cut_pool.reactivate(solved.values,master.matrix,set_cut_rhs);
            master_records.back()["dormant_cuts_reactivated"]=restored;
            // This point was optimal/feasible only for the relaxed active set.
            // Keep its certified lower bound, then reoptimize before using it.
            if(restored>0)continue;
        }
        observe_global("integer_master_finished");
        if (gap(upper, lower) <= options.gap) { reason = "gap_limit"; break; }
        if (pending_design.empty() && !solved.has_solution) {
            if (options.adaptive_master && remaining(deadline) > 0 &&
                (solved.status.find("TimeLim") != std::string::npos || solved.status == "time_limit")) {
                ++master_stalled_passes; continue;
            }
            reason = "master_" + solved.status; break;
        }
        if (pending_design.empty() && !feasible_point(master.matrix, solved.values)) {
            reason = "master_candidate_validation_failed"; break;
        }
        if(pending_design.empty())latest_master_point=solved.values;
        const bool resumed_design = !pending_design.empty();
        active_candidate = resumed_design ? pending_candidate : (stabilized_candidate?"stabilized:":"integer:") + std::to_string(integer_master_solves);
        // Add both cut families before reoptimizing the integer master. If CG
        // needs another slice, retain this candidate and its already valid cut.
        const double candidate_theta = resumed_design ? pending_theta : solved.values[theta];
        std::vector<double> design = pending_design;
        if (design.empty()) {
            design.resize(master.layout.count);
            for (int j = 0; j < master.layout.count; ++j) design[j] = std::round(solved.values[master.design_vars[j]]);
        }
        // Check every positive task branch at its selected departure base
        // before a costly recourse solve. These dominance rows remove empty
        // assignment encodings, not operational all-courier solutions.
        if (!resumed_design && options.path_support_cuts && path_support_cache.complete_domain &&
            path_support_dynamic < static_cast<std::size_t>(options.path_support_cut_limit) && remaining(deadline) > 0) {
            const auto allowance = std::min<std::size_t>(options.path_support_max_cuts,
                options.path_support_cut_limit-path_support_dynamic);
            auto separated = separate_path_support(d,path_support_cache,design,
                std::min(deadline,wall_time()+options.path_support_seconds),allowance);
            path_support_seconds += separated.seconds;
            path_support_checks += separated.checks;
            path_support_blocked += separated.blocked_groups;
            int added = 0;
            for (auto& cut : separated.cuts) {
                if (!cut.valid || !cut.violated || !path_support_keys.insert(cut.key).second) continue;
                cut.row.name += "_"+std::to_string(path_support_dynamic);
                master.matrix.rows.push_back(cut.row); solver.add_row(cut.row);
                Json record = cut.metadata;
                record.update({{"stage","integer_master"},{"candidate_id",active_candidate},
                    {"accepted",true},{"key",cut.key},{"violation",cut.violation},
                    {"row_name",cut.row.name},{"sense",std::string(1,cut.row.sense)},
                    {"rhs",cut.row.rhs},{"coefficients",cut.row.coefficients}});
                path_support_records.push_back(std::move(record)); ++path_support_dynamic; ++added;
            }
            Json attempt = separated.metadata;
            attempt.update({{"candidate_id",active_candidate},{"added",added},
                {"skipped_integer_subproblem",added>0}});
            path_support_attempts.push_back(attempt);
            if (options.progress) { attempt["event"]="path_support_separation";options.progress->record(attempt); }
            if (added>0) {
                ++path_support_skipped;
                if(options.search_scheduling) {supported_cg_passes=0;closed_without_progress=0;}
                ++support_rejections_since_refresh;
                // Refresh the integer-valid LP relaxation after repeated
                // unsupported integer candidates. No operational variables
                // are introduced; all retained master rows are valid on the
                // canonical integer domain used for the global optimum.
                const double refresh_left=.05*options.seconds-fractional_refresh_seconds;
                if(fractional_support_separator && support_rejections_since_refresh>=8 &&
                    refresh_left>.01 && remaining(deadline)>0) {
                    const double refresh_started=wall_time();
                    const double refresh_deadline=std::min(deadline,refresh_started+std::min(3.0,refresh_left));
                    support_rejections_since_refresh=0;
                    for(int pass=0;pass<3 && remaining(refresh_deadline)>0;++pass) {
                        const double lp_started=wall_time();
                        LinearModel relaxation=master.matrix;
                        for(auto& v:relaxation.variables)v.binary=false;
                        NativeSolver support_root(relaxation,options,"support_root_refresh");
                        auto point=support_root.optimize(remaining(refresh_deadline));
                        add_time(timing,"support_root_master",lp_started);
                        if(!point.optimal || !point.has_solution || !feasible_point(relaxation,point.values))break;
                        const double certified=point.objective-1e-8*(1+std::abs(point.objective));
                        if(certified>upper+1e-7*(1+std::abs(upper)))
                            throw std::runtime_error("Support root bound contradicts incumbent");
                        if(certified>lower+1e-7*(1+std::abs(lower)))bound_stalled_passes=0;
                        lower=std::max(lower,std::min(upper,certified));
                        observe_global("support_root_refresh_finished");
                        const bool support_added=separate_fractional_support(point.values,"support_root_refresh",refresh_deadline);
                        const bool queue_added=separate_queue(point.values,"support_root_refresh",refresh_deadline);
                        if(!support_added && !queue_added)break;
                    }
                    fractional_refresh_seconds+=wall_time()-refresh_started;
                }
                continue;
            }
        }
        const bool capacity_added = !resumed_design && separate_capacity(solved.values,"integer_master");
        if(!resumed_design) ++supported_candidates;
        const double resource_left=.12*options.seconds-integer_resource_seconds;
        const bool run_resources=!resumed_design && (!options.search_scheduling ||
            (resource_left>.01 && (supported_candidates-1)%3==0));
        const double resource_started=wall_time();
        const double resource_deadline=options.search_scheduling
            ? std::min(deadline,resource_started+std::max(0.0,resource_left)) : deadline;
        const bool resource_added=run_resources && separate_resource(solved.values,"integer_master",resource_deadline);
        const bool task_resource_added=run_resources && separate_task_resource(solved.values,"integer_master",resource_deadline);
        const bool airline_resource_added=run_resources && separate_airline_resource(solved.values,"integer_master",resource_deadline);
        // Cached queue separation never postpones this candidate's complete recourse.
        if(!resumed_design) separate_queue(solved.values,"integer_master",deadline);
        if(run_resources) integer_resource_seconds+=wall_time()-resource_started;
        else if(!resumed_design && options.search_scheduling) ++resource_budget_skips;
        const bool resource_reject=task_resource_added || airline_resource_added;
        const bool defer_cg=resource_reject && (!options.search_scheduling ||
            defer_resource_recourse(true,resource_restart_streak));
        if(options.search_scheduling && !resumed_design) {
            Json record={{"candidate_id",active_candidate},{"stage","resource_schedule"},
                {"separated",run_resources},{"resource_cuts_added",resource_reject},
                {"skip_complete_cg",defer_cg},{"restart_streak",resource_restart_streak},
                {"integer_resource_seconds",integer_resource_seconds},{"resource_budget_seconds",.12*options.seconds},
                {"bound_stalled_passes",bound_stalled_passes}};
            search_schedule_records.push_back(record);
            if(options.progress){record["event"]="search_schedule";options.progress->record(record);}
        }
        if(defer_cg) {
            if(task_resource_added) ++task_resource_skipped;
            if(airline_resource_added) ++airline_resource_skipped;
            ++resource_restart_streak;
            continue;
        }
        if(options.search_scheduling && resource_reject) ++forced_cg_passes;
        // Combine branch certificates before individual support cuts so a
        // candidate blocked at several bases receives their joint cost bound.
        if (!resumed_design && branch_joint_enabled && branch_joint_cache.support_complete &&
            static_cast<int>(branch_joint_records.size()) < options.branch_joint_cut_limit && remaining(deadline) > 0) {
            const int allowance = std::min(options.branch_joint_max_cuts,
                options.branch_joint_cut_limit - static_cast<int>(branch_joint_records.size()));
            auto separated = separate_branch_joint_cost(d, paths, assignment_support_cache, branch_joint_cache,
                design, theta, candidate_theta, std::min(deadline, wall_time() + options.branch_joint_seconds),
                options.double_support_cuts, options.joint_base_cost_cuts,
                static_cast<std::size_t>(allowance), options.branch_joint_floor);
            branch_joint_seconds += separated.seconds;
            branch_joint_numerical_skips += separated.numerical_skips;
            branch_joint_certificates += separated.certificates.size();
            int inserted = 0;
            for (auto& cut : separated.cuts) {
                if (!cut.valid || !cut.violated || !branch_joint_keys.insert(cut.key).second) continue;
                cut.row.name += "_" + std::to_string(branch_joint_records.size());
                branch_joint_row_indices.insert(master.matrix.rows.size());
                master.matrix.rows.push_back(cut.row);
                solver.add_row(cut.row);
                Json record = cut.metadata;
                record.update({{"candidate_id",active_candidate},{"accepted",true},{"key",cut.key},{"kind",cut.kind},
                    {"rhs",cut.row.rhs},{"coefficients",cut.row.coefficients},{"integer_only",true},
                    {"blocking_arcs",cut.blocking_arcs},{"triggers",cut.triggers},{"origin_bases",cut.origin_bases}});
                branch_joint_records.push_back(record);
                if (cut.kind == "double_branch") ++double_support_count;
                else if (cut.kind == "joint_bases") ++joint_base_count;
                ++inserted;
            }
            Json record = separated.metadata;
            record.update({{"candidate_id",active_candidate},{"accepted_cuts",inserted},
                {"seconds",separated.seconds},{"integer_only",true}});
            branch_joint_attempts.push_back(record);
            if (options.progress) {
                record["event"] = "branch_joint_separation";
                options.progress->record(record);
            }
            if (inserted > 0) {
                if(!options.search_scheduling || defer_resource_recourse(true,resource_restart_streak)) {
                    ++branch_joint_skipped; ++resource_restart_streak;
                    master_stalled_passes = 0;
                    continue;
                }
                ++forced_cg_passes;
            }
        }
        // Check mandatory single-order support before the more expensive
        // assignment-independent topology oracle. Each accepted row uses only
        // one assignment trigger and a complete, cached single-path domain.
        if (!resumed_design && options.assignment_support_cuts && assignment_support_cache.complete_domain &&
            static_cast<int>(assignment_support_records.size()) < options.assignment_support_cut_limit && remaining(deadline) > 0) {
            const int allowance = std::min(options.assignment_support_max_cuts,
                options.assignment_support_cut_limit - static_cast<int>(assignment_support_records.size()));
            auto separated = separate_assignment_support_cost(d, assignment_support_cache, design,
                theta, candidate_theta, std::min(deadline, wall_time() + options.assignment_support_seconds),
                static_cast<std::size_t>(allowance), options.assignment_support_floor);
            assignment_support_seconds += separated.seconds;
            assignment_support_numerical_skips += separated.numerical_skips;
            int inserted = 0;
            for (auto& cut : separated.cuts) {
                if (cut.valid) ++assignment_support_certificates;
                if (!cut.valid || !cut.violated || !assignment_support_keys.insert(cut.key).second) continue;
                cut.row.name += "_" + std::to_string(assignment_support_records.size());
                assignment_support_row_indices.insert(master.matrix.rows.size());
                master.matrix.rows.push_back(cut.row);
                solver.add_row(cut.row);
                Json record = cut.metadata;
                record.update({{"candidate_id",active_candidate},{"accepted",true},{"key",cut.key},
                    {"rhs",cut.row.rhs},{"coefficients",cut.row.coefficients},{"integer_only",true},
                    {"blocking_arcs",cut.blocking_arcs}});
                assignment_support_records.push_back(record);
                ++inserted;
            }
            Json record = separated.metadata;
            record.update({{"candidate_id",active_candidate},{"accepted_cuts",inserted},
                {"seconds",separated.seconds},{"integer_only",true}});
            assignment_support_attempts.push_back(record);
            if (options.progress) {
                record["event"] = "assignment_support_separation";
                options.progress->record(record);
            }
            if (inserted > 0) {
                if(!options.search_scheduling || defer_resource_recourse(true,resource_restart_streak)) {
                    ++assignment_support_skipped; ++resource_restart_streak;
                    master_stalled_passes = 0;
                    continue;
                }
                ++forced_cg_passes;
            }
        }
        // A structural cost certificate covers every assignment on this
        // blocked topology. Reoptimize before paying for the SOCP; the cut is
        // integer-valid and cannot enter the original continuous diagnostics.
        if (!resumed_design && options.topology_cuts &&
            static_cast<int>(topology_records.size()) < options.topology_cut_limit && remaining(deadline) > 0) {
            auto topology = separate_topology_cost(d, paths, design, theta, candidate_theta,
                std::min(deadline, wall_time() + options.topology_seconds));
            topology_seconds += topology.seconds;
            if (topology.zero_service_certified) ++topology_certificates;
            const bool inserted = topology.valid && topology.violated && topology_keys.insert(topology.key).second;
            Json record = topology.metadata;
            record.update({{"candidate_id",active_candidate},{"accepted",inserted},{"valid",topology.valid},
                {"violated",topology.violated},{"status",topology.status},{"seconds",topology.seconds},
                {"zero_service_certified",topology.zero_service_certified},{"timed_out",topology.timed_out},
                {"shrink_complete",topology.shrink_complete},{"violation",topology.violation},
                {"certificate_stage",topology.certificate_stage},{"blocking_count",topology.blocking_count},
                {"initial_closed_count",topology.initial_closed_count},{"integer_only",true}});
            topology_attempts.push_back(record);
            if (inserted) {
                topology.row.name += "_" + std::to_string(topology_records.size());
                topology_row_indices.insert(master.matrix.rows.size());
                master.matrix.rows.push_back(topology.row);
                solver.add_row(topology.row);
                record["rhs"] = topology.row.rhs;
                record["coefficients"] = topology.row.coefficients;
                record["blocking_arcs"] = topology.blocking_arcs;
                topology_records.push_back(record);
                ++topology_skipped;
                master_stalled_passes = 0;
            }
            if (options.progress) {
                record["event"] = "topology_separation";
                options.progress->record(record);
            }
            if (inserted && (!options.search_scheduling || defer_resource_recourse(true,resource_restart_streak))) {
                ++resource_restart_streak;
                continue;
            }
        }
        const bool repeated_design = !resumed_design && !integer_designs.insert(design).second;
        if (repeated_design) ++repeated_integer_designs;
        std::string design_key;
        for (double bit : design) design_key.push_back(bit > .5 ? '1' : '0');
        if(options.primal_alns)run_primal_window(false);
        pending_design.clear();
        pending_candidate.clear();
        std::vector<int> selected_p(pool_p.begin(), pool_p.end()), selected_e(pool_e.begin(), pool_e.end());
        SolverOptions sub_options = options;
        const double pareto_left = options.seconds * options.pareto_budget_fraction - pareto_seconds;
        const int cg_call = integer_cg_calls;
        resource_restart_streak=0;
        const bool try_pareto = options.pareto && cg_call >= next_pareto_call && pareto_left > .01;
        // Successful ordinary dual work gives a problem-size dependent scale.
        // Increase effort after a timed-out auxiliary rather than suppressing
        // all future attempts as though no useful Pareto cut existed.
        double previous_dual_seconds = 1.0;
        if (!iterations.empty() && !iterations.back().at("cg_trace").empty())
            previous_dual_seconds = iterations.back().at("cg_trace").back().value("dual_seconds", 1.0);
        const double pareto_allowance = std::min(options.pareto_call_seconds,
            std::max(2.0, 3 * previous_dual_seconds) * std::pow(2.0, std::min(4, pareto_budget_failures)));
        sub_options.pareto_call_seconds = std::min({pareto_allowance, std::max(0.0,pareto_left), .2 * remaining(deadline)});
        const double sub_deadline = options.primal_alns
            ? std::min(deadline,wall_time()+std::max(.01,primal_schedule.cg_slice(wall_time(),remaining(deadline))))
            : options.restricted_mip ? std::min(deadline,wall_time()+std::max(1.0,.45*options.seconds)) : deadline;
        const int cuts_before = cuts;
        const double upper_before = upper;
        auto cached=options.design_certificate_cache && !resumed_design
            ?design_memo.reuse(d,paths,design,candidate_theta,2*options.cg_duality_tolerance):std::nullopt;
        const bool certificate_reused=cached.has_value();
        if(!certificate_reused)++integer_cg_calls;
        auto sub=certificate_reused?std::move(*cached):subproblem(d, paths, design, sub_options, sub_deadline, selected_p, selected_e, try_pareto ? &core : nullptr,retain_recourse,
            false,false,false,-1,-1,std::numeric_limits<double>::quiet_NaN(),
            resumed_design && !pending_primal.is_null()?&pending_primal:nullptr,&pricing_graph_cache,&recourse_state_cache);
        // Capture the original fixed-design point BEFORE incumbent cleanup can
        // close unused facilities and change its strategic design.
        if(options.design_certificate_cache && !certificate_reused) {
            const double memo_started=wall_time();
            design_memo.remember(d,paths,design,sub);
            timing["design_certificate_store"]=timing.value("design_certificate_store",0.0)+wall_time()-memo_started;
        }
        if(options.design_certificate_cache)design_memo_records.push_back({{"design",design},
            {"cache_hit",certificate_reused},{"action",sub.reason},{"master_theta",candidate_theta}});
        if(options.cg_primal_feedback && sub.primal_feedback) {
            const double feedback_started=wall_time();
            if(primal_alns)cg_feedback_records.push_back(primal_alns->observe_cg_columns(*sub.primal_feedback,std::min(deadline,wall_time()+.25)));
            else pending_feedback=sub.primal_feedback;
            const double feedback_seconds=wall_time()-feedback_started;
            timing["cg_primal_feedback"]=timing.value("cg_primal_feedback",0.0)+feedback_seconds;
            if(primal_alns) {
                primal_schedule.charge_auxiliary(feedback_seconds);
                primal_alns->synchronize_used_seconds(primal_schedule.spent());
                heuristic_seconds+=feedback_seconds;
            }
        }
        pending_primal=nullptr;
        Json resume_primal=options.primal_alns ? sub.solution : Json(nullptr);
        account(sub);
        if(options.queue_price_search && !options.queue_fractional_separation) {
            std::vector<double> price_target(master.matrix.variables.size(),0);
            for(int j=0;j<master.layout.count;++j) price_target[master.design_vars[j]]=design[j];
            price_target[theta]=candidate_theta;
            separate_queue(price_target,"integer_native_prices",deadline,&sub.ordinary.queue_prices);
        }
        if (try_pareto && sub.pareto.value("attempted",false)) {
            const bool timed_out = sub.pareto.value("timeout", false);
            pareto_budget_failures = timed_out ? pareto_budget_failures + 1 : 0;
            next_pareto_call = cg_call + options.pareto_frequency *
                (timed_out ? 1 : (1 << std::min(4,pareto_failures)));
        }
        const double cut_violation = sub.ordinary.valid ? value(sub.ordinary, design) - candidate_theta : 0;
        const bool all_courier_candidate = sub.ordinary.valid && !sub.solution.is_null() &&
            sub.solution.value("served", 0.0) <= 1e-7;
        if (all_courier_candidate) ++all_courier_candidates;
        if(!options.search_scheduling)
            bound_stalled_passes = updated_bound_stalls(lower_before_iteration,lower,bound_stalled_passes);
        if (sub.ordinary.valid) {
            const double tolerance = 2 * options.cg_duality_tolerance * (1 + std::abs(candidate_theta));
            const bool progress = capacity_added || resource_added || cut_violation > tolerance ||
                                  upper < upper_before - tolerance;
            master_stalled_passes = progress ? 0 : master_stalled_passes + 1;
            if(options.search_scheduling) {
                ++supported_cg_passes;
                closed_without_progress=progress ? 0 : closed_without_progress+1;
            }
        }
        iterations.push_back({{"iteration", iteration + 1}, {"objective", upper}, {"bound", lower},
                              {"elapsed_seconds",options.progress?options.progress->elapsed():wall_time()-started},
                              {"candidate_id",active_candidate},{"cg_call",certificate_reused?Json(nullptr):Json(cg_call+1)},{"resumed_design",resumed_design},
                              {"capacity_cuts_added",capacity_added ? 1 : 0},{"benders_cuts_added",cuts-cuts_before},
                              {"resource_lp_cuts_added",resource_added ? 1 : 0},
                              {"master_theta",candidate_theta},{"ordinary_cut_violation",cut_violation},
                              {"design_key",design_key},{"repeated_design",repeated_design},{"design_certificate_reused",certificate_reused},
                              {"all_courier_candidate",all_courier_candidate},
                              {"operating_primal_upper",sub.solution.is_null()?Json(nullptr):sub.solution.at("continuous_cost")},
                              {"operating_certified_lower",sub.ordinary.valid?Json(value(sub.ordinary,design)):Json(nullptr)},
                              {"bound_stalled_passes",bound_stalled_passes},
                              {"master_stalled_passes",master_stalled_passes},
                              {"gap", gap(upper, lower)}, {"pricing_complete", sub.ordinary.valid},
                              {"recourse_optimal",sub.recourse_optimal},
                              {"cg_iterations", sub.iterations}, {"service_columns", sub.service.size()},
                              {"empty_columns", sub.empty.size()}, {"certified_cuts", cuts},
                              {"pareto", sub.pareto}, {"timings", sub.timings}, {"wall", sub.wall},
                              {"termination_reason", sub.reason}, {"cg_trace", sub.trace}});
        if(certificate_reused && sub.reason=="cached_design_separation") {
            if(cuts==cuts_before)design_memo.entries.erase(design);
            continue; // Re-solve the global master after restoring the valid cut.
        }
        if (!options.search_scheduling && options.restricted_mip && (cg_call == 0 || (cg_call + 1) % options.heuristic_frequency == 0 || !sub.ordinary.valid))
            restricted_search("generated_column_mip", -1);
        if (!options.search_scheduling && options.restricted_mip && options.local_branching && (cg_call + 1) % (2 * options.heuristic_frequency) == 0)
            restricted_search("local_branching", options.local_branching_radius);
        if(options.primal_alns && !sub.recourse_optimal && remaining(deadline)>0 &&
           wall_time()>=sub_deadline-.05 && sub_deadline<deadline) {
            pending_design=design;pending_candidate=active_candidate;pending_theta=candidate_theta;
            pending_primal=std::move(resume_primal);++primal_yields;
            iterations.back()["termination_reason"]="yield_for_primal";
            iterations.back()["cg_termination_reason"]=sub.reason;
            iterations.back()["resume_primal_retained"]=!pending_primal.is_null();
            if(options.progress)options.progress->record({{"event","yield_for_primal"},
                {"candidate_id",active_candidate},{"service_columns",sub.service.size()},
                {"retained_primal",!pending_primal.is_null()},{"pricing_complete",sub.ordinary.valid}});
            continue;
        }
        if (!sub.ordinary.valid) {
            // A budgeted CG call may stop after adding useful columns. Resume
            // the unchanged design on the next visit using the enlarged pool;
            // do not issue a cut or claim that this subproblem converged.
            if (options.reuse_columns && remaining(deadline)>0 &&
                (sub.service.size()>selected_p.size() || sub.empty.size()>selected_e.size())) {
                pending_design=design; pending_candidate=active_candidate; pending_theta=candidate_theta;
                if(options.primal_alns)pending_primal=std::move(resume_primal);
                continue;
            }
            reason = sub.reason; break;
        }
        for (std::size_t j = 0; j < core.size(); ++j) core[j] = .50 * core[j] + .49 * design[j] + .01 * original_core[j];
        const double interior_available=.05*options.seconds-interior_cut_seconds;
        if (options.interior_cuts && cg_call+1>=next_interior_call &&
            (bound_stalled_passes>=5 || !sub.recourse_optimal) && interior_available>.01 && remaining(deadline)>.01) {
            const double allowance=std::min({5.0,interior_available,remaining(deadline)});
            auto anchor=design;
            for (std::size_t j=0;j<anchor.size();++j)anchor[j]=.9*design[j]+.1*core[j];
            std::vector<int> interior_p(pool_p.begin(),pool_p.end()),interior_e(pool_e.begin(),pool_e.end());
            auto interior_options=options;interior_options.pareto=false;
            const auto previous_candidate=active_candidate;
            active_candidate="interior:"+std::to_string(interior_cut_records.size()+1);
            const int prior_cuts=cuts;
            auto separated=subproblem(d,paths,anchor,interior_options,wall_time()+allowance,
                interior_p,interior_e,nullptr,{},true,false,true,-1,-1,
                std::numeric_limits<double>::quiet_NaN(),nullptr,&pricing_graph_cache);
            interior_cut_seconds+=separated.wall;
            const double current_value=separated.ordinary.valid?value(separated.ordinary,design):-INFINITY;
            const double anchor_value=separated.ordinary.valid?value(separated.ordinary,anchor):-INFINITY;
            account(separated,true,"interior_cut");
            interior_cut_records.push_back({{"candidate_id",active_candidate},{"design",anchor},
                {"source_design",design},{"core_weight",.1},{"allocated_seconds",allowance},
                {"seconds",separated.wall},{"benders_cuts_added",cuts-prior_cuts},
                {"bound_valid",separated.ordinary.valid},{"recourse_optimal",separated.recourse_optimal},
                {"current_design_cut_value",std::isfinite(current_value)?Json(current_value):Json(nullptr)},
                {"current_master_theta",candidate_theta},
                {"anchor_cut_value",std::isfinite(anchor_value)?Json(anchor_value):Json(nullptr)},
                {"termination_reason",separated.reason},{"cg_trace",separated.trace},
                {"model_size",separated.model_size},
                {"direct_global_bound_update",false}});
            active_candidate=previous_candidate;
            next_interior_call=cg_call+11;
        }
        if (gap(upper, lower) <= options.gap) { reason = "gap_limit"; break; }
        // A shifted floating-point certificate can leave a small positive
        // gap even after an exact master returns the same fully priced design.
        // Report the unresolved numerical gap instead of duplicating cuts or
        // pretending that a requested zero gap was reached.
        if (options.adaptive_master && !resumed_design && solved.optimal && master_stalled_passes >= 3 &&
            !master_records.empty() && master_records.back().at("relative_gap_target").get<double>() == 0 &&
            upper - lower <= 4 * options.cg_duality_tolerance * (1 + std::abs(upper))) {
            reason = "certificate_tolerance_limit"; break;
        }
        if(options.cut_pool_management && pending_design.empty() &&
           latest_master_point.size()==master.matrix.variables.size())
            certified_cut_pool.suspend(latest_master_point,options.cut_pool_active_limit,
                                       options.cut_pool_age,master.matrix,set_cut_rhs);
        if ((cg_call + 1) % options.heuristic_frequency == 0) separate_neighborhood();
        if (!options.primal_alns && options.primal_search && !options.restricted_mip && (cg_call + 1) % 5 == 0 && remaining(deadline) > 0
            && options.seconds*options.heuristic_budget_fraction>heuristic_seconds) {
            auto incumbent_design = solution_design(d, best);
            if (incumbent_design != last_neighbor_design) {
                last_neighbor_design = incumbent_design;
                auto neighbor = neighbor_design(d, paths, best);
                if (!neighbor.empty()) {
                    ++primal_search_attempts;
                    active_candidate = "primal_search:" + std::to_string(primal_search_attempts);
                    std::vector<int> reused_p(pool_p.begin(), pool_p.end()), reused_e(pool_e.begin(), pool_e.end());
                    double search_deadline = std::min(deadline, wall_time() + std::min({10.0, 0.1 * remaining(deadline),
                        options.seconds*options.heuristic_budget_fraction-heuristic_seconds}));
                    auto proposal = subproblem(d, paths, neighbor, options, search_deadline, reused_p, reused_e, nullptr,retain_recourse,
                        false,false,false,-1,-1,std::numeric_limits<double>::quiet_NaN(),nullptr,&pricing_graph_cache);
                    heuristic_seconds += proposal.wall;
                    account(proposal, true, "primal_search");
                    heuristic_records.push_back({{"stage", "primal_search"}, {"termination_reason", proposal.reason},
                                                 {"candidate_id",active_candidate},
                                                 {"pricing_complete", proposal.ordinary.valid}, {"wall", proposal.wall},
                                                 {"cg_iterations", proposal.iterations}, {"cg_trace", proposal.trace}});
                }
            }
        }
    }
    } catch (const RunBudgetExceeded& error) {
        // A late incremental model build must not erase the certified bounds,
        // cuts and diagnostics already obtained by the decomposition.
        reason = "time_limit";
        pricing_complete = false;
        budget_interruption = {{"stage", error.stage}, {"sizes", error.sizes},
            {"candidate_id", active_candidate}, {"error", error.what()}};
        last_subproblem_termination = "model_construction_time_limit";
        if (root_master_solves > 0 && integer_master_solves == 0 && root_reason == "round_limit")
            root_reason = "model_construction_time_limit";
        if (options.progress) options.progress->record({{"event", "benders_budget_interruption"},
            {"details", budget_interruption}, {"bound", lower}, {"objective", upper}});
    }
    const auto cut_pool_at_termination=certified_cut_pool.metadata();
    certified_cut_pool.restore_all(master.matrix,set_cut_rhs);
    synchronize_result_paths(d, paths, best);
    best["certified_cut_pool_summary"]=certified_cut_pool.metadata();
    best["certified_cut_pool_summary"].update({{"deduplication_enabled",options.cut_pool_deduplication},
        {"management_enabled",options.cut_pool_management},{"before_final_restore",cut_pool_at_termination},
        {"active_limit",options.cut_pool_active_limit},{"minimum_inactive_age",options.cut_pool_age}});
    best["pricing_graph_cache_summary"]=pricing_graph_cache.metadata();
    best["recourse_state_cache_summary"]=recourse_state_cache.metadata();
    best["design_certificate_cache_summary"]=design_memo.metadata();
    best["design_certificate_cache_records"]=design_memo_records;
    best["cg_primal_feedback_records"]=cg_feedback_records;
    best["stabilized_master_records"]=stabilized_master_records;
    best["stabilized_master_summary"]={{"enabled",options.stabilized_master},{"seconds",stabilized_master_seconds},
        {"calls",stabilized_master_records.size()},{"trial_bound_used_globally",false}};
    best["bound"] = lower;
    best["gap"] = gap(upper, lower);
    best["termination_reason"] = reason;
    best["budget_interruption"] = budget_interruption;
    best["pricing_complete"] = pricing_complete;
    best["pricing_complete_scope"] = "last_processed_subproblem";
    best["last_subproblem_stage"] = last_subproblem_stage;
    best["last_subproblem_termination_reason"] = last_subproblem_termination;
    best["catalogue_complete"] = paths.complete;
    best["support_complete"] = paths.support_complete;
    best["construction_complete"] = paths.support_complete;
    best["generated_columns"]={{"service",d.preprocessed_demand?used_p.size():paths.service_paths.size()},
        {"empty",d.preprocessed_demand?used_e.size():paths.empty_paths.size()}};
    best["generated_columns_scope"]=d.preprocessed_demand?
        "completed_recourse_column_union_including_continuous_root":"registered_generated_catalogue";
    best["reusable_columns"]={{"service",pool_p.size()},{"empty",pool_e.size()}};
    best["initial_columns"] = seeds.metadata;
    best["initial_service_columns"] = seeds.service.size();
    best["initial_empty_columns"] = seeds.empty.size();
    best["pricing_method"] = options.column_generation ? (options.catalogue_pricing && paths.complete?"exact_catalogue_scan":"exact_resource_labeling") : "all_columns";
    best["cg_max_columns"] = options.cg_max_columns;
    best["pricing_tolerance"] = options.pricing_tolerance;
    best["cg_duality_tolerance"] = options.cg_duality_tolerance;
    best["phase_one"] = "unnecessary_validated_all_courier_complete_recourse";
    best["certified_cuts"] = cuts;
    best["benders_cuts"] = cut_records;
    best["iterations"] = iterations;
    best["master_solves"] = master_records;
    best["continuous_root"] = continuous_root;
    best["root_diagnostics"] = {{"termination_reason", root_reason},
        {"scope", root_scope},
        {"lower_bound", root_lower},
        {"continuous_upper_bound", integer_strengthened_root ? Json(nullptr) : Json(root_upper)},
        {"gap", integer_strengthened_root ? Json(nullptr) : Json(gap(root_upper, root_lower))},
        {"converged", !integer_strengthened_root && root_upper-root_lower <= 2*options.cg_duality_tolerance*(1+std::abs(root_upper))},
        {"original_continuous_value_closed",continuous_root_closed},
        {"integer_feasible_upper_bound",upper},
        {"gap_to_integer_incumbent",gap(upper,root_lower)},
        {"projection_enabled",options.root_projection},{"inout_enabled",options.root_projection && options.root_inout},
        {"candidate_epigraph_certified",root_reason=="candidate_epigraph_certified"},
        {"original_candidate_recourse_consistent",root_reason=="candidate_epigraph_certified" ||
            root_reason=="original_candidate_recourse_consistent"},
        {"rounds", root_records}};
    best["topology_cut_records"] = topology_records;
    best["topology_attempts"] = topology_attempts;
    best["topology_summary"] = {{"enabled",options.topology_cuts},{"cuts",topology_records.size()},
        {"attempts",topology_attempts.size()},{"skipped_integer_subproblems",topology_skipped},
        {"zero_service_certificates",topology_certificates},{"seconds",topology_seconds},
        {"master_added_variables",0},{"validity_scope","binary_designs_all_assignments"}};
    best["assignment_support_preprocessing"] = assignment_support_cache.metadata;
    best["path_support_preprocessing"] = path_support_cache.metadata;
    best["path_support_cut_records"] = path_support_records;
    best["path_support_attempts"] = path_support_attempts;
    best["path_support_summary"] = path_support_summary();
    best["fractional_path_support_attempts"]=fractional_support_attempts;
    best["fractional_path_support_records"]=fractional_support_records;
    best["fractional_path_support_summary"]={{"enabled",options.fractional_path_support},
        {"cuts",fractional_support_cuts},{"attempts",fractional_support_attempts.size()},
        {"seconds",fractional_support_seconds},{"refresh_seconds",fractional_refresh_seconds},
        {"master_added_variables",0},{"validity_scope","optimality_preserving_canonical_assignments"}};
    best["assignment_support_cut_records"] = assignment_support_records;
    best["assignment_support_attempts"] = assignment_support_attempts;
    best["assignment_support_summary"] = {{"enabled",options.assignment_support_cuts},
        {"cuts",assignment_support_records.size()},{"attempts",assignment_support_attempts.size()},
        {"skipped_integer_subproblems",assignment_support_skipped},{"certificates",assignment_support_certificates},
        {"numerical_skips",assignment_support_numerical_skips},{"seconds",assignment_support_seconds},
        {"preprocessing_seconds",assignment_support_cache.seconds},{"master_added_variables",0},
        {"validity_scope","binary_designs_single_assignment_trigger"}};
    best["resource_lp_cuts"] = resource_records;
    best["task_savings_metadata"]=task_savings_metadata;
    best["task_resource_cut_records"]=task_resource_records;
    best["task_resource_attempts"]=task_resource_attempts;
    best["task_resource_metadata"]=task_resource_relaxation ? task_resource_relaxation->metadata() : Json{{"enabled",false}};
    best["task_resource_summary"]={{"enabled",options.task_resource_cuts},{"cuts",task_resource_records.size()},
        {"attempts",task_resource_attempts.size()},{"skipped_integer_subproblems",task_resource_skipped},
        {"master_added_variables",0},{"seconds",timing.value("task_resource_separation",0.0)}};
    best["airline_connectivity_metadata"]=airline_connectivity_metadata;
    attach_queue_records(best);
    best["design_face_summary"]={{"enabled",options.design_face_cuts},{"calls",design_face_records.size()},
        {"cuts",design_face_count},{"seconds",design_face_seconds},{"budget_fraction",.08},
        {"call_limit_seconds",12.0},{"cooldown_seconds",60.0},{"direct_global_bound_update",false},
        {"scope","complete_domain_multistate_task_faces"},{"bases",d.n_l},{"state_count",d.n_l+1}};
    best["queue_joint_summary"]={{"enabled",options.queue_joint_cuts},{"cuts",joint_queue_count},
        {"cached_separation_seconds",joint_queue_seconds},{"cached_budget_fraction",.01},
        {"fractional_budget_source","shared_existing_queue_price_budget"},
        {"direct_global_bound_update",false}};
    best["primal_seed_portfolio_summary"]={{"enabled",options.primal_seed_portfolio},
        {"calls",primal_portfolio_records.size()},{"global_lower_bound_from_portfolio",false},
        {"budget_source","existing_heuristic_budget"},{"frontload_fraction",.03},{"call_limit_seconds",18.0}};
    best["primal_coupled_neighborhood_summary"]={{"enabled",options.primal_coupled_neighborhood},
        {"calls",coupled_neighborhood_records.size()},{"global_lower_bound_from_neighborhood",false},
        {"budget_source","existing_heuristic_budget"}};
    best["queue_price_search_summary"]={{"enabled",options.queue_price_search},
        {"calls",queue_price_records.size()},{"seconds",queue_price_seconds},{"budget_fraction",.02},
        {"call_limit_seconds",.5},{"max_evaluations",7},{"auxiliary_dual_optimization",false}};
    best["target_master_refinement_summary"]={{"enabled",options.target_master_refinement},
        {"calls",target_master_records.size()},{"seconds",target_master_seconds},{"budget_fraction",.02},
        {"call_limit_seconds",5},{"cooldown_seconds",30},{"direct_candidate_bound_update",false}};
    best["queue_fractional_summary"]={{"enabled",options.queue_fractional_separation},
        {"calls",queue_fractional_records.size()},{"lp_seconds",queue_fractional_lp_seconds},
        {"price_seconds",options.queue_fractional_separation?queue_price_seconds:0},
        {"seconds",queue_fractional_wall},{"lp_budget_fraction",.02},{"price_budget_fraction",.02},
        {"call_limit_seconds",2},{"cooldown_seconds",5},{"max_lp_solves",3},{"max_new_cuts",2},
        {"marginal_relative_threshold",1e-5},{"donor_refreshes",deferred_queue_serial},
        {"bound_scope","optimality_preserving_master_relaxation"},{"direct_cut_value_bound_update",false}};
    best["queue_polymatroid_summary"]={{"enabled",options.queue_polymatroid_cuts},
        {"cuts",queue_records.size()},{"attempts",queue_attempts.size()},
        {"master_added_variables",0},{"cut_limit",1000},{"max_profiles",8},
        {"seconds",timing.value("queue_polymatroid_build",0.0)+timing.value("queue_polymatroid_profiles",0.0)+timing.value("queue_polymatroid_separation",0.0)},
        {"direct_global_bound_update",false},{"skipped_integer_subproblems",0}};
    best["airline_resource_cut_records"]=airline_resource_records;
    best["airline_resource_attempts"]=airline_resource_attempts;
    best["airline_resource_metadata"]=airline_resource_relaxation ? airline_resource_relaxation->metadata() : Json{{"enabled",false}};
    best["airline_resource_summary"]={{"enabled",options.airline_resource_cuts},{"cuts",airline_resource_records.size()},
        {"attempts",airline_resource_attempts.size()},{"skipped_integer_subproblems",airline_resource_skipped},
        {"master_added_variables",0},{"seconds",timing.value("airline_resource_separation",0.0)}};
    best["branch_joint_preprocessing"] = branch_joint_cache.metadata;
    best["branch_joint_cut_records"] = branch_joint_records;
    best["branch_joint_attempts"] = branch_joint_attempts;
    best["branch_joint_summary"] = branch_joint_summary();
    best["branch_joint_summary"]["completed_double_groups"] = branch_joint_cache.completed_groups;
    best["branch_joint_summary"]["double_candidates_checked"] = branch_joint_cache.candidates_checked;
    best["resource_lp_attempts"] = resource_attempts;
    best["resource_lp_metadata"] = resource_relaxation ? resource_relaxation->metadata() : Json{{"enabled", false}};
    best["od_airline_summary"] = {{"enabled",od_strengthened_root},
        {"resource_cuts",resource_records.size()},
        {"cuts_with_active_od_prices",std::count_if(resource_records.begin(),resource_records.end(),
            [](const Json& record){return record.value("active_od_airline_prices",0)>0;})},
        {"master_added_variables",0},
        {"validity_scope","integer_design_convex_hull"}};
    best["acceleration_records"] = heuristic_records;
    best["master_bound_metadata"] = bound_metadata;
    best["capacity_cut_metadata"] = capacity_enabled
        ? capacity_network.metadata : Json{{"enabled",false},{"reason","disabled"}};
    best["capacity_cuts"] = capacity_records;
    best["cut_schedule"] = {{"policy",options.search_scheduling?"bounded_resource_and_complete_recourse":"same_candidate_capacity_and_cg"},
        {"root_round_unit","root_master_and_cg"},
        {"frequency_unit",options.search_scheduling?"certified_bounds_and_elapsed_time":"integer_cg_calls"}};
    best["master_structure"] = {{"form", "design_binaries_plus_operating_cost"},
        {"variables", master.matrix.variables.size()}, {"design_binaries", master.design_vars.size()},
        {"operating_cost_variables", 1}, {"operational_auxiliary_variables", 0},
        {"initial_rows", initial_master_rows}, {"final_rows", master.matrix.rows.size()},
        {"cones", master.matrix.cones.size()}};
    best["master_model_size"]={{"variables",master.matrix.variables.size()},
        {"linear_rows",master.matrix.rows.size()},{"cones",master.matrix.cones.size()}};
    best["subproblem_model_size"]=subproblem_size;
    best["subproblem_model_size_peak"]=subproblem_peak;
    best["subproblem_model_size_scope"]="integer_recourse_including_heuristic_designs_excluding_fractional_root";
    best["demand_formulation"]=d.preprocessed_demand?"preprocessed":options.conditional_demand?"conditional":"lifted";
    best["structural_strengthening"]=structural_metadata(options);
    best["tight_airline_bounds"]=options.tight_airline_bounds;
    best["demand_formulation_scope"]={{"integer_recourse",d.preprocessed_demand?"preprocessed":options.conditional_demand?"conditional":"lifted"},
        {"fractional_recourse",d.preprocessed_demand?"preprocessed":"lifted"},
        {"continuous_root",d.preprocessed_demand?"preprocessed":"lifted"},
        {"restricted_mip",d.preprocessed_demand?"preprocessed":"lifted"},
        {"pareto_core",options.conditional_demand?"fixed_assignment_face":"global_design_domain"}};
    best["timings"] = timing;
    best["timings"]["topology_separation"] = topology_seconds;
    best["timings"]["assignment_support_separation"] = assignment_support_seconds;
    best["timings"]["path_support_separation"] = path_support_seconds;
    best["timings"]["branch_joint_separation"] = branch_joint_seconds;
    best["timings"]["restricted_mip"] = restricted_seconds;
    best["timings"]["neighborhood_cg"] = neighborhood_seconds;
    best["heuristic_budget_used"] = heuristic_seconds;
    best["primal_design_polish_records"] = primal_design_polish_records;
    best["primal_design_polish_summary"] = {{"enabled",options.primal_polish && options.restricted_mip},
        {"attempts",primal_design_polish_records.size()},
        {"budget_fraction",options.heuristic_budget_fraction},
        {"global_lower_bound_from_polish",false}};
    best["search_schedule_records"]=search_schedule_records;
    best["search_schedule_summary"]={{"enabled",options.search_scheduling},
        {"ordinary_dual_source","native_cplex"},{"root_support_and_recourse_same_round",true},
        {"supported_candidates",supported_candidates},{"supported_cg_passes",supported_cg_passes},
        {"bound_stalled_passes",bound_stalled_passes},{"closed_without_progress",closed_without_progress},
        {"resource_skipped_by_schedule",resource_budget_skips},{"forced_cg_passes",forced_cg_passes},
        {"integer_resource_seconds",integer_resource_seconds},{"resource_budget_fraction",.12},
        {"primal_interval_seconds",std::max(30.0,.20*options.seconds)},
        {"scheduled_restricted_calls",scheduled_restricted_calls},
        {"ordinary_restricted_attempts",ordinary_restricted_attempts},
        {"restricted_search_revision","20261105"},
        {"restricted_search_policy",restricted_search_policy(options.primal_alns,options.restricted_mip,
            options.search_scheduling,options.local_branching)},
        {"full_recourse_certificate_unchanged",true}};
    best["local_branching_summary"]={{"requested",options.local_branching},
        {"enabled",options.restricted_mip && options.local_branching && !options.primal_alns},
        {"attempts",local_branching_attempts},{"radius",options.local_branching_radius},
        {"global_lower_bound_from_neighborhood",false}};
    best["strengthening_parameters"] = {{"root_cut_rounds",options.root_cut_rounds},
        {"root_budget_fraction",options.root_budget_fraction},{"heuristic_budget_fraction",options.heuristic_budget_fraction},
        {"pareto_frequency",options.pareto_frequency},{"pareto_budget_fraction",options.pareto_budget_fraction},
        {"pareto_call_seconds",options.pareto_call_seconds},{"local_branching_radius",options.local_branching_radius},
        {"capacity_cut_limit",options.capacity_cut_limit},{"topology_cut_limit",options.topology_cut_limit},
        {"topology_seconds",options.topology_seconds}};
    best["strengthening_parameters"].update({{"assignment_support_seconds",options.assignment_support_seconds},
        {"assignment_support_cut_limit",options.assignment_support_cut_limit},
        {"assignment_support_max_cuts",options.assignment_support_max_cuts},
        {"assignment_support_floor",options.assignment_support_floor}});
    best["strengthening_parameters"].update({{"branch_joint_seconds",options.branch_joint_seconds},
        {"branch_joint_preprocessing_seconds",options.branch_joint_preprocessing_seconds},
        {"branch_joint_cut_limit",options.branch_joint_cut_limit},{"branch_joint_max_cuts",options.branch_joint_max_cuts},
        {"branch_joint_floor",options.branch_joint_floor}});
    best["strengthening_parameters"].update({{"master_initial_gap",options.master_initial_gap},
        {"master_slice_seconds",options.master_slice_seconds},{"resource_lp_seconds",options.resource_lp_seconds},
        {"resource_lp_cut_limit",options.resource_lp_cut_limit}});
    best["strengthening_parameters"].update({{"path_support_seconds",options.path_support_seconds},
        {"path_support_preprocessing_seconds",options.path_support_preprocessing_seconds},
        {"path_support_cut_limit",options.path_support_cut_limit},{"path_support_max_cuts",options.path_support_max_cuts}});
    best["strengthening_parameters"].update({{"task_resource_seconds",options.task_resource_seconds},
        {"task_resource_cut_limit",options.task_resource_cut_limit}});
    best["strengthening_parameters"].update({{"continuous_root_seconds",options.continuous_root_seconds},
        {"continuous_root_max_columns",options.continuous_root_max_columns},
        {"continuous_root_budget_fraction",options.continuous_root_budget_fraction},
        {"root_inout_weight",options.root_inout_weight}});
    best["strengthening_parameters"].update({{"heuristic_pool_size",options.heuristic_pool_size},
        {"neighborhood_cut_seconds",options.neighborhood_cut_seconds},
        {"neighborhood_cut_budget_fraction",options.neighborhood_cut_budget_fraction}});
    best["algorithm_wall"] = wall_time() - started;
    double accounted_wall = restricted_seconds + topology_seconds + assignment_support_seconds + branch_joint_seconds + path_support_seconds;
    for (const char* key : {"master_build", "greedy_initial_columns", "master_optimize", "master_update",
                           "subproblem_wall", "root_master", "capacity_cut_build", "capacity_cut_separation",
                           "resource_lp_build", "resource_lp_separation", "task_resource_build", "task_resource_separation", "airline_resource_build", "airline_resource_separation", "queue_polymatroid_build", "queue_polymatroid_profiles", "queue_polymatroid_separation", "queue_price_search", "target_master_refinement", "incumbent_cleanup", "continuous_root",
                           "assignment_support_preprocessing", "branch_joint_preprocessing", "path_support_preprocessing", "support_root_master", "queue_fractional_master", "design_face_separation", "design_certificate_store", "cg_primal_feedback", "stabilized_master"})
        accounted_wall += timing.value(key, 0.0);
    best["timings"]["bookkeeping_and_interrupted_work"] =
        std::max(0.0, best["algorithm_wall"].get<double>() - accounted_wall);
    best["accelerations"] = {{"pareto_cuts", options.pareto}, {"reuse_columns", options.reuse_columns},
                             {"column_generation", options.column_generation}, {"reuse_dual_lp",false},
                             {"courier_savings_bound", options.courier_savings_bound},
                             {"initial_design", options.initial_design}, {"warm_start", options.warm_start},
                             {"incumbent_cleanup", options.incumbent_cleanup}, {"primal_search", options.primal_search}};
    best["accelerations"]["primal_design_polish"] = options.primal_polish && options.restricted_mip;
    best["accelerations"].update({{"eliminate_zero_task_columns",options.eliminate_zero_task_columns},
        {"compact_task_heads",options.compact_task_heads},{"cache_pricing_graph",options.cache_pricing_graph},
        {"persist_recourse_model",options.persist_recourse_model},{"stabilized_pricing",options.stabilized_pricing},
        {"primal_support_completion",options.primal_support_completion},{"grouped_local_branching",options.grouped_local_branching},
        {"cut_pool_deduplication",options.cut_pool_deduplication},{"cut_pool_management",options.cut_pool_management}});
    best["accelerations"]["streamlined_recourse"]=options.streamlined_recourse;
    best["accelerations"]["catalogue_pricing"]=options.catalogue_pricing;
    best["accelerations"]["diverse_pricing"]=options.diverse_pricing;
    best["accelerations"]["reuse_recourse_rhs"]=options.reuse_recourse_rhs;
    best["accelerations"]["design_certificate_cache"]=options.design_certificate_cache;
    best["accelerations"]["adaptive_pricing"]=options.adaptive_pricing;
    best["accelerations"]["cg_primal_feedback"]=options.cg_primal_feedback;
    best["accelerations"]["indexed_support_completion"]=options.indexed_support_completion;
    best["accelerations"]["stabilized_master"]=options.stabilized_master;
    best["accelerations"]["queue_polymatroid_cuts"]=options.queue_polymatroid_cuts;
    best["accelerations"]["queue_price_search"]=options.queue_price_search;
    best["accelerations"]["queue_fractional_separation"]=options.queue_fractional_separation;
    best["accelerations"]["design_face_cuts"]=options.design_face_cuts;
    best["accelerations"]["queue_joint_cuts"]=options.queue_joint_cuts;
    best["accelerations"]["primal_coupled_neighborhood"]=options.primal_coupled_neighborhood;
    best["accelerations"]["primal_seed_portfolio"]=options.primal_seed_portfolio;
    best["accelerations"]["primal_alns"]=options.primal_alns;
    best["accelerations"]["target_master_refinement"]=options.target_master_refinement;
    best["accelerations"]["weak_dual_cuts"]=options.weak_dual_cuts;
    best["accelerations"]["interior_cuts"]=options.interior_cuts;
    best["accelerations"]["primal_asset_neighborhood"]=options.primal_asset_neighborhood;
    best["interior_cut_records"]=interior_cut_records;
    best["interior_cut_summary"]={{"enabled",options.interior_cuts},{"attempts",interior_cut_records.size()},
        {"seconds",interior_cut_seconds},{"budget_fraction",.05},{"call_limit_seconds",5},
        {"ordinary_auxiliary_dual_optimization",false},{"direct_global_bound_update",false}};
    best["accelerations"]["search_scheduling"]=options.search_scheduling;
    best["accelerations"]["dual_lp_first"]=false;
    best["ordinary_dual_source"]="native_cplex";
    best["ordinary_auxiliary_dual_optimization"]=false;
    best["accelerations"]["fractional_path_support"]=options.fractional_path_support;
    best["accelerations"]["airline_resource_balance"]=options.airline_resource_balance;
    best["accelerations"].update({{"branch_covering_seeds",options.branch_covering_seeds},
        {"resource_bounds",options.courier_savings_bound && options.resource_bounds},
        {"capacity_cuts",capacity_enabled},
        {"resource_lp_cuts",resource_enabled},{"adaptive_master",options.adaptive_master},
        {"neighborhood_cuts",options.neighborhood_cuts && options.restricted_mip},
        {"continuous_root",options.continuous_root},
        {"branch_pricing_correction",options.branch_pricing_correction},
        {"cg_seed_reuse",options.cg_seed_reuse},
        {"staged_continuous_root",options.staged_continuous_root},
        {"fast_pricing",options.fast_pricing},{"batch_columns",options.batch_columns},
        {"root_projection",options.root_projection},{"root_inout",options.root_projection && options.root_inout},
        {"auxiliary_valid_cuts",options.auxiliary_valid_cuts},
        {"resource_airline_bounds",resource_enabled && options.resource_airline_bounds},
        {"od_airline_bounds",od_strengthened_root},
        {"tight_airline_bounds",options.tight_airline_bounds},{"topology_cuts",options.topology_cuts},
        {"assignment_incident_clique",options.assignment_incident_clique},
        {"shared_rho_joint",options.shared_rho_joint},{"queue_perspective",options.queue_perspective},
        {"assignment_support_cuts",options.assignment_support_cuts},
        {"path_support_cuts",options.path_support_cuts},
        {"task_savings_bound",options.task_savings_bound},{"task_resource_cuts",options.task_resource_cuts},
        {"airline_connectivity_cuts",options.airline_connectivity_cuts},{"airline_resource_cuts",options.airline_resource_cuts},
        {"task_airline_bounds",options.task_airline_bounds},
        {"double_support_cuts",options.double_support_cuts},{"joint_base_cost_cuts",options.joint_base_cost_cuts},
        {"root_cuts",options.root_cuts},{"restricted_mip",options.restricted_mip},
        {"local_branching",options.restricted_mip && options.local_branching && !options.primal_alns},
        {"incumbent_cutoff",options.incumbent_cutoff}});
    best["acceleration_statistics"] = {{"ordinary_cuts", ordinary_count}, {"pareto_cuts", pareto_count},
                                       {"pareto_attempts", pareto_attempts}, {"cg_iterations", cg_iterations},
                                       {"ordinary_cg_iterations", ordinary_cg_iterations}, {"heuristic_cg_iterations", heuristic_cg_iterations},
                                       {"master_starts", master_starts},
                                       {"cleanup_improvements", cleanup_improvements}, {"cleanup_savings", cleanup_savings},
                                       {"primal_search_attempts", primal_search_attempts},
                                       {"reused_service_pool", pool_p.size()}, {"reused_empty_pool", pool_e.size()}};
    best["acceleration_statistics"].update({{"root_cuts",root_count},{"restricted_mip_attempts",restricted_attempts},
        {"local_branching_attempts",local_branching_attempts},
        {"root_master_solves",root_master_solves},{"root_cg_calls",root_cg_calls},
        {"integer_master_solves",integer_master_solves},{"integer_cg_calls",integer_cg_calls},
        {"restricted_mip_improvements",restricted_improvements},{"pareto_consecutive_failures",pareto_failures},
        {"pareto_budget_used",pareto_seconds},{"capacity_cuts",capacity_records.size()}});
    best["acceleration_statistics"].update({{"resource_lp_cuts",resource_records.size()},
        {"resource_lp_attempts",resource_attempts.size()},{"master_stalled_passes",master_stalled_passes}});
    best["acceleration_statistics"].update({{"task_resource_cuts",task_resource_records.size()},
        {"task_resource_attempts",task_resource_attempts.size()},
        {"task_resource_skipped_integer_subproblems",task_resource_skipped}});
    best["acceleration_statistics"]["continuous_bound_rows"] = continuous_bound_rows;
    best["acceleration_statistics"].update({{"auxiliary_valid_cuts",auxiliary_count},
        {"unique_integer_designs",integer_designs.size()},{"repeated_integer_designs",repeated_integer_designs},
        {"all_courier_candidates",all_courier_candidates},{"bound_stalled_passes",bound_stalled_passes}});
    best["acceleration_statistics"].update({{"neighborhood_cg_calls",neighborhood_cg_calls},
        {"neighborhood_cuts",neighborhood_cut_count},{"neighborhood_budget_used",neighborhood_seconds}});
    best["acceleration_statistics"].update({{"conditional_cuts",conditional_cuts},
        {"conditional_subproblems",conditional_subproblems},{"conditional_assignment_patterns",assignment_patterns.size()}});
    observe_global("benders_finished");
    if (options.progress) options.progress->checkpoint(best,"benders_finished",true);
    return best;
}
} // namespace mlnd
