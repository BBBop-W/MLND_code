#include "mlnd/design_face_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace mlnd {
namespace {
bool supported(const Input& d) {
    return d.preprocessed_demand && d.origin_assignment() && d.n_l==2 && d.max_orders==2 &&
        d.flow_policy=="aggregate" && !d.first_leg_consistency && d.route_mode=="direct" && !d.has_candidate_routes;
}
bool certified_face(const Json& face,int head,int state,double& lower) {
    try {
        if(!face.is_object() || !face.value("bound_valid",false) ||
           face.value("assignment_scope",std::string())!="task" ||
           face.value("method",std::string())!="head_design_face_cg" ||
           face.value("scope",std::string())!="complete_domain_head_design_face" ||
           face.value("head",-1)!=head || face.value("state",-1)!=state ||
           !face.value("objective_includes_fixed_costs",false) || !face.value("other_designs_free",false) ||
           !face.value("support_complete",false) || !face.value("catalogue_complete",false) ||
           face.value("is_integer_solution",true) || face.value("direct_global_bound_update",true) ||
           face.value("face_fix_rows",0)!=2)return false;
        lower=face.at("certified_lower_bound").get<double>();
        const auto& certificate=face.at("lower_bound_certificate");
        if(!std::isfinite(lower) || certificate.value("method",std::string())!=
            "native_cplex_dual_with_complete_label_correction")return false;
        const auto& pricing=certificate.at("pricing");
        if(!pricing.value("complete",false))return false;
        for(const char* key:{"minimum_reduced_cost","minimum_service_reduced_cost","minimum_empty_reduced_cost"})
            if(!std::isfinite(pricing.at(key).get<double>()))return false;
        for(const char* key:{"pricing_correction","total_correction","stationarity_error"}) {
            const double value=certificate.at(key).get<double>();
            if(!std::isfinite(value) || value<0)return false;
        }
        return true;
    } catch(const Json::exception&) {return false;}
}
}

TaskSavingsCut assemble_design_face_cut(const Input& d,int head,const std::array<Json,3>& faces,
        double global_lower,const std::vector<double>& design,double theta_value,int theta) {
    const double started=wall_time();TaskSavingsCut cut;cut.status="invalid_design_face_input";
    const DesignLayout layout(d);
    if(!supported(d) || head<0 || head>=d.task_count() || !std::isfinite(global_lower) ||
       !std::isfinite(theta_value) || theta<layout.count || design.size()!=static_cast<std::size_t>(layout.count))return cut;
    for(double x:design)if(!std::isfinite(x) || x<0 || x>1)return cut;
    if(design[layout.assignment_index(head,0)]+design[layout.assignment_index(head,1)]>1+1e-8)return cut;
    const auto master=design_model(d,false);
    std::array<double,3> bounds{global_lower,global_lower,global_lower};
    Json states=Json::array();int certified=0,improved=0;
    for(int state=0;state<3;++state) {
        double lower=0;const bool valid=certified_face(faces[state],head,state,lower);
        if(valid){++certified;bounds[state]=std::max(global_lower,lower);}
        if(bounds[state]>global_lower)++improved;
        states.push_back({{"state",state},{"face_certificate_valid",valid},
            {"face_lower",valid?Json(lower):Json(nullptr)},{"used_lower",bounds[state]},
            {"fallback_to_global",!valid}});
    }
    std::vector<long double> coefficients(layout.count);
    long double absolute=1+std::abs(master.matrix.constant),conversion=0;
    for(int j=0;j<layout.count;++j) {
        const double cost=master.matrix.variables[j].cost;
        if(!std::isfinite(cost))return cut;
        coefficients[j]=cost;absolute+=std::abs(cost);
    }
    for(int l=0;l<2;++l)
        coefficients[layout.assignment_index(head,l)]-=static_cast<long double>(bounds[l+1])-bounds[0];
    cut.row={"head_design_face_cut",'G',0,{{theta,1}}};
    for(int j=0;j<layout.count;++j)if(coefficients[j]!=0) {
        const double rounded=static_cast<double>(coefficients[j]);
        if(!std::isfinite(rounded))return cut;
        conversion+=std::abs(coefficients[j]-rounded);
        cut.row.coefficients.emplace_back(j,rounded);
    }
    for(double bound:bounds)absolute+=std::abs(bound);
    const long double margin=1e-8L*absolute+conversion;
    cut.row.rhs=std::nextafter(static_cast<double>(static_cast<long double>(bounds[0])-
        master.matrix.constant-margin),-std::numeric_limits<double>::infinity());
    if(!std::isfinite(cut.row.rhs))return cut;
    long double violation=cut.row.rhs-theta_value;
    std::ostringstream key;key<<"design_face:"<<head<<':'<<std::scientific<<std::setprecision(12)<<cut.row.rhs<<';';
    for(const auto& [j,c]:cut.row.coefficients)if(j!=theta) {violation-=static_cast<long double>(c)*design[j];key<<j<<':'<<c<<';';}
    cut.violation=static_cast<double>(violation);
    if(!std::isfinite(cut.violation))return cut;
    cut.valid=true;cut.key=key.str();cut.fallback=certified<3;
    cut.status=improved?"certified_head_design_face_cut":"global_bound_fallback_only";
    cut.violated=cut.violation>1e-7*(1+std::abs(global_lower)+std::abs(theta_value));
    cut.certificate={{"kind","head_design_face"},{"assignment_scope","task"},{"head",head},{"states",states},
        {"global_lower",global_lower},{"face_bounds",bounds},{"certified_faces",certified},
        {"improved_faces",improved},{"complete_domain",true},{"global_integer_valid",true},
        {"validity_scope","all_original_integer_designs_and_their_convex_hull"},
        {"objective_includes_fixed_costs",true},{"fixed_objective_constant",master.matrix.constant},
        {"numerical_margin",static_cast<double>(margin)},{"direct_global_bound_update",false},
        {"separation_design",design},{"separation_theta",theta_value},
        {"value_at_separation_point",theta_value+cut.violation}};
    cut.seconds=wall_time()-started;return cut;
}

DesignFaceResult separate_design_face(const Input& d,PathCollection& paths,const SolverOptions& options,
        int head,const std::vector<double>& design,double theta_value,int theta,double global_lower,double seconds,
        CgColumnSeeds* seeds) {
    const double started=wall_time();DesignFaceResult result;
    result.diagnostics={{"head",head},{"assignment_scope","task"},{"calls",0},{"global_lower",std::isfinite(global_lower)?Json(global_lower):Json(nullptr)},
        {"budget_seconds",std::isfinite(seconds)?Json(seconds):Json(nullptr)},{"direct_global_bound_update",false}};
    auto finish=[&](const char* status) {
        result.seconds=wall_time()-started;result.diagnostics["status"]=status;
        result.diagnostics["seconds"]=result.seconds;return result;
    };
    if(!std::isfinite(seconds) || seconds<=0)return finish("budget_exhausted");
    if(!supported(d) || !paths.complete || !paths.support_complete || head<0 || head>=d.task_count())
        return finish("unsupported_design_face_scope");
    if(!std::isfinite(global_lower))return finish("global_lower_unavailable");
    const double deadline=started+seconds;
    std::array<Json,3> faces{Json::object(),Json::object(),Json::object()};
    for(int state=0;state<3;++state) {
        const double left=deadline-wall_time();
        if(left<=0)break;
        SolverOptions face_options=options;face_options.seconds=left/(3-state);face_options.log=false;
        // The public oracle packages expected budget/enumeration/solver
        // failures. Invalid configuration and logical contradictions must
        // remain visible instead of masquerading as missing evidence.
        faces[state]=solve_design_face_cg(d,paths,face_options,head,state,seeds,
            options.cg_seed_reuse?global_lower:std::numeric_limits<double>::quiet_NaN());
        result.faces.push_back(faces[state]);result.diagnostics["calls"]=state+1;
    }
    result.cut=assemble_design_face_cut(d,head,faces,global_lower,design,theta_value,theta);
    return finish(result.cut.valid?"assembled":"invalid_design_face_input");
}
} // namespace mlnd
