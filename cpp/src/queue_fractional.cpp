#include "mlnd/queue_fractional.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

namespace mlnd {
QueueFractionalResult separate_fractional_queue(
        const LinearModel& current_master,const DesignLayout& layout,int theta,
        QueuePolymatroidCuts& pool,const QueueNativePrices& donor,
        const std::set<std::string>& existing_keys,const SolverOptions& options,
        double total_seconds,double lp_budget_seconds,double price_budget_seconds,
        std::size_t max_rounds) {
    QueueFractionalResult result;
    const double started=wall_time();double lp_segment=started;
    Json rounds=Json::array();
    result.diagnostics={{"scope","optimality_preserving_master_relaxation"},
        {"input_master_rows",current_master.rows.size()},
        {"input_master_variables",current_master.variables.size()},
        {"input_master_constant",current_master.constant},{"theta_index",theta},
        {"relaxed_binary_count",0},{"master_added_variables",0},
        {"ordinary_auxiliary_optimization_calls",0},{"maximum_lp_rounds",std::min<std::size_t>(3,max_rounds)},
        {"maximum_new_rows",2},{"maximum_lp_call_seconds",.3},{"maximum_price_call_seconds",.5},
        {"maximum_price_evaluations",7},{"bound_source","validated_optimal_current_master_LP"}};
    auto lp_used=[&](){return result.lp_seconds+wall_time()-lp_segment;};
    auto finish=[&](const std::string& reason) {
        result.lp_seconds+=wall_time()-lp_segment;
        result.diagnostics["reason"]=reason;
        result.diagnostics["rounds"]=rounds;
        result.diagnostics["lp_solves"]=rounds.size();
        result.diagnostics["accepted_rows"]=result.cuts.size();
        result.diagnostics["certified_lower"]=std::isfinite(result.certified_lower)?Json(result.certified_lower):Json(nullptr);
        result.diagnostics["lp_seconds"]=result.lp_seconds;
        result.diagnostics["price_seconds"]=result.price_seconds;
        result.wall=wall_time()-started;
        result.diagnostics["wall"]=result.wall;
        return result;
    };
    if(!std::isfinite(total_seconds) || !std::isfinite(lp_budget_seconds) ||
       !std::isfinite(price_budget_seconds) || total_seconds<=0 || lp_budget_seconds<=0 ||
       price_budget_seconds<0 || max_rounds==0)return finish("budget_exhausted");
    if(!current_master.minimize || !current_master.cones.empty() || layout.count<=0 ||
       theta<layout.count || theta>=static_cast<int>(current_master.variables.size()) ||
       !std::isfinite(current_master.constant))
        return finish("invalid_current_master");
    const double deadline=started+std::min(2.0,total_seconds);
    max_rounds=std::min<std::size_t>(3,max_rounds);
    LinearModel relaxation=current_master;
    int relaxed=0;
    for(auto& variable:relaxation.variables)if(variable.binary){variable.binary=false;++relaxed;}
    result.diagnostics["relaxed_binary_count"]=relaxed;
    auto keys=existing_keys;
    SolverOptions lp_options=options;lp_options.log=false;
    std::unique_ptr<NativeSolver> solver;
    std::string reason="round_limit";
    try {
        if(wall_time()>=deadline || lp_used()>=lp_budget_seconds)return finish("budget_exhausted");
        solver=std::make_unique<NativeSolver>(relaxation,lp_options,"fractional_queue_master_lp");
        for(std::size_t round=0;round<max_rounds;++round) {
            const double round_lp_begin=round==0?0:lp_used();
            const double budget=std::min({.3,deadline-wall_time(),lp_budget_seconds-lp_used()});
            if(budget<=0){reason="lp_budget_exhausted";break;}
            Json record={{"round",round+1},{"status","not_solved"},{"optimal",false},{"has_solution",false},
                {"feasible",false},{"objective",nullptr},{"certified_lower",nullptr},{"theta",nullptr},
                {"fractional_coordinates",0},{"lp_time_budget",budget},{"lp_seconds",0.0},
                {"rows_before",relaxation.rows.size()},{"rows_after",relaxation.rows.size()},
                {"cut_violation",nullptr},{"queue_separation_violation",nullptr},
                {"actual_design",nullptr},{"acceptance_threshold",nullptr},{"accepted",false},
                {"reason","not_solved"},{"price_record_index",nullptr}};
            const auto solved=solver->optimize(budget);
            record["status"]=solved.status;record["optimal"]=solved.optimal;
            record["has_solution"]=solved.has_solution;
            bool feasible=false;
            if(solved.has_solution && solved.values.size()==relaxation.variables.size()) {
                try {
                    const auto residuals=point_residuals(relaxation,solved.values);
                    feasible=residuals.at("passed").get<bool>();record["residuals"]=residuals;
                } catch(const std::exception& error) {record["validation_error"]=error.what();}
            }
            record["feasible"]=feasible;
            auto complete_round=[&](const std::string& why) {
                record["reason"]=why;record["rows_after"]=relaxation.rows.size();
                record["lp_seconds"]=std::max(0.0,lp_used()-round_lp_begin);
                rounds.push_back(record);reason=why;
            };
            if(!solved.optimal || !solved.has_solution || !feasible || !std::isfinite(solved.objective)) {
                complete_round("lp_not_certified");break;
            }
            const double margin=1e-7*(1+std::abs(solved.objective));
            const double certified=solved.objective-margin;
            result.certified_lower=std::max(result.certified_lower,certified);
            record["objective"]=solved.objective;record["certified_lower"]=certified;
            record["bound_margin"]=margin;record["theta"]=solved.values[theta];
            const double threshold=1e-5*(1+std::abs(solved.objective));
            record["acceptance_threshold"]=threshold;
            std::vector<double> design(layout.count);int fractional=0;
            record["actual_design"]=std::vector<double>(solved.values.begin(),solved.values.begin()+layout.count);
            for(int j=0;j<layout.count;++j) {
                design[j]=std::clamp(solved.values[j],0.0,1.0);
                if(std::abs(design[j]-std::round(design[j]))>1e-7)++fractional;
            }
            record["fractional_coordinates"]=fractional;
            if(round+1>=max_rounds || result.cuts.size()>=2) {
                complete_round("final_lp_certification");break;
            }
            if(fractional==0){complete_round("integer_lp_point");break;}
            if(!donor.valid){complete_round("native_donor_unavailable");break;}
            const double lp_left=lp_budget_seconds-lp_used();
            // Reserve one remaining LP call before spending the price budget.
            const double reserve=std::min(.3,std::max(0.0,lp_left));
            const double price_budget=std::min({.5,price_budget_seconds-result.price_seconds,
                                               deadline-wall_time()-reserve});
            if(price_budget<=0 || lp_left<=0){complete_round("price_budget_exhausted");break;}
            const double price_started=wall_time();
            result.lp_seconds+=price_started-lp_segment;
            QueuePriceSearchResult searched;std::string price_error;
            TaskSavingsCut joint;double joint_seconds=0;std::string joint_error;
            if(options.queue_joint_cuts) {
                const double joint_started=wall_time();
                try {joint=pool.separate_joint(design,solved.values[theta],theta,std::min(.05,.2*price_budget));}
                catch(const std::exception& error) {joint_error=error.what();}
                joint_seconds=wall_time()-joint_started;
            }
            try {
                const double left=price_budget-(wall_time()-price_started);
                if(left>0)searched=pool.search_prices(donor.prices,design,solved.values[theta],theta,left,7);
            } catch(const std::exception& error) {price_error=error.what();}
            catch(...) {price_error="unavailable_price_search";}
            Json price={{"diagnostics",price_error.empty()?searched.diagnostics:
                Json({{"status","price_search_error"},{"error",price_error}})}};
            price["time_budget"]=price_budget;price["accepted"]=false;
            price["trigger"]="fractional_master";price["lp_round"]=round+1;
            if(options.queue_joint_cuts)price["joint"]={{"seconds",joint_seconds},{"status",joint.status},
                {"valid",joint.valid},{"violated",joint.violated},{"violation",joint.violation},
                {"error",joint_error}};
            const std::size_t price_index=result.price_records.size();
            result.price_records.push_back(std::move(price));
            const double price_finished=wall_time();
            const double spent=price_finished-price_started;
            result.price_seconds+=spent;lp_segment=price_finished;
            result.price_records.back()["seconds"]=spent;
            record["price_record_index"]=price_index;
            auto cut=searched.cut;
            if(joint.valid && joint.violated && !keys.count(joint.key) &&
               (!cut.valid || keys.count(cut.key) || joint.violation>cut.violation))cut=std::move(joint);
            if(!price_error.empty() && !cut.valid){complete_round("price_search_error");break;}
            record["cut_kind"]=cut.certificate.value("kind",std::string("unavailable"));
            if(!cut.valid || !cut.certificate.value("complete_domain",false) ||
               !cut.certificate.value("global_integer_valid",false)) {
                complete_round("queue_row_not_certified");break;
            }
            // Test the actual LP point, not just the clamped separation point or
            // the price pool's improvement over its old profile.
            long double violation=cut.row.rhs;
            for(const auto& [j,coefficient]:cut.row.coefficients)
                violation-=static_cast<long double>(coefficient)*solved.values.at(j);
            const double actual_violation=static_cast<double>(violation);
            record["cut_violation"]=actual_violation;
            record["queue_separation_violation"]=cut.violation;
            if(!cut.violated || !std::isfinite(actual_violation) ||
               std::min(actual_violation,cut.violation)<=threshold) {
                complete_round("current_master_envelope_closed");break;
            }
            if(keys.count(cut.key)){complete_round("duplicate_queue_row");break;}
            cut.certificate["fractional_master"]={{"round",round+1},{"objective",solved.objective},
                {"theta",solved.values[theta]},{"actual_violation",actual_violation},
                {"acceptance_threshold",threshold},{"rows_before",relaxation.rows.size()},
                {"bound_requires_lp_reoptimization",true}};
            solver->add_row(cut.row);relaxation.rows.push_back(cut.row);keys.insert(cut.key);
            result.cuts.push_back(std::move(cut));
            record["accepted"]=true;result.price_records.back()["accepted"]=true;
            complete_round("queue_row_added_reoptimization_required");
        }
    } catch(const IloException& error) {
        result.diagnostics["error"]=error.getMessage();reason="lp_solver_error";
    } catch(const std::exception& error) {
        result.diagnostics["error"]=error.what();reason="fractional_separation_error";
    }
    solver.reset();
    return finish(reason);
}
} // namespace mlnd
