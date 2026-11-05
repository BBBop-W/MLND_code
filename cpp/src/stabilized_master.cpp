#include "mlnd/stabilized_master.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace mlnd {
namespace {
StabilizedMasterResult select_stabilized_design_impl(const Input& d,const Formulation& master,
        int theta,const std::vector<double>& binary_center,double global_lb,double incumbent_ub,
        const SolverOptions& options,double deadline) {
    const double started=wall_time();
    StabilizedMasterResult result;
    result.candidate.bound=-std::numeric_limits<double>::infinity();
    result.metadata={{"scope","candidate_selection_only"},{"bound_scope","not_global"},
        {"global_bound_exported",false},{"solver_calls",0},{"model_built",false},
        {"candidate_validated",false},{"trial_bound",nullptr},{"trial_objective",nullptr},
        {"distance","categorical_task_capacity_plus_small_asset_hamming"},
        {"timings",{{"copy",0.0},{"prepare",0.0},{"build",0.0},{"solve",0.0},{"validation",0.0}}}};
    auto finish=[&](const char* status) {
        result.seconds=wall_time()-started;
        result.metadata["status"]=status;result.metadata["seconds"]=result.seconds;
        result.candidate.optimal=false;result.candidate.bound=-std::numeric_limits<double>::infinity();
        if(!result.candidate.has_solution)result.candidate.status=status;
        return result;
    };
    if(std::isnan(deadline)||!std::isfinite(options.stabilized_master_seconds)||
       options.stabilized_master_seconds<0)return finish("invalid_budget");
    const double local_deadline=std::min(deadline,started+options.stabilized_master_seconds);
    auto remaining=[&] {
        double seconds=std::max(0.0,local_deadline-wall_time());
        if(options.progress)seconds=std::min(seconds,options.progress->remaining());
        return seconds;
    };
    if(remaining()<=0)return finish("no_budget");
    if(!d.origin_assignment()||!master.matrix.minimize||!master.matrix.cones.empty())
        return finish("unsupported_scope");
    const DesignLayout layout(d);
    if(layout.count!=master.layout.count || layout.capacity!=master.layout.capacity ||
       layout.assignment!=master.layout.assignment || layout.airline!=master.layout.airline ||
       master.design_vars.size()!=static_cast<std::size_t>(layout.count) ||
       master.matrix.variables.size()<static_cast<std::size_t>(layout.count) ||
       theta<layout.count || static_cast<std::size_t>(theta)>=master.matrix.variables.size() ||
       master.matrix.variables[theta].binary || master.matrix.variables[theta].cost<=0)
        return finish("invalid_master_layout");
    if(binary_center.size()!=static_cast<std::size_t>(layout.count))return finish("invalid_center");
    if(!std::isfinite(global_lb)||!std::isfinite(incumbent_ub)||global_lb>=incumbent_ub)
        return finish("no_open_finite_level");
    if(!std::isfinite(options.stabilized_master_level)||options.stabilized_master_level<=0||
       options.stabilized_master_level>=1)return finish("invalid_level_fraction");
    const long double exact_level=static_cast<long double>(global_lb)+
        static_cast<long double>(options.stabilized_master_level)*(static_cast<long double>(incumbent_ub)-global_lb);
    const double level=static_cast<double>(exact_level);
    if(!std::isfinite(level)||!std::isfinite(master.matrix.constant)||
       !std::isfinite(level-master.matrix.constant))return finish("invalid_level_value");
    result.metadata["global_lb_input"]=global_lb;result.metadata["incumbent_ub_input"]=incumbent_ub;
    result.metadata["level_fraction"]=options.stabilized_master_level;result.metadata["cost_level"]=level;
    std::vector<int> center(layout.count);
    for(int j=0;j<layout.count;++j) {
        if(j%128==0&&remaining()<=0)return finish("deadline_before_copy");
        const double value=binary_center[j];
        if(master.design_vars[j]!=j||!master.matrix.variables[j].binary)return finish("unsupported_design_columns");
        if(!std::isfinite(value)||std::min(std::abs(value),std::abs(value-1))>1e-7)
            return finish("invalid_center");
        center[j]=value>.5;
    }
    for(int r=0;r<d.assignment_entities();++r) {
        int active=0;for(int l=0;l<d.n_l;++l)active+=center[layout.assignment_index(r,l)];
        if(active>1)return finish("invalid_center_head_state");
    }
    for(int h=0;h<d.n_h;++h) {
        int active=0;for(int s=0;s<d.n_levels;++s)active+=center[layout.capacity+h*d.n_levels+s];
        if(active>1)return finish("invalid_center_capacity_state");
    }
    try {
        const double copy_started=wall_time();
        LinearModel trial;trial.minimize=true;
        trial.variables.reserve(master.matrix.variables.size());trial.rows.reserve(master.matrix.rows.size()+1);
        Coefficients costs;costs.reserve(master.matrix.variables.size());
        for(std::size_t j=0;j<master.matrix.variables.size();++j) {
            if(j%128==0&&remaining()<=0)return finish("deadline_during_copy");
            const auto& variable=master.matrix.variables[j];
            if(!std::isfinite(variable.cost))return finish("invalid_master_cost");
            if(variable.cost!=0)costs.emplace_back(static_cast<int>(j),variable.cost);
            trial.variables.push_back(variable);trial.variables.back().cost=0;
        }
        std::size_t copied_nonzeros=0;
        for(const auto& row:master.matrix.rows) {
            if(remaining()<=0)return finish("deadline_during_copy");
            RowRecord copy{row.name,row.sense,row.rhs,{}};
            copy.coefficients.reserve(row.coefficients.size());
            for(const auto& coefficient:row.coefficients) {
                if(copied_nonzeros++%512==0&&remaining()<=0)return finish("deadline_during_copy");
                copy.coefficients.push_back(coefficient);
            }
            trial.rows.push_back(std::move(copy));
        }
        result.metadata["timings"]["copy"]=wall_time()-copy_started;
        const double prepare_started=wall_time();
        // Include the ORIGINAL objective constant in the level RHS. Every
        // current cut is already present in trial; no local bound is imported.
        trial.row("stabilized_master_original_cost_level",'L',level-master.matrix.constant,std::move(costs));
        auto add_group=[&](const std::vector<int>& indices) {
            int chosen=-1;for(int j:indices)if(center[j])chosen=j;
            if(chosen>=0) {trial.constant+=1;trial.variables[chosen].cost-=1;}
            else for(int j:indices)trial.variables[j].cost+=1;
        };
        for(int r=0;r<d.assignment_entities();++r) {
            if(remaining()<=0)return finish("deadline_during_prepare");
            std::vector<int> group;for(int l=0;l<d.n_l;++l)group.push_back(layout.assignment_index(r,l));
            add_group(group);
        }
        for(int h=0;h<d.n_h;++h) {
            if(remaining()<=0)return finish("deadline_during_prepare");
            std::vector<int> group;for(int s=0;s<d.n_levels;++s)group.push_back(layout.capacity+h*d.n_levels+s);
            add_group(group);
        }
        const std::size_t asset_count=static_cast<std::size_t>(d.n_nodes())+
            static_cast<std::size_t>(d.n_nodes())*d.n_nodes();
        const double tie_weight=.25/(1+static_cast<double>(asset_count));
        auto add_asset=[&](int j) {
            trial.variables[j].cost+=center[j]?-tie_weight:tie_weight;
            if(center[j])trial.constant+=tie_weight;
        };
        for(int j=layout.facility;j<layout.capacity;++j)add_asset(j);
        for(int j=layout.airline;j<layout.count;++j) {
            if(j%128==0&&remaining()<=0)return finish("deadline_during_prepare");
            add_asset(j);
        }
        result.metadata["tie_weight"]=tie_weight;result.metadata["tie_total_upper"]=tie_weight*asset_count;
        result.metadata["copied_rows"]=master.matrix.rows.size();result.metadata["copied_nonzeros"]=copied_nonzeros;
        result.metadata["timings"]["prepare"]=wall_time()-prepare_started;
        if(remaining()<=0)return finish("deadline_before_build");
        // Native extraction is atomic with respect to this local slice. Its
        // existing global construction guards remain active, and its entire
        // wall time is deducted before deciding whether optimization may run.
        const double build_started=wall_time();
        auto solver=std::make_unique<NativeSolver>(trial,options,"stabilized_master_candidate_only");
        result.metadata["timings"]["build"]=wall_time()-build_started;
        result.metadata["model_built"]=true;
        const double after_build=remaining();
        if(after_build<=0)return finish("deadline_after_build");
        const double validation_reserve=std::min(.05,.1*after_build);
        const double solve_budget=after_build-validation_reserve;
        result.metadata["validation_reserve_seconds"]=validation_reserve;
        result.metadata["solve_budget"]=solve_budget;result.metadata["solver_calls"]=1;
        const auto outcome=solver->optimize(solve_budget);
        result.metadata["timings"]["solve"]=outcome.seconds;
        result.metadata["trial_status"]=outcome.status;result.metadata["trial_optimal"]=outcome.optimal;
        result.metadata["trial_bound"]=std::isfinite(outcome.bound)?Json(outcome.bound):Json(nullptr);
        result.metadata["trial_objective"]=outcome.candidate_available?Json(outcome.objective):Json(nullptr);
        if(!outcome.has_solution||!outcome.candidate_available)return finish("no_trial_candidate");
        if(remaining()<=0)return finish("deadline_before_validation");
        const double validation_started=wall_time();
        const auto trial_assessment=assess_point(trial,outcome.values);
        const auto original_assessment=assess_point(master.matrix,outcome.values);
        result.metadata["timings"]["validation"]=wall_time()-validation_started;
        result.metadata["trial_validation"]=trial_assessment.diagnostics();
        result.metadata["original_validation"]=original_assessment.diagnostics();
        if(!trial_assessment.feasible||!original_assessment.feasible)
            return finish("candidate_validation_failed");
        const double original_cost=objective_value(master.matrix,outcome.values);
        if(!std::isfinite(original_cost))return finish("candidate_cost_nonfinite");
        result.candidate=outcome;result.candidate.objective=original_cost;
        result.candidate.optimal=false;result.candidate.bound=-std::numeric_limits<double>::infinity();
        result.candidate.status="validated_stabilized_master_candidate";
        result.metadata["candidate_validated"]=true;result.metadata["candidate_original_cost"]=original_cost;
        result.metadata["distance_objective"]=objective_value(trial,outcome.values);
        result.metadata["timings"]["validation"]=wall_time()-validation_started;
        // Destroy the trial solver before accounting the complete slice time.
        solver.reset();
        return finish("candidate");
    } catch(const RunBudgetExceeded& error) {
        result.metadata["interruption_stage"]=error.stage;return finish("global_budget_exhausted");
    } catch(const IloException& error) {
        result.metadata["error"]=error.getMessage();return finish("solver_exception");
    } catch(const std::exception& error) {
        result.metadata["error"]=error.what();return finish("helper_exception");
    }
}
}

StabilizedMasterResult select_stabilized_design(const Input& d,const Formulation& master,
        int theta,const std::vector<double>& center,double global_lb,double incumbent_ub,
        const SolverOptions& options,double deadline) {
    const double started=wall_time();
    auto result=select_stabilized_design_impl(d,master,theta,center,global_lb,incumbent_ub,options,deadline);
    // Include trial/model/solver cleanup on success, rejection and exceptions.
    result.seconds=wall_time()-started;result.metadata["seconds"]=result.seconds;
    return result;
}
} // namespace mlnd
