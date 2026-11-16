// Deterministic replay of K2 stabilized:94's numerical recovery failure.
// Real tiny matrices, residual gates, subproblem state and outer Benders run;
// only this translation unit's optimizer outcomes and clock are controlled.
#include "mlnd/benders.hpp"
#include "mlnd/benders_control.hpp"
#include "mlnd/certified_cut_pool.hpp"
#include "mlnd/continuous_root_control.hpp"
#include "mlnd/pricing.hpp"
#include "mlnd/path_identity.hpp"
#include "mlnd/adaptive_pricing.hpp"
#include "mlnd/stabilized_master.hpp"
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
#include <iostream>
namespace mlnd {
namespace recovery_replay {
double now=wall_time();
int initial_calls=0,recovery_calls=0;
int bad_recoveries=0;
bool always_invalid=false,short_first=true;
double recovery_build_seconds=.06;
std::vector<double> allowances;
}
double replay_wall_time(){return recovery_replay::now;}
class ReplayNativeSolver : public NativeSolver {
    const LinearModel& source_;
    std::string role_;
public:
    ReplayNativeSolver(const LinearModel& m,const SolverOptions& o,std::string role="auxiliary")
        :NativeSolver(m,o,role),source_(m),role_(std::move(role)) {
        if(role_=="integer_rmp_recovery")recovery_replay::now+=recovery_replay::recovery_build_seconds;
    }
    SolveOutcome optimize(double seconds) {
        using namespace recovery_replay;
        if(role_!="integer_rmp"&&role_!="integer_rmp_recovery")return NativeSolver::optimize(seconds);
        const bool recovery=role_=="integer_rmp_recovery";
        if(recovery) {
            ++recovery_calls;allowances.push_back(seconds);
            if(seconds<1) {
                now+=seconds;SolveOutcome timeout;timeout.status="AbortTimeLim";timeout.seconds=seconds;return timeout;
            }
        } else {
            ++initial_calls;
            now+=short_first&&initial_calls==1?std::max(.214,seconds-.426):.214;
        }
        auto point=NativeSolver::optimize(2);
        if(!point.has_solution)throw std::runtime_error("Tiny replay reference did not solve");
        if(!recovery||always_invalid||recovery_calls<=bad_recoveries) {
            const auto& c=source_.cones.at(0);
            point.values[c.one]=std::sqrt(c.factor*point.values[c.up]*point.values[c.down]+4.574274766966369e-6);
            point.objective=objective_value(source_,point.values);
            point.optimal=true;point.status="Optimal";
            if(recovery&&recovery_calls<=bad_recoveries)now+=1.2;
        }
        now+=.02;return point;
    }
};
class ReplayDualRayWorkspace : public DualRayWorkspace {
public:
    ReplayNativeSolver& synchronize(const LinearModel&,const SolverOptions&,Json&) {
        throw std::runtime_error("Replay must not invoke auxiliary dual workspace");
    }
};
}
#define NativeSolver ReplayNativeSolver
#define DualRayWorkspace ReplayDualRayWorkspace
#define wall_time replay_wall_time
#define main performance_probe_unused_main
#include "performance_state_probe.cpp"
#undef main
#undef wall_time
#undef NativeSolver
#undef DualRayWorkspace

int main(int argc,char** argv){try {
    using namespace mlnd;
    const std::string mode=argc>1?argv[1]:"resume";
    auto d=performance_probe::fixture();auto paths=enumerate_paths(d);auto o=performance_probe::options();
    if(mode=="outer"||mode=="outer-failure"||mode=="outer-budget-failure") {
        recovery_replay::always_invalid=mode=="outer-failure";
        if(mode=="outer-budget-failure")recovery_replay::recovery_build_seconds=30;
        o.seconds=120;o.max_iterations=20;o.primal_alns=true;
        o.initial_design=o.continuous_root=o.root_cuts=o.root_projection=false;
        o.restricted_mip=o.neighborhood_cuts=o.primal_search=o.capacity_cuts=o.resource_lp_cuts=false;
        o.branch_covering_seeds=false;
        const auto result=solve_benders(d,paths,o);
        int yields=0;bool resumed=false;int failed_cuts=0;
        for(const auto& iteration:result.at("iterations")) {
            yields+=iteration.value("termination_reason",std::string())=="yield_for_primal_recovery";
            resumed=resumed||iteration.value("resumed_design",false);
            if(iteration.value("cg_termination_reason",iteration.value("termination_reason",std::string()))=="primal_validation_failed")
                failed_cuts+=iteration.value("benders_cuts_added",0);
        }
        const Json report={{"mode",mode},{"termination_reason",result.at("termination_reason")},
            {"recovery_yields",yields},{"resumed_design",resumed},{"failed_candidate_cuts",failed_cuts},
            {"recovery_calls",recovery_replay::recovery_calls},{"recovery_allowances",recovery_replay::allowances},
            {"objective",result.at("objective")},{"bound",result.at("bound")},
            {"native_model_validated",result.at("native_model_validated")}};
        std::cout<<report.dump(2)<<std::endl;
        performance_probe::require(yields==1&&resumed,"Outer Benders did not grant exactly one bounded recovery resume");
        performance_probe::require(failed_cuts==0,"Rejected numerical point produced a Benders cut");
        performance_probe::require(result.at("native_model_validated").get<bool>()&&
            result.at("bound").get<double>()<=result.at("objective").get<double>()+1e-6,"Invalid final primal/bound ordering");
        if(recovery_replay::always_invalid) {
            performance_probe::require(result.at("termination_reason")=="primal_validation_failed"&&
                recovery_replay::recovery_calls==3,"Persistent failure did not terminate after three funded profiles");
        } else if(mode=="outer-budget-failure") {
            performance_probe::require(result.at("termination_reason")=="primal_validation_failed"&&
                recovery_replay::recovery_calls==0,"Repeated budget deferral did not stop after one dedicated resume");
        } else performance_probe::require(result.at("termination_reason")!="primal_validation_failed",
            "Recoverable failure still terminated global Benders");
        return 0;
    }
    if(mode=="build-budget") {
        recovery_replay::short_first=false;recovery_replay::recovery_build_seconds=4;
    }
    if(mode=="partial") {
        recovery_replay::short_first=false;recovery_replay::bad_recoveries=1;
    }
    const DesignLayout layout(d);std::vector<double> design(layout.count,0);const std::vector<int> none;
    SubproblemStateCache cache;LabelPricingGraphCache graph;
    int promoted=0;
    auto call=[&](double seconds){return subproblem(d,paths,design,o,recovery_replay::now+seconds,
        none,none,nullptr,[&](const Json&){++promoted;},false,false,false,-1,-1,NAN,nullptr,&graph,&cache);};
    const auto first=call(mode=="build-budget"||mode=="partial"?5:.64);
    const int first_retries=cache.state->primal_retries_for_columns;
    performance_probe::require(first.solution.at("objective").get<double>()==courier_cost(d),
        "Rejected point replaced the all-courier fallback");
    performance_probe::require(promoted==0,"Rejected point reached the incumbent promotion callback");
    performance_probe::require(first.trace.back().at("rmp_residuals").at("worst_tolerance_ratio").get<double>()>7,
        "Original cone residual gate was not exercised");
    recovery_replay::recovery_build_seconds=.06;
    const auto second=call(30);
    Json report={{"first_reason",first.reason},{"first_retries",first_retries},
        {"resumed_reason",second.reason},{"recovery_calls",recovery_replay::recovery_calls},
        {"recovery_allowances",recovery_replay::allowances},{"resumed_certified",second.ordinary.valid},
        {"resumed_recourse_optimal",second.recourse_optimal}};
    std::cout<<report.dump(2)<<std::endl;
    performance_probe::require(!first.ordinary.valid&&!first.selected.valid&&!first.auxiliary.valid&&
        !first.recourse_optimal,"Bad point generated a certificate");
    performance_probe::require(first_retries<3,"Short slice consumed all numerical recovery profiles");
    if(mode=="partial") {
        performance_probe::require(first_retries==1&&
            second.trace.front().at("primal_attempts").at(1).at("numerical_profile")==2,
            "Same-column resume reset a fully tried numerical profile");
    }
    performance_probe::require(second.ordinary.valid&&second.recourse_optimal,
        "Same-column resume failed to recover and certify despite 30 seconds");
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
