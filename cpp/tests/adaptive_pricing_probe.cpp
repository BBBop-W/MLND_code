#include "mlnd/adaptive_pricing.hpp"
#include "mlnd/primal_schedule.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
using mlnd::AdaptivePricingControl;
namespace {
int checks=0;
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);++checks;}
void interval_and_deadline_events() {
    AdaptivePricingControl c;c.initialize(.5,4,64);
    c.initialize(.1,1,1024);
    require(c.weight==.5 && c.interval==4 && c.batch==64,"Resume initialization overwrote controller state");
    c.exact(2);
    require(c.since_exact==0 && !c.exact_due(100),"Exact pass did not reset the bounded discovery interval");
    require(c.exact_due(2.9) && !c.exact_due(3.1),"Observed exact-pricing time did not protect the last certificate window");
    c.discovery(10,9);
    require(c.weight==.5 && c.interval==4,"High acceptance without certified progress relaxed the policy");
    require(!c.exact_due(100),"One discovery exhausted a four-round interval");
    c.discovery(10,9);require(!c.exact_due(100),"Two discoveries exhausted a four-round interval");
    c.discovery(10,9);require(c.exact_due(100),"Bounded interval failed to require the next original-dual pass");
    c.exact(4);
    require(std::abs(c.exact_seconds-2.6)<1e-12 && c.since_exact==0,"Exact timing estimate or counter reset is incorrect");
    c.exact(-1);c.exact(std::numeric_limits<double>::quiet_NaN());
    require(std::abs(c.exact_seconds-2.6)<1e-12,"Invalid timing replaced the finite exact-pricing estimate");
}
void fallback_and_certified_recovery() {
    AdaptivePricingControl c;c.initialize(.5,4,64);
    // Each failed discovery is followed by the existing original-dual
    // fallback. These events model separate, valid pricing iterations.
    for(int i=0;i<3;++i) {c.discovery(0,0);c.exact(1);}
    require(c.interval==1 && c.weight==1 && c.exact_due(100),"Repeated empty discoveries failed to reach exact fallback");
    c.certificate(100);
    require(c.certified_progress && c.certified_lower==100,"Finite complete-domain progress was not recorded");
    require(c.interval==2 && !c.exact_due(100),"Interval-one fallback became an absorbing state despite certified progress");
    c.certificate(99);
    require(!c.certified_progress && c.certified_lower==100,"A worse certificate counted as progress or degraded the stored bound");
    const double old_weight=c.weight;c.discovery(10,9);
    require(c.weight==old_weight && c.interval==2,"Discovery-only success loosened a stalled certificate policy");
    c.exact(1);c.certificate(101);c.discovery(10,9);
    require(std::abs(c.weight-.9)<1e-12 && c.interval==3,"Certified progress plus useful discovery failed to relax smoothing conservatively");
    for(int i=0;i<25;++i) {c.exact(1);c.certificate(102+i);c.discovery(10,10);}
    require(c.weight>=.25 && c.weight<=1 && c.interval==c.max_interval,
            "Adaptive smoothing or interval exceeded its configured range");
    const auto saved=c.certified_lower;
    c.certificate(std::numeric_limits<double>::infinity());c.certificate(std::numeric_limits<double>::quiet_NaN());
    require(c.certified_lower==saved,"Nonfinite observation was recorded as a certificate");
    c.certificate(saved+1e-9);
    require(!c.certified_progress,"Sub-tolerance certificate variation counted as meaningful progress");
    AdaptivePricingControl exact_only;exact_only.initialize(.5,1,64);exact_only.exact(1);exact_only.certificate(1);
    require(exact_only.interval==1 && exact_only.exact_due(100),"Configured exact-only policy was silently expanded");
}
void participation_and_state_reset() {
    AdaptivePricingControl c;c.initialize(.5,4,64);
    c.participation(64,0,1,2);require(c.batch==64,"Low participation without expensive RMP work reduced the batch");
    c.participation(64,0,2,1);require(c.batch==32,"Low participation and expensive RMP work did not reduce the batch");
    c.participation(32,0,2,1);require(c.batch==16,"Second reduction did not reach the ordinary batch floor");
    c.participation(16,0,2,1);require(c.batch==16,"Batch reduction crossed its configured floor");
    c.participation(16,10,2,1);require(c.batch==32,"Useful new columns did not allow a larger batch");
    c.participation(32,20,2,1);require(c.batch==64,"Batch failed to recover to the configured maximum");
    c.participation(64,64,2,1);require(c.batch==64,"Adaptive batch exceeded its maximum");
    c.participation(0,0,2,1);require(c.batch==64,"An empty observation altered the batch");
    for(std::size_t size:{1u,8u}) {
        AdaptivePricingControl small;small.initialize(.5,4,size);small.participation(size,0,2,1);
        require(small.batch==size,"The ordinary minimum batch enlarged a deliberately small configured maximum");
    }
    c.exact(3);c.certificate(100);c.discovery(10,9);
    const auto before=c;c.initialize(.7,2,32);
    require(c.batch==before.batch && c.weight==before.weight && c.certified_lower==before.certified_lower &&
            c.since_exact==before.since_exact,"Same-design continuation lost accumulated control history");
    // The caller resets the whole controller on a changed RHS/design. This
    // verifies the reset primitive; integration must also test that call site.
    c=AdaptivePricingControl{};c.initialize(.4,3,32);
    require(c.weight==.4 && c.interval==3 && c.batch==32 && c.since_exact==0 && c.exact_seconds==0 &&
            !c.certified_progress && std::isinf(c.certified_lower) && c.certified_lower<0,
            "Cross-design reset carried stale progress, timing, or batch state");
}
void auxiliary_budget_events() {
    using mlnd::PrimalPhase;
    mlnd::PrimalSchedule schedule(600,100);
    schedule.charge(PrimalPhase::startup,10,100);
    require(schedule.allowance(PrimalPhase::regular,144,400)==0 &&
            schedule.allowance(PrimalPhase::regular,145,400)==6,
            "Reference regular window does not follow its 45-second cooldown");
    schedule.charge_auxiliary(3);
    require(schedule.spent()==13 && schedule.used(PrimalPhase::regular)==3,
            "CG feedback did not consume the shared total and regular allowances");
    require(schedule.allowance(PrimalPhase::regular,144,400)==0 &&
            schedule.allowance(PrimalPhase::regular,145,400)==6 && schedule.cg_slice(144,400)==1,
            "Auxiliary feedback reset the cooldown and delayed the regular window");
    bool negative=false,nonfinite=false;
    try {schedule.charge_auxiliary(-1);}catch(const std::invalid_argument&) {negative=true;}
    try {schedule.charge_auxiliary(std::numeric_limits<double>::quiet_NaN());}
    catch(const std::invalid_argument&) {nonfinite=true;}
    require(negative && nonfinite && schedule.spent()==13,"Invalid auxiliary time changed the budget ledger");
    // Recorded work can overrun a slice; future allowances must clamp at zero
    // rather than exporting negative solve times or inventing budget credit.
    schedule.charge_auxiliary(200);
    require(schedule.spent()==213 && schedule.allowance(PrimalPhase::regular,200,400)==0 &&
            schedule.allowance(PrimalPhase::elite,200,400)==0 &&
            schedule.allowance(PrimalPhase::startup,200,400)==0,
            "Auxiliary overrun produced a negative or positive exhausted allowance");
    require(schedule.cg_slice(200,400)==400 && schedule.cg_slice(200,-1)==0,
            "Exhausted primal quota left a negative CG slice or kept interrupting CG");
}
}
int main() {try {
    interval_and_deadline_events();fallback_and_certified_recovery();participation_and_state_reset();auxiliary_budget_events();
    std::cout<<nlohmann::json({{"passed",true},{"checks",checks},{"solver_called",false},
        {"scope","controller_event_sequences_not_end_to_end_certification"}}).dump(2)<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
