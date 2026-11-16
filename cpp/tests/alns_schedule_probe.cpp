#include "mlnd/primal_schedule.hpp"
#include <iostream>
#include <stdexcept>
using namespace mlnd;
int main(){try {
    auto require=[](bool ok,const char* why){if(!ok)throw std::runtime_error(why);};
    PrimalSchedule s(600,0);
    require(s.allowance(PrimalPhase::startup,0,600)==30,"Startup must share the 180 second primal budget");
    s.charge(PrimalPhase::startup,30,30);
    require(s.allowance(PrimalPhase::regular,74,526)==0,"Window opened before 45 second cooldown");
    require(s.allowance(PrimalPhase::regular,75,525)==6,"Pending CG must permit due primal window");
    require(s.cg_slice(75,525)==0,"Due primal search starved behind pending CG");
    s.charge(PrimalPhase::regular,6,81);
    require(s.cg_slice(81,519)==30,"Integer CG slice did not yield within 30 seconds");
    require(s.cg_slice(120,480)==6,"CG crossed the next primal window");
    require(s.allowance(PrimalPhase::elite,126,474)==15,"Elite refinement not bounded at 15 seconds");
    s.charge(PrimalPhase::elite,15,141);s.charge(PrimalPhase::elite,15,156);
    require(s.allowance(PrimalPhase::elite,201,399)==0,"More than 30 seconds spent on elites");
    s.charge(PrimalPhase::regular,114,315);
    require(s.spent()==180 && s.allowance(PrimalPhase::regular,400,200)==0,"Old and new primal work exceeded shared cap");
    PrimalSchedule late(600,0);
    require(late.allowance(PrimalPhase::regular,590,10)==0,"Started repair in final ten seconds");
    require(late.cg_slice(590,10)==10,"Unused primal allowance must return to Benders");
    std::cout<<"{\"passed\":true,\"checks\":12,\"optimizer_calls\":0}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
