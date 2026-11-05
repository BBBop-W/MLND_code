#include "mlnd/continuous_root_control.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    try {
        using namespace mlnd;
        auto require=[](bool ok,const char* message){if(!ok)throw std::runtime_error(message);};
        ContinuousRootProgress p;
        require(continuous_root_slice_budget(120,600,p)==30,"Initial stage did not leave time for integer search");
        p.initialized=true;p.spent=30;p.since_last=60;
        require(continuous_root_slice_budget(120,500,p)==0,"Resumed before an integer candidate was processed");
        p.integer_passes=1;
        require(continuous_root_slice_budget(120,500,p)==15,"Safe resumed stage has wrong cap");
        p.pending=true;
        require(continuous_root_slice_budget(120,500,p)==0,"Interrupted a pending integer recourse");
        p.pending=false;p.since_last=59;
        require(continuous_root_slice_budget(120,500,p)==0,"Ignored stage cooldown");
        p.since_last=60;p.spent=116;
        require(continuous_root_slice_budget(120,500,p)==4,"Exceeded cumulative root allowance");
        p.spent=120;
        require(continuous_root_slice_budget(120,500,p)==0,"Reused an exhausted allowance");
        p.spent=30;
        require(continuous_root_slice_budget(120,30,p)==0,"Stole final integer/primal reserve");
        require(continuous_root_slice_budget(120,31,p)==1,"Crossed final thirty-second reserve");
        p.closed=true;
        require(continuous_root_slice_budget(120,500,p)==0,"Repeated an already closed continuous solve");
        p.closed=false;p.spent=std::numeric_limits<double>::quiet_NaN();
        require(continuous_root_slice_budget(120,500,p)==0,"Accepted a nonfinite spent budget");
        std::cout<<"{\"passed\":true,\"checks\":11,\"optimizer_calls\":0}\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
