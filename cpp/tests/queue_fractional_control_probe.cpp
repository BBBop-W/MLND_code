#include "mlnd/queue_fractional_control.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    int checks=0;
    auto require=[&](bool ok,const char* message){if(!ok)throw std::runtime_error(message);++checks;};
    // A new donor at the next eligible master point gets two bounded phases.
    // A 600-second production budget permits 12 seconds for each phase in total.
    const auto initial=mlnd::queue_fractional_budget(600,600,0,0,5,true,false);
    require(initial.has_value(),"Fresh donor was rejected at the cooldown boundary");
    require(initial->seconds==2 && initial->lp_seconds==1 && initial->price_seconds==1,
            "Fresh donor did not get the bounded one-second LP and price phases");
    auto request=[](double remaining=30,double total=600,double lp=0,double price=0,
                    double elapsed=5,bool fresh=true,bool pending=false) {
        return mlnd::queue_fractional_budget(remaining,total,lp,price,elapsed,fresh,pending);
    };
    require(!request(30,600,0,0,5,false),"The same donor enabled another call");
    require(!request(30,600,0,0,5,true,true),"A pending recourse design was interrupted");
    require(!request(30,600,0,0,std::nextafter(5.0,0.0)),"Cooldown ended before five seconds");
    require(!request(30,600,0,0,-1),"Negative elapsed time enabled a call");
    require(!request(0) && !request(-1),"No global wall time enabled a call");
    require(!request(30,0) && !request(30,-1),"Invalid total budget enabled a call");
    require(!request(30,600,-1) && !request(30,600,0,-1),"Negative phase accounting enabled a call");
    const double bad=std::numeric_limits<double>::quiet_NaN();
    const double inf=std::numeric_limits<double>::infinity();
    for(double value:{bad,inf,-inf}) {
        require(!request(value),"Nonfinite remaining wall time enabled a call");
        require(!request(30,value),"Nonfinite total budget enabled a call");
        require(!request(30,600,value),"Nonfinite LP accounting enabled a call");
        require(!request(30,600,0,value),"Nonfinite price accounting enabled a call");
        require(!request(30,600,0,0,value),"Nonfinite cooldown time enabled a call");
    }
    require(!request(30,600,12,0),"Exhausted LP allowance still launched price work");
    require(!request(30,600,0,12),"Exhausted price allowance still launched an LP");
    require(!request(30,600,12.1,0) && !request(30,600,0,12.1),
            "Actual phase overrun was ignored by the next allocation");
    const auto short_wall=request(.5);
    require(short_wall && short_wall->seconds<=.5 && short_wall->lp_seconds==.25 &&
            short_wall->price_seconds==.25,"Short global wall time was not shared between both phases");
    const auto lp_end=request(30,600,11.75,0);
    require(lp_end && lp_end->lp_seconds<=.25 && lp_end->lp_seconds>.249999 &&
            lp_end->price_seconds==1 && lp_end->seconds<=1.25,
            "LP allocation exceeded its independent remaining allowance");
    const auto price_end=request(30,600,0,11.75);
    require(price_end && price_end->price_seconds<=.25 && price_end->price_seconds>.249999 &&
            price_end->lp_seconds==1 && price_end->seconds<=1.25,
            "Price allocation exceeded its independent remaining allowance");
    const auto small_budget=request(30,10,0,0);
    require(small_budget && small_budget->lp_seconds<=.2 && small_budget->price_seconds<=.2 &&
            small_budget->seconds<=.4,"Dedicated budgets did not scale with the total run time");
    require(!request(std::numeric_limits<double>::denorm_min()),
            "An unrepresentably small phase allowance enabled a call");
    std::cout<<"{\"passed\":true,\"checks\":"<<checks<<"}\n";
}
