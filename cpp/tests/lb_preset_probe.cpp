#include "mlnd/solver_presets.hpp"
#include <iostream>
#include <set>
#include <stdexcept>

int main() {
    using namespace mlnd;
    auto require=[](bool ok,const char* text){if(!ok)throw std::runtime_error(text);};
    SolverConfiguration c;c.apply_preset("baseline-20261029");c.validate();
    auto baseline=c.metadata();
    c.apply_preset("lb-20261030");c.validate();auto candidate=c.metadata();
    const std::set<std::string> changes={"branch_pricing_correction","cg_seed_reuse",
        "staged_continuous_root","design_face_cuts","queue_joint_cuts"};
    for(const auto& item:baseline.at("flags").items()) {
        if(changes.count(item.key())) {
            require(!item.value().get<bool>() && candidate.at("flags").at(item.key()).get<bool>(),
                    "LB preset omitted a lower-bound method");
        } else require(item.value()==candidate.at("flags").at(item.key()),"LB preset changed unrelated/upper-bound policy");
    }
    candidate["flags"]=baseline["flags"];candidate["preset"]=baseline["preset"];
    require(candidate==baseline,"LB preset changed base numerical/model settings");
    for(const char* preset:{"legacy","baseline-20261029","full-cplex-20261029","bounds-20261030"}) {
        c.apply_preset(preset);const auto metadata=c.metadata();
        for(const char* flag:{"branch_pricing_correction","cg_seed_reuse","staged_continuous_root"})
            require(!metadata.at("flags").at(flag).get<bool>(),"Historical preset enabled new behavior");
    }
    c.apply_preset("lb-20261030");c.plain();c.validate();
    const auto plain=c.metadata();
    for(const auto& item:plain.at("flags").items())require(!item.value().get<bool>(),"Plain retained a switch");
    c.apply_preset("lb-20261030");c.set_acceleration("cg_seed_reuse",false);
    bool rejected=false;try{c.validate();}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Staged root accepted disabled seed reuse");
    std::cout<<"{\"passed\":true,\"scope\":\"lb_only_preset_and_historical_isolation\"}\n";
}
