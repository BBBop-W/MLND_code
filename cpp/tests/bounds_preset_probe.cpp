#include "mlnd/solver_presets.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    mlnd::SolverConfiguration c;
    c.apply_preset("bounds-20261030");
    c.validate();
    const auto candidate=c.metadata();
    for(const char* name:{"design_face_cuts","queue_joint_cuts","primal_coupled_neighborhood","primal_seed_portfolio"})
        if(!candidate.at("flags").at(name).get<bool>())throw std::runtime_error("Candidate omitted a requested method");
    c.apply_preset("baseline-20261029");
    const auto baseline=c.metadata();
    for(const char* name:{"design_face_cuts","queue_joint_cuts","primal_coupled_neighborhood","primal_seed_portfolio"})
        if(baseline.at("flags").at(name).get<bool>())throw std::runtime_error("Existing baseline policy changed");
    if(c.options.seconds!=600||c.options.gap!=.01||c.method!="benders_cg")
        throw std::runtime_error("Existing baseline budget changed");
    c.apply_preset("bounds-20261030");
    c.plain(); c.validate();
    const auto plain=c.metadata();
    for(const auto& item:plain.at("flags").items())
        if(item.value().get<bool>())throw std::runtime_error("Plain candidate retained a strategy");
    std::cout<<"{\"passed\":true,\"scope\":\"candidate_policy_and_baseline_isolation\"}\n";
}
