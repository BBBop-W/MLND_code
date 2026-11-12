#pragma once

#include "mlnd/benders.hpp"
#include "mlnd/task_savings.hpp"
#include <array>
#include <vector>

namespace mlnd {

// The retained argument/certificate name `head` denotes a dense task ID q.
struct DesignFaceResult {
    TaskSavingsCut cut;
    Json diagnostics = Json::object();
    Json faces = Json::array();
    double seconds = 0;
};

// Assemble only the certified total-cost face bounds, independently checking
// scope and complete native/pricing evidence. An unusable state falls back to
// the caller's certified global_lower, never its restricted primal objective.
TaskSavingsCut assemble_design_face_cut(const Input&, int head,
    const std::vector<Json>& face_certificates, double global_lower,
    const std::vector<double>& design, double theta_value, int theta);

// Retain fixed-array callers; the implementation still checks for L+1 states.
template<std::size_t N>
TaskSavingsCut assemble_design_face_cut(const Input& d, int head,
    const std::array<Json,N>& faces, double global_lower,
    const std::vector<double>& design, double theta_value, int theta) {
    return assemble_design_face_cut(d,head,std::vector<Json>(faces.begin(),faces.end()),
        global_lower,design,theta_value,theta);
}

// Run up to L+1 bounded complete-domain continuous CG oracles, then assemble
// one integer-valid total-cost row. Local face values never update global LB.
DesignFaceResult separate_design_face(const Input&, PathCollection&, const SolverOptions&,
    int head, const std::vector<double>& design, double theta_value, int theta,
    double global_lower, double seconds, CgColumnSeeds* seeds = nullptr);
} // namespace mlnd
