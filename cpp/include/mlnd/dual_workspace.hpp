#pragma once

#include "mlnd/linear_model.hpp"
#include <unordered_map>
#include <limits>

namespace mlnd {
// Persistent auxiliary dual model, never a retained cut or certificate.
// LPs may reuse a simplex basis; SOCPs retain their original cones and reuse
// only the native model. Returned values follow the current source order and
// must be validated against that source before any certificate is accepted.
class DualWorkspace {
public:
    NativeSolver& synchronize(const LinearModel&, const SolverOptions&, Json& diagnostics,
                              int numerical_profile = 0,
                              double deadline = std::numeric_limits<double>::infinity());
    SolveOutcome optimize(double seconds, Json& diagnostics);
    Json metadata() const;
private:
    std::unique_ptr<NativeSolver> solver_;
    LinearModel stored_;
    std::unordered_map<std::string,int> variable_index_, row_index_;
    std::vector<int> source_to_native_;
    std::string options_identity_, cone_identity_;
    const RunProgress* progress_ = nullptr;
    std::size_t synchronizations_ = 0, builds_ = 0, hits_ = 0;
    std::size_t lp_synchronizations_ = 0, socp_synchronizations_ = 0;
    void clear();
};
// Keep the historical source name; the implementation now also accepts the
// unchanged conic Pareto dual. This alias does not imply LP or basis reuse.
using DualRayWorkspace = DualWorkspace;
} // namespace mlnd
