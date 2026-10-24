#pragma once

#include "mlnd/linear_model.hpp"
#include <unordered_map>

namespace mlnd {
// Persistent LP implementation only. It stores a simplex basis and matrix,
// never a conic certificate or a full-path lower bound. Returned values are
// restored to the supplied source model's order for independent validation.
class DualRayWorkspace {
public:
    NativeSolver& synchronize(const LinearModel&, const SolverOptions&, Json& diagnostics);
    SolveOutcome optimize(double seconds, Json& diagnostics);
private:
    std::unique_ptr<NativeSolver> solver_;
    LinearModel stored_;
    std::unordered_map<std::string,int> variable_index_, row_index_;
    std::vector<int> source_to_native_;
    void clear();
};
} // namespace mlnd
