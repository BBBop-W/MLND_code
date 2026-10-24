#pragma once

#include "mlnd/formulation.hpp"

namespace mlnd {
// Append-only generated pools may grow after an incumbent was saved. Extend
// absent columns with zero flow and refresh physical records before export.
void synchronize_result_paths(const Input&, const PathCollection&, Json& result);
// Native JSON and CSV writer. It never invokes a Python interpreter.
void write_result(const std::filesystem::path& directory, const Input& input,
                  const PathCollection& paths, const Json& result);
} // namespace mlnd
