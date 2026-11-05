#pragma once

#include "mlnd/formulation.hpp"

namespace mlnd {
// Append-only generated pools may grow after an incumbent was saved. Extend
// absent columns with zero flow and refresh physical records before export.
// Aggregate fixed-demand records retain nonzero flows plus exact task/arc
// statistic witnesses; dense decision arrays and original path IDs stay intact.
void synchronize_result_paths(const Input&, const PathCollection&, Json& result);
// Check dense or indexed sparse physical records before synchronizing against
// the current pool. Omitted zero-flow records are permitted; nonzero ones are not.
void validate_result_path_records(const PathCollection&, const Json& result);
// Native JSON and CSV writer. It never invokes a Python interpreter.
void write_result(const std::filesystem::path& directory, const Input& input,
                  const PathCollection& paths, const Json& result);
// Optional audit export, separate from compact incumbent records. Streams all
// registered paths and refuses an incomplete catalogue or support domain.
Json write_complete_catalogue(const std::filesystem::path& file, const Input&,
                              const PathCollection&);
} // namespace mlnd
