"""Verify frozen benchmark files without rebuilding inputs or invoking a solver."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.data.instances import load_saved


def main():
    manifest = json.loads((HERE / "benchmark_manifest.json").read_text())
    for name, entry in manifest["copied_files"].items():
        actual = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        assert actual == entry["sha256"], f"Frozen file changed: {name}"
    config = manifest["config_snapshot"]
    assert hashlib.sha256((ROOT / config["path"]).read_bytes()).hexdigest() == config["sha256"]
    frozen = load_saved(HERE)
    assert (frozen["n_l"], frozen["n_h"], len(frozen["demand"])) == (2, 20, 50)
    assert frozen["demand"].sum() == manifest["total_demand_per_hour"]
    assert np.count_nonzero(frozen["kappa"]) == manifest["retained_kappa_pairs"]
    checked, absent = [], []
    references = [manifest["source_instance"]]
    references += [manifest["reference_runs"][f"h20_K{k}"]["path"] for k in (1, 2)]
    for relative in references:
        source = ROOT / relative
        if not (source / "input_metadata.json").exists():
            absent.append(relative)
            continue
        data = load_saved(source)
        assert set(data) == set(frozen), relative
        for key, value in frozen.items():
            if key in ("max_orders", "metadata"):
                continue
            equal = np.array_equal(value, data[key]) if isinstance(value, np.ndarray) else value == data[key]
            assert equal, f"Reference differs: {relative}: {key}"
        checked.append(relative)
    print(json.dumps({"passed": True, "benchmark": manifest["benchmark_id"],
                      "frozen_files_verified": len(manifest["copied_files"]),
                      "historical_inputs_checked": checked,
                      "historical_inputs_unavailable": absent}, indent=2))


if __name__ == "__main__":
    main()
