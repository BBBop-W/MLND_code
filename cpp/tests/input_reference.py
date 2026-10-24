"""Native input parity and malformed-instance checks, without a solver license."""

# ruff: noqa: E402 -- standalone script adds the repository root before local imports.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.standard import load_standard
from src.model.locker_capacity import occupancy_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--instance", type=Path, default=ROOT / "examples/tsl_2l20h50od"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary, instance = args.binary.resolve(), args.instance.resolve()
    checks = []

    def run(path, k=2):
        return subprocess.run(
            [str(binary), str(path), str(k)], text=True, capture_output=True
        )

    for k in (1, 2):
        source = load_standard(instance, case=f"K{k}")
        actual = run(instance, k)
        if actual.returncode:
            raise AssertionError(actual.stderr)
        actual = json.loads(actual.stdout)
        assert actual["max_orders"] == k
        assert actual["effective_delta"] == (source["delta"] if k == 2 else 0)
        expected_arrays = {
            key: value for key, value in source.items() if isinstance(value, np.ndarray)
        }
        assert set(actual["arrays"]) == set(expected_arrays)
        for key, expected in expected_arrays.items():
            np.testing.assert_array_equal(
                np.asarray(actual["arrays"][key]).reshape(expected.shape),
                expected,
                err_msg=f"K{k} {key}",
            )
        np.testing.assert_allclose(
            np.asarray(actual["occupancy_limit"]).reshape(source["slot_counts"].shape),
            occupancy_limits(source["slot_counts"], source["slot_epsilon"][:, None]),
            atol=1e-12,
            rtol=1e-12,
        )
        checks.append({"name": f"K{k}_all_25_arrays_exact", "passed": True})

    def edit_csv(path, edit):
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        edit(rows)
        with path.open("w", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)

    def edit_json(path, edit):
        document = json.loads(path.read_text())
        edit(document)
        path.write_text(json.dumps(document))

    cases = [
        (
            "unknown_hub",
            "coefficients/od_hub.csv",
            lambda p: edit_csv(p, lambda rows: rows[1].__setitem__(1, "H_UNKNOWN")),
            False,
        ),
        (
            "duplicate_link",
            "coefficients/links.csv",
            lambda p: edit_csv(p, lambda rows: rows.__setitem__(2, rows[1])),
            False,
        ),
        (
            "missing_kappa_row",
            "coefficients/kappa.csv",
            lambda p: edit_csv(p, lambda rows: rows.pop()),
            False,
        ),
        (
            "wrong_unit",
            "model.json",
            lambda p: edit_json(
                p, lambda m: m["units"].__setitem__("time_coefficients", "minute")
            ),
            False,
        ),
        (
            "invalid_shape",
            "model.json",
            lambda p: edit_json(
                p, lambda m: m["arrays"]["kappa"].__setitem__("shape", [50, 49])
            ),
            False,
        ),
        (
            "invalid_kappa",
            "coefficients/kappa.csv",
            lambda p: edit_csv(p, lambda rows: rows[1].__setitem__(2, "1.1")),
            False,
        ),
        (
            "wrong_capacity",
            "coefficients/locker_levels.csv",
            lambda p: edit_csv(p, lambda rows: rows[1].__setitem__(5, "99")),
            False,
        ),
        (
            "bad_observed_count",
            "od_demand.csv",
            lambda p: edit_csv(p, lambda rows: rows[1].__setitem__(3, "161")),
            False,
        ),
        (
            "nan_coefficient",
            "coefficients/od_hub.csv",
            lambda p: edit_csv(p, lambda rows: rows[1].__setitem__(2, "nan")),
            False,
        ),
        (
            "reordered_rows",
            "coefficients/od_hub.csv",
            lambda p: edit_csv(
                p, lambda rows: rows.__setitem__(slice(1, None), rows[:0:-1])
            ),
            True,
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="mlnd_cpp_input_") as temp:
        for name, relative, edit, should_pass in cases:
            path = Path(temp) / name
            shutil.copytree(instance, path)
            edit(path / relative)
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["files"][relative] = hashlib.sha256(
                (path / relative).read_bytes()
            ).hexdigest()
            (path / "manifest.json").write_text(json.dumps(manifest))
            result = run(path)
            assert (result.returncode == 0) == should_pass, (name, result.stderr)
            if should_pass:
                baseline = json.loads(run(instance).stdout)
                assert json.loads(result.stdout) == baseline
            checks.append({"name": name, "passed": True})
        path = Path(temp) / "stale_hash"
        shutil.copytree(instance, path)
        with (path / "sites.csv").open("a") as stream:
            stream.write("\n")
        result = run(path)
        assert result.returncode and "SHA-256 mismatch" in result.stderr
        checks.append({"name": "stale_hash", "passed": True})
    report = {
        "passed": True,
        "checks": checks,
        "binary": str(binary),
        "instance": str(instance),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
