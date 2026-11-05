"""Small independent end-to-end task-assignment and courier-home regression.

Creates only an L2 H2 OD2 fixture. Solves each variant serially for at most
15 seconds; compares current CPLEX/Benders bounds and Python physical checks.
Historical experiment inputs and executable files are never modified.
"""

# ruff: noqa: E402 -- standalone test imports repository modules.
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.instances import home_receipt_mean, to_planning_data
from src.data.standard import BASE_HEADERS, export_standard, load_standard
from src.experiments.native import validate_native
from src.experiments.storage import save_inputs, sha256, write_json
from src.model.paths import enumerate_paths
from path_reference import compare, signature


def table(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def fixture():
    """Explicit resolved coefficients; market observation is one hour."""
    def full(shape, value):
        return np.full(shape, value, dtype=float)
    flight = full((4, 4), .05)
    np.fill_diagonal(flight, 0)
    flight[2, 3] = flight[3, 2] = .1
    links = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(links, False)
    return dict(
        n_l=2, n_h=2, n_od=2, max_orders=2, name="task_migration_L2H2OD2",
        model_version="consecutive", route_mode="direct", flow_policy="aggregate",
        candidate_routes=None, delta=.04, operating_limit=1.5, fleet_cost=1.,
        flight_cost=.2, waiting_cost=.3, demand=np.array([4., 7.]),
        ground_cost=np.array([20., 20.]), deadline=np.array([.46, .46]),
        flight_times=flight, allowed_links=links, link_cost=full((4, 4), .01),
        base_cost=np.array([.5, .5]), rider_mean=full((2, 2), .08),
        first_cost=full((2, 2), 1.), terminal_cost=full((2, 2), .5),
        receipt_mean=full((2, 2), .13), home_receipt_mean=full((2, 2), .25),
        removal_mean=full((2, 2), .03), pickup_share=full((2, 2), .6),
        last_rider_cost=full((2, 2), .5), last_rider_mean=full((2, 2), .20),
        slot_counts=np.full((2, 1), 20, dtype=np.int64),
        slot_cost=full((2, 1), .1), slot_epsilon=np.array([.05, .05]),
        kappa=np.array([[0., 1.], [0., 0.]]),
        bases=np.array([[0., 0.], [1., 1.]]), hubs=np.array([[0., 1.], [1., 0.]]),
        origins=np.array([[.1, .1], [.9, .9]]),
        destinations=np.array([[.8, .2], [.2, .8]]),
        od_origin=np.array([0, 1], dtype=np.int64),
        od_destination=np.array([0, 1], dtype=np.int64),
        od_pairs=[["O0", "D0"], ["O1", "D1"]],
        metadata={
            "source_metadata": {"coordinate_system": "local_km", "selected_days": ["2026-11-05"],
                "demand_hour_window": [0, 1], "demand_orders_in_area": 11,
                "demand_exposure_hours": 1.},
            "resolved_config": {"terminal": {"courier_receipt_minutes": 15.,
                "courier_removal_minutes": 3.}},
            "purpose": "small task and courier-home migration regression only",
        },
    )


def prepare(output):
    data = fixture()
    market = output / "market"
    table(market / "od_nodes.csv", BASE_HEADERS["od_nodes.csv"],
          [(f"{prefix}{i}", role, *xy, "")
           for prefix, role, key in (("O", "origin", "origins"), ("D", "destination", "destinations"))
           for i, xy in enumerate(data[key])])
    table(market / "od_demand.csv", BASE_HEADERS["od_demand.csv"],
          [(f"R{i}", f"O{i}", f"D{i}", int(rate), 1, rate, 1.)
           for i, rate in enumerate(data["demand"])])
    table(market / "periods.csv", ("day", "demand_exposure_hours", "demand_orders"),
          [("2026-11-05", 1, 11)])
    instances = {}
    for schema in (1, 2):
        instance = output / f"schema{schema}"
        if not instance.exists():
            source = dict(data)
            if schema == 1:
                source.pop("home_receipt_mean")
            snapshot = output / f"snapshot{schema}"
            save_inputs(source, snapshot)
            export_standard(snapshot, market, instance)
        load_standard(instance)
        instances[schema] = instance
    return instances


def parity(instance, k, binary_dir, output):
    raw = load_standard(instance, case=f"K{k}")
    planning = to_planning_data(raw)
    completed = subprocess.run([str(binary_dir / "mlnd_input_probe"), str(instance), str(k)],
                               text=True, capture_output=True, check=True)
    native_input = json.loads(completed.stdout)
    arrays = {key: value for key, value in raw.items() if isinstance(value, np.ndarray)}
    arrays["home_receipt_mean"] = home_receipt_mean(raw)
    assert arrays.keys() == native_input["arrays"].keys()
    for key, expected in arrays.items():
        np.testing.assert_array_equal(np.asarray(native_input["arrays"][key]).reshape(expected.shape), expected)
    target = output / f"{instance.name}_K{k}_paths.json"
    subprocess.run([str(binary_dir / "mlnd_path_catalogue"), str(instance), str(k), str(target)],
                   text=True, capture_output=True, check=True)
    native_paths = json.loads(target.read_text())
    assert native_paths["paths"].pop("support_complete") is True
    reference = asdict(enumerate_paths(planning))
    differences = dict(float_values_compared=0, float_values_with_roundoff=0,
                       max_absolute_difference=0., largest_difference_at=None)
    compare(native_paths["paths"], reference, "paths", differences)
    assert signature(native_paths["paths"]) == signature(reference)
    # The fixture must actually distinguish courier-home and mixed receipt
    # feasibility. This is a coefficient-level counterfactual, not a solver run.
    mixed_paths = enumerate_paths(replace(planning, home_receipt_mean=planning.receipt_mean))
    if k == 2:
        assert len(mixed_paths.service_paths) > len(reference["service_paths"])
    return dict(schema=instance.name, k=k, passed=True, arrays_compared=len(arrays),
                paths=len(reference["service_paths"]), mixed_receipt_paths=len(mixed_paths.service_paths),
                physical_signature_sha256=signature(reference), **differences)


def solve(instance, k, preset, binary, output, seconds):
    target = output / "runs" / f"{instance.name}_K{k}_{preset}"
    command = [str(binary), "--input", str(instance), "--out", str(target),
               "--preset", preset, "--k", str(k), "--seconds", str(seconds),
               "--gap", "0.000001", "--threads", "1", "--seed", "43", "--export-catalogue"]
    if k == 1 and preset == "performance-20261105":
        for feature in ("primal_alns", "primal_support_completion", "indexed_support_completion", "cg_primal_feedback"):
            command.extend(["--disable", feature])
    provenance = dict(command=command, binary_sha256=sha256(binary))
    started = time.perf_counter()
    if (target / "result.json").exists():
        previous = json.loads((target / "result.json").read_text())
        if not (target / "solution.json").exists() or previous.get("objective") is None:
            target.rename(target.with_name(target.name + f"_failed_{time.time_ns()}"))
    if (target / "result.json").exists():
        assert json.loads((target / "test_command.json").read_text()) == provenance
    else:
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "test_command.json", provenance)
        completed = subprocess.run(command, text=True, capture_output=True, timeout=seconds + 45)
        (target / "test_stdout.log").write_text(completed.stdout)
        (target / "test_stderr.log").write_text(completed.stderr)
        assert completed.returncode == 0, completed.stderr
    validation = validate_native(target, instance, max_orders=k)
    assert validation["passed"], validation["checks"]
    result = json.loads((target / "result.json").read_text())
    solution = json.loads((target / "solution.json").read_text())
    q_count = 2 if k == 1 else 6
    assert np.asarray(solution["assignment"]).shape == (q_count, 2)
    assert result["assignment_scope"] == solution["assignment_scope"] == "task"
    assert result["delivery_time_scope"] == solution["delivery_time_scope"] == "courier_home"
    assert result["task_indexing"] == solution["task_indexing"] == "singles_then_row_major_pairs"
    assert result["task_table"] == solution["task_table"]
    assert len(result["task_table"]) == q_count
    for q, task in enumerate(result["task_table"]):
        assert task["task"] == q
        assert task["first_od"] == (q if q < 2 else (q - 2) // 2)
        assert task["second_od"] == (-1 if q < 2 else (q - 2) % 2)
    if k == 2:
        assert sum(t["rate"] for t in result["task_table"][2:]) > 0
    upper, lower = result["objective"], result["bound"]
    assert isinstance(upper, (int, float)) and isinstance(lower, (int, float))
    assert lower <= upper + 1e-5 * (1 + abs(upper))
    return dict(schema=instance.name, k=k, preset=preset, passed=True, validated=True,
                objective=upper, bound=lower, gap=result["gap"],
                termination_reason=result.get("termination_reason"),
                target_gap_met=result.get("target_gap_met"),
                wall=result.get("wall"), test_wall=time.perf_counter() - started,
                assignment_shape=[q_count, 2], result=str(target / "result.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", type=Path, default=Path("/tmp/mlnd-task-assignment-20261105-build"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/task_assignment_migration_20261105/end_to_end")
    parser.add_argument("--seconds", type=float, default=15)
    args = parser.parse_args()
    assert 0 < args.seconds <= 15
    output, binary_dir = args.output.resolve(), args.binary_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    instances = prepare(output)
    old, new = (load_standard(instances[schema]) for schema in (1, 2))
    np.testing.assert_array_equal(home_receipt_mean(old), home_receipt_mean(new))
    np.testing.assert_array_equal(old["receipt_mean"], new["receipt_mean"])
    assert np.any(home_receipt_mean(new) != new["receipt_mean"])
    rows, path_checks = [], []
    for schema, instance in instances.items():
        for k in (1, 2):
            path_checks.append(parity(instance, k, binary_dir, output))
            for preset in ("full-cplex-20261029", "performance-20261105"):
                row = solve(instance, k, preset, binary_dir / "mlnd_cpp", output, args.seconds)
                rows.append(row)
                print(json.dumps(row), flush=True)
    comparisons = []
    for k in (1, 2):
        selected = [row for row in rows if row["k"] == k]
        best_upper = min(row["objective"] for row in selected)
        best_lower = max(row["bound"] for row in selected)
        assert best_lower <= best_upper + 2e-5 * (1 + abs(best_upper))
        refs = [row for row in selected if row["preset"] == "full-cplex-20261029"]
        assert all(row["gap"] <= 1e-5 for row in refs)
        assert abs(refs[0]["objective"] - refs[1]["objective"]) <= 2e-5 * (1 + best_upper)
        signatures = {p["physical_signature_sha256"] for p in path_checks if p["k"] == k}
        assert len(signatures) == 1
        comparisons.append(dict(k=k, passed=True, maximum_certified_lower_bound=best_lower,
                                minimum_validated_upper_bound=best_upper,
                                cplex_reference_objective=refs[0]["objective"]))
    report = dict(passed=True, schema="task-migration-end-to-end/1", dimensions=dict(n_l=2, n_h=2, n_od=2),
                  protocol=dict(seconds=args.seconds, threads=1, seed=43, target_gap=1e-6, serial=True),
                  schema1_frozen_home_matches_schema2=True, mixed_receipt_preserved=True,
                  binary_sha256=sha256(binary_dir / "mlnd_cpp"),
                  path_checks=path_checks, runs=rows, bound_comparisons=comparisons)
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
