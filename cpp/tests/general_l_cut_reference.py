"""Matched small L2/L3 regression for arbitrary-base face and two-task cuts.

Uses the same current task-assignment model in frozen-before and after binaries.
No historical head-assignment result is used as a comparator.
"""
# ruff: noqa: E402 -- standalone test imports repository modules.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from task_migration_reference import fixture, table
from src.data.standard import BASE_HEADERS, export_standard, load_standard
from src.experiments.native import validate_native
from src.experiments.storage import save_inputs, sha256, write_json


def prepare(root, bases, size):
    data = fixture()
    hubs = ods = size
    nodes = bases + hubs
    rng = np.random.default_rng(43 + size)
    data.update(n_l=bases, n_h=hubs, n_od=ods,
                name=f"L{bases}H{hubs}OD{ods}",
                bases=np.column_stack((np.arange(bases), np.zeros(bases))),
                hubs=np.column_stack((np.arange(hubs), np.ones(hubs))),
                origins=rng.random((ods, 2)), destinations=rng.random((ods, 2)),
                od_origin=np.arange(ods), od_destination=np.arange(ods),
                od_pairs=[[f"O{i}", f"D{i}"] for i in range(ods)],
                demand=np.arange(ods, dtype=float) * 3 + 4,
                ground_cost=np.full(ods, 20.), deadline=np.full(ods, .8),
                base_cost=.5 + np.arange(bases) * .15,
                flight_times=rng.uniform(.04, .10, (nodes, nodes)),
                allowed_links=np.ones((nodes, nodes), dtype=bool),
                link_cost=rng.uniform(.01, .04, (nodes, nodes)),
                slot_counts=np.full((hubs, 1), 12, dtype=np.int64),
                slot_cost=np.full((hubs, 1), .2), slot_epsilon=np.full(hubs, .05),
                kappa=np.ones((ods, ods)) - np.eye(ods))
    for name, center in (("rider_mean", .08), ("first_cost", 1.),
                         ("terminal_cost", .5), ("receipt_mean", .13),
                         ("home_receipt_mean", .25), ("removal_mean", .03),
                         ("pickup_share", .6), ("last_rider_cost", .5),
                         ("last_rider_mean", .20)):
        data[name] = np.full((ods, hubs), center)
    np.fill_diagonal(data["flight_times"], 0)
    np.fill_diagonal(data["allowed_links"], False)
    if size == 4:
        # Additional small capacity-constrained case with opening-cost tradeoffs.
        data.update(base_cost=5. + np.arange(bases), fleet_cost=3.,
                    ground_cost=5. + np.arange(ods),
                    slot_counts=np.tile([2, 6], (hubs, 1)).astype(np.int64),
                    slot_cost=np.tile([2., 4.], (hubs, 1)))
    demand_sum = int(sum(data["demand"]))
    data["metadata"]["source_metadata"].update(selected_days=["2026-11-09"],
                                                demand_orders_in_area=demand_sum)
    data["metadata"]["purpose"] = "small arbitrary-L cut validity and matched bound comparison"
    directory = root / "instances" / data["name"]
    if directory.exists():
        return directory
    market = root / "markets" / data["name"]
    table(market / "od_nodes.csv", BASE_HEADERS["od_nodes.csv"],
          [(f"{p}{i}", role, *xy, "") for p, role, key in
           (("O", "origin", "origins"), ("D", "destination", "destinations"))
           for i, xy in enumerate(data[key])])
    table(market / "od_demand.csv", BASE_HEADERS["od_demand.csv"],
          [(f"R{i}", f"O{i}", f"D{i}", int(v), 1, v, 1.) for i, v in enumerate(data["demand"])])
    table(market / "periods.csv", ("day", "demand_exposure_hours", "demand_orders"),
          [("2026-11-09", 1, demand_sum)])
    snapshot = root / "snapshots" / data["name"]
    save_inputs(data, snapshot)
    export_standard(snapshot, market, directory)
    load_standard(directory)
    return directory


def run(instance, label, binary, output, seconds):
    target = output / "runs" / instance.name / label
    command = [str(binary), "--preset", "full-cplex-20261029" if label == "cplex" else "performance-20261105",
               "--input", str(instance), "--k", "2", "--seconds", str(seconds),
               "--gap", "0.000001", "--threads", "1", "--seed", "43",
               "--out", str(target), "--export-catalogue"]
    # Equal upper-bound machinery across L2/L3 and both algorithm versions.
    if label != "cplex":
        for feature in ("primal_alns", "primal_support_completion", "indexed_support_completion", "cg_primal_feedback"):
            command += ["--disable", feature]
    provenance = {"command": command, "binary_sha256": sha256(binary),
                  "input_hashes": {str(p.relative_to(instance)): sha256(p)
                                   for p in sorted(instance.rglob("*")) if p.is_file()}}
    if (target / "result.json").exists():
        assert json.loads((target / "command.json").read_text()) == provenance
    else:
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "command.json", provenance)
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=seconds + 60)
        (target / "stdout.log").write_text(completed.stdout)
        (target / "stderr.log").write_text(completed.stderr)
        write_json(target / "process.json", {"returncode": completed.returncode, "elapsed": time.perf_counter() - started})
        assert completed.returncode == 0, completed.stderr
    validation = validate_native(target, instance, max_orders=2)
    write_json(target / "independent_validation.json", validation)
    assert validation["passed"], validation
    result = json.loads((target / "result.json").read_text())
    assert result["assignment_scope"] == "task"
    assert result["bound"] <= result["objective"] + 1e-5 * (1 + abs(result["objective"]))
    return dict(instance=instance.name, method=label, ub=result["objective"], lb=result["bound"],
                gap=result["gap"], wall=result.get("wall", result.get("wall_before_output")),
                termination=result.get("termination_reason"), validated=True,
                face=result.get("design_face_summary"), joint=result.get("queue_joint_summary"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/general_l_cuts_20261109/comparison")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=("before", "after", "cplex"), default=["before", "after", "cplex"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    instances = [prepare(args.output, bases, size) for bases in (2, 3) for size in (2, 3, 4)]
    if args.prepare_only:
        print(json.dumps([str(p) for p in instances]))
        return
    rows = []
    for instance in instances:
        for method, binary in (("before", args.before), ("after", args.after), ("cplex", args.after)):
            if method not in args.methods:
                continue
            row = run(instance, method, binary.resolve(), args.output, args.seconds)
            rows.append(row)
            write_json(args.output / "results.json", rows)
            print(json.dumps(row), flush=True)
        case = [r for r in rows if r["instance"] == instance.name]
        assert max(r["lb"] for r in case) <= min(r["ub"] for r in case) + 2e-5 * (1 + min(r["ub"] for r in case))
    with (args.output / "results.csv").open("w", newline="") as stream:
        fields = [k for k in rows[0] if k not in ("face", "joint")]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
