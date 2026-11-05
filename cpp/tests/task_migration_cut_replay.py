"""Read-only mathematical replay of the tiny migration results; never invokes a solver."""
import argparse
import sys
from collections import Counter
from decimal import localcontext
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.instances import to_planning_data
from src.data.standard import load_standard
from src.experiments import baseline_cut_audit_20261029 as audit
from src.experiments import task_cost_audit as prior
from src.operations.validation import _physical, _segment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Output directory produced by task_migration_reference.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.directory.resolve()
    records = []
    for schema in ("schema1", "schema2"):
        for K in (1, 2):
            payloads, points = [], []
            for preset in ("full-cplex-20261029", "performance-20261105"):
                folder = base / "runs" / f"{schema}_K{K}_{preset}"
                payload = {name: prior.read(folder / (name + ".json")) for name in
                           ("result", "solution", "input_reference", "preprocessed_demand", "path_flows")}
                audit.require(prior.read(folder / "native_validation.json")["passed"], "Saved physical validation failed")
                payloads.append((folder, payload))
                points.append((preset, prior.design(payload["solution"]), payload["result"]["continuous_cost"]))
            full_folder, full = payloads[0]
            raw = prior.read(full_folder / "full_catalogue.json")
            result, reference = full["result"], full["input_reference"]
            audit.require(prior.sha(full_folder / "full_catalogue.json") == result["complete_catalogue_export"]["sha256"], "Catalogue hash mismatch")
            audit.require(raw["complete"] and raw["assignment_scope"] == "task"
                          and raw["counts"] == result["catalogue"], "Complete catalogue identity mismatch")
            catalogue = dict(segments=raw["segments"],
                missions=[dict(path=p, flow=full["solution"]["flow"][p["id"]]) for p in raw["service"]],
                relocations=[dict(path=p, flow=full["solution"]["empty_flow"][p["id"]]) for p in raw["empty"]])
            instance = Path(reference["instance_path"])
            data = to_planning_data(load_standard(instance, case=f"K{K}"))
            for segment in raw["segments"]:
                _segment(data, segment)
            for path in raw["service"]:
                _physical(data, path)
            for path in raw["empty"]:
                _physical(data, path, empty=True)
            constants = audit.input_constants(reference["model"], full["preprocessed_demand"], catalogue, instance,
                                              assignment_scope="task", max_orders=K)
            for folder, payload in payloads:
                result = payload["result"]
                audit.require(all(payload["preprocessed_demand"][key] == full["preprocessed_demand"][key]
                    for key in ("single_rates", "pair_rates", "head_rates")), "Paired task rates differ")
                coverage = prior._check_record_coverage(payload, catalogue)
                rows, task_evidence = audit.task_rows(result, constants, catalogue, payload)
                static_checks = audit.cross_check_rows(rows, points)
                audit.require(static_checks["passed"], str(static_checks["errors"]))
                queue = []
                for cut in result.get("queue_polymatroid_cut_records", []):
                    profile = cut["certificate"]["profile"]
                    profile_check = audit.validate_profile(profile, constants, payload["preprocessed_demand"], catalogue)
                    queue.append(dict(profile=profile_check, replay=audit.replay_queue_row(cut, constants)))
                queue_cross = audit.cross_check_rows([(str(i), row) for i, row in enumerate(result.get("queue_polymatroid_cut_records", []))], points)
                audit.require(queue_cross["passed"], str(queue_cross["errors"]))
                maximum, ordinary_checks = 0., 0
                for cut in result.get("benders_cuts", []):
                    audit.require(cut.get("condition") is None and all(cut[k] for k in
                        ("catalogue_complete", "pricing_complete", "support_complete")), "Incomplete ordinary certificate")
                    audit.require(len(cut["coefficients"]) == constants["design_count"], "Ordinary cut layout")
                    for label, design, cost in points:
                        violation = audit.D(cut["intercept"]) + sum(audit.D(c) * audit.D(x)
                            for c, x in zip(cut["coefficients"], design)) - audit.D(cost)
                        audit.require(violation <= audit.D(1e-7) * (1 + abs(audit.D(cost))), "Ordinary cut excludes " + label)
                        maximum = max(maximum, float(violation))
                        ordinary_checks += 1
                airline = audit.cross_check_rows([(str(i), row) for i, row in enumerate(result.get("airline_resource_cut_records", []))], points)
                audit.require(airline["passed"], str(airline["errors"]))
                canonical = []
                for (_, other), point in zip(payloads, points):
                    normalized, evidence = audit.canonicalize_support_point(reference["model"], full["preprocessed_demand"],
                        catalogue, other["solution"], point)
                    audit.require(evidence["passed"], str(evidence["errors"]))
                    canonical.append(normalized)
                support = audit.cross_check_rows([(str(i), row) for i, row in enumerate(result.get("fractional_path_support_records", []))], canonical)
                audit.require(support["passed"], str(support["errors"]))
                record = dict(directory=str(folder), passed=True, K=K, schema=schema, tasks=constants["Q"],
                    result_sha256=prior.sha(folder / "result.json"), solution_sha256=prior.sha(folder / "solution.json"),
                    full_catalogue_sha256=prior.sha(full_folder / "full_catalogue.json"), physical_paths_rebuilt=len(raw["service"]) + len(raw["empty"]),
                    coverage=coverage, task_certificates=task_evidence, static_crosspoint=static_checks,
                    queue_kinds=dict(Counter(c["certificate"]["kind"] for c in result.get("queue_polymatroid_cut_records", []))),
                    queue_replay=queue, queue_crosspoint=queue_cross,
                    ordinary_metadata_and_crosspoint=dict(cuts=len(result.get("benders_cuts", [])), checks=ordinary_checks,
                                                         maximum_violation=maximum, full_dual_reconstruction=False),
                    airline_crosspoint=airline, fractional_support_crosspoint=support)
                records.append(record)
    output = dict(passed=True, solver_invoked=False, source_files_read_only=True, runs=records,
                  audit_source_sha256={name: prior.sha(Path(prior.__file__).with_name(name)) for name in
                                       ("task_cost_audit.py", "baseline_cut_audit_20261029.py")})
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(passed=True, runs=len(records), report=str(target))))


if __name__ == "__main__":
    with localcontext() as context:
        context.prec = 60
        main()
