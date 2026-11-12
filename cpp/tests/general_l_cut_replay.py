"""Independent saved-output replay for the matched small general-L comparison.

Rebuilds queue profiles from every physical path and verifies every joint
state and lifting coefficient with Decimal arithmetic. Face rows are rebuilt
from their state certificates and checked at all three saved physical designs;
the native face dual certificates are scope-checked, not independently solved.
Never invokes an optimization solver.
"""
# ruff: noqa: E402 -- standalone test imports repository modules.
from __future__ import annotations

import argparse
from collections import Counter
from decimal import localcontext
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.instances import to_planning_data
from src.data.standard import load_standard
from src.experiments import baseline_cut_audit_20261029 as audit
from src.experiments import task_cost_audit as prior
from src.experiments.storage import write_json
from src.operations.validation import _physical, _segment


def face_row(record, constants, raw):
    """Verify the L+1 disjunction and reconstruct its total-cost affine row."""
    cert = record.get("certificate", {})
    if not cert:
        audit.require(not record.get("accepted", False), "Accepted face has no certificate")
        return None, dict(status="no_certified_row")
    L, Q, count, start = (constants[k] for k in ("L", "Q", "design_count", "assignment"))
    audit.require(cert["kind"] == "head_design_face" and cert["assignment_scope"] == "task"
                  and cert["complete_domain"] and cert["global_integer_valid"]
                  and cert["objective_includes_fixed_costs"] and not cert["direct_global_bound_update"],
                  "Face row scope")
    task = cert["head"]
    audit.require(type(task) is int and 0 <= task < Q, "Face task ID")
    bounds = [audit.D(v) for v in cert["face_bounds"]]
    states = cert["states"]
    audit.require(len(bounds) == len(states) == L + 1 and all(v.is_finite() for v in bounds),
                  "Face state dimensions")
    baseline = audit.D(cert["global_lower"])
    oracle = {f["state"]: f for f in record.get("faces", []) if "state" in f}
    certified = 0
    for state, entry in enumerate(states):
        audit.require(entry["state"] == state, "Face state ordering")
        if entry["face_certificate_valid"]:
            f = oracle[state]
            audit.require(f["bound_valid"] and f["assignment_scope"] == "task"
                          and f["method"] == "head_design_face_cg"
                          and f["scope"] == "complete_domain_head_design_face"
                          and f["head"] == task and f["state"] == state
                          and f["face_fix_rows"] == L and f["objective_includes_fixed_costs"]
                          and f["other_designs_free"] and f["support_complete"] and f["catalogue_complete"]
                          and not f["is_integer_solution"] and not f["direct_global_bound_update"],
                          "Face oracle domain/cost scope")
            native = f["lower_bound_certificate"]
            audit.require(native["method"] == "native_cplex_dual_with_complete_label_correction"
                          and native["pricing"]["complete"], "Face full-domain certificate")
            for key in ("pricing_correction", "total_correction", "stationarity_error"):
                audit.require(math.isfinite(native[key]) and native[key] >= 0, "Face correction")
            for key in ("minimum_reduced_cost", "minimum_service_reduced_cost", "minimum_empty_reduced_cost"):
                audit.require(math.isfinite(native["pricing"][key]), "Face pricing finite value")
            lower = audit.D(f["certified_lower_bound"])
            audit.close(entry["face_lower"], lower, "Face bound provenance")
            expected = max(baseline, lower)
            certified += 1
        else:
            audit.require(entry["fallback_to_global"], "Missing face requires global fallback")
            expected = baseline
        audit.close(entry["used_lower"], expected, "Used face lower")
        audit.close(bounds[state], expected, "Stored face lower")
    audit.require(cert["certified_faces"] == certified, "Certified face count")
    fixed = ([audit.D(v) for v in raw["base_cost"]] + [audit.ZERO] * raw["n_h"]
             + [audit.D(v) for v in raw["slot_cost"].flat] + [audit.ZERO] * (Q * L)
             + [audit.D(v) for v in raw["link_cost"].flat])
    audit.require(len(fixed) == count, "Fixed-cost layout")
    for base_id in range(L):
        fixed[start + task * L + base_id] -= bounds[base_id + 1] - bounds[0]
    margin = audit.D(cert["numerical_margin"])
    audit.require(margin.is_finite() and margin >= 0, "Face numerical margin")
    audit.close(cert["fixed_objective_constant"], audit.ZERO, "Fixed objective constant")
    rhs = bounds[0] - audit.D(cert["fixed_objective_constant"]) - margin
    row = dict(rhs=float(rhs), coefficients=[[j, float(a)] for j, a in enumerate(fixed) if a] + [[count, 1.]])
    if record.get("accepted", False):
        audit.close(record["rhs"], rhs, "Face row rhs")
        entries = dict(record["coefficients"])
        audit.require(len(entries) == len(record["coefficients"]) and entries.get(count) == 1, "Face sparse row")
        for j, a in enumerate(fixed):
            audit.close(entries.get(j, 0), a, "Face coefficient", relative=2e-11)
        row = record
    design = [audit.D(v) for v in cert["separation_design"]]
    audit.require(len(design) == count, "Face separation design")
    value = rhs - sum(a * x for a, x in zip(fixed, design))
    audit.close(cert["value_at_separation_point"], value, "Face point value")
    audit.close(record["violation"], value - audit.D(cert["separation_theta"]), "Face violation")
    return row, dict(task=task, states=L + 1, certified_states=certified,
                     accepted=record.get("accepted", False), full_dual_reconstruction=False)


def replay_case(folder):
    payloads, points = [], []
    for method in ("before", "after", "cplex"):
        run = folder / method
        payload = {name: prior.read(run / (name + ".json")) for name in
                   ("result", "solution", "input_reference", "preprocessed_demand", "path_flows")}
        audit.require(prior.read(run / "independent_validation.json")["passed"], "Physical validation failed")
        payloads.append((run, payload))
        points.append((method, prior.design(payload["solution"]), payload["result"]["continuous_cost"]))
    full_folder, full = payloads[-1]
    export = prior.read(full_folder / "full_catalogue.json")
    audit.require(prior.sha(full_folder / "full_catalogue.json") == full["result"]["complete_catalogue_export"]["sha256"],
                  "Catalogue hash mismatch")
    audit.require(export["complete"] and export["assignment_scope"] == "task"
                  and export["counts"] == full["result"]["catalogue"], "Catalogue identity")
    catalogue = dict(segments=export["segments"],
        missions=[dict(path=p, flow=full["solution"]["flow"][p["id"]]) for p in export["service"]],
        relocations=[dict(path=p, flow=full["solution"]["empty_flow"][p["id"]]) for p in export["empty"]])
    instance = Path(full["input_reference"]["instance_path"])
    raw = load_standard(instance, case="K2")
    planning = to_planning_data(raw)
    for segment in export["segments"]:
        _segment(planning, segment)
    for path in export["service"]:
        _physical(planning, path)
    for path in export["empty"]:
        _physical(planning, path, empty=True)
    constants = audit.input_constants(full["input_reference"]["model"], full["preprocessed_demand"], catalogue,
                                      instance, assignment_scope="task", max_orders=2)
    records = []
    for run, payload in payloads:
        result = payload["result"]
        audit.require(all(payload["preprocessed_demand"][key] == full["preprocessed_demand"][key]
                          for key in ("single_rates", "pair_rates", "head_rates")), "Matched task rates differ")
        audit.require(result["assignment_scope"] == "task", "Result assignment scope")
        queue = result.get("queue_polymatroid_cut_records", [])
        profiles, replays = {}, []
        for row in queue:
            profile = row["certificate"]["profile"]
            key = json.dumps(profile, sort_keys=True)
            if key not in profiles:
                profiles[key] = audit.validate_profile(profile, constants, payload["preprocessed_demand"], catalogue)
            replays.append(audit.replay_queue_row(row, constants))
        queue_checks = audit.cross_check_rows([(str(i), row) for i, row in enumerate(queue)], points)
        audit.require(queue_checks["passed"], str(queue_checks["errors"]))
        face_records, face_rows = [], []
        for i, record in enumerate(result.get("design_face_cut_records", [])):
            row, evidence = face_row(record, constants, raw)
            face_records.append(evidence)
            if row is not None:
                face_rows.append((str(i), row))
        face_checks = audit.cross_check_rows(face_rows, points)
        audit.require(face_checks["passed"], str(face_checks["errors"]))
        records.append(dict(instance=folder.name, method=run.name, passed=True,
            bases=constants["L"], tasks=constants["Q"], result_sha256=prior.sha(run / "result.json"),
            solution_sha256=prior.sha(run / "solution.json"), catalogue_sha256=prior.sha(full_folder / "full_catalogue.json"),
            physical_paths_rebuilt=len(export["service"]) + len(export["empty"]),
            profile_replays=list(profiles.values()), queue_kinds=dict(Counter(row["certificate"]["kind"] for row in queue)),
            queue_replays=replays, queue_crosspoint=queue_checks, face_replays=face_records, face_crosspoint=face_checks))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = sorted(p for p in (args.directory / "runs").iterdir() if p.is_dir())
    audit.require(cases, "No comparison runs")
    runs = []
    for case in cases:
        runs.extend(replay_case(case))
    report = dict(passed=True, solver_invoked=False, runs=runs,
        audit_source_sha256={str(p.relative_to(ROOT)): prior.sha(p) for p in
            (Path(__file__), Path(audit.__file__), Path(prior.__file__))},
        limitations="Queue profiles and rows replayed independently; face native dual certificates scope-checked, not re-solved. Saved-point checks alone are not a universal validity proof.")
    write_json(args.output, report)
    print(json.dumps(dict(passed=True, runs=len(runs), queue_rows=sum(sum(r["queue_kinds"].values()) for r in runs),
                         face_rows=sum(r["face_crosspoint"]["row_count"] for r in runs), report=str(args.output))))


if __name__ == "__main__":
    with localcontext() as context:
        context.prec = 60
        main()
