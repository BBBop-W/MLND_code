#!/usr/bin/env python3
"""Read-only, solver-free audit of the frozen structural-strengthening runs.

This intentionally does not import the experiment/reporting implementation.
Raw restricted-master values are not bounds. Root primal values are not
integer incumbents. Nested stage timers are never summed with their parents.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / "outputs/structural_strengthening_20261022"
FLAGS = ("assignment_incident_clique", "shared_rho_joint", "queue_perspective")
ROOT_VARIANTS = {"baseline": (False, False, False), "joint": (True, True, False),
                 "perspective": (False, False, True), "all": (True, True, True)}
DISJOINT_TIMERS = (
    "master_build", "greedy_initial_columns", "master_optimize", "master_update",
    "subproblem_wall", "root_master", "capacity_cut_build", "capacity_cut_separation",
    "resource_lp_build", "resource_lp_separation", "incumbent_cleanup", "continuous_root",
    "assignment_support_preprocessing", "branch_joint_preprocessing", "restricted_mip",
    "topology_separation", "assignment_support_separation", "branch_joint_separation",
    "bookkeeping_and_interrupted_work",
)
NESTED_TIMERS = ("root_projection_cg", "neighborhood_cg", "subproblem")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a, b, *, atol=1e-7, rtol=1e-9):
    return number(a) and number(b) and abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def gap(upper, lower):
    return max(0.0, upper - lower) / max(1.0, abs(upper))


def argument(command, flag):
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


class Audit:
    def __init__(self):
        self.errors, self.warnings, self.pending = [], [], []
        self.checks = 0
        self.hash_checks = 0

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.errors.append(label)
        return bool(condition)

    def hash(self, path, expected, label):
        self.hash_checks += 1
        return self.check(Path(path).is_file() and digest(path) == expected, label)

    def frozen(self, batch, manifest):
        for name, expected in manifest["frozen_sha256"].items():
            self.hash(batch / name, expected, f"{batch.name}: frozen hash {name}")

    def instance(self, directory, expected):
        self.hash(directory / "manifest.json", expected, f"input manifest {directory}")
        manifest = read(directory / "manifest.json")
        for name, value in manifest["files"].items():
            self.hash(directory / name, value, f"input file {directory / name}")
        return manifest

    def nonnegative(self, value, label):
        if isinstance(value, dict):
            for key, item in value.items():
                self.nonnegative(item, label + "." + key)
        elif isinstance(value, (int, float)):
            self.check(number(value) and value >= -1e-9, f"nonnegative finite timer {label}")


def root_certificate(audit, result, label):
    lo, up = result.get("certified_lower_bound"), result.get("continuous_objective")
    valid_lb = result.get("bound_valid") is True
    valid_up = result.get("primal_valid") is True
    if valid_lb:
        audit.check(number(lo), label + ": finite certified root lower bound")
        cert = result.get("lower_bound_certificate") or {}
        audit.check(cert.get("pricing", {}).get("complete") is True,
                    label + ": complete-domain pricing certificate")
        rhs = cert.get("rmp_dual_objective")
        correction = cert.get("pricing_correction")
        audit.check(number(rhs) and number(correction) and close(lo, rhs - correction),
                    label + ": lower bound equals dual minus pricing correction")
    if valid_up:
        audit.check(number(up) and result.get("primal_residuals", {}).get("passed") is True,
                    label + ": validated continuous primal")
    own_gap = None
    if valid_lb and valid_up and number(lo) and number(up):
        audit.check(lo <= up + 1e-7 * (1 + abs(up)), label + ": root weak duality")
        own_gap = gap(up, lo)
        if result.get("pricing_complete"):
            tol = result.get("cg_duality_tolerance", 1e-6)
            audit.check(up - lo <= 2 * tol * (1 + abs(up)), label + ": converged root certificate gap")
    trace = result.get("cg_trace") or []
    raw = [r["certified_full_lower_bound"] for r in trace if number(r.get("certified_full_lower_bound"))]
    if raw and valid_lb:
        audit.check(lo >= max(raw) - 1e-7 * (1 + abs(lo)), label + ": retained best complete-domain bound")
    return {"lower_bound": lo if valid_lb else None, "primal_objective": up if valid_up else None,
            "gap": own_gap, "pricing_complete": result.get("pricing_complete"),
            "raw_certificate_count": len(raw),
            "raw_bound_decreases_allowed": sum(y < x - 1e-7 for x, y in zip(raw, raw[1:])),
            "is_integer_upper_bound": False}


def audit_roots(audit, batch):
    manifest = read(batch / "manifest.json")
    audit.frozen(batch, manifest)
    source = audit.instance(batch / "instance", manifest["input_manifest_sha256"])
    expected_jobs = {(k, variant) for k in (1, 2) for variant in ROOT_VARIANTS}
    audit.check({(j["K"], j["variant"]) for j in manifest["jobs"]} == expected_jobs
                and len(manifest["jobs"]) == 8, "root: exact K1/K2 by four variants")
    audit.check(manifest.get("threads") == 1 and manifest.get("seed") == 43,
                "root: single thread and shared seed")
    audit.check(manifest.get("root_certificates_supplied_to_production") is False,
                "root: no free initialization supplied to production")
    rows = []
    for job in manifest["jobs"]:
        label = "root:" + job["id"]
        output = Path(job["output"])
        if not output.is_file():
            audit.pending.append(label)
            continue
        result, saved = read(output), read(output.parent / "audit.json")
        execution, request = read(output.parent / "execution.json"), read(output.parent / "request.json")
        audit.hash(output, saved.get("result_sha256"), label + ": result validation hash")
        audit.hash(batch / "manifest.json", request.get("manifest_sha256"), label + ": request manifest hash")
        binary = batch / "frozen/mlnd_continuous_probe"
        audit.hash(binary, saved.get("frozen_binary_sha256"), label + ": probe identity")
        audit.check(execution.get("command") == job["command"] == request.get("job", {}).get("command"),
                    label + ": actual command matches frozen request")
        audit.check(execution.get("returncode") == 0, label + ": process exited successfully")
        audit.check(result.get("K") == job["K"] and result.get("input_manifest") == source
                    and result.get("input_model_sha256") == digest(batch / "instance/model.json"),
                    label + ": input and K identity")
        features = dict(zip(FLAGS, ROOT_VARIANTS[job["variant"]]))
        audit.check(job["features"] == features, label + ": feature plan")
        audit.check(all(result.get("structural_strengthening", {}).get(k) is v for k, v in features.items()),
                    label + ": actual features")
        scope = "structurally_strengthened_continuous_relaxation" if any(features.values()) else "full_path_continuous_relaxation"
        audit.check(result.get("method") == "full_continuous_cg" and result.get("scope") == scope
                    and result.get("objective_includes_fixed_costs") is True
                    and result.get("is_integer_solution") is False, label + ": continuous bound scope")
        row = dict(id=job["id"], K=job["K"], variant=job["variant"], **root_certificate(audit, result, label))
        timing = result.get("timings", {})
        audit.nonnegative(timing, label)
        accounted = sum(v for v in timing.values() if number(v))
        wall = result.get("algorithm_wall")
        audit.check(close(accounted, wall, atol=.01, rtol=1e-5), label + ": nonoverlapping root phase sum")
        audit.check(result.get("end_to_end_seconds", -1) + .01 >= wall + result.get("preprocessing_seconds", 0),
                    label + ": end-to-end includes root and preprocessing")
        row.update(phase_sum=accounted, algorithm_wall=wall, model_size=result.get("model_size"))
        rows.append(row)
    comparisons = []
    for k in (1, 2):
        base = next((r for r in rows if r["K"] == k and r["variant"] == "baseline"), None)
        if not base:
            continue
        for row in (r for r in rows if r["K"] == k and r["variant"] != "baseline"):
            # Interval separation is evidence of measured improvement. Merely
            # comparing incomplete lower bounds is not a monotonicity proof.
            if number(row["primal_objective"]) and number(base["lower_bound"]):
                audit.check(row["primal_objective"] >= base["lower_bound"] - 1e-5,
                            f"root K{k} {row['variant']}: strengthened optimum not below baseline certificate")
            comparisons.append(dict(K=k, variant=row["variant"],
                certified_improvement=(row["lower_bound"] - base["primal_objective"])
                    if number(row["lower_bound"]) and number(base["primal_objective"]) else None,
                lower_bound_difference=(row["lower_bound"] - base["lower_bound"])
                    if number(row["lower_bound"]) and number(base["lower_bound"]) else None))
    return dict(rows=rows, comparisons=comparisons, input_manifest_sha256=manifest["input_manifest_sha256"],
                source_hashes=manifest.get("source_hashes", {}))


def production_curve(audit, directory, label, result):
    path = directory / "progress.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    curve = [r for r in records if r.get("event") in ("global_bounds", "run_finished")
             and r.get("bound_scope") == "global" and number(r.get("lower_bound")) and number(r.get("upper_bound"))]
    audit.check(bool(curve), label + ": global bound curve available")
    for r in curve:
        lo, up = r["lower_bound"], r["upper_bound"]
        audit.check(lo <= up + 1e-7 * (1 + abs(up)), label + ": curve weak duality")
        audit.check(close(r.get("gap"), gap(up, lo)), label + ": curve own-incumbent gap")
        audit.check(r.get("validated_incumbent") is True, label + ": curve incumbent validated")
    for a, b in zip(curve, curve[1:]):
        audit.check(b["elapsed_seconds"] >= a["elapsed_seconds"] - 1e-6, label + ": curve clock monotone")
        audit.check(b["lower_bound"] >= a["lower_bound"] - 1e-7 * (1 + abs(a["lower_bound"])), label + ": global LB monotone")
        audit.check(b["upper_bound"] <= a["upper_bound"] + 1e-7 * (1 + abs(a["upper_bound"])), label + ": global UB monotone")
    if curve:
        audit.check(close(curve[-1]["lower_bound"], result.get("bound")), label + ": final curve LB")
        audit.check(close(curve[-1]["upper_bound"], result.get("objective")), label + ": final curve UB")
    return len(curve)


def design_for_cross_check(audit, solution, size, result, label):
    """Mirror only the documented DesignLayout ordering, not model logic."""
    l, h, r = size["bases"], size["lockers"], size["orders"]
    n = l + h
    levels = solution.get("capacity_level", [])
    q = len(levels[0]) if levels and isinstance(levels[0], list) else 0
    shapes = (("facility", (n,)), ("capacity_level", (h, q)),
              ("assignment", (r, l, l)), ("airline", (n, n)))

    def flatten(value, shape):
        if not shape:
            return [value] if number(value) and value in (0, 1) else None
        if not isinstance(value, list) or len(value) != shape[0]:
            return None
        pieces = [flatten(item, shape[1:]) for item in value]
        if any(piece is None for piece in pieces):
            return None
        return [item for piece in pieces for item in piece]

    design = []
    for name, shape in shapes:
        part = flatten(solution.get(name), shape)
        if not audit.check(part is not None and q > 0, label + ": exact binary layout " + name):
            return None
        design.extend(part)
    expected_size = n + h * q + r * l * l + n * n
    audit.check(len(design) == expected_size == result.get("master_structure", {}).get("design_binaries"),
                label + ": design dimension matches master layout")
    return design, n + h * q, n + h * q + r * l * l


def cross_check_benders_cuts(audit, candidates):
    """Necessary finite-point consistency check; never a global validity proof.

    The compared operating cost belongs to a validated feasible solution,
    hence upper-bounds the fixed-design recourse optimum. No solver is run.
    Conditional cuts are already globally gated in the stored coefficients;
    evaluate them unchanged even on different assignment fingerprints.
    """
    count = conditional_count = changed_assignment_count = 0
    maximum_signed = None
    maximum_excess = 0.0
    worst = None
    pairs = []
    for source in candidates:
        cuts = source["result"].get("benders_cuts")
        if not audit.check(isinstance(cuts, list), source["id"] + ": stored Benders cut array"):
            continue
        valid_cuts = []
        for index, cut in enumerate(cuts):
            coefficients = cut.get("coefficients")
            valid = (number(cut.get("intercept")) and isinstance(coefficients, list)
                     and len(coefficients) == len(source["design"]) and all(number(x) for x in coefficients))
            audit.check(valid, f"{source['id']}: Benders cut {index} complete affine representation")
            audit.check(cut.get("pricing_complete") is True and cut.get("support_complete") is True,
                        f"{source['id']}: Benders cut {index} full-domain certificate flags")
            if valid:
                valid_cuts.append((index, cut))
        for target in candidates:
            if source["group"] != target["group"]:
                continue
            if not audit.check(len(source["design"]) == len(target["design"]),
                               source["id"] + "/" + target["id"] + ": shared cut layout"):
                continue
            pair_maximum = None
            pair_count = 0
            for index, cut in valid_cuts:
                terms = [cut["intercept"], *(c * x for c, x in zip(cut["coefficients"], target["design"]))]
                value = math.fsum(terms)
                operating = target["operating_cost"]
                # The small scale term permits existing primal validation
                # noise; the explicit sum term bounds cancellation in gates.
                tolerance = 1e-5 + 1e-7 * (1 + abs(operating)) + 64 * math.ulp(1.0) * math.fsum(abs(x) for x in terms)
                violation = value - operating
                audit.check(number(value) and violation <= tolerance,
                            f"{source['id']}: Benders cut {index} at {target['id']} <= validated operating cost")
                count += 1
                pair_count += 1
                pair_maximum = violation if pair_maximum is None else max(pair_maximum, violation)
                maximum_excess = max(maximum_excess, violation - tolerance)
                condition = cut.get("condition")
                if condition is not None:
                    conditional_count += 1
                    fingerprint = "".join(str(int(x)) for x in target["design"][target["assignment_start"]:target["assignment_end"]])
                    changed_assignment_count += condition.get("assignment_fingerprint") != fingerprint
                if maximum_signed is None or violation > maximum_signed:
                    maximum_signed = violation
                    worst = dict(source=source["id"], target=target["id"], cut_index=index,
                                 kind=cut.get("kind"), cut_value=value, operating_cost=operating,
                                 signed_violation=violation, tolerance=tolerance,
                                 conditional=condition is not None)
            pairs.append(dict(source=source["id"], target=target["id"], checks=pair_count,
                              maximum_signed_violation=pair_maximum))
    return dict(checks=count, conditional_checks=conditional_count,
                changed_assignment_conditional_checks=changed_assignment_count,
                validated_incumbents=len(candidates),
                maximum_signed_violation=maximum_signed,
                maximum_positive_violation=max(0.0, maximum_signed) if maximum_signed is not None else None,
                maximum_excess_over_tolerance=maximum_excess, worst_check=worst, pairs=pairs,
                design_order=["facility", "capacity_level[h][level]", "assignment[r][l][b]", "airline[i][j]"],
                comparator="validated feasible solution continuous_cost; not the unknown recourse optimum",
                scope="necessary checks at this batch's validated final integer solutions; not a proof over all designs",
                conditional_handling="stored globally gated affine coefficients evaluated unchanged on every matching-K/input solution",
                tolerance="1e-5 + 1e-7*(1+abs(continuous_cost)) + 64*machine_epsilon*sum(abs(affine_terms))")


def audit_production(audit, batch, roots):
    manifest, state = read(batch / "experiment_manifest.json"), read(batch / "state.json")
    audit.frozen(batch, manifest)
    protocol = manifest["protocol"]
    audit.check(protocol.get("threads_per_job") == 1 and protocol.get("parallel_jobs", 99) <= 3,
                "production: single thread and concurrency cap")
    audit.check(protocol.get("external_warm_start") is None and protocol.get("fixed_design") is None,
                "production: independent starts")
    for version, binary in manifest["binaries"].items():
        audit.hash(batch / binary["path"], binary["sha256"], f"production: {version} binary")
    expected = {(k, form, version) for k in (1, 2) for form in ("lifted", "conditional") for version in ("old", "new")}
    for origin in manifest["input_origins"]:
        jobs = [j for j in manifest["jobs"] if j["instance"] == origin["instance"]]
        observed = {(j["K"], j["method"].removeprefix("benders_"), j["version"]) for j in jobs}
        audit.check(len(jobs) == 8 and observed == expected, "production: exact four implementations for each K")
        audit.instance(batch / origin["instance"], origin["manifest_sha256"])
        audit.check(origin["manifest_sha256"] == roots["input_manifest_sha256"], "root/production: identical input")
    new_sources = manifest["binaries"]["new"].get("source_hashes", {})
    matched = 0
    for name, expected_hash in roots["source_hashes"].items():
        if name in new_sources:
            audit.check(new_sources[name] == expected_hash, f"root/production source identity {name}")
            matched += 1
    audit.check(matched > 0, "root/production: shared C++ source hash evidence")
    rows, candidates = [], []
    for job in manifest["jobs"]:
        label = job["id"]
        progress = state["jobs"].get(label, {})
        relative = progress.get("result_directory") or progress.get("directory")
        directory = batch / relative if relative else None
        if not directory or not (directory / "result.json").is_file() or not (directory / "native_validation.json").is_file():
            audit.pending.append(label)
            rows.append(dict(id=label, status=progress.get("status", "pending"), audited=False))
            continue
        result, validation = read(directory / "result.json"), read(directory / "native_validation.json")
        execution = read(directory / "execution.json")
        guard = read(directory / "guard_status.json")
        request = read(directory / "job.json")
        command = guard.get("command", [])
        audit.hash(batch / "experiment_manifest.json", request.get("experiment_manifest_sha256"),
                   label + ": request manifest hash")
        audit.check(command == request.get("command") and request.get("job") == job,
                    label + ": guarded command equals requested job")
        audit.check(execution.get("returncode") == 0, label + ": process exited successfully")
        physical_ok = audit.check(validation.get("passed") is True, label + ": physical validation passed")
        for name, field in (("result.json", "result_sha256"), ("solution.json", "solution_sha256"), ("path_flows.json", "path_flows_sha256")):
            physical_ok = audit.hash(directory / name, validation.get(field), label + ": physical validation covers " + name) and physical_ok
        physical_ok = audit.hash(batch / job["instance"] / "manifest.json", validation.get("manifest_sha256"), label + ": physical validation input hash") and physical_ok
        binary = manifest["binaries"][job["version"]]
        audit.check(result.get("binary_sha256") == binary["sha256"], label + ": result executable identity")
        form = job["method"].removeprefix("benders_")
        audit.check(bool(command) and Path(command[0]) == batch / binary["path"], label + ": executed binary path")
        expected_args = {"--input": str(batch / job["instance"]), "--method": "benders_cg",
                         "--k": str(job["K"]), "--threads": "1", "--seed": str(protocol["seed"]),
                         "--demand-formulation": form, "--out": str(directory)}
        audit.check(all(argument(command, flag) == value for flag, value in expected_args.items()),
                    label + ": actual execution parameters")
        audit.check(("--structural-strengthening" in command) is (job["version"] == "new"),
                    label + ": structural execution flag")
        audit.check(argument(command, "--warm-start") is None and argument(command, "--fixed-design") is None,
                    label + ": no external start or fixed-design execution")
        audit.hash(batch / job["instance"] / "manifest.json", result.get("input_manifest_sha256"),
                   label + ": result input manifest hash")
        audit.check(result.get("max_orders") == job["K"] and result.get("demand_formulation") == form
                    and result.get("method") == "benders_cg" and result.get("solver") == "CPLEX", label + ": K/method identity")
        audit.check(result.get("threads") == 1 and result.get("seed") == protocol["seed"]
                    and result.get("budget") == protocol["seconds_per_job"], label + ": resource protocol")
        for flag in FLAGS:
            actual = result.get("structural_strengthening", {}).get(flag, result.get("accelerations", {}).get(flag))
            if actual is None and job["version"] == "old" and result.get("binary_sha256") == binary["sha256"]:
                actual = False
            audit.check(actual is (job["version"] == "new"), label + ": actual feature " + flag)
        lo, up = result.get("bound"), result.get("objective")
        audit.check(number(lo) and number(up) and lo <= up + 1e-7 * (1 + abs(up)), label + ": valid own bounds")
        own_gap = gap(up, lo) if number(up) and number(lo) else None
        audit.check(close(result.get("gap"), own_gap), label + ": own-incumbent final gap")
        curve_points = production_curve(audit, directory, label, result)
        timing = result.get("timings", {})
        audit.nonnegative(timing, label)
        accounted = sum(timing.get(k, 0) for k in DISJOINT_TIMERS)
        wall = result.get("algorithm_wall")
        audit.check(close(accounted, wall, atol=.03, rtol=1e-4), label + ": disjoint production stage sum")
        sub = timing.get("subproblem", {})
        audit.check(sum(v for v in sub.values() if number(v)) <= timing.get("subproblem_wall", 0) + .03,
                    label + ": subproblem components fit parent wall")
        for nested in ("root_projection_cg", "neighborhood_cg"):
            audit.check(timing.get(nested, 0) <= timing.get("subproblem_wall", 0) + .03, label + ": nested timer " + nested)
        root = result.get("continuous_root") or {}
        if root.get("bound_valid"):
            root_certificate(audit, root, label + ": in-budget root")
        structure = result.get("master_structure") or {}
        audit.check(structure.get("operational_auxiliary_variables") == 0 and structure.get("cones") == 0,
                    label + ": compact master retained")
        operating = result.get("continuous_cost")
        opening = result.get("opening_cost")
        cost_ok = audit.check(number(operating) and operating >= 0 and number(opening)
                              and close(up, opening + operating, atol=1e-5), label + ": validated operating-cost decomposition")
        if physical_ok and cost_ok:
            size = next(o["size"] for o in manifest["input_origins"] if o["instance"] == job["instance"])
            layout = design_for_cross_check(audit, read(directory / "solution.json"), size, result, label)
            if layout is not None:
                design, start, end = layout
                candidates.append(dict(id=label, group=(validation["manifest_sha256"], job["K"]),
                                       design=design, assignment_start=start, assignment_end=end,
                                       operating_cost=operating, result=result))
        rows.append(dict(id=label, status=progress.get("status"), audited=True, lower_bound=lo, upper_bound=up,
                         gap=own_gap, phase_sum=accounted, algorithm_wall=wall, curve_points=curve_points,
                         nested_timers_excluded=list(NESTED_TIMERS), result=str(directory / "result.json")))
    return dict(rows=rows, shared_cpp_source_hashes=matched,
                benders_cut_cross_checks=cross_check_benders_cuts(audit, candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT)
    parser.add_argument("--root-only", action="store_true")
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.base = args.base.resolve()
    audit = Audit()
    roots = audit_roots(audit, args.base / "root_ablations")
    production = None if args.root_only else audit_production(audit, args.base / "comparison", roots)
    result = dict(schema="mlnd-structural-independent-run-audit/1", created_utc=datetime.now(timezone.utc).isoformat(),
                  passed=not audit.errors and (not audit.pending or args.allow_pending),
                  complete=not audit.pending, root_only=args.root_only,
                  checks=audit.checks, hash_checks=audit.hash_checks,
                  errors=audit.errors, warnings=audit.warnings, pending=audit.pending,
                  roots={k:v for k,v in roots.items() if k != "source_hashes"}, production=production,
                  audit_script_sha256=digest(__file__), solver_invoked=False)
    output = args.out or args.base / "audit" / ("root_run_audit.json" if args.root_only else "independent_run_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("passed", "complete", "checks", "hash_checks", "errors", "pending")}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
