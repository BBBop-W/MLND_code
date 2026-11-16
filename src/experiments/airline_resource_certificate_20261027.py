"""Freeze pre-fix evidence and compare a repaired baseline with proposed cuts.

No solve is implicit. ``snapshot`` preserves the verified previous executable,
sources, input cases, commands and failed designs. ``prepare`` freezes a new
binary for six controlled runs. Only explicit ``launch`` or ``run`` dispatches.
The task-cost runner supplies scheduling, guards, validation and metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import statistics

from src.experiments import task_cost_benchmark_20261027 as task
from src.experiments.storage import write_json

base = task.base
ROOT = task.ROOT
MODULE = "src.experiments.airline_resource_certificate_20261027"
DEFAULT_ROOT = ROOT / "outputs/airline_resource_certificate_20261027"
L1_BATCH = ROOT / "outputs/l1_airline_expanded_20261027"
L2_BATCH = ROOT / "outputs/l2_airline_comparison_20261027"
PRIMAL_BATCH = ROOT / "outputs/l2_primal_improvement_20261027"
PRIMAL_FLAG = "--primal-polish"
PRIMAL_SUITE = "primal-comparison"
OPTIMIZATION_BATCH = ROOT / "outputs/l2_algorithm_optimization_20261027"
OPTIMIZATION_SUITE = "optimization-comparison"
REFINEMENT_BATCH = ROOT / "outputs/l2_algorithm_refinement_20261027"
REFINEMENT_SUITE = "refinement-comparison"
SCHEDULING_FLAGS = ("--search-scheduling", "--dual-lp-first")
STRUCTURAL_FLAGS = ("--fractional-path-support", "--airline-resource-balance")
VALIDATED_REFERENCE = DEFAULT_ROOT / "comparison_bulk_update_20261027"
PREVIOUS = task.EXPANDED_BATCH
CASES = (("tsl_L02_M01", "double", 2), ("tsl_L01_M02", "double", 2),
         ("tsl_L01_M03", "double", 2))
BASELINE_FLAGS = task.VARIANTS["task_airline"]
PROPOSED_FLAGS = ("--airline-connectivity-cuts", "--airline-resource-cuts")
L1_CASES = tuple((identity, "double", 2) for identity in (
    "tsl_L01_M04", "tsl_L01_M05", "tsl_L01_M06",
    "nanshan_L01_M02", "nanshan_L01_M03", "nanshan_L01_M04"))
L2_CASES = tuple((identity, "double", 2) for identity in (
    "tsl_L02_M01", "tsl_L02_M07", "tsl_L02_M08"))
EXPANSION_SUITES = {"l1-expanded": (L1_BATCH, L1_CASES),
                    "l2-comparison": (L2_BATCH, L2_CASES)}
FRESH_SUITES = {**EXPANSION_SUITES, PRIMAL_SUITE: (PRIMAL_BATCH, L2_CASES),
                OPTIMIZATION_SUITE: (OPTIMIZATION_BATCH, L2_CASES),
                REFINEMENT_SUITE: (REFINEMENT_BATCH, L2_CASES)}


def design_from_key(key, model, assignment_scope):
    """Decode the saved binary DesignLayout, never an incumbent substitute."""
    if not key or set(key) - {"0", "1"}:
        raise ValueError("Failed candidate must have a binary design_key")
    values, ids = model["values"], model["ids"]
    l, h, orders = (int(values[k]) for k in ("n_l", "n_h", "n_od"))
    levels, nodes = len(ids["levels"]), l + h
    if assignment_scope != "origin":
        raise ValueError("This replay decoder supports the current origin-assignment model only")
    expected = nodes + h * levels + orders * l + nodes * nodes
    if len(key) != expected:
        raise ValueError(f"Design length {len(key)} does not match saved layout {expected}")
    bits = [int(v) for v in key]
    at = 0

    def take(count):
        nonlocal at
        result = bits[at:at + count]
        at += count
        return result

    return dict(assignment_scope=assignment_scope, facility=take(nodes),
                capacity_level=[take(levels) for _ in range(h)],
                assignment=[take(l) for _ in range(orders)],
                airline=[take(nodes) for _ in range(nodes)])


def snapshot(root=DEFAULT_ROOT, *, previous=PREVIOUS):
    root, previous = Path(root).resolve(), Path(previous).resolve()
    target = root / "pre_fix"
    if target.exists():
        raise FileExistsError(f"Refusing to replace pre-fix evidence: {target}")
    old = base.verify(previous)
    identities = {case[0] for case in CASES}
    selected = [j for j in old["jobs"] if Path(j["instance"]).name in identities]
    if len(selected) != len(CASES) * len(task.VARIANTS):
        raise ValueError("Previous controlled suite does not contain all expected jobs")
    # Freeze the source actually paired with the old binary, even if ongoing
    # repair work has already changed the live checkout.
    shutil.copytree(previous / "frozen", target / "frozen")
    for identity in sorted(identities):
        shutil.copytree(previous / "instances" / identity, target / "instances" / identity)
    shutil.copy2(previous / "experiment_manifest.json", target / "reference_manifest.json")
    state = base.read(previous / "state.json")
    commands, replays = [], []
    for job in selected:
        saved = state["jobs"][job["id"]]
        source = previous / saved["result_directory"]
        job_file, result_file = source / "job.json", source / "result.json"
        original = base.read(job_file)
        destination = target / "commands" / job["id"]
        destination.mkdir(parents=True)
        shutil.copy2(job_file, destination / "reference_job.json")
        commands.append(dict(job_id=job["id"], instance=job["instance"],
            original_command=original["command"], reference_job=str(job_file),
            reference_job_sha256=base.sha256(job_file), result=str(result_file),
            result_sha256=base.sha256(result_file)))
        if Path(job["instance"]).name != "tsl_L02_M01":
            continue
        result, input_ref = base.read(result_file), base.read(source / "input_reference.json")
        failed = [i for i in result.get("iterations", [])
                  if i.get("termination_reason") == "rmp_duality_gap_failed"]
        if not failed:
            raise ValueError(f"No saved certificate failure for {job['id']}")
        candidate = failed[-1]
        design = design_from_key(candidate["design_key"], input_ref["model"],
                                 input_ref["assignment_scope"])
        replay_dir = target / "replay" / job["variant"]
        replay_dir.mkdir(parents=True)
        write_json(replay_dir / "fixed_design.json", design)
        write_json(replay_dir / "failed_candidate.json", candidate)
        # The old result saves pool counts, not individual indices. Preserve
        # those counts as evidence, not an exact RMP warm-start specification.
        pools = dict(initial=result.get("initial_columns"),
                     final_generated=result.get("generated_columns"),
                     final_reusable=result.get("reusable_columns"),
                     exact_failed_rmp_indices_available=False,
                     failed_service_count=candidate.get("service_columns"),
                     failed_empty_count=candidate.get("empty_columns"))
        write_json(replay_dir / "column_pool_evidence.json", pools)
        command = list(original["command"])
        command[0] = str(target / "frozen/bin/mlnd_cpp")
        overrides = {"--input": str(target / job["instance"]), "--seconds": "600",
                     "--out": str(root / "diagnostics" / job["variant"] / "pre_fix")}
        for key, value in overrides.items():
            command[command.index(key) + 1] = value
        command += ["--fixed-design", str(replay_dir / "fixed_design.json")]
        replays.append(dict(variant=job["variant"], candidate_id=candidate["candidate_id"],
            design_key=candidate["design_key"], fixed_design=str(replay_dir / "fixed_design.json"),
            evidence=str(replay_dir / "failed_candidate.json"),
            column_pool_evidence=str(replay_dir / "column_pool_evidence.json"),
            source_result=str(result_file), source_result_sha256=base.sha256(result_file),
            command=command, scope="fixed_design_cg_recomputation_not_exact_rmp_replay",
            solver_started=False))
    write_json(target / "reference_commands.json", commands)
    write_json(target / "replay_cases.json", replays)
    manifest = dict(schema="pre-fix-certificate-evidence/1", created_utc=base.existing.utc(),
        source_batch=str(previous), source_manifest_sha256=base.sha256(previous / "experiment_manifest.json"),
        binary_sha256=base.sha256(target / "frozen/bin/mlnd_cpp"),
        cases=[c[0] for c in CASES], old_runs_reused=True, solver_started=False,
        frozen_sha256={str(p.relative_to(target)): base.sha256(p)
                       for p in sorted(target.rglob("*")) if p.is_file()})
    write_json(target / "snapshot_manifest.json", manifest)
    return dict(snapshot=str(target), binary_sha256=manifest["binary_sha256"],
                historical_jobs=len(commands), replay_cases=len(replays), solver_started=False)


def verify_snapshot(root):
    target = Path(root).resolve() / "pre_fix"
    manifest = base.read(target / "snapshot_manifest.json")
    for name, digest in manifest["frozen_sha256"].items():
        path = target / name
        if not path.is_file() or base.sha256(path) != digest:
            raise ValueError(f"Pre-fix evidence changed: {name}")
    return manifest


def validate_flags(flags):
    if not isinstance(flags, list) or not flags or any(not isinstance(f, str) for f in flags):
        raise ValueError("Explicit nonempty proposed boolean flags are required")
    if len(set(flags)) != len(flags):
        raise ValueError("Duplicate proposed flag")
    forbidden = set(BASELINE_FLAGS) | {"--path-support-cuts", "--plain", "--start", "--fixed-design"}
    if any(not f.startswith("--") or "=" in f or f in forbidden or
           any(ch.isspace() for ch in f) for f in flags):
        raise ValueError("Proposed flags must be additional boolean solver switches")
    # Only the two strengthening families belong to this controlled experiment;
    # this prevents silent changes to demand, assignment, budgets or tolerances.
    if any(not f.startswith(("--airline-resource-", "--airline-connectivity-")) for f in flags):
        raise ValueError("Only airline resource/connectivity switches may differ")
    return tuple(flags)


def prepare(root=DEFAULT_ROOT, *, binary, proposed_flags, cases=CASES,
            comparison_name="comparison", include_baseline=True):
    root = Path(root).resolve()
    old = verify_snapshot(root)
    flags = validate_flags(proposed_flags)
    if base.sha256(Path(binary)) == old["binary_sha256"]:
        raise ValueError("The repaired-binary comparison requires a new binary")
    batch = root / comparison_name
    variants = {"fixed_baseline": BASELINE_FLAGS} if include_baseline else {}
    variants["proposed"] = (*BASELINE_FLAGS, *flags)
    task.prepare(batch, binary=binary, suite="certificate", cases=cases, variants=variants)
    manifest = base.read(batch / "experiment_manifest.json")
    manifest["schema"] = "airline-resource-certificate/1"
    manifest["protocol"].update(pre_fix_binary_sha256=old["binary_sha256"],
        pre_fix_comparison="verified previous expanded first attempts; not rerun",
        controlled_comparison="same repaired binary; only declared proposed flags differ")
    (batch / "run.py").write_text("from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\nsys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\nmain([*sys.argv[1:], '--root', str(batch.parent), '--batch', str(batch)])\n")
    manifest["frozen_sha256"]["run.py"] = base.sha256(batch / "run.py")
    write_json(batch / "experiment_manifest.json", manifest)
    return dict(batch=str(batch), jobs=len(manifest["jobs"]),
                variants={name: list(value) for name, value in variants.items()},
                binary_sha256=manifest["binary_sha256"], solver_started=False)


def prepare_bulk(root=DEFAULT_ROOT, *, binary, proposed_flags):
    """Three proposed regressions; retain prior flag-disabled baselines verbatim."""
    root = Path(root).resolve()
    reference = root / "comparison"
    previous = base.verify(reference)
    state = base.read(reference / "state.json")
    candidates = [j for j in previous["jobs"] if j["variant"] == "fixed_baseline"
                  and Path(j["instance"]).name in {c[0] for c in CASES}]
    if len(candidates) != len(CASES):
        raise ValueError("Expected one saved fixed_baseline per selected case")
    evidence = []
    for job in candidates:
        row = record(reference, job, state["jobs"][job["id"]])
        if not row["validation_passed"] or row["termination_reason"] not in {"time_limit", "gap_limit"}:
            raise ValueError("Each reused baseline must be validated and normally terminated")
        folder = reference / state["jobs"][job["id"]]["result_directory"]
        evidence.append(dict(batch=str(reference), job=job,
            result_directory=str(folder), result_sha256=base.sha256(folder / "result.json"),
            solution_sha256=base.sha256(folder / "solution.json"),
            input_manifest_sha256=base.sha256(reference / job["instance"] / "manifest.json"),
            binary_sha256=previous["binary_sha256"], row=row,
            reused_not_rerun=True,
            control_scope="Only the enabled airline-resource LP objective-update implementation and diagnostics change; baseline disables this LP"))
    binary_hash = base.sha256(Path(binary))
    if binary_hash == previous["binary_sha256"]:
        raise ValueError("Bulk-update test requires the rebuilt binary")
    result = prepare(root, binary=binary, proposed_flags=proposed_flags,
        comparison_name="comparison_bulk_update_20261027", include_baseline=False)
    batch = Path(result["batch"])
    target = batch / "provenance/reused_baselines.json"
    write_json(target, evidence)
    manifest = base.read(batch / "experiment_manifest.json")
    manifest["frozen_sha256"][str(target.relative_to(batch))] = base.sha256(target)
    manifest["protocol"].update(reused_baselines=str(target.relative_to(batch)),
        controlled_comparison=evidence[0]["control_scope"],
        same_binary_for_all_compared_results=False, new_runs=len(CASES))
    write_json(batch / "experiment_manifest.json", manifest)
    result["reused_baselines"] = [item["result_directory"] for item in evidence]
    return result


def prepare_l1(batch=L1_BATCH, *, binary, proposed_flags=PROPOSED_FLAGS):
    return prepare_expansion(batch, binary=binary, proposed_flags=proposed_flags,
                             suite="l1-expanded")


def prepare_l2(batch=L2_BATCH, *, binary, proposed_flags=PROPOSED_FLAGS):
    return prepare_expansion(batch, binary=binary, proposed_flags=proposed_flags,
                             suite="l2-comparison")


def prepare_preserving_analysis(batch, **kwargs):
    """Keep pre-run diagnosis files; never replace an existing experiment."""
    batch = Path(batch).resolve()
    if not batch.exists():
        return task.prepare(batch, **kwargs)
    if (batch / "experiment_manifest.json").exists():
        raise FileExistsError(f"Refusing to replace existing experiment: {batch}")
    if any(path.name != "analysis" or not path.is_dir() for path in batch.iterdir()):
        raise FileExistsError(f"Only a pre-existing analysis directory may be preserved: {batch}")
    # Prepare beside the destination. All runner-owned identities are relative
    # to its own __file__; original input/command provenance remains unchanged.
    # A failure keeps the diagnosis directory untouched and retains the staging
    # evidence rather than overwriting an existing partial experiment.
    staging = batch.with_name(batch.name + ".prepare")
    result = task.prepare(staging, **kwargs)
    children = list(staging.iterdir())
    if any((batch / path.name).exists() for path in children):
        raise FileExistsError("Prepared files overlap the retained diagnosis directory")
    for path in children:
        shutil.move(str(path), batch / path.name)
    staging.rmdir()
    return dict(result, batch=str(batch))


def prepare_primal(batch=PRIMAL_BATCH, *, binary):
    """Compare only the opt-in primal heuristic on one freshly built binary."""
    batch, binary = Path(batch).resolve(), Path(binary).resolve()
    previous = base.verify(VALIDATED_REFERENCE)
    if base.sha256(binary) == previous["binary_sha256"]:
        raise ValueError("Primal comparison requires a newly compiled binary")
    history = historical_full_cplex(L2_CASES)
    common = (*BASELINE_FLAGS, *PROPOSED_FLAGS)
    variants = dict(fixed_baseline=common, proposed=(*common, PRIMAL_FLAG))
    source_before = {str(path.relative_to(ROOT / "cpp")): base.sha256(path)
        for path in (ROOT / "cpp").rglob("*") if path.is_file()
        and not any(part.startswith("build") or part == "__pycache__" for part in path.relative_to(ROOT / "cpp").parts)
        and path.name != ".DS_Store" and path.suffix != ".pyc"}
    prepare_preserving_analysis(batch, binary=binary, suite=PRIMAL_SUITE,
                                cases=L2_CASES, variants=variants)
    manifest = base.read(batch / "experiment_manifest.json")
    frozen_cpp = batch / "frozen/cpp"
    frozen_hashes = {str(path.relative_to(frozen_cpp)): base.sha256(path)
                     for path in frozen_cpp.rglob("*") if path.is_file()}
    if frozen_hashes != source_before:
        raise ValueError("C++ source changed during preparation; do not launch this batch")
    changes = []
    old_cpp = VALIDATED_REFERENCE / "frozen/cpp"
    for name, digest in sorted(frozen_hashes.items()):
        old = old_cpp / name
        old_digest = base.sha256(old) if old.is_file() else None
        if digest != old_digest:
            changes.append(dict(path=name, prior_sha256=old_digest, current_sha256=digest))
    evidence = dict(suite=PRIMAL_SUITE, cases=[case[0] for case in L2_CASES],
        prior_validated_batch=str(VALIDATED_REFERENCE), prior_binary_sha256=previous["binary_sha256"],
        binary_sha256=manifest["binary_sha256"], source_origin="current_checkout_at_preparation",
        frozen_cpp_sha256=frozen_hashes, source_changes_from_adffe=changes,
        common_algorithm_flags=list(common), sole_variant_flag=PRIMAL_FLAG,
        model_and_operating_assumptions_unchanged=True,
        external_warm_start=None, external_root_certificate=None,
        old_results_are_controlled_baselines=False, solver_started=False)
    for name, payload in (("primal_preflight.json", evidence),
                          ("historical_full_cplex.json", dict(rows=history,
                              comparison_scope="Historical reference only; binary and time budget differ"))):
        target = batch / "provenance" / name
        write_json(target, payload)
        manifest["frozen_sha256"][str(target.relative_to(batch))] = base.sha256(target)
    manifest["schema"] = "l2-primal-comparison/1"
    manifest["protocol"].update(pre_fix_reference=False, cases_predeclared=True,
        controlled_comparison="fresh off/on runs on the same new binary; only --primal-polish differs",
        comparison_feature=PRIMAL_FLAG, common_lower_bound_policy="all previous task and airline strengthening enabled in both variants",
        external_warm_start=None, external_root_certificate=None, reused_baselines=None,
        historical_full_cplex_reference="provenance/historical_full_cplex.json",
        input_dimensions="2 candidate bases, 20 candidate lockers, 50 OD pairs")
    (batch / "run.py").write_text("from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\nsys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\nmain([*sys.argv[1:], '--suite', {PRIMAL_SUITE!r}, '--batch', str(batch)])\n")
    manifest["frozen_sha256"]["run.py"] = base.sha256(batch / "run.py")
    write_json(batch / "experiment_manifest.json", manifest)
    return dict(batch=str(batch), jobs=len(manifest["jobs"]), suite=PRIMAL_SUITE,
                binary_sha256=manifest["binary_sha256"], solver_started=False)


def optimization_variants():
    common = (*BASELINE_FLAGS, *PROPOSED_FLAGS, PRIMAL_FLAG)
    return dict(baseline=common, schedule=(*common, *SCHEDULING_FLAGS),
                full=(*common, *SCHEDULING_FLAGS, *STRUCTURAL_FLAGS))


def prepare_optimization(batch=OPTIMIZATION_BATCH, *, binary):
    """Fresh three-stage ablation; previous polished results are references only."""
    batch, binary = Path(batch).resolve(), Path(binary).resolve()
    previous = base.verify(PRIMAL_BATCH)
    if base.sha256(binary) == previous["binary_sha256"]:
        raise ValueError("Algorithm optimization requires a newly compiled binary")
    variants = optimization_variants()
    history = historical_full_cplex(L2_CASES)
    source_before = {str(p.relative_to(ROOT / "cpp")): base.sha256(p)
        for p in (ROOT / "cpp").rglob("*") if p.is_file()
        and not any(part.startswith("build") or part == "__pycache__"
                    for part in p.relative_to(ROOT / "cpp").parts)
        and p.name != ".DS_Store" and p.suffix != ".pyc"}
    prepare_preserving_analysis(batch, binary=binary, suite=OPTIMIZATION_SUITE,
                                cases=L2_CASES, variants=variants)
    manifest = base.read(batch / "experiment_manifest.json")
    frozen_cpp = batch / "frozen/cpp"
    frozen_hashes = {str(p.relative_to(frozen_cpp)): base.sha256(p)
                     for p in frozen_cpp.rglob("*") if p.is_file()}
    if frozen_hashes != source_before:
        raise ValueError("C++ source changed during preparation; do not launch this batch")
    changes = []
    for name, digest in sorted(frozen_hashes.items()):
        old = PRIMAL_BATCH / "frozen/cpp" / name
        old_digest = base.sha256(old) if old.is_file() else None
        if digest != old_digest:
            changes.append(dict(path=name, prior_sha256=old_digest, current_sha256=digest))
    evidence = dict(suite=OPTIMIZATION_SUITE, cases=[c[0] for c in L2_CASES],
        prior_validated_batch=str(PRIMAL_BATCH), prior_binary_sha256=previous["binary_sha256"],
        binary_sha256=manifest["binary_sha256"], source_origin="current_checkout_at_preparation",
        frozen_cpp_sha256=frozen_hashes, source_changes_from_primal=changes,
        algorithm_variants={name: list(flags) for name, flags in variants.items()},
        baseline_definition="prior primal-polish configuration rerun on the shared new binary",
        stage_contrasts={"schedule_minus_baseline": list(SCHEDULING_FLAGS),
                         "full_minus_schedule": list(STRUCTURAL_FLAGS)},
        expected_default_budgets=dict(integer_resource_fraction=.12,
            max_resource_restarts=2, supported_candidate_check_interval=3,
            primal_wall_interval_rule="max(30, .2 * seconds_per_job)"),
        model_and_operating_assumptions_unchanged=True,
        external_warm_start=None, external_root_certificate=None,
        old_results_are_controlled_baselines=False, solver_started=False)
    for name, payload in (("optimization_preflight.json", evidence),
                          ("historical_full_cplex.json", dict(rows=history,
                              comparison_scope="Historical reference only; binary and time budget differ"))):
        target = batch / "provenance" / name
        write_json(target, payload)
        manifest["frozen_sha256"][str(target.relative_to(batch))] = base.sha256(target)
    manifest["schema"] = "l2-algorithm-optimization/1"
    manifest["protocol"].update(pre_fix_reference=False, cases_predeclared=True,
        controlled_comparison="fresh baseline/schedule/full runs on the same new binary",
        ordered_variants=list(variants),
        dispatch_order="case-major: M01 baseline/schedule/full, M07 baseline/schedule/full, M08 baseline/schedule/full",
        stage_contrasts=evidence["stage_contrasts"],
        common_baseline_flags=list(variants["baseline"]),
        external_warm_start=None, external_root_certificate=None, reused_baselines=None,
        historical_full_cplex_reference="provenance/historical_full_cplex.json",
        input_dimensions="2 candidate bases, 20 candidate lockers, 50 OD pairs")
    (batch / "run.py").write_text("from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\nsys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\nmain([*sys.argv[1:], '--suite', {OPTIMIZATION_SUITE!r}, '--batch', str(batch)])\n")
    manifest["frozen_sha256"]["run.py"] = base.sha256(batch / "run.py")
    write_json(batch / "experiment_manifest.json", manifest)
    return dict(batch=str(batch), jobs=len(manifest["jobs"]), suite=OPTIMIZATION_SUITE,
                binary_sha256=manifest["binary_sha256"], solver_started=False)


def prior_algorithm_results(previous):
    """Pin all completed stages before dispatching a separate refinement batch."""
    previous = Path(previous)
    manifest, state = base.verify(previous), base.read(previous / "state.json")
    if len(manifest["jobs"]) != 9 or any(state["jobs"][j["id"]]["status"] != "completed"
                                          for j in manifest["jobs"]):
        raise ValueError("Refinement requires all nine original jobs completed before preparation")
    fields = ("upper_bound", "lower_bound", "gap", "reported_gap", "solve_seconds", "termination_reason",
              "opening_cost", "continuous_cost", "all_service_paths", "generated_service_columns",
              "master_variables", "master_linear_rows", "operations_opened", "operations_opened_links")
    rows = []
    for job in manifest["jobs"]:
        current = state["jobs"][job["id"]]
        folder = previous / current.get("result_directory", current.get("directory", "_missing"))
        row = record(previous, job, current)
        if not row.get("validation_passed"):
            raise ValueError("Prior algorithm reference failed validation: " + job["id"])
        rows.append(dict(instance=Path(job["instance"]).name, variant=job["variant"],
            job_id=job["id"], source_directory=str(folder),
            result_sha256=base.sha256(folder / "result.json"),
            native_validation_sha256=base.sha256(folder / "native_validation.json"),
            artifact_audit_sha256=base.sha256(folder / "artifact_audit.json"),
            input_manifest_sha256=base.sha256(previous / job["instance"] / "manifest.json"),
            binary_sha256=manifest["binary_sha256"],
            **{field: row.get(field) for field in fields}))
    return manifest, rows


def prepare_refinement(batch=REFINEMENT_BATCH, *, binary):
    batch, binary = Path(batch).resolve(), Path(binary).resolve()
    previous, references = prior_algorithm_results(OPTIMIZATION_BATCH)
    if base.sha256(binary) == previous["binary_sha256"]:
        raise ValueError("Refinement requires a newly compiled binary")
    source_before = {str(p.relative_to(ROOT / "cpp")): base.sha256(p)
        for p in (ROOT / "cpp").rglob("*") if p.is_file()
        and not any(part.startswith("build") or part == "__pycache__"
                    for part in p.relative_to(ROOT / "cpp").parts)
        and p.name != ".DS_Store" and p.suffix != ".pyc"}
    changes = []
    old_cpp = OPTIMIZATION_BATCH / "frozen/cpp"
    old_hashes = {str(p.relative_to(old_cpp)): base.sha256(p) for p in old_cpp.rglob("*") if p.is_file()}
    for name in sorted(source_before.keys() | old_hashes.keys()):
        digest, prior = source_before.get(name), old_hashes.get(name)
        if prior != digest:
            changes.append(dict(path=name, prior_sha256=prior, current_sha256=digest))
    if any(c["path"] != "src/benders.cpp" and not c["path"].startswith("tests/") for c in changes):
        raise ValueError("Algorithm refinement contains an undeclared production-source change")
    flags = optimization_variants()["full"]
    prepare_preserving_analysis(batch, binary=binary, suite=REFINEMENT_SUITE,
                                cases=L2_CASES, variants=dict(full_refined=flags))
    manifest = base.read(batch / "experiment_manifest.json")
    frozen_cpp = batch / "frozen/cpp"
    frozen_hashes = {str(p.relative_to(frozen_cpp)): base.sha256(p)
                     for p in frozen_cpp.rglob("*") if p.is_file()}
    if frozen_hashes != source_before:
        raise ValueError("C++ source changed during preparation; do not launch this batch")
    for identity, _, _ in L2_CASES:
        if base.sha256(batch / "instances" / identity / "manifest.json") != base.sha256(
                OPTIMIZATION_BATCH / "instances" / identity / "manifest.json"):
            raise ValueError("Refinement input changed: " + identity)
    evidence = dict(suite=REFINEMENT_SUITE, source_changes_from_optimization=changes,
        prior_batch=str(OPTIMIZATION_BATCH), prior_binary_sha256=previous["binary_sha256"],
        prior_manifest_sha256=base.sha256(OPTIMIZATION_BATCH / "experiment_manifest.json"),
        binary_sha256=manifest["binary_sha256"], frozen_cpp_sha256=frozen_hashes,
        changes=["Do not restart a root round after fractional support separation; continue root CG and retain added-cut stop conditions",
                 "Keep the original dual-profile order for initial_design and initialization neighborhood; LP-first remains enabled for subsequent formal CG"],
        all_prior_jobs_completed=True, same_input=True, solver_started=False)
    payloads = (("refinement_preflight.json", evidence),
        ("prior_algorithm_results.json", dict(rows=references,
            comparison_scope="Original stages retained with their own binary; full_refined alone rerun after root scheduling correction")),
        ("historical_full_cplex.json", dict(rows=historical_full_cplex(L2_CASES),
            comparison_scope="Historical reference only; binary and time budget differ")))
    for name, payload in payloads:
        target = batch / "provenance" / name
        write_json(target, payload)
        manifest["frozen_sha256"][str(target.relative_to(batch))] = base.sha256(target)
    manifest["schema"] = "l2-algorithm-refinement/1"
    manifest["protocol"].update(pre_fix_reference=False, cases_predeclared=True,
        controlled_comparison="three fresh full_refined jobs; original baseline/schedule/full retained as pinned references",
        ordered_variants=["full_refined"], external_warm_start=None, external_root_certificate=None,
        reused_baselines=None, prior_algorithm_reference="provenance/prior_algorithm_results.json",
        historical_full_cplex_reference="provenance/historical_full_cplex.json",
        input_dimensions="2 candidate bases, 20 candidate lockers, 50 OD pairs")
    (batch / "run.py").write_text("from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\nsys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\nmain([*sys.argv[1:], '--suite', {REFINEMENT_SUITE!r}, '--batch', str(batch)])\n")
    manifest["frozen_sha256"]["run.py"] = base.sha256(batch / "run.py")
    write_json(batch / "experiment_manifest.json", manifest)
    return dict(batch=str(batch), jobs=len(manifest["jobs"]), suite=REFINEMENT_SUITE,
                binary_sha256=manifest["binary_sha256"], solver_started=False)


def historical_full_cplex(cases):
    """Read checked historical results without treating them as paired runs."""
    reference = task.REFERENCE
    manifest = base.read(reference / "experiment_manifest.json")
    state = base.read(reference / "state.json")
    rows = []
    for identity, mode, cardinality in cases:
        selected = [job for job in manifest["jobs"] if job["instance"] == "instances/" + identity
                    and job["method"] == "full_cplex" and job["K"] == cardinality
                    and job["task_mode"] == mode]
        if len(selected) != 1:
            raise ValueError("Expected one historical full CPLEX job: " + identity)
        job = selected[0]
        current = state["jobs"][job["id"]]
        folder = reference / current.get("result_directory", current.get("directory", "_missing"))
        row = task.unified.record(reference, job, current)
        if (current["status"] != "completed" or not row["validation_passed"]
                or not base.optional_read(folder / "artifact_audit.json").get("passed")
                or row.get("upper_bound") is None or row.get("lower_bound") is None):
            raise ValueError("Historical full CPLEX result is not validated: " + identity)
        fields = ("upper_bound", "lower_bound", "gap", "solve_seconds", "termination_reason",
                  "solver_status", "all_service_paths", "all_empty_paths", "all_paths",
                  "full_model_variables", "full_model_linear_rows", "full_model_cones")
        rows.append(dict(instance=identity, task_mode=mode, K=cardinality, job_id=job["id"],
            **{key: row.get(key) for key in fields}, validation_passed=True,
            source_directory=str(folder), input_manifest_sha256=base.sha256(reference / job["instance"] / "manifest.json"),
            result_sha256=base.sha256(folder / "result.json"),
            native_validation_sha256=base.sha256(folder / "native_validation.json"),
            artifact_audit_sha256=base.sha256(folder / "artifact_audit.json"),
            binary_sha256=manifest["binary_sha256"],
            seconds_per_job=manifest["protocol"]["seconds_per_job"],
            threads_per_job=manifest["protocol"]["threads_per_job"],
            seed=manifest["protocol"]["seed"],
            relative_gap_target=manifest["protocol"]["relative_gap_target"],
            historical_reference_only=True))
    return rows


def prepare_expansion(batch, *, binary, proposed_flags, suite):
    """Fresh paired runs with the last validated executable and its paired C++."""
    _, cases = EXPANSION_SUITES[suite]
    batch, binary = Path(batch).resolve(), Path(binary).resolve()
    reference = base.verify(VALIDATED_REFERENCE)
    if base.sha256(binary) != reference["binary_sha256"]:
        raise ValueError("Expansion requires the latest validated binary")
    source_differences, source_difference_hashes = [], []
    frozen_cpp = VALIDATED_REFERENCE / "frozen/cpp"
    for path in frozen_cpp.rglob("*"):
        if path.is_file():
            relative = path.relative_to(frozen_cpp)
            current = ROOT / "cpp" / relative
            if not current.is_file() or base.sha256(current) != base.sha256(path):
                source_differences.append(str(relative))
                source_difference_hashes.append(dict(path=str(relative),
                    validated_sha256=base.sha256(path),
                    live_sha256=base.sha256(current) if current.is_file() else None))
    flags = validate_flags(list(proposed_flags))
    if set(flags) != set(PROPOSED_FLAGS):
        raise ValueError("Proposed must add exactly the two validated airline switches")
    variants = dict(fixed_baseline=BASELINE_FLAGS, proposed=(*BASELINE_FLAGS, *flags))
    solver_references = historical_full_cplex(cases) if suite == "l2-comparison" else None
    task.prepare(batch, binary=binary, suite=suite, cases=cases, variants=variants)
    manifest = base.read(batch / "experiment_manifest.json")
    # Other tasks may edit the checkout without rebuilding this executable.
    # Preserve the exact verified source paired with it, not that live checkout.
    target_cpp = batch / "frozen/cpp"
    if target_cpp.exists():
        shutil.rmtree(target_cpp)
    shutil.copytree(frozen_cpp, target_cpp)
    manifest["frozen_sha256"] = {name: value for name, value in manifest["frozen_sha256"].items()
                                 if not name.startswith("frozen/cpp/")}
    manifest["frozen_sha256"].update({str(path.relative_to(batch)): base.sha256(path)
        for path in target_cpp.rglob("*") if path.is_file()})
    source_manifest = base.read(task.REFERENCE / "experiment_manifest.json")
    source_state = base.read(task.REFERENCE / "state.json")
    history = []
    for identity, _, _ in cases:
        selected = [j for j in source_manifest["jobs"] if j["instance"] == "instances/" + identity]
        history.append(dict(instance=identity, input_hashes_checked_by_load_standard=True,
            old_jobs=[dict(job=j["id"], status=source_state["jobs"][j["id"]]["status"],
                attempts=len(source_state["jobs"][j["id"]].get("attempts", []))) for j in selected]))
    selection = ("predeclared map order, not selected on outcomes" if suite == "l1-expanded" else
        "predeclared L2 maps with validated historical full CPLEX results; includes a time-limited result")
    evidence = dict(cases=history, selection=selection,
        validated_reference_batch=str(VALIDATED_REFERENCE),
        validated_reference_manifest_sha256=base.sha256(VALIDATED_REFERENCE / "experiment_manifest.json"),
        binary_sha256=reference["binary_sha256"], frozen_cpp_matches_validated_snapshot=True,
        source_origin="validated_frozen_snapshot",
        cpp_source_origin=str(frozen_cpp),
        production_cpp_matches_validated_snapshot=not source_differences,
        live_checkout_differences=source_differences,
        live_checkout_difference_hashes=source_difference_hashes,
        old_results_are_controlled_baselines=False, solver_started=False)
    target = batch / ("provenance/l1_preflight.json" if suite == "l1-expanded" else "provenance/l2_preflight.json")
    write_json(target, evidence)
    manifest["frozen_sha256"][str(target.relative_to(batch))] = base.sha256(target)
    if solver_references is not None:
        target = batch / "provenance/historical_full_cplex.json"
        write_json(target, dict(rows=solver_references,
            comparison_scope="Historical reference; binary and time budget differ from the new paired runs"))
        manifest["frozen_sha256"][str(target.relative_to(batch))] = base.sha256(target)
        manifest["protocol"]["historical_full_cplex_reference"] = str(target.relative_to(batch))
    manifest["schema"] = ("l1-airline-expanded/1" if suite == "l1-expanded" else "l2-airline-comparison/1")
    manifest["protocol"].update(pre_fix_reference=False,
        controlled_comparison="fresh fixed_baseline and proposed per case, same validated binary",
        external_warm_start=None, external_root_certificate=None, reused_baselines=None,
        cases_predeclared=True, validated_binary_sha256=reference["binary_sha256"])
    (batch / "run.py").write_text("from pathlib import Path\nimport sys\n"
        "batch = Path(__file__).resolve().parent\nsys.path.insert(0, str(batch / 'frozen'))\n"
        f"from {MODULE} import main\nmain([*sys.argv[1:], '--suite', {suite!r}, '--batch', str(batch)])\n")
    manifest["frozen_sha256"]["run.py"] = base.sha256(batch / "run.py")
    write_json(batch / "experiment_manifest.json", manifest)
    return dict(batch=str(batch), jobs=len(manifest["jobs"]), suite=suite,
        binary_sha256=manifest["binary_sha256"], solver_started=False)


def verify_context(batch, root):
    manifest = base.verify(batch)
    if manifest["protocol"].get("suite") not in FRESH_SUITES:
        verify_snapshot(root)
    return manifest


def guarded_execute(batch, manifest, job_id, command, attempt, cancel, **kwargs):
    command = list(command)
    if "-m" in command and command[command.index("-m") + 1] == base.__name__:
        command[command.index("-m") + 1] = MODULE
    return task.unified._BASE_GUARD(batch, manifest, job_id, command, attempt, cancel, **kwargs)


def record(batch, job, state):
    row = task.record(batch, job, state)
    folder = Path(batch) / state.get("result_directory", state.get("directory", "_missing"))
    result = base.optional_read(folder / "result.json")
    native_passed = base.optional_read(folder / "native_validation.json").get("passed")
    artifact_passed = base.optional_read(folder / "artifact_audit.json").get("passed")
    row["native_validation_passed"] = native_passed
    row["artifact_audit_passed"] = artifact_passed
    verified = row["validation_passed"] and native_passed is True and artifact_passed is True
    row["validation_passed"] = verified
    if result:
        summary = self_gap_summary(result)
        if verified:
            row.update(summary)
            row.update(target_time_evidence(folder, summary, row.get("solve_seconds")))
            row.update(incumbent_improvement_evidence(folder))
        else:
            row["raw_result_self_gap_diagnostic"] = summary
            for key in ("upper_bound", "lower_bound", "gap", "time_to_target_seconds",
                        "observed_target_time_upper_bound_seconds", "computed_target_met", "target_gap_met"):
                row[key] = None
        row["certificate_recovery"] = certificate_recovery_summary(result)
        row["search_diagnostics"] = search_diagnostics(result, folder, verified=verified)
        row["airline_resource_execution"] = resource_execution_summary(result)
        row["opening_cost"] = result.get("opening_cost") if verified else None
        row["continuous_cost"] = result.get("continuous_cost") if verified else None
        polish = result.get("primal_design_polish_summary", {})
        row["primal_design_polish_summary"] = polish
        for name, value in polish.items():
            if not isinstance(value, (dict, list)):
                row["primal_design_polish_" + name] = value
        records = result.get("primal_design_polish_records", [])
        row["primal_design_polish_records"] = records
        first = [item.get("elapsed_seconds") for item in records
            if item.get("validated") and isinstance(item.get("upper_before"), (int, float))
            and isinstance(item.get("upper_after"), (int, float))
            and item["upper_after"] < item["upper_before"] - 1e-7 * (1 + abs(item["upper_before"]))
            and isinstance(item.get("elapsed_seconds"), (int, float))]
        row["first_primal_polish_improvement_seconds"] = min(first) if first and verified else None
    for name, value in result.items():
        if not name.startswith(("airline_resource", "airline_connectivity", "fractional_path_support",
                                "search_scheduling", "search_schedule", "scheduler")):
            continue
        if isinstance(value, list):
            row[name + "_count"] = len(value)
        elif isinstance(value, dict):
            for key, scalar in value.items():
                if not isinstance(scalar, (dict, list)):
                    row[name + "_" + key] = scalar
        else:
            row[name] = value
    return row


def search_diagnostics(result, folder, *, verified=True):
    """Keep observed global-LB progress separate from controller stall counters."""
    events = []
    progress = Path(folder) / "progress.jsonl"
    if progress.is_file():
        with progress.open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    masters = [e for e in events if e.get("event") == "master_control"]
    if not masters:
        masters = result.get("master_solves", [])
    bounds = [e for e in events if verified and e.get("event") == "global_bounds"
        and e.get("bound_scope") == "global" and e.get("validated_incumbent") is True
        and all(isinstance(e.get(k), (int, float)) and math.isfinite(e[k])
                for k in ("lower_bound", "upper_bound", "elapsed_seconds"))
        and e["lower_bound"] <= e["upper_bound"] + 1e-7 * (1 + abs(e["upper_bound"]))]
    bounds.sort(key=lambda e: e["elapsed_seconds"])
    improvements, observed = [], None
    for event in bounds:
        lb = event["lower_bound"]
        if observed is None or lb > observed + 1e-7 * (1 + abs(lb)):
            improvements.append(dict(elapsed_seconds=event["elapsed_seconds"], lower_bound=lb,
                                     upper_bound=event["upper_bound"], stage=event.get("stage")))
            observed = lb
    checkpoints = []
    for seconds in (60, 120, 180, 240, 300, 420, 540, 600):
        eligible = [event for event in bounds if event["elapsed_seconds"] <= seconds]
        if eligible:
            event = eligible[-1]
            checkpoints.append(dict(requested_seconds=seconds, **{k: event.get(k)
                for k in ("elapsed_seconds", "lower_bound", "upper_bound", "gap", "stage")}))
    support_skips, resource_accepts, resource_skips = set(), set(), set()
    for event in events:
        candidate = event.get("candidate_id")
        if not isinstance(candidate, str) or not candidate.startswith("integer:"):
            continue
        name = event.get("event", "")
        if name == "path_support_separation" and event.get("skipped_integer_subproblem"):
            support_skips.add(candidate)
        if name in {"task_resource_separation", "airline_resource_separation"}:
            if event.get("accepted"):
                resource_accepts.add(candidate)
            if event.get("skipped_integer_subproblem") is True:
                resource_skips.add(candidate)
    exact = [m for m in masters if m.get("relative_gap_target") == 0]
    timing, acceleration = result.get("timings", {}), result.get("acceleration_statistics", {})
    final_elapsed = result.get("wall_before_output", result.get("wall"))
    last = improvements[-1]["elapsed_seconds"] if improvements else None
    return dict(master_count=len(masters), master_seconds=sum(m.get("seconds", 0) for m in masters),
        master_reported_nodes=sum(m.get("nodes", 0) for m in masters),
        master_statuses=dict(Counter(m.get("status") for m in masters)),
        zero_gap_master_count=len(exact), zero_gap_master_seconds=sum(m.get("seconds", 0) for m in exact),
        integer_cg_calls=acceleration.get("integer_cg_calls"),
        recorded_integer_cg_candidates=len(result.get("iterations", [])),
        controller_bound_stalled_passes=acceleration.get("bound_stalled_passes"),
        controller_master_stalled_passes=acceleration.get("master_stalled_passes"),
        support_skipped_candidates=len(support_skips), resource_accepted_candidates=len(resource_accepts),
        explicitly_reported_resource_skipped_candidates=len(resource_skips),
        resource_skip_scope="Only explicit skipped_integer_subproblem=true; accepting a resource cut does not imply skipping CG",
        airline_resource_seconds=timing.get("airline_resource_separation"),
        subproblem_wall_seconds=timing.get("subproblem_wall"),
        dual_seconds=timing.get("subproblem", {}).get("dual"),
        restricted_mip_seconds=timing.get("restricted_mip"),
        global_lb_improvement_events=improvements, global_bound_checkpoints=checkpoints,
        last_observed_lb_improvement_seconds=last,
        seconds_since_last_observed_lb_improvement=None if last is None or not isinstance(final_elapsed, (int, float))
            else max(0, final_elapsed-last),
        lb_progress_scope="Validated global progress events only; first event establishes the reference, later improvements use 1e-7*(1+abs(LB))",
        algorithm_diagnostics={k: v for k, v in result.items()
            if k.startswith(("search_scheduling", "search_schedule", "scheduler", "fractional_path_support"))
            or k in ("airline_resource_metadata", "airline_resource_summary")})


def incumbent_improvement_evidence(folder):
    """Use only validated global incumbent events; elapsed is not call time."""
    progress = Path(folder) / "progress.jsonl"
    initial = first = None
    if progress.is_file():
        for line in progress.open():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if (event.get("event") != "global_bounds" or event.get("bound_scope") != "global"
                    or event.get("validated_incumbent") is not True):
                continue
            ub, seconds = event.get("upper_bound"), event.get("elapsed_seconds")
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (ub, seconds)):
                continue
            item = dict(elapsed_seconds=seconds, upper_bound=ub,
                        stage=event.get("stage"), sequence=event.get("sequence"))
            if initial is None:
                initial = item
            elif ub < initial["upper_bound"] - 1e-7 * (1 + abs(initial["upper_bound"])):
                first = item
                break
    return dict(initial_progress_upper_bound=None if initial is None else initial["upper_bound"],
        first_incumbent_improvement_seconds=None if first is None else first["elapsed_seconds"],
        first_incumbent_improvement_evidence=first,
        first_incumbent_improvement_scope="First validated global UB decrease from the earliest validated global UB event; not time to optimality")


def target_time_evidence(folder, summary, final_solve_seconds):
    reached = summary.get("computed_target_met") is True
    first = None
    progress = Path(folder) / "progress.jsonl"
    target = summary.get("requested_gap")
    if reached and progress.is_file() and isinstance(target, (int, float)):
        with progress.open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if (event.get("event") != "global_bounds" or event.get("bound_scope") != "global"
                        or event.get("validated_incumbent") is not True):
                    continue
                ub, lb, seconds = (event.get(k) for k in ("upper_bound", "lower_bound", "elapsed_seconds"))
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (ub, lb, seconds)):
                    continue
                if lb > ub + 1e-7 * (1+abs(ub)):
                    continue
                gap = max(0.0, ub-lb) / max(1.0, abs(ub))
                if gap <= target and (first is None or seconds < first["elapsed_seconds"]):
                    first = dict(elapsed_seconds=seconds, gap=gap,
                                 upper_bound=ub, lower_bound=lb, sequence=event.get("sequence"))
    return dict(time_to_target_seconds=None if first is None else first["elapsed_seconds"],
        time_to_target_evidence=first,
        observed_target_time_upper_bound_seconds=final_solve_seconds if reached else None,
        budget_censored=not reached and summary.get("termination_reason") == "time_limit",
        target_time_scope="first certified global-bound progress event; final solve time is only an upper bound")


def resource_execution_summary(result):
    grouped = defaultdict(list)
    for item in result.get("airline_resource_attempts", []):
        grouped[item.get("stage", "unspecified")].append(item)
    out = []
    for stage, attempts in grouped.items():
        status = Counter(item.get("status") for item in attempts)
        out.append(dict(stage=stage, attempts=len(attempts), statuses=dict(status),
            actual_optimize_calls=sum(item.get("solver_seconds", 0) > 0
                and item.get("status") != "budget_exhausted"
                and not str(item.get("status", "")).startswith("solver_exception") for item in attempts),
            solver_exception_attempts=sum(str(item.get("status", "")).startswith("solver_exception") for item in attempts),
            budget_exhausted=status["budget_exhausted"], valid_certificates=sum(bool(item.get("valid")) for item in attempts),
            accepted_cuts=sum(bool(item.get("accepted")) for item in attempts),
            separation_seconds=sum(item.get("seconds", 0) for item in attempts),
            solver_seconds=sum(item.get("solver_seconds", 0) for item in attempts)))
    return out


def self_gap_summary(result):
    upper, lower = result.get("objective"), result.get("bound")
    reported, target = result.get("gap"), result.get("requested_gap")
    computed = None
    if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (upper, lower)):
        computed = max(0.0, upper - lower) / max(1.0, abs(upper))
    reason = result.get("termination_reason")
    return dict(upper_bound=upper, lower_bound=lower, reported_gap=reported,
        computed_self_gap=computed,
        self_gap_consistent=computed is not None and isinstance(reported, (int, float))
            and math.isfinite(reported) and abs(reported-computed) <= 1e-10 * (1+abs(computed)),
        bound_order_consistent=computed is not None and lower <= upper + 1e-7 * (1+abs(upper)),
        requested_gap=target, target_gap_met=result.get("target_gap_met"),
        computed_target_met=None if computed is None or not isinstance(target, (int, float))
            else computed <= target,
        termination_reason=reason,
        normal_budget_or_gap_termination=reason in {"time_limit", "gap_limit", "optimal"},
        certificate_failure=isinstance(reason, str) and "failed" in reason
            and any(word in reason for word in ("dual", "certificate", "validation")),
        last_subproblem_termination=result.get("last_subproblem_termination_reason"),
        pricing_complete=result.get("pricing_complete"),
        pricing_complete_scope=result.get("pricing_complete_scope"))


def certificate_recovery_summary(result):
    """Read final-result traces only; merge repeated candidate/call/iteration logs."""
    traces = {}

    def visit(node, context, location):
        if isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, context, f"{location}/{index}")
            return
        if not isinstance(node, dict):
            return
        ctx = dict(context)
        for key in ("candidate_id", "cg_call", "anchor_kind", "stage", "round", "attempt"):
            if key in node:
                ctx[key] = node[key]
        trace = node.get("cg_trace")
        if isinstance(trace, list):
            candidate = ctx.get("candidate_id")
            if candidate is None:
                # Initial/continuous/primal-search calls sometimes omit an ID.
                # Their trace clock and stage identify repeated serialized copies.
                clock = trace[0].get("elapsed_seconds") if trace else None
                candidate = (ctx.get("stage", location.split("/")[0]),
                             ctx.get("round"), ctx.get("attempt"), clock)
            for index, entry in enumerate(trace):
                key = (str(candidate), ctx.get("cg_call"), ctx.get("anchor_kind"),
                       entry.get("iteration", index))
                item = traces.setdefault(key, dict(candidate_id=candidate,
                    cg_call=ctx.get("cg_call"), anchor_kind=ctx.get("anchor_kind"),
                    iteration=entry.get("iteration", index), attempts={},
                    tolerances=set(), terminations=set(), selected_methods=set(), copies=0))
                item["copies"] += 1
                tolerance = entry.get("duality_tolerance")
                if isinstance(tolerance, (int, float)) and math.isfinite(tolerance):
                    item["tolerances"].add(tolerance)
                if entry.get("termination_reason"):
                    item["terminations"].add(entry["termination_reason"])
                if entry.get("dual_certificate_method"):
                    item["selected_methods"].add(entry["dual_certificate_method"])
                for attempt in entry.get("dual_attempts", []):
                    signature = json.dumps(attempt, sort_keys=True, separators=(",", ":"))
                    item["attempts"][signature] = attempt
        for name, child in node.items():
            if name != "cg_trace":
                visit(child, ctx, f"{location}/{name}" if location else name)

    visit(result, {}, "")
    attempts, failures, duplicates = [], 0, 0
    profiles = defaultdict(list)
    for item in traces.values():
        failures += "rmp_duality_gap_failed" in item["terminations"]
        duplicates += max(0, item["copies"] - 1)
        # A mismatch is retained as evidence; use the tighter recorded tolerance
        # rather than silently relaxing the original acceptance rule.
        tolerance = min(item["tolerances"]) if item["tolerances"] else None
        for attempt in item["attempts"].values():
            profiles[(attempt.get("numerical_profile"), attempt.get("method"))].append(attempt)
            if (attempt.get("numerical_profile") != 5
                    and attempt.get("method") != "minimum_norm_cone_ray_lp"):
                continue
            gap = attempt.get("corrected_gap")
            passed = (attempt.get("valid") is True and attempt.get("optimal") is True
                and isinstance(gap, (int, float)) and math.isfinite(gap)
                and tolerance is not None and -tolerance <= gap <= tolerance)
            attempts.append(dict(candidate_id=item["candidate_id"], cg_call=item["cg_call"],
                anchor_kind=item["anchor_kind"], iteration=item["iteration"],
                numerical_profile=attempt.get("numerical_profile"), method=attempt.get("method"),
                valid=attempt.get("valid"), optimal=attempt.get("optimal"),
                status=attempt.get("status"), corrected_gap=gap,
                original_duality_tolerance=tolerance, passed_original_gap=passed,
                selected="minimum_norm_cone_ray_lp" in item["selected_methods"],
                seconds=attempt.get("seconds"), correction=attempt.get("correction")))
    return dict(source="final result.json only; repeated candidate/cg-call/anchor/CG-iteration traces merged",
        unique_cg_iterations=len(traces), duplicate_trace_copies_merged=duplicates,
        minimum_norm_attempts=len(attempts),
        minimum_norm_original_gap_passes=sum(a["passed_original_gap"] for a in attempts),
        minimum_norm_selected_iterations=sum("minimum_norm_cone_ray_lp" in i["selected_methods"]
                                             for i in traces.values()),
        rmp_duality_gap_failed_iterations=failures,
        final_termination_reason=result.get("termination_reason"), attempts=attempts,
        all_profile_summary=[dict(numerical_profile=key[0], method=key[1], attempts=len(values),
            valid=sum(v.get("valid") is True for v in values),
            statuses=dict(Counter(v.get("status") for v in values)),
            seconds=sum(v.get("seconds", 0) or 0 for v in values),
            build_seconds=sum(v.get("solver_build_seconds", 0) or 0 for v in values),
            optimize_seconds=sum(v.get("optimize_seconds", 0) or 0 for v in values))
            for key, values in profiles.items()])


def cross_check_rows(rows, points, *, tolerance=2e-5):
    """Necessary saved-point checks, not a claim of universal cut validity."""
    errors, checked, maximum, worst = [], 0, 0.0, None
    for label, row in rows:
        for point_label, design, operating in points:
            activity = 0.0
            try:
                terms = []
                for index, coefficient in row["coefficients"]:
                    if not isinstance(index, int) or not 0 <= index <= len(design):
                        raise ValueError("coefficient index outside design/epigraph layout")
                    value = operating if index == len(design) else design[index]
                    if not all(math.isfinite(v) for v in (coefficient, value, row["rhs"])):
                        raise ValueError("non-finite cut or point")
                    terms.append(coefficient * value)
                activity = math.fsum(terms)
                sense = row.get("sense", "G")
                if sense == "G":
                    violation = row["rhs"] - activity
                elif sense == "L":
                    violation = activity - row["rhs"]
                elif sense == "E":
                    violation = abs(activity - row["rhs"])
                else:
                    raise ValueError(f"unknown row sense {sense}")
            except (ValueError, TypeError, KeyError) as error:
                errors.append(f"{label} / {point_label}: {error}")
                continue
            checked += 1
            if violation > maximum:
                maximum, worst = violation, [label, point_label]
            if violation > tolerance:
                errors.append(f"{label} excludes {point_label}: violation={violation:.12g}")
    return dict(passed=not errors, row_count=len(rows), final_points=len(points),
        checks=checked, maximum_violation=maximum, worst_row_point=worst, errors=errors)


def canonicalize_support_point(model, demand, catalogue, solution, point):
    """Remove only unsupported zero-availability assignments, with zero flow.

    This preserves operations exactly. Positive/tiny rho is never silently
    rounded to zero for an audit; a point requiring that change is rejected.
    """
    label, design, cost = point
    values, ids = model["values"], model["ids"]
    R, L, H = (values[k] for k in ("n_od", "n_l", "n_h"))
    offset = L + H + H * len(ids["levels"])
    supported, nonzero = set(), set()
    for item in catalogue["missions"]:
        path = item["path"]
        key = (path["first_od"], path["b"][0], path["second_od"])
        if all(solution["airline"][u][v] >= .5 for u, v, _ in path["arc_counts"]):
            supported.add(key)
        if solution["flow"][path["id"]] != 0:
            nonzero.add(key[:2])
    normalized, changes, errors = list(design), [], []
    for r in range(R):
        branches = [s for s, rate in [(None, demand["single_rates"][r]),
                    *enumerate(demand["pair_rates"][r])] if rate > 0]
        for l in range(L):
            if not solution["assignment"][r][l]:
                continue
            missing = [s for s in branches if (r, l, s) not in supported]
            if not missing:
                continue
            evidence = dict(od=r, origin_base=l, missing_positive_branches=missing,
                            rho=solution["rho"][l], has_nonzero_flow=(r, l) in nonzero)
            if solution["rho"][l] != 0 or (r, l) in nonzero:
                errors.append(dict(**evidence,
                    reason="Unsupported assignment cannot be removed while preserving the saved operations exactly"))
                continue
            normalized[offset + r*L + l] = 0
            changes.append(dict(**evidence,
                reason="Zero-availability and exactly zero-flow assignment removed; operations and cost unchanged"))
    return (label + ":canonical", normalized, cost), dict(point=label, passed=not errors,
        changed_assignments=changes, errors=errors,
        scope="Current fixed-task origin/aggregate model; optional cost-free assignments; no rounding of rho or flows")


def audit(batch, *, root, require_complete=False):
    from src.experiments import task_cost_audit as previous_audit
    batch, root = Path(batch).resolve(), Path(root).resolve()
    manifest = verify_context(batch, root)
    # Retain the established independent demand/catalogue/task-cut audit.
    inherited = previous_audit.audit(batch, require_complete)
    manifest, state = base.verify(batch), base.read(batch / "state.json")
    rows_by_case, points_by_case = defaultdict(list), defaultdict(list)
    connectivity_by_case, canonical_point_labels = defaultdict(list), set()
    fractional_by_case, fractional_applied_labels = defaultdict(list), set()
    point_sources = {}
    errors, summaries = [], []

    def saved_point(folder, label, expected_sha=None):
        result, solution = base.read(folder / "result.json"), base.read(folder / "solution.json")
        validation = base.read(folder / "native_validation.json")
        artifact = base.read(folder / "artifact_audit.json")
        actual_sha = base.sha256(folder / "result.json")
        if (not validation.get("passed") or not artifact.get("passed")
                or validation.get("result_sha256") != actual_sha
                or validation.get("solution_sha256") != base.sha256(folder / "solution.json")
                or expected_sha is not None and expected_sha != actual_sha):
            raise ValueError(f"Saved feasible point failed identity/validation: {label}")
        point_sources[label] = (Path(folder), solution)
        return result, (label, previous_audit.design(solution), result["continuous_cost"])

    for job in manifest["jobs"]:
        current = state["jobs"][job["id"]]
        if current["status"] != "completed":
            summaries.append(dict(job=job["id"], instance=job["instance"],
                scheduler_status=current["status"], termination_reason=None))
            continue
        folder = batch / current.get("result_directory", current.get("directory", "_missing"))
        result, point = saved_point(folder, job["id"])
        key = job["instance"], job["K"]
        points_by_case[key].append(point)
        for index, cut in enumerate(result.get("airline_resource_cut_records", [])):
            label = f"{job['id']}:airline_resource:{index}"
            if not cut.get("valid") or cut.get("max_dual_violation_after", math.inf) > 0:
                errors.append(label + ": missing or failing recorded dual-feasibility check")
            rows_by_case[key].append((label, cut))
        connectivity = result.get("airline_connectivity_metadata", {})
        if connectivity.get("enabled"):
            canonical_point_labels.add(job["id"])
        for index, cut in enumerate(connectivity.get("rows", [])):
            connectivity_by_case[key].append((f"{job['id']}:connectivity:{index}", cut))
        fractional_rows = result.get("fractional_path_support_records", [])
        if result.get("fractional_path_support_summary", {}).get("enabled"):
            fractional_applied_labels.add(job["id"])
        for index, cut in enumerate(fractional_rows):
            if not cut.get("valid"):
                errors.append(f"{job['id']}:fractional_support:{index}: missing valid certificate flag")
            fractional_by_case[key].append((f"{job['id']}:fractional_support:{index}", cut))
        summary = dict(job=job["id"], instance=job["instance"], variant=job["variant"],
            scheduler_status=current["status"], **self_gap_summary(result))
        summary["certificate_recovery"] = certificate_recovery_summary(result)
        summaries.append(summary)
        if not summary["self_gap_consistent"] or not summary["bound_order_consistent"]:
            errors.append(job["id"] + ": reported bound/self-gap inconsistency")
        if summary["target_gap_met"] != summary["computed_target_met"]:
            errors.append(job["id"] + ": target-gap flag differs from own UB/LB")

    # Include all four old final incumbents per physical case. They are useful
    # falsification points even if their algorithms terminated numerically.
    historical = [] if manifest["protocol"].get("suite") in FRESH_SUITES else base.read(root / "pre_fix/reference_commands.json")
    for item in historical:
        key = item["instance"], 2
        if key not in points_by_case:
            continue
        old_input = root / "pre_fix" / item["instance"] / "manifest.json"
        if base.sha256(old_input) != base.sha256(batch / item["instance"] / "manifest.json"):
            raise ValueError("Old and new input identities differ: " + item["instance"])
        _, point = saved_point(Path(item["result"]).parent, "pre_fix:" + item["job_id"],
                               item["result_sha256"])
        points_by_case[key].append(point)
    reused = manifest["protocol"].get("reused_baselines")
    if reused:
        for evidence in base.read(batch / reused):
            item = evidence["job"]
            key = item["instance"], item["K"]
            if evidence["input_manifest_sha256"] != base.sha256(batch / item["instance"] / "manifest.json"):
                raise ValueError("Reused baseline input identity changed")
            _, point = saved_point(Path(evidence["result_directory"]), "reused:" + item["id"],
                                   evidence["result_sha256"])
            if key in points_by_case:
                points_by_case[key].append(point)
    cases, catalogue_cache = [], {}
    for key, points in points_by_case.items():
        check = cross_check_rows(rows_by_case[key], points)
        connectivity_all = cross_check_rows(connectivity_by_case[key], points)
        connectivity_applied = cross_check_rows(connectivity_by_case[key],
            [point for point in points if point[0] in canonical_point_labels])
        fractional_all = cross_check_rows(fractional_by_case[key], points)
        fractional_applied = cross_check_rows(fractional_by_case[key],
            [point for point in points if point[0] in fractional_applied_labels])
        canonical_evidence, canonical_points = [], []
        if fractional_by_case[key]:
            folder = point_sources[points[0][0]][0]
            catalogue, _ = previous_audit.resolve_full_catalogue(batch, key, folder, cache=catalogue_cache)
            model = base.read(folder / "input_reference.json")["model"]
            demand = base.read(folder / "preprocessed_demand.json")
            for point in points:
                normalized, evidence = canonicalize_support_point(model, demand, catalogue,
                    point_sources[point[0]][1], point)
                canonical_evidence.append(evidence)
                if evidence["passed"]:
                    canonical_points.append(normalized)
                else:
                    errors.append(f"{point[0]}: cannot canonicalize unsupported assignments without changing operations")
        fractional_canonical = cross_check_rows(fractional_by_case[key], canonical_points)
        # Connectivity rows intentionally canonicalize redundant assignments
        # and unused assets. An old noncanonical encoding is not a counterexample
        # to preservation of an optimum. Applied runs must satisfy their rows.
        cases.append(dict(instance=key[0], K=key[1], **check,
            connectivity_all_final_points=connectivity_all,
            connectivity_applied_final_points=connectivity_applied,
            fractional_all_final_points=fractional_all,
            fractional_applied_final_points=fractional_applied,
            fractional_canonical_final_points=fractional_canonical,
            fractional_canonicalization=canonical_evidence,
            fractional_scope="optimality-preserving canonical supported assignments; exclusions of other encodings are separately reported",
            connectivity_scope="optimality-preserving canonical encodings; old-point exclusions are reported, not labelled universally invalid"))
        errors.extend(check["errors"])
        errors.extend(connectivity_applied["errors"])
        errors.extend(fractional_applied["errors"])
        errors.extend(fractional_canonical["errors"])
    out = dict(schema="airline-certificate-saved-point-audit/1", solver_invoked=False,
        inherited_task_audit=inherited, passed=not errors, errors=errors, cases=cases,
        termination_counts=dict(Counter(row["termination_reason"] for row in summaries)),
        runs=summaries, validity_scope="Necessary checks on saved validated final points; not a universal proof")
    output = batch / "analysis"
    write_json(output / "airline_certificate_audit.json", out)
    lines = ["# 航线割与求解终止审计", "",
        "复用既有需求守恒、完整路径目录、任务收益/资源割审计；新增航线资源成本割对同实例所有新旧最终可行点逐一检查。该跨点检查是排错条件，不替代通用有效性证明。",
        "连通性行会规范化无服务分配及闲置资产，并不保留全部冗余编码。旧点排除单独记录；已启用这些行的新运行，其最终点必须满足这些行。", "",
        f"通过：{out['passed']}；终止原因：{out['termination_counts']}。", "",
        "| 运行 | 调度状态 | 算法终止原因 | 原报告gap | 自身UB/LB重算gap | 达到目标 |",
        "|---|---|---|---:|---:|---|"]
    for row in summaries:
        fmt = lambda key: "—" if row.get(key) is None else f"{100*row[key]:.5f}%"
        lines.append(f"| {row['job']} | {row['scheduler_status']} | {row['termination_reason']} | "
            f"{fmt('reported_gap')} | {fmt('computed_self_gap')} | {row.get('computed_target_met')} |")
    lines += ["", "共同最佳UB仅能用于另行比较下界，不替代此表每次运行自身的gap或停止条件。算法提前失败与time_limit/gap_limit分别记录。", "",
        "| 运行 | 最小范数恢复尝试 | 通过原gap门槛 | RMP gap失败轮数 | 最终终止原因 |",
        "|---|---:|---:|---:|---|"]
    for row in summaries:
        recovery = row.get("certificate_recovery", {})
        lines.append(f"| {row['job']} | {recovery.get('minimum_norm_attempts', '—')} | "
            f"{recovery.get('minimum_norm_original_gap_passes', '—')} | "
            f"{recovery.get('rmp_duality_gap_failed_iterations', '—')} | {row['termination_reason']} |")
    lines += ["", "恢复统计仅读最终result中的iterations/root/acceleration等cg_trace；同candidate、cg_call、anchor、CG iteration的重复副本合并，不叠加checkpoint或progress快照。尝试未触发时记录零，不把轨迹避开失败称为恢复成功。通过次数要求valid、optimal和修正gap同时满足原始duality_tolerance。"]
    if any(c["fractional_all_final_points"]["row_count"] for c in cases):
        lines += ["", "## 分数路径支持割", "",
            "这些行在规范的受支持分配上有效。对所有阶段最终点进行检查；其他编码的排除保留报告，已启用这些行的full阶段最终点必须通过。此跨点检查不替代完整域证书和有效性证明。", "",
            "| 算例 | 行数 | 全部最终点检查 | 全部点最大违反 | 已应用点检查 | 已应用点通过 |",
            "|---|---:|---:|---:|---:|---|"]
        for case in cases:
            all_points, applied = case["fractional_all_final_points"], case["fractional_applied_final_points"]
            lines.append(f"| {Path(case['instance']).name} | {all_points['row_count']} | {all_points['checks']} | "
                f"{all_points['maximum_violation']:.9g} | {applied['checks']} | {applied['passed']} |")
    (output / "AIRLINE_CERTIFICATE_AUDIT.md").write_text("\n".join(lines)+"\n")
    if errors:
        raise RuntimeError("Airline/certificate audit failed; inspect airline_certificate_audit.json")
    return dict(passed=True, report=str(output / "AIRLINE_CERTIFICATE_AUDIT.md"),
                cross_point_checks=sum(c["checks"] for c in cases),
                fractional_cross_point_checks=sum(c["fractional_all_final_points"]["checks"] for c in cases),
                runs=len(summaries))


def report(batch):
    result = task.report(batch, record_fn=record)
    output = Path(batch) / "analysis"
    rows = base.read(output / "comparison.json")["rows"]
    manifest = base.read(Path(batch) / "experiment_manifest.json")
    l1_expanded = manifest["protocol"].get("suite") == "l1-expanded"
    l2_comparison = manifest["protocol"].get("suite") == "l2-comparison"
    primal_comparison = manifest["protocol"].get("suite") == PRIMAL_SUITE
    optimization_comparison = manifest["protocol"].get("suite") == OPTIMIZATION_SUITE
    refinement_comparison = manifest["protocol"].get("suite") == REFINEMENT_SUITE
    reused = manifest["protocol"].get("reused_baselines")
    if reused:
        evidence = base.read(Path(batch) / reused)
        baselines = [dict(item["row"], reused_not_rerun=True) for item in evidence]
        write_json(output / "controlled_comparison.json", dict(
            control_scope=evidence[0]["control_scope"], rows=[*baselines, *rows]))
        rows = [*baselines, *rows]
    title = ("L1六地图航线强化：受控比较" if l1_expanded else
             "L2三地图航线强化：受控比较" if l2_comparison else
             "L2调度与初始化修正：补充比较" if refinement_comparison else
             "L2算法优化：三阶段受控比较" if optimization_comparison else
             "L2可行解构造改善：受控比较" if primal_comparison else "证书修复与航线资源强化：受控比较")
    selection = ("六张地图按事先确定的顺序选取，不复用旧baseline、warm start或root证书。" if l1_expanded else
        "三张地图仅重新运行full_refined；原三阶段结果保留为带二进制和输出哈希的参考。" if refinement_comparison else
        "三张地图的baseline/schedule/full均重新运行，不复用旧baseline、外部warm start或root证书。" if optimization_comparison else
        "三张地图均有历史全路径CPLEX结果；两个算法均重新运行，不复用旧baseline、外部warm start或root证书。" if l2_comparison or primal_comparison else
        "旧pre-fix结果仅作为已冻结的历史参考，没有重复运行。")
    lines = ["# " + title, "",
        ("本次三个实例仅重跑proposed，fixed_baseline直接复用前轮同实例已验证结果；二进制变化仅为已启用外部资源LP的目标系数批量更新及诊断，baseline禁用该LP，未重复运行。"
         if reused else
         "本批full_refined包含两项修正：分数支持割加入后继续同一候选的root CG；初始化保留原对偶配置顺序，后续正式CG仍优先LP恢复。三个作业使用同一新二进制，原9次作业不覆盖、不重跑。两个修正联合测试，不能单独归因于其中一个。" if refinement_comparison else
         f"{len(rows)}组使用同一新编译二进制。baseline保留前轮全部加强和primal-polish；schedule增加搜索调度与dual-LP优先；full再增加分数路径支持与航线资源基地平衡。模型和运营假设相同。" if optimization_comparison else
         f"{len(rows)}组使用同一新编译二进制；两组均保留原任务及航线加强，proposed只增加 --primal-polish。下界策略、模型和运营假设相同。原有内部warm start与启发式保留。" if primal_comparison else
         f"{len(rows)}组使用同一验证后二进制；每例两个方法均重新运行，fixed_baseline保留原task_airline设置，proposed仅增加两个航线资源/连通性开关。模型和运营假设相同。"),
        "每组600秒、单线程、seed43、gap1%，最多四组并行。" + selection,
        "调度器completed只表示执行和输出校验完成。是否正常求解必须同时查看算法终止原因；可行解验证不等于收敛。", "",
        "| 实例 | 方法 | UB | 全局LB | 自身gap | 求解秒 | 首次证据达标秒 | 预算删失 | 调度状态 | 算法终止原因 | 验证 |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|---|"]
    fmt = lambda v: "—" if v is None else f"{v:.6f}" if isinstance(v, float) else str(v)
    for row in rows:
        values = [Path(row["instance"]).name, row["variant"], row.get("upper_bound"),
            row.get("lower_bound"), None if row.get("gap") is None else f"{100*row['gap']:.4f}%",
            row.get("solve_seconds"), row.get("time_to_target_seconds"), row.get("budget_censored"),
            row["scheduler_status"], row.get("termination_reason"),
            row["validation_passed"]]
        lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
    summary_rows = []
    for variant in dict.fromkeys(row["variant"] for row in rows):
        selected = [row for row in rows if row["variant"] == variant]
        verified = [row for row in selected if row["scheduler_status"] == "completed" and row["validation_passed"]]
        reached = [row for row in verified if row.get("computed_target_met") is True]
        observed = [row["time_to_target_seconds"] for row in reached if row.get("time_to_target_seconds") is not None]
        summary_rows.append(dict(variant=variant, total=len(selected), completed_double_validated=len(verified),
            target_met=len(reached), timeout_censored=sum(bool(row.get("budget_censored")) for row in verified),
            abnormal_terminations=sum(row["scheduler_status"] in {"failed", "interrupted"}
                or row["scheduler_status"] == "completed" and
                    (not row["validation_passed"] or not row.get("normal_budget_or_gap_termination", False))
                for row in selected),
            target_time_observed=len(observed),
            median_time_to_target_on_reached_subset=statistics.median(observed) if observed else None))
    paired = defaultdict(dict)
    for row in rows:
        paired[row["instance"]][row["variant"]] = row
    required_variants = (("full_refined",) if refinement_comparison else
        ("baseline", "schedule", "full") if optimization_comparison else ("fixed_baseline", "proposed"))
    both_reached = sum(all(pair.get(v, {}).get("validation_passed")
        and pair.get(v, {}).get("computed_target_met") for v in required_variants)
        for pair in paired.values())
    write_json(output / "target_attainment_summary.json", dict(rows=summary_rows,
        compared_variants=list(required_variants), all_variants_reached_cases=both_reached,
        both_variants_reached_cases=both_reached if len(required_variants) == 2 else None,
        time_scope="Median is only among reached runs with first certified event evidence; not a timeout-imputed runtime"))
    lines += ["", "| 方法 | 计划数 | 完成且双验证 | 1%达标 | timeout删失 | 异常终止 | 有首次时间证据的达标数 | 达标子集时间中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        values = [row[key] for key in ("variant", "total", "completed_double_validated", "target_met",
            "timeout_censored", "abnormal_terminations", "target_time_observed", "median_time_to_target_on_reached_subset")]
        lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", f"全部{len(required_variants)}个方法均达标的地图数：{both_reached}。",
        "首次达标秒仅取已验证全局UB/LB进度事件首次达到目标的时刻。未达标时为空；time_limit且未达标标记预算删失，不能把600秒称为收敛时间。若最终达标而缺少进度证据，最终求解秒只另存为观察到的时间上界。",
        "完整规模、root LB、最终LB、生成列、候选、认证、资源和运营指标见comparison.csv/JSON；原始文件保留于各attempt。"]
    if primal_comparison or optimization_comparison or refinement_comparison:
        lines += primal_report_sections(rows, fmt)
    if optimization_comparison or refinement_comparison:
        lines += optimization_report_sections(rows, fmt)
    if refinement_comparison:
        references = base.read(Path(batch) / manifest["protocol"]["prior_algorithm_reference"])["rows"]
        lines += ["", "## 原三阶段参考", "",
            "以下为前一个二进制的原始结果，未替换本批停止gap。原schedule/full的初始化也采用LP优先，其数值不能替代本轮保留原始初始化顺序后的结果。源二进制、输入和结果哈希均冻结在provenance/prior_algorithm_results.json。", "",
            "| 实例 | 原阶段 | UB | LB | gap | 秒 | 终止 |", "|---|---|---:|---:|---:|---:|---|"]
        contrasts = []
        for reference in references:
            if base.sha256(Path(reference["source_directory"]) / "result.json") != reference["result_sha256"]:
                raise ValueError("Pinned prior algorithm result changed: " + reference["job_id"])
            values = [reference["instance"], reference["variant"], reference["upper_bound"], reference["lower_bound"],
                f"{100*reference['gap']:.4f}%", reference["solve_seconds"], reference["termination_reason"]]
            lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
            current = next((r for r in rows if Path(r["instance"]).name == reference["instance"]
                and r.get("validation_passed") and r.get("scheduler_status") == "completed"), None)
            if current:
                contrasts.append(dict(instance=reference["instance"], prior_stage=reference["variant"],
                    new_stage="full_refined", prior_binary_sha256=reference["binary_sha256"],
                    new_binary_sha256=manifest["binary_sha256"],
                    upper_bound_reduction=reference["upper_bound"]-current["upper_bound"],
                    lower_bound_increase=current["lower_bound"]-reference["lower_bound"],
                    gap_percentage_points_reduction=100*(reference["gap"]-current["gap"])))
        write_json(output / "refinement_contrasts.json", dict(rows=contrasts,
            scope="Original stages keep their prior binary and timestamps; only full_refined freshly run"))
    if optimization_comparison:
        contrasts = []
        for identity, stages in paired.items():
            for old, new in (("baseline", "schedule"), ("schedule", "full"), ("baseline", "full")):
                left, right = stages.get(old, {}), stages.get(new, {})
                if not all(r.get("validation_passed") and r.get("scheduler_status") == "completed"
                           for r in (left, right)):
                    continue
                contrasts.append(dict(instance=identity, baseline_stage=old, new_stage=new,
                    upper_bound_reduction=left["upper_bound"]-right["upper_bound"],
                    lower_bound_increase=right["lower_bound"]-left["lower_bound"],
                    gap_percentage_points_reduction=100*(left["gap"]-right["gap"]),
                    old_termination=left.get("termination_reason"), new_termination=right.get("termination_reason")))
        write_json(output / "stage_contrasts.json", dict(rows=contrasts,
            scope="Same-input same-binary fresh stages; no historical incumbent substituted into per-run gaps"))
    historical_reference = manifest["protocol"].get("historical_full_cplex_reference")
    if historical_reference:
        history = base.read(Path(batch) / historical_reference)
        lines += ["", "## 历史全路径 CPLEX 参考", "",
            "同一物理输入与双单任务模式；旧二进制、3600秒上限、单线程、seed43、gap1%。这些结果未重跑，不能当作本轮600秒同版本对照。保留每个结果的原始终止原因，限时结果不视作达到1%。",
            "", "| 实例 | UB | 全局LB | gap | 求解秒 | 终止 | 完整服务路径数 |",
            "|---|---:|---:|---:|---:|---|---:|"]
        for row in history["rows"]:
            values = [row["instance"], row["upper_bound"], row["lower_bound"],
                f"{100*row['gap']:.4f}%", row["solve_seconds"], row["termination_reason"], row["all_service_paths"]]
            lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
        lines += ["", "来源目录、结果和验证哈希、二进制与运行参数见 provenance/historical_full_cplex.json。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def optimization_report_sections(rows, fmt):
    lines = ["", "## 主问题与完整CG机会", "",
        "完整整数CG调用与最终记录的整数候选数分别报告。资源割被接受不必然表示跳过CG；仅将明确的skip事件统计为跳过。节点数为各次求解器报告数累计。", "",
        "| 实例 | 阶段 | master次数 | master秒 | 零gap请求次数/秒 | 整数CG调用 | 支持跳过 | 资源接受候选 | 距最近LB提升秒 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        d = row.get("search_diagnostics", {})
        values = [Path(row["instance"]).name, row["variant"], d.get("master_count"),
            d.get("master_seconds"), f"{d.get('zero_gap_master_count', 0)} / {d.get('zero_gap_master_seconds', 0):.3f}",
            d.get("integer_cg_calls"), d.get("support_skipped_candidates"), d.get("resource_accepted_candidates"),
            d.get("seconds_since_last_observed_lb_improvement")]
        lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", "LB提升采用已验证全局界事件独立重算，首次事件作为参照，不把控制器的stall计数直接解释为下界停滞。60/120/180/240/300/420/540/600秒界值见comparison.json的search_diagnostics。", "",
        "## 对偶配置耗时", "",
        "仅统计最终result的CG trace，并按候选/CG调用/锚点/迭代合并重复序列。是否valid与是否满足完整CG认证条件仍以原始记录为准。", "",
        "| 实例 | 阶段 | profile | 方法 | 调用 | valid | 秒 | 状态 |",
        "|---|---|---|---|---:|---:|---:|---|"]
    for row in rows:
        for profile in row.get("certificate_recovery", {}).get("all_profile_summary", []):
            values = [Path(row["instance"]).name, row["variant"], profile.get("numerical_profile"),
                profile.get("method"), profile.get("attempts"), profile.get("valid"),
                profile.get("seconds"), json.dumps(profile.get("statuses"), ensure_ascii=False)]
            lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", "分数支持、基地平衡、调度预算和触发记录按原字段保存在comparison.json及每次运行的result/progress中。三阶段UB/LB差分见stage_contrasts.json。对新割的跨点审计另行运行，不能用更低gap代替有效性检查。"]
    return lines


def primal_report_sections(rows, fmt):
    lines = ["", "## 可行解、设施与路径规模", "",
        "三个实例均为2个候选基地、20个候选柜点、50个OD。完整服务路径目录、实际进入CG列池的服务列以及启发式模型采用的列分别报告。",
        "首次UB改善时间取已验证global_bounds事件，相对于最早记录的已验证UB；它通常包含初始构造改善，不等于新启发式贡献。新启发式首次改善另列，只有记录了全局elapsed_seconds的已验证改进才显示时间。", "",
        "| 实例 | 方法 | 开放成本 | 连续成本 | 开放柜点 | 开放航线 | 完整服务路径 | CG服务列 | 首次UB改善秒 | 新启发式首次改善秒 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        opened = row.get("operations_opened") or {}
        if isinstance(opened, str):
            try:
                opened = json.loads(opened)
            except ValueError:
                opened = {}
        values = [Path(row["instance"]).name, row["variant"], row.get("opening_cost"),
            row.get("continuous_cost"), opened.get("lockers"), row.get("operations_opened_links"),
            row.get("all_service_paths"), row.get("generated_service_columns"),
            row.get("first_incumbent_improvement_seconds"), row.get("first_primal_polish_improvement_seconds")]
        lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", "## 新启发式调用", "",
        "这些调用只用于构造经验证的可行上界。受限列模型的best bound不作为全局下界；固定分配标记表示该次调用固定当前分配，并非主问题新增约束。逐阶段构建、求解、校验时间和完整诊断保留在comparison.json的primal_design_polish_records中。", "",
        "| 实例 | 方法 | 阶段 | 固定分配 | 分配预算秒 | 实际秒 | 验证 | UB前 | UB后 | 目录路径 | 请求列 | 模型变量/行 | 状态 |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---|"]
    count = 0
    for row in rows:
        for item in row.get("primal_design_polish_records", []):
            count += 1
            size = item.get("model_size", {})
            size_label = json.dumps(size, ensure_ascii=False, sort_keys=True) if size else "—"
            columns = item.get("requested_service_columns")
            if isinstance(columns, list):
                columns = len(columns)
            values = [Path(row["instance"]).name, row["variant"], item.get("stage"),
                item.get("fixed_assignment"), item.get("allocated_seconds"), item.get("seconds"),
                item.get("validated"), item.get("upper_before"), item.get("upper_after"),
                item.get("catalogue_service_paths"), columns, size_label, item.get("status")]
            lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
    if not count:
        lines.append("| — | — | 尚无调用记录 | — | — | — | — | — | — | — | — | — | — |")
    return lines


@contextmanager
def adapters():
    with task.adapters():
        old_guard, old_report, old_record = base.guarded_execute, base.report, base._record
        try:
            base.guarded_execute, base.report, base._record = guarded_execute, report, record
            yield
        finally:
            base.guarded_execute, base.report, base._record = old_guard, old_report, old_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "snapshot", "prepare", "prepare-bulk", "verify", "launch",
                                           "run", "stop", "status", "report", "audit", "audit-worker"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--suite", choices=("certificate", *FRESH_SUITES), default="certificate")
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--proposed-flags-json", type=Path)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    batch = args.batch or (FRESH_SUITES[args.suite][0] if args.suite in FRESH_SUITES else args.root / "comparison")
    with adapters():
        if args.action == "inspect":
            cases = FRESH_SUITES[args.suite][1] if args.suite in FRESH_SUITES else CASES
            common = (*BASELINE_FLAGS, *PROPOSED_FLAGS) if args.suite == PRIMAL_SUITE else BASELINE_FLAGS
            repeats = 1 if args.suite == REFINEMENT_SUITE else 3 if args.suite == OPTIMIZATION_SUITE else 2
            result = dict(cases=[c[0] for c in cases], new_runs=repeats*len(cases),
                baseline_flags=list(common),
                proposed_flags=[*common, PRIMAL_FLAG] if args.suite == PRIMAL_SUITE else "pending explicit input",
                seconds=600, threads=1, seed=43, gap=.01, parallel=4, solver_started=False)
            if args.suite == OPTIMIZATION_SUITE:
                result.update(algorithm_variants=optimization_variants(),
                    baseline_flags=list(optimization_variants()["baseline"]),
                    proposed_flags=None, dispatch_order="case-major baseline/schedule/full")
            if args.suite == REFINEMENT_SUITE:
                result.update(algorithm_variants=dict(full_refined=optimization_variants()["full"]),
                    baseline_flags=None, proposed_flags=list(optimization_variants()["full"]),
                    dispatch_order="M01 full_refined, M07 full_refined, M08 full_refined")
        elif args.action == "snapshot":
            result = snapshot(args.root)
        elif args.action in {"prepare", "prepare-bulk"}:
            if not args.binary or (not args.proposed_flags_json and args.suite not in FRESH_SUITES):
                parser.error("prepare requires --binary and --proposed-flags-json")
            flags = base.read(args.proposed_flags_json) if args.proposed_flags_json else list(PROPOSED_FLAGS)
            if args.suite == REFINEMENT_SUITE:
                if args.action != "prepare" or args.proposed_flags_json:
                    parser.error("Refinement comparison uses prepare and the fixed full-refined flag map")
                result = prepare_refinement(batch, binary=args.binary)
            elif args.suite == OPTIMIZATION_SUITE:
                if args.action != "prepare" or args.proposed_flags_json:
                    parser.error("Optimization comparison uses prepare and the fixed three-stage flag map")
                result = prepare_optimization(batch, binary=args.binary)
            elif args.suite == PRIMAL_SUITE:
                if args.action != "prepare" or args.proposed_flags_json:
                    parser.error("Primal comparison uses prepare and the fixed --primal-polish flag")
                result = prepare_primal(batch, binary=args.binary)
            elif args.suite in EXPANSION_SUITES:
                if args.action != "prepare":
                    parser.error("Expansion suite uses prepare, not prepare-bulk")
                result = prepare_expansion(batch, binary=args.binary, proposed_flags=flags,
                                           suite=args.suite)
            else:
                builder = prepare_bulk if args.action == "prepare-bulk" else prepare
                result = builder(args.root, binary=args.binary, proposed_flags=flags)
        elif args.action == "verify":
            result = dict(passed=bool(verify_context(batch, args.root)), solver_started=False)
        elif args.action == "launch":
            verify_context(batch, args.root)
            result = base.launch(batch)
        elif args.action == "run":
            result = base.run(batch, only=args.jobs, retry_failed=args.retry_failed)
        elif args.action == "stop":
            result = task.unified.stop(batch)
        elif args.action == "status":
            result = base.existing.status_summary(batch)
        elif args.action == "report":
            result = report(batch)
        elif args.action == "audit":
            result = audit(batch, root=args.root, require_complete=args.require_complete)
        else:
            if not args.job_id or not args.attempt:
                parser.error("audit-worker requires --job-id and --attempt")
            result = base.audit_worker(batch, args.job_id, args.attempt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
