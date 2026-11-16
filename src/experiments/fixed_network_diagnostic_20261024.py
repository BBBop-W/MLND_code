"""Paired full-lifted assignment diagnostics on saved, distinct integer networks.

Selection and preparation never invoke a solver or generate an instance.
Only ``run``/``launch`` start the frozen probe under the existing resource guard.
The reported lower bounds concern a fixed network, never the network-design optimum.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import threading
import time

import numpy as np

from src.data.instances import to_planning_data
from src.data.standard import load_standard
from src.experiments import conditional_benchmark_20261022 as base
from src.experiments.storage import write_json
from src.model.paths import enumerate_paths


ROOT = base.ROOT
MODULE = "src.experiments.fixed_network_diagnostic_20261024"
DEFAULT_REFERENCE = ROOT / "outputs/branch_joint_benchmark_20261022"
DEFAULT_BATCH = ROOT / "outputs/fixed_network_diagnostic_20261024"
NETWORK_FIELDS = ("facility", "capacity_level", "airline")
MODES = ("integer", "relaxed")
NUMERICAL_PROFILES = ("original", "strict", "strict_no_presolve")


def _binary(value):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all() or np.any(np.abs(array - np.rint(array)) > 1e-6):
        raise ValueError("A saved integer network must contain finite binary decisions")
    array = np.rint(array).astype(int)
    if np.any((array < 0) | (array > 1)):
        raise ValueError("Binary decisions must be zero or one")
    return array


def extract_network(solution):
    """Discard assignment/operations and canonicalize saved network binaries."""
    values = {key: _binary(solution[key]) for key in NETWORK_FIELDS}
    n = values["facility"].size
    if (values["facility"].ndim != 1 or values["capacity_level"].ndim != 2
            or values["airline"].shape != (n, n)):
        raise ValueError("Network dimensions must be facility[N], capacity[H,Q], airline[N,N]")
    return {key: value.tolist() for key, value in values.items()}


def network_from_design(design, *, n_nodes, n_h, n_levels, n_od, n_l):
    """Decode the historical x,z,alpha,y layout, retaining x,z,y only."""
    vector = _binary(design)
    assignment = n_nodes + n_h * n_levels
    airline = assignment + n_od * n_l * n_l
    if vector.ndim != 1 or vector.size != airline + n_nodes * n_nodes:
        raise ValueError("Historical design length does not match the input dimensions")
    return extract_network(dict(facility=vector[:n_nodes],
        capacity_level=vector[n_nodes:assignment].reshape(n_h, n_levels),
        airline=vector[airline:].reshape(n_nodes, n_nodes)))


def network_fingerprint(network):
    normalized = extract_network(network)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def design_for_probe(network, *, n_od, n_l):
    """Use an all-courier alpha placeholder; the probe leaves alpha unfixed."""
    network = extract_network(network)
    return (np.asarray(network["facility"]).ravel().tolist()
            + np.asarray(network["capacity_level"]).ravel().tolist()
            + [0] * (n_od * n_l * n_l)
            + np.asarray(network["airline"]).ravel().tolist())


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def assignment_gap_interval(integer, relaxed):
    """Certified interval for z_integer(network)-z_relaxed(network)."""
    def endpoint(row, kind):
        value = row.get(kind + "_bound")
        flag = "lower_bound_certified" if kind == "lower" else "upper_bound_validated"
        return float(value) if row.get(flag) is True and _number(value) else None
    li, ui = endpoint(integer, "lower"), endpoint(integer, "upper")
    lc, uc = endpoint(relaxed, "lower"), endpoint(relaxed, "upper")
    tolerance = 1e-7 * max([1] + [abs(x) for x in (li, ui, lc, uc) if x is not None])
    errors = []
    for name, lower, upper in (("integer", li, ui), ("relaxed", lc, uc)):
        if lower is not None and upper is not None and lower > upper + tolerance:
            errors.append(f"{name}_lower_exceeds_upper")
    raw_lower = li - uc if li is not None and uc is not None else None
    raw_upper = ui - lc if ui is not None and lc is not None else None
    if raw_upper is not None and raw_upper < -tolerance:
        errors.append("integer_upper_below_relaxed_lower")
    lower = max(0.0, raw_lower) if raw_lower is not None else None
    upper = max(0.0, raw_upper) if raw_upper is not None and raw_upper >= -tolerance else None
    if lower is not None and upper is not None and lower > upper + tolerance:
        errors.append("assignment_gap_interval_reversed")
    return dict(lower=None if errors else lower, upper=None if errors else upper,
                valid=not errors, errors=errors, raw_lower=raw_lower, raw_upper=raw_upper,
                tolerance=tolerance, scope="fixed_network_assignment_integrality_gap")


def _dimensions(data):
    return dict(n_nodes=data.n_nodes, n_h=data.n_h, n_levels=data.slot_counts.shape[1],
                n_od=data.n_od, n_l=data.n_l)


def _validated_source(reference, case, variant):
    state = base.read(reference / "state.json")
    job_id = f"{case}_K2_{variant}"
    state = state["jobs"][job_id]
    if state["status"] in ("pending", "running"):
        raise ValueError(f"Reference not completed: {job_id}")
    directory = reference / state.get("result_directory", state["directory"])
    validation = base.read(directory / "native_validation.json")
    if validation.get("passed") is not True or validation.get("native_manifest_matches") is not True:
        raise ValueError(f"Source incumbent has not passed native validation: {job_id}")
    for name in ("result", "solution", "path_flows"):
        if validation.get(name + "_sha256") != base.sha256(directory / (name + ".json")):
            raise ValueError(f"Source hash changed: {job_id}/{name}")
    if validation.get("manifest_sha256") != base.sha256(reference / "instances" / case / "manifest.json"):
        raise ValueError(f"Source input mismatch: {job_id}")
    return directory, base.read(directory / "result.json"), base.read(directory / "solution.json")


def network_support(data, paths, network):
    """Read-only physical support and a closed singleton-pool service witness."""
    x, z, y = (np.asarray(network[key]) for key in NETWORK_FIELDS)
    slots = (z * data.slot_counts).sum(axis=1)
    supported = [p for p in paths.service_paths if x[p.b[0]] and x[p.b[1]]
                 and all(y[u, v] for u, v, count in p.arc_counts if count)
                 and all(slots[i] > 0 for i, duration in p.slot_times if duration > 0)]
    single = {p.segment_id: p for p in supported if p.second_od is None}
    self_double = {p.segment_id: p for p in supported if p.second_od == p.first_od}
    witness = None
    for segment, path in single.items():
        r = path.first_od
        if path.b[0] != path.b[1] or data.demand[r] <= 0:
            continue
        needs_double = data.effective_delta > 0 and data.kappa[r, r] > 0
        if needs_double and segment not in self_double:
            continue
        witness = dict(base_pair=list(path.b), od=r, segment=segment, single_path=path.id,
                       double_path=self_double[segment].id if needs_double else None,
                       interpretation="Positive arbitrarily small flow with this OD alone in a same-base pool; no optimality claim")
        break
    return dict(complete_domain=bool(paths.complete), full_service_paths=len(paths.service_paths),
                full_empty_paths=len(paths.empty_paths), supported_service_paths=len(supported),
                supported_single_paths=len(single), supported_double_paths=sum(p.second_od is not None for p in supported),
                positive_service_witness=witness)


def select_networks(reference=DEFAULT_REFERENCE):
    """Select recorded networks in a fixed order, without fitting to new outcomes."""
    reference = Path(reference).resolve()
    manifest = base.verify(reference)
    if manifest.get("schema") != "branch-joint-benchmark/1":
        raise ValueError("Selection requires the completed branch-joint reference")
    for name in ("src/model/paths.py", "src/model/resources.py", "src/model/types.py"):
        if base.sha256(ROOT / name) != base.sha256(reference / "frozen" / name):
            raise ValueError(f"Read-only path inspection differs from frozen reference: {name}")
    selected, notes = [], []
    for case in ("L01", "L02"):
        data = to_planning_data(load_standard(reference / "instances" / case, case="K2"))
        paths = enumerate_paths(data)
        directory, result, solution = _validated_source(reference, case, "lifted_branch_joint")
        first = extract_network(solution)
        support = network_support(data, paths, first)
        root = result["continuous_root"].get("applied_total_cost_lower_bound")
        selected.append(dict(id=f"{case}_N1", case=case, role="validated_lifted_incumbent", network=first,
            fingerprint=network_fingerprint(first), dimensions=_dimensions(data), support=support,
            source=dict(directory=str(directory), pointer="solution.json", variant="lifted_branch_joint",
                        validated_incumbent=True, objective=result["objective"])))
        candidates = []
        for index, item in enumerate(result.get("iterations", [])):
            if "design_key" not in item:
                continue
            candidate = network_from_design([int(x) for x in item["design_key"]], **_dimensions(data))
            cid = item.get("candidate_id", "")
            number = int(cid.split(":")[1]) if cid.startswith("integer:") else None
            master = result["master_solves"][number - 1] if number else {}
            objective = master.get("candidate_objective")
            at_floor = _number(root) and _number(objective) and abs(objective-root) <= 1e-6*max(1, abs(root))
            candidates.append((not at_floor, index, candidate,
                dict(directory=str(directory), pointer=f"result.json#/iterations/{index}/design_key",
                     variant="lifted_branch_joint", candidate_id=cid, historical_master_objective=objective,
                     historical_root_applied=root, historical_master_at_floor=at_floor,
                     historical_master_theta=item.get("master_theta"), validated_incumbent=False)))
        for ai, item in enumerate(result.get("acceleration_records", [])):
            for pi, pool in enumerate(item.get("design_pool", [])):
                candidates.append((True, len(candidates), network_from_design(pool["design"], **_dimensions(data)),
                    dict(directory=str(directory), pointer=f"result.json#/acceleration_records/{ai}/design_pool/{pi}/design",
                         variant="lifted_branch_joint", validated_incumbent=pool.get("validated") is True,
                         historical_master_at_floor=False, objective=pool.get("objective"))))
        second = None
        for _, _, network, source in sorted(candidates, key=lambda item: (item[0], item[1])):
            if network_fingerprint(network) == network_fingerprint(first):
                continue
            support = network_support(data, paths, network)
            if support["positive_service_witness"] is not None:
                second = network, source, support
                break
        if second is None:
            other_dir, other_result, other_solution = _validated_source(reference, case, "conditional_branch_joint")
            network = extract_network(other_solution)
            support = network_support(data, paths, network)
            if network_fingerprint(network) != network_fingerprint(first) and support["positive_service_witness"] is not None:
                second = network, dict(directory=str(other_dir), pointer="solution.json",
                    variant="conditional_branch_joint", validated_incumbent=True, objective=other_result["objective"],
                    fallback_reason="No different service-capable integer network saved in lifted design records; use same-batch validated conditional incumbent"), support
        if second is not None:
            network, source, support = second
            selected.append(dict(id=f"{case}_N2", case=case, role="recorded_alternative_network", network=network,
                fingerprint=network_fingerprint(network), dimensions=_dimensions(data), source=source, support=support))
        else:
            notes.append(f"{case}: only one distinct service-capable integer network could be recovered; no network was fabricated")
    return dict(reference=str(reference), reference_manifest_sha256=base.sha256(reference/"experiment_manifest.json"),
                selection_rule="Per case: validated lifted incumbent; earliest saved floor candidate with a positive-service witness; then saved lifted pool; then same-batch validated conditional incumbent. Network identity excludes alpha.",
                networks=selected, notes=notes, solver_started=False, input_generation=False)


def jobs_for(networks):
    return [dict(id=f"{network['id']}_{mode}", case=network["case"], network_id=network["id"],
                 network_fingerprint=network["fingerprint"], instance=f"instances/{network['case']}",
                 design=f"networks/{network['id']}.json", mode=mode, K=2,
                 method="full_lifted_fixed_network", output=f"runs/{network['id']}/{mode}")
            for network in networks for mode in MODES]


def command_for(batch, manifest, job, attempt):
    batch, attempt = Path(batch), Path(attempt)
    command = [str(batch / "frozen/bin/fixed_network_probe"), str(batch / job["instance"]),
            str(job["K"]), str(batch / job["design"]), job["mode"], str(attempt / "result.json"),
            str(manifest["protocol"]["seconds_per_job"])]
    profile = manifest["protocol"].get("numerical_profile", "original")
    if profile != "original":
        command.append(profile)
    return command


def prepare(batch, *, binary, reference=DEFAULT_REFERENCE, probe_source=None,
            numerical_profile="original", only=None):
    batch, reference = Path(batch).resolve(), Path(reference).resolve()
    if batch.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {batch}")
    if numerical_profile not in NUMERICAL_PROFILES:
        raise ValueError(f"Numerical profile must be one of {NUMERICAL_PROFILES}")
    selection = select_networks(reference)
    jobs = jobs_for(selection["networks"])
    if only:
        chosen = set(only)
        if chosen - {job["id"] for job in jobs}:
            raise ValueError("Unknown prepare job selection")
        jobs = [job for job in jobs if job["id"] in chosen]
    base.prepare(batch, [reference / "instances" / case for case in ("L01", "L02")],
                 binary=binary, seconds=600, threads=1, parallel=4, gap=1e-6, seed=43,
                 job_rss_gib=4, batch_rss_gib=8, system_available_gib=2, hard_wall_seconds=720)
    manifest = base.read(batch / "experiment_manifest.json")
    old_binary = batch / "frozen/bin/mlnd_cpp"
    old_binary.rename(batch / "frozen/bin/fixed_network_probe")
    manifest["schema"] = "fixed-network-diagnostic/1"
    manifest["protocol"].update(full_lifted_model=True, tight_airline_bounds=True,
        bound_scope="fixed_network_only", fixed_network_fields=list(NETWORK_FIELDS), assignment_variable=True,
        compared_modes=list(MODES), source_assignment_used_as_start=False,
        budget_scope="input read, full path enumeration, model construction, optimization and certificates within 600 seconds",
        objective_scope="total cost includes fixed network cost exactly once",
        existing_global_bound_used_as_fixed_network_cut=False, numerical_profile=numerical_profile)
    manifest["jobs"] = jobs
    manifest["selection"] = selection
    (batch / "reference").mkdir()
    shutil.copy2(reference / "experiment_manifest.json", batch / "reference/experiment_manifest.json")
    for network in selection["networks"]:
        write_json(batch / "networks" / (network["id"] + ".json"),
            dict(design=design_for_probe(network["network"], n_od=network["dimensions"]["n_od"], n_l=network["dimensions"]["n_l"]),
                 network=network["network"], network_fingerprint=network["fingerprint"],
                 alpha_placeholder="zero; not fixed, not copied from source, not a warm start"))
        directory = Path(network["source"]["directory"])
        copied = batch / "reference" / network["id"]
        copied.mkdir()
        for name in ("result.json", "solution.json", "native_validation.json", "path_flows.json"):
            shutil.copy2(directory / name, copied / name)
        network["source"]["snapshot"] = str(copied.relative_to(batch))
        network["source"]["sha256"] = {p.name:base.sha256(p) for p in copied.iterdir()}
    if probe_source:
        source = Path(probe_source).resolve()
        shutil.copy2(source, batch / "frozen/bin" / source.name)
        manifest["probe_source"] = dict(source=str(source), copied=f"frozen/bin/{source.name}", sha256=base.sha256(source))
    write_json(batch / "provenance.json", selection)
    manifest["frozen_sha256"] = {str(path.relative_to(batch)):base.sha256(path)
        for folder in ("frozen", "instances", "networks", "reference")
        for path in sorted((batch / folder).rglob("*")) if path.is_file()}
    manifest["frozen_sha256"]["provenance.json"] = base.sha256(batch / "provenance.json")
    write_json(batch / "experiment_manifest.json", manifest)
    write_json(batch / "state.json", dict(status="prepared_not_started", updated_utc=base.existing.utc(),
        jobs={job["id"]:dict(status="pending", attempts=[]) for job in manifest["jobs"]}))
    (batch / "run.py").write_text("from pathlib import Path\nimport sys\n"
        "batch=Path(__file__).resolve().parent\nsys.path.insert(0,str(batch/'frozen'))\n"
        f"from {MODULE} import main\nmain([*sys.argv[1:], '--batch', str(batch)])\n", encoding="utf-8")
    report(batch)
    return dict(batch=str(batch), networks=len(selection["networks"]), jobs=len(manifest["jobs"]),
                solver_started=False, input_generation=False)


def summarize_result(result, execution, job):
    """Only explicit validated/certified outputs become displayed bounds."""
    catalogue, model, solver = (result.get(key) or {} for key in ("catalogue", "model", "solver"))
    timings, settings = result.get("timings") or {}, result.get("settings") or {}
    errors = []
    if result:
        for name, actual, expected in (("scope", result.get("scope"), "conditional_on_fixed_network"),
                ("mode", result.get("mode"), job["mode"]), ("K", result.get("K"), job.get("K", 2)),
                ("tight_airline_bounds", settings.get("tight_airline_bounds"), True),
                ("conditional_demand", settings.get("conditional_demand"), False),
                ("extra_strengthening", settings.get("extra_strengthening"), False),
                ("assignment_incident_clique", settings.get("assignment_incident_clique"), False),
                ("shared_rho_joint", settings.get("shared_rho_joint"), False),
                ("queue_perspective", settings.get("queue_perspective"), False)):
            if actual != expected:
                errors.append(f"{name}_mismatch")
    domain_ok = catalogue.get("complete") is True and catalogue.get("support_complete") is True
    validation = result.get("validation") or {}
    selected_valid = (validation.get("selected_candidate") or {}).get("passed") is True
    start_valid = ((result.get("warm_start") or {}).get("validation") or {}).get("passed") is True
    upper_ok = (not errors and domain_ok and (selected_valid or start_valid)
                and result.get("upper_bound_validated") is True and _number(result.get("total_upper_bound")))
    certificate = result.get("bound_certificate") or {}
    status_eligible = job["mode"] == "integer" or (solver.get("optimal") is True and solver.get("status") == "Optimal")
    lower_ok = (not errors and domain_ok and status_eligible and result.get("lower_bound_certified") is True
                and certificate.get("valid") is True and certificate.get("scope") == "conditional_on_fixed_network"
                and _number(result.get("total_lower_bound")))
    upper = result.get("total_upper_bound") if upper_ok else None
    lower = result.get("total_lower_bound") if lower_ok else None
    guard = execution.get("resource_guard") or {}
    row = dict(job=job["id"], case=job["case"], network_id=job["network_id"], mode=job["mode"],
        network_fingerprint=job["network_fingerprint"], upper_bound=upper, lower_bound=lower,
        upper_bound_validated=upper_ok, lower_bound_certified=lower_ok,
        gap=max(0, upper-lower)/max(1, abs(upper)) if upper_ok and lower_ok else None,
        raw_objective=solver.get("objective"), raw_bound=solver.get("raw_bound"),
        raw_status=result.get("status"), termination_reason=result.get("termination"),
        process_exit=execution.get("returncode"), stop_reason=execution.get("stop_reason"),
        resource_limit_kind=execution.get("resource_limit_kind"), peak_rss_bytes=guard.get("peak_sampled_rss_bytes"),
        external_wall_seconds=guard.get("external_wall_seconds"), bound_scope="fixed_network_only", evidence_errors=errors,
        elapsed_seconds=timings.get("total_seconds"), optimization_seconds=timings.get("solver_seconds"),
        construction_seconds=sum(timings.get(key, 0) for key in ("input_seconds", "enumeration_seconds", "formulation_seconds", "solver_construction_seconds")) if timings else None,
        catalogue_complete=catalogue.get("complete"), full_service_paths=catalogue.get("service_paths"),
        full_empty_paths=catalogue.get("empty_paths"), full_paths=catalogue.get("total_paths"),
        initial_segments=catalogue.get("initial_segments"), path_scope=catalogue.get("scope"),
        model_variables=model.get("variables"), model_binary_variables=model.get("binary_variables"),
        model_linear_rows=model.get("linear_constraints"), model_cones=model.get("quadratic_constraints"),
        solver_optimal=solver.get("optimal"), optimal_certificate=result.get("optimal_certificate"))
    for key in ("fixed_cost", "operating_upper_bound", "operating_lower_bound", "validation", "settings"):
        row[key] = result.get(key)
    return row


def audit_worker(batch, job_id, attempt):
    """Read saved candidates in a guarded process, never call an optimizer."""
    from src.operations.validation import build_operations
    batch, attempt = Path(batch).resolve(), Path(attempt).resolve()
    manifest = base.read(batch / "experiment_manifest.json")
    job = next(item for item in manifest["jobs"] if item["id"] == job_id)
    result = base.read(attempt / "result.json")
    design_file = batch / job["design"]
    network = base.read(design_file)["network"]
    errors = []
    expected = command_for(batch, manifest, job, attempt)
    if base.read(attempt / "job.json")["command"] != expected:
        errors.append("actual_command_does_not_match_frozen_protocol")
    provenance = result.get("provenance") or {}
    if provenance.get("design_sha256") != base.sha256(design_file):
        errors.append("design_sha256_mismatch")
    expected_vector = [int(v) for key in NETWORK_FIELDS for v in np.asarray(network[key]).ravel()]
    if provenance.get("fixed_network_vector") != expected_vector:
        errors.append("fixed_network_vector_mismatch")
    if result.get("input") != str(batch / job["instance"]):
        errors.append("input_path_mismatch")
    if provenance.get("input_manifest") != base.read(batch / job["instance"] / "manifest.json"):
        errors.append("input_manifest_mismatch")
    if provenance.get("source_assignment_ignored") is not True:
        errors.append("source_assignment_not_ignored")
    settings = result.get("settings") or {}
    for key, expected_setting in (("seconds",600),("threads",1),("seed",43),("gap",1e-6)):
        if settings.get(key) != expected_setting:
            errors.append(f"protocol_{key}_mismatch")
    if settings.get("numerical_profile", "original") != manifest["protocol"].get("numerical_profile", "original"):
        errors.append("protocol_numerical_profile_mismatch")
    preliminary = summarize_result(result, {}, job)
    errors += preliminary["evidence_errors"]
    audit = dict(passed=False, mode=job["mode"], result_sha256=base.sha256(attempt / "result.json"),
                 design_sha256=base.sha256(design_file), scope="saved_fixed_network_result_identity_and_candidate_validation")
    values_path = attempt / "result.json.values.json"
    if values_path.is_file():
        values = base.read(values_path)
        selected = dict(zip(values["variable_names"], values["selected_values"], strict=True))
        for key in NETWORK_FIELDS:
            array = np.asarray(network[key])
            for index in np.ndindex(array.shape):
                name = key + "_" + "_".join(map(str, index))
                if name not in selected or abs(selected[name] - array[index]) > 1e-6:
                    errors.append("saved_candidate_fixed_network_mismatch")
                    break
        audit["values_sha256"] = base.sha256(values_path)
    else:
        audit["values_unavailable"] = True
        if result.get("status") == "completed":
            errors.append("completed_result_missing_saved_values")
    data = load_standard(batch / job["instance"], case="K2")
    planning = to_planning_data(data)
    fixed_cost = float(np.dot(network["facility"][:planning.n_l],planning.base_cost)
        +np.sum(np.asarray(network["capacity_level"])*planning.slot_cost)
        +np.sum(np.asarray(network["airline"])*planning.link_cost))
    audit["independent_fixed_cost"] = fixed_cost
    if _number(result.get("fixed_cost")):
        if abs(result["fixed_cost"]-fixed_cost) > 1e-7*max(1,abs(fixed_cost)):
            errors.append("fixed_cost_mismatch")
        for kind in ("upper", "lower"):
            total, operating = result.get(f"total_{kind}_bound"), result.get(f"operating_{kind}_bound")
            if _number(total) and _number(operating) and abs(total-fixed_cost-operating) > 1e-7*max(1,abs(total)):
                errors.append(f"{kind}_cost_decomposition_mismatch")
    if job["mode"] == "integer" and preliminary["upper_bound_validated"]:
        solution_path = attempt / "result.json.solution.json"
        if solution_path.is_file():
            package = base.read(solution_path)
            checks = build_operations(data, package)["summary"]["checks"]
            audit["independent_operations_checks"] = checks
            audit["solution_sha256"] = base.sha256(solution_path)
            if checks.get("passed") is not True:
                errors.append("independent_integer_operations_validation_failed")
            if network_fingerprint(package["_solution"]) != job["network_fingerprint"]:
                errors.append("integer_package_network_mismatch")
        elif result.get("status") == "completed":
            errors.append("completed_integer_result_missing_solution_package")
    audit.update(errors=errors, passed=not errors)
    write_json(attempt / "artifact_audit.json", audit)
    return audit


def report(batch):
    batch = Path(batch).resolve()
    manifest = base.verify(batch)
    state = base.read(batch / "state.json")
    rows = []
    for job in manifest["jobs"]:
        status = state["jobs"][job["id"]]
        attempt = batch / status.get("directory", "_missing")
        row = summarize_result(base.optional_read(attempt / "result.json"), base.optional_read(attempt / "execution.json"), job)
        row["scheduler_status"] = status["status"]
        row["result_sha256"] = base.sha256(attempt / "result.json") if (attempt / "result.json").is_file() else None
        row["command_matches_frozen"] = (base.optional_read(attempt / "job.json").get("command")
                                          == command_for(batch, manifest, job, attempt)) if attempt.exists() else None
        audit = base.optional_read(attempt / "artifact_audit.json")
        audit_valid = audit.get("passed") is True and audit.get("result_sha256") == row["result_sha256"]
        for key, suffix in (("values_sha256","values"),("solution_sha256","solution")):
            if audit.get(key):
                path = attempt / f"result.json.{suffix}.json"
                audit_valid = audit_valid and path.is_file() and audit[key] == base.sha256(path)
        row["artifact_audit_passed"] = audit_valid
        row["artifact_audit_errors"] = audit.get("errors")
        if not audit_valid or row["command_matches_frozen"] is not True:
            row.update(lower_bound=None, upper_bound=None, lower_bound_certified=False, upper_bound_validated=False, gap=None)
        rows.append(row)
    intervals = []
    for network in manifest["selection"]["networks"]:
        paired = {row["mode"]:row for row in rows if row["network_id"] == network["id"]}
        if all(mode in paired for mode in MODES):
            intervals.append(dict(network_id=network["id"], **assignment_gap_interval(paired["integer"], paired["relaxed"])))
    payload = dict(protocol=manifest["protocol"], provenance=manifest["selection"], rows=rows,
                   assignment_gap_intervals=intervals, updated_utc=base.existing.utc())
    write_json(batch / "analysis/comparison.json", payload)
    for name, records in (("comparison", rows), ("assignment_gap", intervals)):
        fields = list(dict.fromkeys(key for row in records for key in row))
        temporary = batch / "analysis" / (name + ".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k:json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k,v in row.items()} for row in records)
        temporary.replace(batch / "analysis" / (name + ".csv"))
    def fmt(x):
        return "—" if x is None else f"{x:,.6f}" if isinstance(x, float) else str(x)
    lines = ["# 固定网络下的分配整数性诊断", "", "固定设施、格口等级和航线，只比较订单分配为二元变量或连续变量；其余运营变量均重新优化。",
        "完整 lifted 模型、原始路径与运营假设保持一致。每组含构造 600 秒，单线程；最多四组并发。",
        "此处 LB 仅属于给定网络的问题，不能替代原网络选址问题的全局下界。", "", "## 网络来源", ""]
    lines.insert(5, f"数值配置：`{manifest['protocol'].get('numerical_profile', 'original')}`。")
    active_networks = {job["network_id"] for job in manifest["jobs"]}
    for network in manifest["selection"]["networks"]:
        if network["id"] not in active_networks:
            continue
        src, support = network["source"], network["support"]
        lines += [f"- **{network['id']}**：{network['role']}；来源 `{src['directory']}/{src['pointer']}`。",
            f"  网络指纹 `{network['fingerprint']}`；支持服务路径 {support['supported_service_paths']} 条；"
            f"单订单池正服务见证：`{support['positive_service_witness']}`。"]
        if src.get("fallback_reason"):
            lines += ["  lifted 日志未保存可恢复的另一套整数网络，采用同批次 conditional 的已验证最终网络；来源算法不改变本轮完整 lifted 比较的定义。"]
    lines += manifest["selection"]["notes"]
    lines += ["", "## 同一网络上的结果", "", "| 网络 | 分配 | 状态 | 已验证 UB | 认证 LB | gap | 总耗时 s | 退出码 | 资源原因 |",
              "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        values = [row["network_id"],row["mode"],row["termination_reason"] or row["scheduler_status"],row["upper_bound"],
            row["lower_bound"],None if row["gap"] is None else f"{row['gap']:.3%}",row.get("elapsed_seconds"),row["process_exit"],row["resource_limit_kind"]]
        lines.append("| "+" | ".join(map(fmt, values))+" |")
    lines += ["", "## 分配整数性差距", "", "以同一网络下整数最优值减连续松弛最优值为差距；固定成本在两者之间抵消。",
        "认证区间为 `[max(0, LI−UC), UI−LC]`，只使用已验证上界与认证下界。缺失端点保持空缺，未求完不等于无差距。",
        "", "| 网络 | 差距下界 | 差距上界 | 证据一致 | 诊断 |", "|---|---:|---:|---|---|"]
    for item in intervals:
        lines.append("| "+" | ".join(fmt(item[key]) for key in ("network_id","lower","upper","valid","errors"))+" |")
    if not intervals:
        lines += ["", "本批次仅含所选单侧任务，不独立构造整数性差距区间；与原批次的配对比较须明确记录来源和数值配置。"]
    lines += ["", "## 路径与模型规模", "", "| 网络 | 分配 | 完整服务/空飞路径 | 变量 | 二元变量 | 线性行 | 锥 | 固定费用 |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["network_id"], row["mode"], f"{fmt(row['full_service_paths'])}/{fmt(row['full_empty_paths'])}",
                  row["model_variables"], row["model_binary_variables"], row["model_linear_rows"],row["model_cones"],row["fixed_cost"]]
        lines.append("| "+" | ".join(map(fmt, values))+" |")
    lines += ["", "连续模式验证完整矩阵与原锥约束；不会将分数分配交给强制二元分配的运营导出器。",
        "数值失败、构造中断、资源终止和缺失证书均保留原状态，不自动记作最优或将缺失下界记作零。",
        "", "[结果 CSV](comparison.csv) / [完整 JSON](comparison.json) / [差距区间 CSV](assignment_gap.csv)。",
        "输入、网络、源码、可执行程序及参考证据的哈希见 `../experiment_manifest.json`。"]
    (batch / "analysis/REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return dict(report=str(batch / "analysis/REPORT.md"), jobs=len(rows),
                completed=sum(row["scheduler_status"] == "completed" for row in rows))


def run(batch, *, only=None):
    batch = Path(batch).resolve()
    manifest = base.verify(batch)
    cancel, lock = threading.Event(), threading.Lock()
    with base.existing.batch_lock(batch):
        state = base.read(batch / "state.json")
        if any(item["status"] == "running" for item in state["jobs"].values()):
            raise RuntimeError("Existing running state must be reconciled before resuming")
        selected = set(only or state["jobs"])
        if selected - state["jobs"].keys():
            raise ValueError("Unknown job selection")
        def save():
            state["updated_utc"] = base.existing.utc()
            write_json(batch / "state.json", state)
            report(batch)
        def one(job):
            if cancel.is_set():
                return
            with lock:
                item = state["jobs"][job["id"]]
                relative = f"{job['output']}/attempts/{len(item['attempts'])+1:03d}"
                attempt = batch / relative
                attempt.mkdir(parents=True)
                record = dict(directory=relative, status="running", started_utc=base.existing.utc())
                item["attempts"].append(record)
                item.update(status="running", directory=relative)
                save()
            command = command_for(batch, manifest, job, attempt)
            write_json(attempt / "job.json", dict(job=job, command=command,
                experiment_manifest_sha256=base.sha256(batch / "experiment_manifest.json")))
            print(json.dumps(dict(event="job_started",job=job["id"],utc=base.existing.utc())), flush=True)
            started = time.monotonic()
            try:
                execution = base.guarded_execute(batch, manifest, job["id"], command, attempt, cancel,
                                                stage_directory=attempt / "result.json.diagnostics")
                result = base.optional_read(attempt / "result.json")
                remaining = manifest["protocol"]["hard_wall_seconds"] - (time.monotonic() - started)
                if result and not cancel.is_set() and remaining > 1:
                    postprocess = attempt / "postprocess"
                    postprocess.mkdir()
                    audit_command = [manifest["python"], "-m", MODULE, "audit-worker", "--batch", str(batch),
                                     "--job-id", job["id"], "--attempt", str(attempt)]
                    checked = base.guarded_execute(batch,manifest,job["id"]+"_audit",audit_command,postprocess,
                        cancel,remaining=remaining,initial_phase="validation")
                    audit = base.optional_read(attempt / "artifact_audit.json")
                    audit_ok = audit.get("passed") is True and checked.get("returncode") == 0 and not checked.get("stop_reason")
                else:
                    audit_ok = False
                status = ("interrupted" if cancel.is_set() else "failed" if execution.get("returncode") != 0
                          or execution.get("stop_reason") or not result else "construction_stopped"
                          if result.get("status") in ("construction_stopped", "construction_time_limit") else
                          "completed" if result.get("status") == "completed" and audit_ok else "failed")
            except Exception as exc:
                status = "failed"
                write_json(attempt / "dispatcher_failure.json", dict(error=f"{type(exc).__name__}: {exc}"))
            with lock:
                record.update(status=status, finished_utc=base.existing.utc())
                item.update(status=status)
                save()
            print(json.dumps(dict(event="job_finished",job=job["id"],status=status,utc=base.existing.utc())), flush=True)
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, lambda *_:cancel.set())
        state["status"] = "running"
        save()
        try:
            jobs = [job for job in manifest["jobs"] if job["id"] in selected and state["jobs"][job["id"]]["status"] == "pending"]
            with ThreadPoolExecutor(max_workers=min(4, manifest["protocol"]["parallel_jobs"])) as pool:
                list(pool.map(one, jobs))
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            state["status"] = "interrupted" if cancel.is_set() else "finished_dispatch"
            save()
    return report(batch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "prepare", "verify", "launch", "run", "report", "audit-worker"))
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--probe-source", type=Path)
    parser.add_argument("--numerical-profile", choices=NUMERICAL_PROFILES, default="original")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--job-id")
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args(argv)
    if args.action == "select":
        value = select_networks(args.reference)
    elif args.action == "prepare":
        if not args.binary:
            parser.error("prepare requires the reviewed --binary probe")
        value = prepare(args.batch,binary=args.binary,reference=args.reference,probe_source=args.probe_source,
                        numerical_profile=args.numerical_profile,only=args.only)
    elif args.action == "verify":
        value = dict(verified=True,jobs=len(base.verify(args.batch)["jobs"]))
    elif args.action == "launch":
        value = base.launch(args.batch)
    elif args.action == "run":
        value = run(args.batch,only=args.only)
    elif args.action == "audit-worker":
        value = audit_worker(args.batch,args.job_id,args.attempt)
    else:
        value = report(args.batch)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
