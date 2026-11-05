"""Audit saved task-cost benchmark artifacts without invoking any solver.

Usage: python -m src.experiments.task_cost_audit --batch OUTPUT [--require-complete]
The audit is dimension-independent, includes all return bases and empty moves,
and never compares a cut with a point from a different instance or task mode.
It writes separate audit/paired-comparison artifacts and preserves REPORT.md.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dec(value):
    return Decimal.from_float(float(value))


def table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def fail(errors, condition, message):
    if not condition:
        errors.append(message)


def catalogue_digest(catalogue):
    digest = hashlib.sha256()
    for key in ("missions", "relocations"):
        digest.update(key.encode())
        for item in catalogue[key]:
            digest.update(json.dumps(item["path"], sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verified_payloads(folder, expected=None):
    """Follow saved validation hashes before trusting a catalogue or witness."""
    folder = Path(folder)
    expected = expected or {}
    for name, field in (("result.json", "result_sha256"),
                        ("native_validation.json", "native_validation_sha256"),
                        ("artifact_audit.json", "artifact_audit_sha256")):
        if expected:
            _require(expected.get(field) == sha(folder / name),
                     f"Historical catalogue {name} hash mismatch")
    native, artifact = read(folder / "native_validation.json"), read(folder / "artifact_audit.json")
    _require(native.get("passed") and artifact.get("passed"),
             f"Catalogue source is not validated: {folder}")
    for name in ("result.json", "solution.json", "path_flows.json",
                 "input_reference.json", "preprocessed_demand.json", "native_validation.json"):
        _require(artifact.get("artifact_sha256", {}).get(name) == sha(folder / name),
                 f"Artifact hash mismatch: {folder / name}")
    for name, field in (("result.json", "result_sha256"),
                        ("solution.json", "solution_sha256"),
                        ("path_flows.json", "path_flows_sha256")):
        _require(native.get(field) == sha(folder / name), f"Native validation hash mismatch: {folder / name}")
    return {name: read(folder / (name + ".json")) for name in
            ("result", "solution", "path_flows", "input_reference", "preprocessed_demand")}, native


def _record_index(records, count, label, *, complete):
    index = {}
    for item in records:
        path = item["path"]
        identifier = path.get("id")
        _require(type(identifier) is int and 0 <= identifier < count,
                 label + ": invalid global path id")
        _require(identifier not in index, label + ": duplicate global path id")
        index[identifier] = item
    if complete:
        _require(len(index) == count, label + ": incomplete full catalogue")
    return index


def _check_record_coverage(payload, full_catalogue):
    """Check explicit global IDs, exact paths and every nonzero stored flow."""
    result, solution, records = (payload[k] for k in ("result", "solution", "path_flows"))
    counts = result["catalogue"]
    coverage = {}
    for family, kind, vector_key in (("missions", "service", "flow"),
                                      ("relocations", "empty", "empty_flow")):
        count = counts[kind]
        full = _record_index(full_catalogue[family], count, kind + " reference", complete=True)
        saved = _record_index(records[family], count, kind + " current", complete=False)
        vector = solution[vector_key]
        _require(len(vector) == count, kind + ": full flow vector dimension mismatch")
        _require(all(math.isfinite(value) for value in vector), kind + ": nonfinite flow vector")
        for identifier, item in saved.items():
            _require(item["path"] == full[identifier]["path"],
                     kind + ": path differs from full catalogue at id " + str(identifier))
            _require(math.isfinite(item["flow"]) and item["flow"] == vector[identifier],
                     kind + ": record/vector flow mismatch at id " + str(identifier))
        nonzero = {i for i, value in enumerate(vector) if value != 0}
        _require(nonzero <= saved.keys(), kind + ": omitted nonzero flow records")
        declared = result.get("path_record_counts", {})
        if declared:
            _require(declared.get(kind) == len(saved), kind + ": record count metadata mismatch")
        coverage[kind] = dict(full_count=count, saved_count=len(saved), nonzero_count=len(nonzero))
    # Segment witnesses contain no flow, but all segment IDs and physical data
    # are retained by the portable compact format.
    expected_segments = {p["id"]: p for p in full_catalogue["segments"]}
    current_segments = {p["id"]: p for p in records["segments"]}
    _require(len(expected_segments) == counts["segments"]
             and len(current_segments) == len(records["segments"])
             and current_segments == expected_segments,
             "Segment catalogue mismatch")
    return coverage


def resolve_full_catalogue(batch, key, folder, *, cache=None):
    """Resolve dense physical constants separately from current flow witnesses.

    Compact output is not a full extremum certificate.  Only a pinned, validated
    dense reference with identical input, K and reconstructed task rates can
    supply the missing zero-flow paths.  No solver or reconstruction is run.
    """
    batch, folder = Path(batch), Path(folder)
    current, native = _verified_payloads(folder)
    result, reference = current["result"], current["input_reference"]
    instance = batch / key[0]
    manifest_hash = sha(instance / "manifest.json")
    _require(reference["manifest"] == read(instance / "manifest.json")
             and native.get("manifest_sha256") == manifest_hash
             and result.get("input_manifest_sha256") == manifest_hash,
             "Current catalogue input manifest mismatch")
    _require(reference["max_orders"] == key[1] == result["max_orders"], "Current catalogue K mismatch")
    _require(result.get("catalogue_complete") and result.get("support_complete"),
             "Current physical catalogue is incomplete")
    records, counts = current["path_flows"], result["catalogue"]
    dense = (len(records["missions"]) == counts["service"]
             and len(records["relocations"]) == counts["empty"])
    source = folder
    provenance = dict(resolution="local_full_catalogue", source_directory=str(source))
    if not dense:
        _require(result.get("path_record_storage") == "nonzero_and_statistics_witnesses",
                 "Incomplete path records lack explicit compact storage metadata")
        manifest = read(batch / "experiment_manifest.json")
        name = manifest.get("protocol", {}).get("historical_full_cplex_reference")
        _require(name is not None, "Compact audit requires a verified full catalogue reference")
        _require(manifest.get("frozen_sha256", {}).get(name) == sha(batch / name),
                 "Full catalogue reference provenance is not frozen or has changed")
        candidates = [row for row in read(batch / name)["rows"]
                      if row.get("instance") in (str(key[0]), Path(key[0]).name) and row.get("K") == key[1]]
        _require(len(candidates) == 1, "Full catalogue reference must uniquely identify the instance and K")
        evidence = candidates[0]
        _require(evidence.get("validation_passed") is True, "Historical catalogue reference is not validated")
        source = Path(evidence["source_directory"])
        if not source.is_absolute():
            source = batch / source
        cache_key = (str(source.resolve()), evidence["result_sha256"],
                     evidence["native_validation_sha256"], evidence["artifact_audit_sha256"])
        if cache is not None and cache_key in cache:
            historic, historic_native = cache[cache_key]
        else:
            historic, historic_native = _verified_payloads(source, evidence)
            if cache is not None:
                cache[cache_key] = historic, historic_native
        old_result, old_ref = historic["result"], historic["input_reference"]
        _require(evidence.get("input_manifest_sha256") == manifest_hash
                 == historic_native.get("manifest_sha256") == old_result.get("input_manifest_sha256"),
                 "Historical catalogue input manifest hash mismatch")
        for field in ("manifest", "model", "max_orders", "assignment_scope", "delivery_time_scope", "demand_formulation", "first_leg_consistency"):
            _require(old_ref.get(field) == reference.get(field), "Historical catalogue input mismatch: " + field)
        _require(old_result.get("max_orders") == key[1]
                 and old_result.get("catalogue_complete") and old_result.get("support_complete"),
                 "Historical physical catalogue is incomplete or has different K")
        _require(old_result.get("catalogue") == counts
                 and evidence.get("all_service_paths") == counts["service"]
                 and evidence.get("all_empty_paths") == counts["empty"],
                 "Historical full catalogue size mismatch")
        for field in ("single_rates", "pair_rates", "head_rates"):
            _require(historic["preprocessed_demand"][field] == current["preprocessed_demand"][field],
                     "Historical catalogue task demand mismatch: " + field)
        records = historic["path_flows"]
        _check_record_coverage(historic, records)  # Requires every full global ID.
        provenance = dict(resolution="verified_historical_full_catalogue", source_directory=str(source),
            reference_file=name, reference_sha256=sha(batch / name),
            result_sha256=evidence["result_sha256"], native_validation_sha256=evidence["native_validation_sha256"],
            artifact_audit_sha256=evidence["artifact_audit_sha256"], path_flows_sha256=sha(source / "path_flows.json"))
    provenance["current_record_coverage"] = _check_record_coverage(current, records)
    provenance["full_catalogue_digest"] = catalogue_digest(records)
    return records, provenance


def design(solution):
    return (solution["facility"] + [x for row in solution["capacity_level"] for x in row]
            + [x for row in solution["assignment"] for x in row]
            + [x for row in solution["airline"] for x in row])


def row_violation(row, point, theta):
    return row["rhs"] - math.fsum(c * (theta if j == len(point) else point[j])
                                   for j, c in row["coefficients"])


def assignment_groups(model, demand, *, assignment_scope=None, max_orders=None):
    """Return stable entity -> physical tasks; historical origin artifacts stay readable."""
    scope = assignment_scope or demand.get("assignment_scope") or model.get("assignment_scope") or model["values"].get("assignment_scope", "origin")
    _require(scope in {"origin", "task"}, "Unsupported fixed-demand assignment scope")
    R = model["values"]["n_od"]
    _require(len(demand["single_rates"]) == R and len(demand["pair_rates"]) == R
             and all(len(row) == R for row in demand["pair_rates"]), "Task demand dimension mismatch")
    if scope == "origin":
        return scope, [[(r, None, demand["single_rates"][r]),
                        *((r, s, rate) for s, rate in enumerate(demand["pair_rates"][r]))] for r in range(R)]
    K = max_orders if max_orders is not None else demand.get("max_task_orders", model["values"].get("max_orders"))
    _require(K in {1, 2}, "Independent-task audit requires explicit max_orders")
    _require(K == 2 or not any(rate for row in demand["pair_rates"] for rate in row),
             "K1 task domain contains positive pair rates")
    groups = [[(r, None, demand["single_rates"][r])] for r in range(R)]
    if K == 2:
        groups.extend([(r, s, demand["pair_rates"][r][s])] for r in range(R) for s in range(R))
    return scope, groups


def physical_constants(model, demand, catalogue, instance, *, assignment_scope=None, max_orders=None):
    """Independent complete-domain minima and arc maxima, in Decimal arithmetic."""
    ids, values = model["ids"], model["values"]
    R, L, H = values["n_od"], values["n_l"], values["n_h"]
    scope, groups = assignment_groups(model, demand, assignment_scope=assignment_scope, max_orders=max_orders)
    Q = len(groups)
    N, G, S = L + H, Q * L, len(ids["levels"])
    ground_rows = {row["od_id"]: row for row in table(instance / "coefficients/od_costs.csv")}
    ground = [dec(ground_rows[str(r)]["ground_cost"]) for r in ids["ods"]]
    rate_rows = {row["od_id"]: row for row in table(instance / "od_demand.csv")}
    raw = [dec(rate_rows[str(r)]["lambda_per_hour"]) for r in ids["ods"]]
    cap_rows = {(row["hub_id"], row["level_id"]): row
                for row in table(instance / "coefficients/locker_levels.csv")}
    capacity = [dec(cap_rows[h, str(s)]["occupancy_limit"]) for h in ids["hubs"] for s in ids["levels"]]
    names = {name: i for i, name in enumerate(ids["flight_nodes"])}
    allowed = [False] * (N * N)
    for row in table(instance / "coefficients/links.csv"):
        allowed[names[row["from_site"]] * N + names[row["to_site"]]] = row["allowed_links"] == "1"
    F = dec(values["fleet_cost"])
    branches = {}
    domain = {(r, second) for tasks in groups for r, second, rate in tasks}
    same_od = cross_return = repeated_arc = 0
    for item in catalogue["missions"]:
        p = item["path"]
        key = p["first_od"], p["b"][0], None if p["second_od"] in (None, -1) else p["second_od"]
        _require((key[0], key[2]) in domain and 0 <= key[1] < L, "Physical path lies outside assignment task domain")
        load = [Decimal(0)] * H
        for h, time in p["slot_times"]:
            load[h] += dec(time)
        cost, total = dec(p["cost"]) + F * dec(p["time"]), sum(load)
        arcs = Counter()
        for u, v, count in p["arc_counts"]:
            arcs[u * N + v] += count
        old = branches.get(key)
        if old is None:
            branches[key] = dict(cost=cost, total=total, load=load, arcs=dict(arcs))
        else:
            old["cost"] = min(old["cost"], cost)
            old["total"] = min(old["total"], total)
            old["load"] = [min(a, b) for a, b in zip(old["load"], load)]
            for e, count in arcs.items():
                old["arcs"][e] = max(old["arcs"].get(e, 0), count)
        same_od += p["first_od"] == p["second_od"]
        cross_return += p["b"][0] != p["b"][1]
        repeated_arc += sum(count > 1 for count in arcs.values())
    savings, totals, loads, supported = [], [], [], []
    arc_load = [defaultdict(lambda: Decimal(0)) for _ in range(G)]
    for q, tasks in enumerate(groups):
        for l in range(L):
            saving, total, load, ok = Decimal(0), Decimal(0), [Decimal(0)] * H, True
            for r, second, rate in tasks:
                if rate <= 0:
                    continue
                rate = dec(rate)
                branch = branches.get((r, l, second))
                if branch is None:
                    ok = False
                    continue
                reward = ground[r] + (Decimal(0) if second is None else ground[second])
                saving += rate * (reward - branch["cost"])
                total += rate * branch["total"]
                load = [a + rate * b for a, b in zip(load, branch["load"])]
                for e, count in branch["arcs"].items():
                    arc_load[q * L + l][e] += rate * count
            supported.append(ok)
            savings.append(max(Decimal(0), saving) if ok else Decimal(0))
            totals.append(total if ok else Decimal(0))
            loads.append(load if ok else [Decimal(0)] * H)
    smax = [sum(savings[q * L + l] for q in range(Q)) for l in range(L)]
    eta = [(1 - (F / value).sqrt()) ** 2 if value > F else Decimal(0) for value in smax]
    chord = [eta[l] * savings[q * L + l] for q in range(Q) for l in range(L)]
    D = sum(raw)
    upper, previous = [Decimal(0)] * (N * N), [Decimal(0)] * (N * N)
    for u in range(N):
        for v in range(N):
            e = u * N + v
            if u == v or not allowed[e]:
                continue
            if u < L and v < L:
                # Preserve the original empty-relocation circulation bound.
                upper[e] = previous[e] = D * max(0, L - 1)
            else:
                previous[e] = D
                upper[e] = min(D, sum(max(arc_load[q * L + l].get(e, Decimal(0))
                                          for l in range(L)) for q in range(Q)))
    return dict(R=R, Q=Q, assignment_scope=scope, assignment_groups=groups, L=L, H=H, N=N, G=G, S=S, F=F, raw=raw, capacity=capacity,
                C0=sum(a * b for a, b in zip(raw, ground)), saving=savings, total_load=totals,
                locker_load=loads, supported=supported, chord_coefficient=chord,
                task_upper=upper, previous_upper=previous,
                catalogue_statistics=dict(paths=len(catalogue["missions"]), empty_paths=len(catalogue["relocations"]),
                    same_od_paths=same_od, cross_return_paths=cross_return, repeated_arc_occurrences=repeated_arc))


def audit_allocations(meta, constants, errors, label):
    fail(errors, meta.get("assignment_scope", "origin") == constants.get("assignment_scope", "origin"), label + ": assignment scope mismatch")
    fail(errors, meta["supported"] == constants["supported"], label + ": support mismatch")
    fail(errors, len(meta["locker_load"]) == constants["G"]
         and all(len(row) == constants["H"] for row in meta["locker_load"]),
         label + ": per-locker load dimension")
    checks, maximum_error = 0, 0.0
    for field in ("saving", "total_load", "chord_coefficient"):
        if field not in meta:
            continue
        fail(errors, len(meta[field]) == len(constants[field]), label + ": " + field + " dimension")
        for got, expected in zip(meta[field], constants[field]):
            delta = dec(got) - expected
            maximum_error = max(maximum_error, float(abs(delta)))
            # Catalogue rewards themselves have undergone IEEE addition in C++;
            # this guard covers those final input-rounding differences only.
            tolerance = dec(2e-10) * (1 + abs(expected))
            safe = delta >= -tolerance if field in ("saving", "chord_coefficient") else delta <= tolerance
            fail(errors, safe and abs(delta) <= dec(1e-7) * (1 + abs(expected)), label + ": unsafe " + field)
            checks += 1
    for got_row, expected_row in zip(meta["locker_load"], constants["locker_load"]):
        for got, expected in zip(got_row, expected_row):
            delta = dec(got) - expected
            fail(errors, -dec(1e-7) * (1 + abs(expected)) <= delta <= dec(2e-10) * (1 + abs(expected)),
                 label + ": unsafe per-locker load")
            maximum_error = max(maximum_error, float(abs(delta)))
            checks += 1
    return dict(coefficient_checks=checks, maximum_coefficient_error=maximum_error)


def audit_duals(result, constants, errors, label):
    meta = result.get("task_resource_metadata", {})
    records = result.get("task_resource_cut_records", [])
    if not meta.get("enabled"):
        fail(errors, not records, label + ": resource rows without metadata")
        return dict(cuts=0, column_checks=0)
    audit_allocations(meta, constants, errors, label + ":resource")
    R, L, H, N, G, S = (constants[k] for k in ("R", "L", "H", "N", "G", "S"))
    assignment, theta = N + H * S, N + H * S + G + N * N
    columns = []
    def add(c, rhs, coeff):
        columns.append((dec(c), rhs, coeff))
    for g in range(G):
        rho, alpha = G + g % L, assignment + g
        add(0, [(alpha, dec(1))], [(g, dec(1))])
        add(0, [], [(g, dec(1)), (rho, dec(-1))])
        add(1, [(alpha, dec(-1))], [(rho, dec(1)), (g, dec(-1))])
        if not constants["supported"][g]:
            add(0, [], [(g, dec(1))])
    for l in range(L):
        add(0, [(l, dec(1))], [(G + l, dec(1))])
    for h in range(H):
        add(0, [(N + h * S + s, constants["capacity"][h * S + s]) for s in range(S)],
            [(g, constants["locker_load"][g][h]) for g in range(G) if constants["locker_load"][g][h]])
    add(0, [(N + j, value) for j, value in enumerate(constants["capacity"])],
        [(g, value) for g, value in enumerate(constants["total_load"]) if value])
    for l in range(L):
        for anchor in meta["idle_tangent_anchors"]:
            t = dec(anchor)
            a = 1 / (1 - t) ** 2
            columns.append((a * t * t, [], [(G + l, a), (G + L + l, dec(-1))]))
    # Reconstruct from the complete path catalogue rather than trusting the
    # exported allocation values. The row margin must cover every resulting
    # bounded-column residual, including input/coefficient rounding effects.
    objective = constants["saving"] + [dec(0)] * L + [-constants["F"]] * L
    checks, max_correction, min_margin = 0, 0.0, None
    for rec in records:
        prices = list(map(dec, rec["dual_prices"]))
        fail(errors, len(prices) == len(columns), label + ": dual dimension mismatch")
        if len(prices) != len(columns):
            continue
        activity, rhs, constant = [dec(0)] * (G + 2 * L), defaultdict(lambda: dec(0)), dec(0)
        for price, (c, b, coeff) in zip(prices, columns):
            fail(errors, price.is_finite() and price >= 0, label + ": invalid dual price")
            constant += price * c
            for j, value in b:
                rhs[j] += price * value
            for j, value in coeff:
                activity[j] += price * value
        slack = [a - c for a, c in zip(activity, objective)]
        # v and rho have bounds <= 1, but idle w is unbounded. A negative
        # w-column slack therefore cannot be dismissed as a small tolerance.
        fail(errors, min(slack[G + L:]) >= 0, label + ": unbounded idle dual column infeasible")
        correction = sum(max(dec(0), -value) for value in slack[:G + L])
        max_correction = max(max_correction, float(correction))
        exported = defaultdict(lambda: dec(0))
        for j, value in rec["coefficients"]:
            exported[j] += dec(value)
        fail(errors, exported.pop(theta, None) == 1, label + ": invalid theta coefficient")
        excess = dec(rec["rhs"]) - (constants["C0"] - constant - correction)
        for j in set(rhs) | set(exported):
            excess += max(dec(0), rhs[j] - exported[j])
        fail(errors, excess <= 0, label + ": exported row exceeds corrected dual certificate")
        min_margin = float(-excess) if min_margin is None else min(min_margin, float(-excess))
        checks += len(slack)
    return dict(cuts=len(records), column_checks=checks, maximum_bounded_residual_correction=max_correction,
                minimum_exported_certificate_margin=min_margin)


def flow_checks(catalogue, solution, demand, upper, constants, errors, label):
    R, L, N = (constants[k] for k in ("R", "L", "N"))
    loads, flows = [0.0] * (N * N), defaultdict(float)
    for item in catalogue["missions"]:
        p, value = item["path"], item["flow"]
        second = None if p["second_od"] in (None, -1) else p["second_od"]
        flows[p["first_od"], p["b"][0], second] += value
        for u, v, count in p["arc_counts"]:
            loads[u * N + v] += value * count
    for item in catalogue["relocations"]:
        for u, v, count in item["path"]["arc_counts"]:
            loads[u * N + v] += item["flow"] * count
    max_arc, max_identity = 0.0, 0.0
    for e, load in enumerate(loads):
        rhs = float(upper[e]) * solution["airline"][e // N][e % N]
        max_arc = max(max_arc, load - rhs)
        fail(errors, load - rhs <= 2e-7 * (1 + max(abs(load), abs(rhs))), label + ": incumbent violates task arc " + str(e))
    groups = constants.get("assignment_groups")
    if groups is None:
        groups = [[(r, None, demand["single_rates"][r]),
                   *((r, s, rate) for s, rate in enumerate(demand["pair_rates"][r]))] for r in range(R)]
    fail(errors, len(solution["assignment"]) == len(groups), label + ": assignment entity dimension")
    if len(solution["assignment"]) != len(groups):
        return dict(arc_rows_checked=N * N, maximum_arc_violation=max_arc, maximum_task_flow_identity_error=math.inf)
    fail(errors, solution.get("assignment_scope", "origin") == constants.get("assignment_scope", "origin"), label + ": assignment scope mismatch")
    for q, tasks in enumerate(groups):
        for l in range(L):
            for r, s, rate in tasks:
                expected = rate * solution["assignment"][q][l] * solution["rho"][l]
                actual = flows[r, l, s]
                max_identity = max(max_identity, abs(actual - expected))
                fail(errors, abs(actual - expected) <= 2e-7 * (1 + max(abs(actual), abs(expected))),
                     label + ": task flow identity violation")
    return dict(arc_rows_checked=N * N, maximum_arc_violation=max_arc, maximum_task_flow_identity_error=max_identity)


def audit_case(batch, key, jobs, state):
    errors, runs = [], []
    for job in jobs:
        js = state["jobs"][job["id"]]
        if js["status"] != "completed":
            continue
        folder = batch / js.get("result_directory", js.get("directory", "_missing"))
        required = ["result.json", "solution.json", "path_flows.json", "input_reference.json", "preprocessed_demand.json"]
        if not all((folder / name).exists() for name in required):
            errors.append(job["id"] + ": completed result lacks required artifact")
            continue
        result, solution = read(folder / "result.json"), read(folder / "solution.json")
        first_error = len(errors)
        for name in ("native_validation.json", "artifact_audit.json"):
            fail(errors, (folder / name).exists() and read(folder / name).get("passed"), job["id"] + ": " + name + " not passed")
        # Legacy full-path outputs may omit this in-model flag. The separate
        # saved native/Python validation above is still required in that case.
        fail(errors, result.get("native_model_validated") is not False, job["id"] + ": native model not validated")
        fail(errors, result.get("catalogue_complete") and result.get("support_complete"), job["id"] + ": incomplete physical domain")
        if len(errors) > first_error:
            continue
        runs.append((job, folder, result, solution))
    if not runs:
        return dict(instance=key[0], K=key[1], audited_runs=0, expected_runs=len(jobs), passed=not errors, errors=errors, comparisons=[])
    first = runs[0][1]
    model, demand = read(first / "input_reference.json")["model"], read(first / "preprocessed_demand.json")
    cache, resolved = {}, {}
    try:
        for job, folder, result, solution in runs:
            resolved[job["id"]] = resolve_full_catalogue(batch, key, folder, cache=cache)
        catalogue = resolved[runs[0][0]["id"]][0]
    except (ValueError, KeyError, OSError, TypeError) as error:
        errors.append("Full catalogue provenance/coverage failed: " + str(error))
        return dict(instance=key[0], K=key[1], audited_runs=0, expected_runs=len(jobs),
                    passed=False, errors=errors, comparisons=[])
    constants = physical_constants(model, demand, catalogue, batch / key[0],
        assignment_scope=read(first / "input_reference.json").get("assignment_scope"), max_orders=key[1])
    R, L, H, N, G, S = (constants[k] for k in ("R", "L", "H", "N", "G", "S"))
    ao, yo, theta = N + H * S, N + H * S + G, N + H * S + G + N * N
    digest = catalogue_digest(catalogue)
    fail(errors, len(demand["single_rates"]) == R and len(demand["pair_rates"]) == R
         and all(len(row) == R for row in demand["pair_rates"]), "Task demand dimension mismatch")
    fail(errors, all(rate >= 0 for rate in demand["single_rates"])
         and all(rate >= 0 for row in demand["pair_rates"] for rate in row), "Negative task demand")
    max_conservation = max(abs(float(constants["raw"][r]) - demand["single_rates"][r]
        - math.fsum(demand["pair_rates"][r]) - math.fsum(row[r] for row in demand["pair_rates"])) for r in range(R))
    fail(errors, max_conservation <= 1e-8 * (1 + float(sum(constants["raw"]))), "Mean OD demand is not conserved")
    for r in range(R):
        fail(errors, abs(demand["head_rates"][r] - demand["single_rates"][r] - math.fsum(demand["pair_rates"][r])) <= 1e-8,
             "Head demand accounting mismatch")
    all_rows, points, audits, comparisons = [], [], [], []
    common_ub = min(run[2]["objective"] for run in runs)
    baseline = next((run[2] for run in runs if run[0].get("variant") == "support"), None)
    normal_terminations = {"time_limit", "gap_limit", "optimal"}
    baseline_normal = baseline is not None and baseline.get("termination_reason") in normal_terminations
    for job, folder, result, solution in runs:
        label = job["id"]
        current_demand = read(folder / "preprocessed_demand.json")
        for name in ("single_rates", "pair_rates", "head_rates"):
            fail(errors, current_demand[name] == demand[name], label + ": demand changed across variants")
        current = read(folder / "path_flows.json")
        full_catalogue, catalogue_evidence = resolved[label]
        fail(errors, catalogue_digest(full_catalogue) == digest, label + ": complete catalogue changed across variants")
        point = design(solution)
        fail(errors, len(point) == theta, label + ": strategic layout dimension mismatch")
        structure = result.get("master_structure", {})
        if structure.get("form") == "design_binaries_plus_operating_cost":
            fail(errors, structure.get("variables") == theta + 1
                 and structure.get("design_binaries") == theta
                 and structure.get("operational_auxiliary_variables") == 0,
                 label + ": master no longer contains only the design variables and epigraph")
        points.append((label + ":incumbent", point, result["continuous_cost"]))
        for it in result.get("iterations", []):
            if it.get("operating_primal_upper") is not None:
                points.append((label + ":" + it["candidate_id"], list(map(int, it["design_key"])), it["operating_primal_upper"]))
        meta = result.get("task_savings_metadata", {})
        evidence = dict(job=label, variant=job.get("variant"), catalogue_resolution=catalogue_evidence)
        if meta.get("enabled"):
            evidence["static"] = audit_allocations(meta, constants, errors, label + ":static")
            static_row = dict(rhs=meta["courier_cost"] - meta["numerical_margin"],
                coefficients=[[theta, 1.0], *[[ao + g, c] for g, c in enumerate(meta["chord_coefficient"]) if c]])
            excess = dec(static_row["rhs"]) - constants["C0"] + sum(
                max(dec(0), exact - dec(reported))
                for exact, reported in zip(constants["chord_coefficient"], meta["chord_coefficient"]))
            fail(errors, excess <= 0, label + ": static row exceeds independent chord certificate")
            evidence["static"]["independent_certificate_margin"] = float(-excess)
            all_rows.append((label + ":static", static_row))
        evidence["resource_dual"] = audit_duals(result, constants, errors, label)
        all_rows.extend((label + ":resource:" + str(i), row) for i, row in enumerate(result.get("task_resource_cut_records", [])))
        am = result.get("task_airline_metadata", {})
        if am.get("enabled"):
            fail(errors, len(am["task_upper"]) == N * N, label + ": arc bound dimension")
            differences = [dec(a) - b for a, b in zip(am["task_upper"], constants["task_upper"])]
            for delta, expected in zip(differences, constants["task_upper"]):
                fail(errors, -dec(1e-10) <= delta <= dec(1e-8) * (1 + abs(expected)), label + ": unsafe task arc bound")
            evidence["airline"] = dict(arc_bounds_checked=N * N, minimum_reported_minus_exact=float(min(differences)),
                maximum_reported_minus_exact=float(max(differences)),
                reduced_arc_bounds=sum(a + dec(1e-10) < b for a, b in zip(map(dec, am["task_upper"]), constants["previous_upper"])))
        # Every original incumbent must also satisfy the independently derived
        # tighter bounds, even if that variant did not enable them.
        evidence["incumbent_flow"] = flow_checks(current, solution, demand, constants["task_upper"], constants, errors, label)
        audits.append(evidence)
        iters = result.get("iterations", [])
        reason = result.get("termination_reason", "")
        row = dict(instance=key[0], K=key[1], variant=job.get("variant"), status="completed",
            upper_bound=result["objective"], lower_bound=result["bound"], reported_gap=result["gap"],
            common_upper_bound=common_ub, common_ub_diagnostic_gap=max(0.0, common_ub - result["bound"]) / max(1.0, abs(common_ub)),
            seconds=result.get("wall_before_output", result.get("wall")), termination_reason=reason,
            requested_gap=result.get("requested_gap"), target_gap_met=result.get("target_gap_met"),
            last_subproblem_termination=result.get("last_subproblem_termination_reason"),
            algorithm_failure="failed" in reason or "error" in reason,
            certificate_failure=("failed" in reason and any(word in reason for word in ("dual", "certificate", "validation"))),
            normal_budget_or_gap_termination=reason in normal_terminations,
            support_baseline_normal_termination=baseline_normal,
            root_lower_bound=result.get("root_diagnostics", {}).get("lower_bound"),
            root_bound_scope=result.get("root_diagnostics", {}).get("scope"),
            root_rounds=len(result.get("root_diagnostics", {}).get("rounds", [])),
            lower_bound_gain_over_support=None if not baseline_normal or reason not in normal_terminations
                else result["bound"] - baseline["bound"],
            lower_bound_gain_after_root=None if result.get("root_diagnostics", {}).get("lower_bound") is None
                else result["bound"] - result["root_diagnostics"]["lower_bound"],
            master_solves=len(result.get("master_solves", [])), evaluated_integer_candidates=len(iters),
            assignment_patterns=len({it["design_key"][ao:yo] for it in iters}),
            airline_patterns=len({it["design_key"][yo:] for it in iters}),
            master_variables=result.get("master_structure", {}).get("variables"),
            master_rows=result.get("master_structure", {}).get("final_rows"),
            path_support_cuts=result.get("path_support_summary", {}).get("cuts", 0),
            task_resource_cuts=len(result.get("task_resource_cut_records", [])),
            benders_cuts=len(result.get("benders_cuts", [])))
        values = [it["master_theta"] for it in iters if it.get("master_theta") is not None]
        row["candidate_theta_median"] = statistics.median(values) if values else None
        comparisons.append(row)
    count, maximum, worst = 0, 0.0, None
    for name, row in all_rows:
        for pname, point, operating in points:
            if len(point) != theta:
                errors.append(pname + ": candidate layout dimension mismatch")
                continue
            violation = row_violation(row, point, operating)
            if violation > maximum:
                maximum, worst = violation, [name, pname]
            fail(errors, violation <= 2e-5, name + ": excludes saved operating point " + pname)
            count += 1
    return dict(instance=key[0], K=key[1], audited_runs=len(runs), expected_runs=len(jobs),
        dimensions=dict(bases=L, lockers=H, ods=R, levels=S), catalogue_sha256=digest,
        catalogue_statistics=constants["catalogue_statistics"], maximum_demand_conservation_error=max_conservation,
        cost_rows=len(all_rows), operating_points=len(points), cross_point_checks=count,
        maximum_cost_row_violation=maximum, worst_row_point=worst, audits=audits,
        comparisons=comparisons, passed=not errors, errors=errors)


def audit(batch, require_complete=False):
    batch = Path(batch).resolve()
    manifest, state = read(batch / "experiment_manifest.json"), read(batch / "state.json")
    groups = defaultdict(list)
    for job in manifest["jobs"]:
        groups[job["instance"], job["K"]].append(job)
    frozen_errors = []
    for filename, expected in manifest["frozen_sha256"].items():
        fail(frozen_errors, (batch / filename).exists() and sha(batch / filename) == expected, "Frozen artifact changed: " + filename)
    with localcontext() as ctx:
        ctx.prec = 60
        cases = [audit_case(batch, key, jobs, state) for key, jobs in groups.items()]
    statuses = Counter(job["status"] for job in state["jobs"].values())
    terminal = sum(statuses[key] for key in ("completed", "failed", "interrupted"))
    auditor_source = Path(__file__).resolve()
    out = dict(schema="task-cost-independent-audit/1", batch=str(batch), solver_invoked=False,
        auditor_source_sha256=sha(auditor_source),
        saved_auditor_source="analysis/audit_task_cost_source.py",
        passed=not frozen_errors and all(case["passed"] for case in cases),
        all_jobs_terminal=terminal == len(manifest["jobs"]), expected_jobs=len(manifest["jobs"]),
        audited_completed_jobs=sum(case["audited_runs"] for case in cases), statuses=dict(statuses),
        frozen_errors=frozen_errors, cases=cases)
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    (output / "audit_task_cost_source.py").write_bytes(auditor_source.read_bytes())
    (output / "expanded_task_audit.json").write_text(json.dumps(out, indent=2) + "\n")
    rows = [row for case in cases for row in case["comparisons"]]
    (output / "expanded_paired_comparison.json").write_text(json.dumps(dict(rows=rows,
        common_ub_note="Diagnostic gap using only validated completed incumbents of the same instance and K; not the original reported stopping gap."), indent=2) + "\n")
    if rows:
        with (output / "expanded_paired_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# 多算例任务成本强化：独立审计与配对比较", "",
        "仅读取已经 completed 且通过原生与 Python 校验的运行。未调用求解器；不同实例或任务规模之间不混用割和运营点。", "",
        f"批次状态：{dict(statuses)}；已审计 {out['audited_completed_jobs']}/{out['expected_jobs']} 次运行；独立检查通过：{out['passed']}。", "",
        "| 算例 | 方案 | UB | LB | 原报告 gap | 共同 UB 诊断 gap | 秒 | 相对基线 LB 增量 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        gain = "—" if row["lower_bound_gain_over_support"] is None else f"{row['lower_bound_gain_over_support']:.3f}"
        lines.append(f"| {Path(row['instance']).name} | {row['variant']} | {row['upper_bound']:.3f} | {row['lower_bound']:.3f} | {100*row['reported_gap']:.3f}% | {100*row['common_ub_diagnostic_gap']:.3f}% | {row['seconds']:.2f} | {gain} |")
    lines += ["", "共同 UB 列仅用于比较下界，使用同一算例、同一任务规模的已验证最小上界；不能替换单次运行报告的 gap 或停止条件。",
        "若基线或待比较方案因算法/数值证书问题提前停止，仍保留通过验证的 incumbent 和此前有效 LB，但不列正常时间预算下的 LB 增量。",
        "整数候选 SOCP 评估数为零不表示没有求解任何子问题；initial/root/heuristic 子问题仍可能已经求解。", "",
        "| 算例 | 方案 | 调度器状态 | 算法终止原因 | 达到目标 gap | 证书失败终止 |",
        "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {Path(row['instance']).name} | {row['variant']} | {row['status']} | {row['termination_reason']} | {row['target_gap_met']} | {row['certificate_failure']} |")
    lines += ["", "调度器 completed 只表示该进程及结果保存/校验流程已经结束；算法的终止原因另行列示。独立割审计通过也不能把证书失败终止改称正常收敛。", "",
        "| 算例 | 方案 | root LB | root 后 LB 增量 | 主问题求解数 | 整数候选评估数 | 不同分配 | 不同航线设计 | 候选 theta 中位数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        root = "—" if row["root_lower_bound"] is None else f"{row['root_lower_bound']:.3f}"
        gain = "—" if row["lower_bound_gain_after_root"] is None else f"{row['lower_bound_gain_after_root']:.3f}"
        median = "—" if row["candidate_theta_median"] is None else f"{row['candidate_theta_median']:.3f}"
        lines.append(f"| {Path(row['instance']).name} | {row['variant']} | {root} | {gain} | {row['master_solves']} | {row['evaluated_integer_candidates']} | {row['assignment_patterns']} | {row['airline_patterns']} | {median} |")
    lines += ["", "不同分配/航线设计仅统计已经进入整数候选评估的设计，不代表分支定界树遍历的全部节点。root 的具体有效域标签另存于 comparison 数据的 root_bound_scope 字段。", "",
        "完整目录、订单守恒、全部返回基地、AA、空载调位、任务收益/占用极值、对偶可行性及跨设计检查细节见 expanded_task_audit.json。"]
    (output / "EXPANDED_AUDIT.md").write_text("\n".join(lines) + "\n")
    if require_complete and not out["all_jobs_terminal"]:
        raise RuntimeError("Audit requested a terminal batch, but some jobs are still active or pending")
    if not out["passed"]:
        raise RuntimeError("Independent task-cost artifact audit failed; inspect expanded_task_audit.json")
    return dict(passed=out["passed"], all_jobs_terminal=out["all_jobs_terminal"],
        audited_runs=out["audited_completed_jobs"], expected_runs=out["expected_jobs"],
        statuses=dict(statuses), cases=len(cases), cross_point_checks=sum(case.get("cross_point_checks", 0) for case in cases),
        report=str(output / "EXPANDED_AUDIT.md"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.batch, args.require_complete), indent=2))


if __name__ == "__main__":
    main()
