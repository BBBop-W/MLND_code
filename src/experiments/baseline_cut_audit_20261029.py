"""Independent same-batch cut and full-catalogue audit; never starts a solver.

The Decimal queue replay and mechanism budget checks retain the previously
reviewed mathematics. Complete physical paths now come from this batch's fresh
full CPLEX export, never a historical run or an output-directory import.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path

from src.experiments import conditional_benchmark_20261022 as base
from src.experiments import task_cost_audit as prior
from src.experiments.storage import write_json

D = prior.dec
ZERO = Decimal(0)

def rounding_loss(value, magnitude, operations=64):
    """Enclose the declared 1e-12 guard and binary arithmetic/conversion.

    The directional validity checks remain separate. This is only the maximum
    explainable conservative loss, using absolute terms rather than their sum.
    IEEE double epsilon bounds long-double error on the supported platform too.
    """
    scale = 1 + max(abs(value), magnitude)
    return (Decimal("2e-12") + D(math.ulp(1.0)) * (4 * operations + 16)) * scale


def verified_demand_totals(raw, reported):
    """Separate exact mathematics from the backend's ordered double sum."""
    exact = sum(raw, ZERO)
    accumulated = 0.0
    for value in raw:
        accumulated = float(accumulated + float(value))
    require(math.isfinite(accumulated) and D(reported).is_finite(), "nonfinite total demand")
    require(D(reported) == D(accumulated), "reported total demand differs from ordered binary64 accumulation")
    return exact, D(accumulated)


def check_resource_affine(got, exact, guarded, operations):
    # The danger direction must use independently reconstructed demand/costs.
    # The conservative-loss enclosure may use the verified stored safe cost.
    require(got <= exact + D(2e-12) * (1 + abs(exact)), "unsafe resource affine coefficient")
    require(got <= guarded + D(2e-12) * (1 + abs(guarded)), "unsafe guarded resource affine coefficient")
    require(got >= guarded - rounding_loss(guarded, abs(guarded), operations),
            "resource affine coefficient exceeds propagated conservative-rounding interval")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, relative=2e-9):
    require(math.isfinite(float(actual)), label + ": nonfinite")
    error = abs(D(actual) - expected)
    require(error <= D(relative) * (1 + abs(expected)), label + f": mismatch {float(error):.9g}")
    return float(error)


def conjugate(value, fleet):
    return (value.sqrt() - fleet.sqrt()) ** 2 if value > fleet else ZERO


def load_verified(folder):
    payload, _ = prior._verified_payloads(folder)
    return payload, prior.design(payload["solution"])


def input_constants(model, demand, catalogue, instance, *, assignment_scope=None, max_orders=None):
    constants = prior.physical_constants(model, demand, catalogue, instance,
        assignment_scope=assignment_scope, max_orders=max_orders)
    ground = {r["od_id"]: D(r["ground_cost"])
              for r in prior.table(instance / "coefficients/od_costs.csv")}
    constants["ground"] = [ground[str(r)] for r in model["ids"]["ods"]]
    constants["assignment"] = constants["N"] + constants["H"] * constants["S"]
    constants["airline"] = constants["assignment"] + constants["G"]
    constants["design_count"] = constants["airline"] + constants["N"] ** 2
    return constants


def validate_profile(profile, constants, demand, catalogue):
    """Recompute the mathematical priced minima, not the C++ valid flag."""
    R, L, H, N, S = (constants[k] for k in ("R", "L", "H", "N", "S"))
    count, start, arc_start = (constants[k] for k in ("design_count", "assignment", "airline"))
    Q = constants.get("Q", R)
    require(profile.get("assignment_scope", "origin") == constants.get("assignment_scope", "origin"), "profile assignment scope")
    require(profile["heads"] == Q and profile["bases"] == L, "profile dimensions")
    for field in ("complete_domain", "catalogue_complete", "support_complete"):
        require(profile[field] is True, "profile missing complete-domain field: " + field)
    require(profile["service_paths_scanned"] == len(catalogue["missions"]), "profile service scan incomplete")
    close(profile["fleet_cost"], constants["F"], "fleet cost")
    close(profile["constant"], constants["C0"], "courier constant")
    signed = [D(v) for v in profile["signed_savings"]]
    lower = [D(v) for v in profile["lower_coefficients"]]
    require(len(signed) == Q * L and len(lower) == count and len(profile["unsupported"]) == Q * L,
            "profile vector dimensions")
    require(all(v.is_finite() for v in signed + lower), "profile nonfinite coefficients")
    if profile["kind"] == "nonnegative_savings_epi":
        require(profile["prices"] is None and not any(profile["unsupported"]), "zero profile structure")
        require(profile["empty_paths_scanned"] == 0 and not profile["negative_empty_paths"], "zero profile empty paths")
        require(all(v == 0 for v in lower), "zero profile resource coefficients")
        for got, expected in zip(signed, constants["saving"]):
            require(got >= 0 and got >= expected - D(2e-12) * (1 + abs(expected)), "unsafe S+ saving")
            close(got, expected, "S+ saving", relative=2e-8)
        return dict(kind=profile["kind"], path_checks=len(catalogue["missions"]),
                    group_checks=Q * L, resource_coefficient_checks=count)
    require(profile["kind"] == "resource_price_epi", "unknown queue profile kind")
    require(profile["rounding_relative_guard"] == 1e-12, "unknown profile roundoff guard")
    require(profile["empty_paths_scanned"] == len(catalogue["relocations"]), "profile empty scan incomplete")
    prices = profile["prices"]
    mu, eta, potential = ([D(v) for v in prices[k]] for k in ("locker", "airline", "balance"))
    require((len(mu), len(eta), len(potential)) == (H, N * N, L), "price dimensions")
    require(all(v.is_finite() for v in mu + eta + potential) and all(v >= 0 for v in mu + eta),
            "resource price sign/nonfinite")
    rates = {(r, l, s): D(rate) for r in range(R) for l in range(L)
             for s, rate in [(-1, demand["single_rates"][r]), *enumerate(demand["pair_rates"][r])]
             if rate > 0}
    gate = {}
    expected_lower = [ZERO] * count
    for record in prices["branch_gates"]:
        key = record["head"], record["origin"], record["second"]
        arc, price = record["arc"], D(record["price"])
        require(key in rates and type(arc) is int and 0 <= arc < N * N and price.is_finite() and price >= 0,
                "invalid normalized branch gate price")
        require((key, arc) not in gate, "duplicate merged branch gate")
        gate[key, arc] = price
        expected_lower[arc_start + arc] -= price
    upper = [D(v) for v in profile["airline_upper"]]
    require(len(upper) == N * N, "airline upper dimensions")
    for got, expected in zip(upper, constants["task_upper"]):
        # The existing common task-airline upper bound is independently
        # reconstructed by the inherited audit, including repeated arcs.
        close(got, expected, "airline upper", relative=2e-8)
        require(got >= expected - D(2e-12) * (1 + abs(expected)), "unsafe airline upper")
    total, implementation_total = verified_demand_totals(constants["raw"], profile["total_demand"])
    for h in range(H):
        for level in range(S):
            expected_lower[N + h * S + level] -= mu[h] * constants["capacity"][h * S + level]
    for arc in range(N * N):
        expected_lower[arc_start + arc] -= eta[arc] * upper[arc]
    minima, minimum_lower = {}, {}
    positive_paths = 0
    for record in catalogue["missions"]:
        path = record["path"]
        second = -1 if path["second_od"] is None else path["second_od"]
        head, origin, returned = path["first_od"], path["b"][0], path["b"][1]
        key = head, origin, second
        if key not in rates:
            continue
        positive_paths += 1
        reward = constants["ground"][head] + (ZERO if second < 0 else constants["ground"][second])
        magnitude = (abs(D(path["cost"])) + abs(constants["F"] * D(path["time"])) + abs(reward)
                     + abs(potential[origin]) + abs(potential[returned]))
        adjusted = (D(path["cost"]) + constants["F"] * D(path["time"]) - reward
                    + potential[origin] - potential[returned])
        adjusted += sum(mu[h] * D(time) for h, time in path["slot_times"])
        magnitude += sum(abs(mu[h] * D(time)) for h, time in path["slot_times"])
        present = set()
        for u, v, times in path["arc_counts"]:
            arc = u * N + v
            adjusted += eta[arc] * times
            magnitude += abs(eta[arc] * times)
            present.add(arc)
        adjusted *= rates[key]
        magnitude *= rates[key]
        adjusted += sum(gate.get((key, arc), ZERO) for arc in present)
        magnitude += sum(abs(gate.get((key, arc), ZERO)) for arc in present)
        minima[key] = min(minima.get(key, adjusted), adjusted)
        safe_lower = adjusted - rounding_loss(adjusted, magnitude,
            16 + 3 * len(path["slot_times"]) + 3 * len(path["arc_counts"]) + 2 * len(present))
        minimum_lower[key] = min(minimum_lower.get(key, safe_lower), safe_lower)
    missing = sorted(key for key in rates if key not in minima)
    require(sorted(map(tuple, profile["missing_branches"])) == missing, "missing positive branches")
    groups = constants.get("assignment_groups")
    if groups is None:
        groups = [[(r, None, demand["single_rates"][r]),
                   *((r, second, rate) for second, rate in enumerate(demand["pair_rates"][r]))] for r in range(R)]
    group_index = {(r, -1 if second is None else second): q
                   for q, tasks in enumerate(groups) for r, second, rate in tasks}
    unsupported = [any((r, origin, -1 if second is None else second) in missing
                       for r, second, rate in tasks if rate > 0)
                   for tasks in groups for origin in range(L)]
    require(profile["unsupported"] == unsupported, "unsupported groups")
    require(profile["positive_service_paths_scanned"] == positive_paths, "positive path scan count")
    expected_saving = [ZERO] * (Q * L)
    saving_upper, saving_magnitude = [ZERO] * (Q * L), [ZERO] * (Q * L)
    for (head, origin, second), minimum in minima.items():
        group = group_index[head, second] * L + origin
        low = minimum_lower[head, origin, second]
        expected_saving[group] -= minimum
        saving_upper[group] -= low
        saving_magnitude[group] += max(abs(minimum), abs(low))
    for group, (got, expected) in enumerate(zip(signed, expected_saving)):
        if unsupported[group]:
            require(got == 0, "unsupported group nonzero savings")
            continue
        require(got >= expected - D(2e-12) * (1 + abs(expected)), "unsafe signed savings")
        require(got <= saving_upper[group] + rounding_loss(saving_upper[group], saving_magnitude[group], R + 4),
                "signed savings exceeds propagated conservative-rounding interval")
    negative = {row["path_id"]: row for row in profile["negative_empty_paths"]}
    require(len(negative) == len(profile["negative_empty_paths"]), "duplicate negative-empty record")
    seen_negative = set()
    exact_lower = list(expected_lower)
    for record in catalogue["relocations"]:
        path = record["path"]
        origin, returned = path["b"]
        arc = origin * N + returned
        require(path["arc_counts"] == [[origin, returned, 1]], "empty correction is not a direct arc")
        adjusted = D(path["cost"]) + constants["F"] * D(path["time"]) + eta[arc] + potential[origin] - potential[returned]
        exact_lower[arc_start + arc] += min(ZERO, adjusted) * total
        magnitude = (abs(D(path["cost"])) + abs(constants["F"] * D(path["time"])) + abs(eta[arc])
                     + abs(potential[origin]) + abs(potential[returned]))
        if path["id"] in negative:
            seen_negative.add(path["id"])
            item = negative[path["id"]]
            require(item["arc"] == arc and D(item["adjusted_cost"]) < 0, "negative-empty identity/sign")
            require(D(item["adjusted_cost"]) <= adjusted + D(2e-12) * (1 + abs(adjusted)), "unsafe empty adjusted cost")
            require(D(item["adjusted_cost"]) >= adjusted - rounding_loss(adjusted, magnitude),
                    "empty adjusted cost exceeds propagated conservative-rounding interval")
            close(item["affine_coefficient"], D(item["adjusted_cost"]) * implementation_total, "empty affine")
            expected_lower[arc_start + arc] += D(item["adjusted_cost"]) * implementation_total
        else:
            require(adjusted >= -D(2e-12) * (1 + abs(adjusted)), "missing negative empty correction")
    require(seen_negative == set(negative), "unknown negative empty path")
    for got, exact, guarded in zip(lower, exact_lower, expected_lower):
        check_resource_affine(got, exact, guarded, count + len(gate) + 8)
    return dict(kind=profile["kind"], path_checks=len(catalogue["missions"]) + len(catalogue["relocations"]),
                group_checks=Q * L, resource_coefficient_checks=count, negative_empty_paths=len(negative))


def replay_joint_queue_row(row, constants):
    """Check all two-task categorical vertices and the external-task lift."""
    cert, profile = row["certificate"], row["certificate"]["profile"]
    Q, L, count, start = (constants[k] for k in ("Q", "L", "design_count", "assignment"))
    require(L >= 1 and cert["kind"] == "queue_joint" and cert["complete_domain"] is True,
            "joint queue domain")
    require(cert["global_integer_valid"] is True and cert["direct_global_bound_update"] is False,
            "joint queue validity scope")
    method = cert.get("coefficient_method", "L2_facet_enumeration")
    require(method in {"L2_facet_enumeration", "complete_state_coefficient_lp"}, "joint coefficient method")
    expected_calls = int(method == "complete_state_coefficient_lp")
    require(cert["ordinary_auxiliary_optimization_calls"] == expected_calls, "joint coefficient LP count")
    if expected_calls:
        require(cert["all_states_validated"] is True and cert["states_checked"] == (L + 1) ** 2
                and cert["state_count_per_task"] == L + 1 and cert["bases"] == L,
                "incomplete general-base state validation")
    else:
        require(L == 2, "legacy joint facet domain")
    require(cert.get("assignment_scope", "origin") == profile.get("assignment_scope", "origin")
            == constants["assignment_scope"], "joint queue assignment scope")
    selected = cert["selected_heads"]
    require(len(selected) == 2 and len(set(selected)) == 2
            and all(type(q) is int and 0 <= q < Q for q in selected), "joint selected tasks")
    if constants["assignment_scope"] == "task":
        require(cert["selected_tasks"] == selected, "joint task identities differ")
    signed = [D(v) for v in profile["signed_savings"]]
    unsupported = profile["unsupported"]
    lower = [D(v) for v in profile["lower_coefficients"]]
    affine = [D(v) for v in cert["affine_coefficients"]]
    require(len(signed) == len(unsupported) == L * Q and len(lower) == count
            and len(affine) == 1 + 2 * L and all(v.is_finite() for v in signed + lower + affine),
            "joint profile dimensions/finite values")
    require(len(cert["vertices"]) == (L + 1) ** 2, "joint vertex count")
    for index, vertex in enumerate(cert["vertices"]):
        states = [index // (L + 1), index % (L + 1)]
        require(vertex["states"] == states, "joint vertex identity")
        value = ZERO
        for base in range(L):
            active = [L * q + base for q, state in zip(selected, states) if state == base + 1]
            if not any(unsupported[g] for g in active):
                value -= conjugate(sum((signed[g] for g in active), ZERO), constants["F"])
        close(vertex["queue_value"], value, "joint vertex value")
        safe = D(vertex["safe_queue_value"])
        require(safe.is_finite() and safe <= value + D(2e-12) * (1 + abs(value)), "unsafe joint vertex")
        plane = affine[0] + sum(affine[1 + L * i + state - 1]
                               for i, state in enumerate(states) if state)
        require(plane <= safe + D(2e-12) * (1 + abs(safe) + abs(plane)), "joint plane excludes a vertex")
    require(cert["lifting_rule"] == "JL_positive_marginal_at_full_positive_sum", "joint lifting rule")
    sums = [sum((max(ZERO, signed[L * q + l]) for q in range(Q)
                 if not unsupported[L * q + l]), ZERO) for l in range(L)]
    require(len(cert["external_positive_sums"]) == L and len(cert["external_penalties"]) == L * Q,
            "joint lifting dimensions")
    for got, expected in zip(cert["external_positive_sums"], sums):
        close(got, expected, "joint positive sum")
    for i, q in enumerate(selected):
        for l in range(L):
            lower[start + L * q + l] += affine[1 + L * i + l]
    for q in range(Q):
        for l in range(L):
            g = L * q + l
            got = D(cert["external_penalties"][g])
            expected = ZERO
            if q not in selected and not unsupported[g] and signed[g] > 0:
                expected = conjugate(sums[l], constants["F"]) - conjugate(sums[l] - signed[g], constants["F"])
            require(got.is_finite() and got >= expected - D(2e-12) * (1 + abs(expected)),
                    "unsafe joint external penalty")
            close(got, expected, "joint external penalty", relative=2e-8)
            lower[start + g] -= got
    fields = ("numerical_margin", "conversion_error", "arithmetic_error", "intercept_correction",
              "facet_validation_correction")
    require(all(math.isfinite(cert[k]) and cert[k] >= 0 for k in fields), "joint numerical guard")
    correction = D(cert["intercept_correction"])
    close(cert["numerical_margin"], D(1e-8) * (1 + D(profile["constant"])), "joint numerical margin")
    close(correction, sum((D(cert[k]) for k in ("numerical_margin", "conversion_error", "arithmetic_error")), ZERO),
          "joint intercept correction")
    exact_intercept = D(profile["constant"]) + affine[0] - correction
    require(D(row["rhs"]) <= exact_intercept + D(2e-12) * (1 + abs(exact_intercept)), "unsafe joint intercept")
    close(row["rhs"], exact_intercept, "joint row intercept", relative=2e-8)
    sparse = dict(row["coefficients"])
    require(len(sparse) == len(row["coefficients"]) and sparse.get(count) == 1
            and all(type(j) is int and 0 <= j <= count and math.isfinite(c) for j, c in sparse.items()),
            "joint sparse row")
    for j, expected in enumerate(lower):
        close(sparse.get(j, 0), -expected, "joint row coefficient", relative=2e-11)
    design = [D(v) for v in cert["separation_design"]]
    require(len(design) == count and all(v.is_finite() and 0 <= v <= 1 for v in design), "joint separation design")
    if "design" in row:
        require(row["design"] == cert["separation_design"], "joint record/certificate design differs")
    if "theta" in row:
        require(row["theta"] == cert["separation_theta"], "joint record/certificate theta differs")
    value = D(row["rhs"]) - sum(D(c) * design[j] for j, c in sparse.items() if j != count)
    close(cert["value_at_separation_point"], value, "joint separation value")
    close(row["violation"], value - D(cert["separation_theta"]), "joint separation violation")
    return dict(kind="queue_joint", selected_tasks=selected, vertex_checks=(L + 1) ** 2, lifted_tasks=Q - 2,
                recomputed_violation=float(value - D(cert["separation_theta"])))


def replay_queue_row(row, constants):
    if row["certificate"]["kind"] == "queue_joint":
        return replay_joint_queue_row(row, constants)
    certificate = row["certificate"]
    profile = certificate["profile"]
    require(certificate["kind"] == "queue_polymatroid" and certificate["complete_domain"] is True,
            "queue certificate domain/kind")
    require(certificate["global_integer_valid"] is True and certificate["direct_global_bound_update"] is False,
            "queue validity/bound-update scope")
    require(certificate["ordinary_auxiliary_optimization_calls"] == 0, "queue ordinary auxiliary solve")
    R, L, count, start = (constants[k] for k in ("R", "L", "design_count", "assignment"))
    R = constants.get("Q", R)  # Chain elements are assignment entities (tasks in the current model).
    require(profile.get("assignment_scope", "origin") == constants.get("assignment_scope", "origin"), "queue assignment scope")
    design = [D(v) for v in certificate["separation_design"]]
    require(len(design) == count and all(v.is_finite() and 0 <= v <= 1 for v in design), "separation design")
    theta = D(certificate["separation_theta"])
    require(theta.is_finite(), "nonfinite separation theta")
    if "design" in row:
        require(row["design"] == certificate["separation_design"], "record/certificate design differs")
    if "theta" in row:
        require(row["theta"] == certificate["separation_theta"], "record/certificate theta differs")
    signed = [D(v) for v in profile["signed_savings"]]
    lower = [D(v) for v in profile["lower_coefficients"]]
    intercept = D(profile["constant"])
    constant_exact = constants["C0"]
    old_chord = constants["C0"]
    maximum_error, ideal_value = 0.0, constants["C0"] + sum(c * x for c, x in zip(lower, design))
    require(len(certificate["bases"]) == L, "chain base count")
    for l, chain in enumerate(certificate["bases"]):
        require(chain["base"] == l, "chain base identity")
        required = [bool(profile["unsupported"][r * L + l]) for r in range(R)]
        complement = [required[r] or signed[r * L + l] < 0 for r in range(R)]
        weights = [ZERO if required[r] else abs(signed[r * L + l]) for r in range(R)]
        shift = sum(signed[r * L + l] for r in range(R) if complement[r] and not required[r])
        transformed = [1 - design[start + r * L + l] if complement[r] else design[start + r * L + l]
                       for r in range(R)]
        require(chain["necessary_gates"] == required and chain["complemented"] == complement, "chain complement/gates")
        maximum_error = max(maximum_error, close(chain["shift"], shift, "chain shift"))
        require(len(chain["weights"]) == R and len(chain["transformed_point"]) == R and len(chain["deltas"]) == R,
                "chain vector dimensions")
        for got, expected in zip(chain["weights"], weights):
            maximum_error = max(maximum_error, close(got, expected, "chain weight"))
        for got, expected in zip(chain["transformed_point"], transformed):
            maximum_error = max(maximum_error, close(got, expected, "transformed point", relative=2e-13))
        order = chain["order"]
        require(sorted(order) == list(range(R)), "greedy order is not a permutation")
        require(all(transformed[a] >= transformed[b] - D(1e-13) for a, b in zip(order, order[1:])),
                "greedy order is not descending")
        prefixes = chain["prefix_savings"]
        require(len(prefixes) == R + 1 and prefixes[0] == 0, "greedy prefix dimensions/empty value")
        cumulative, previous, missing = shift, ZERO, sum(required)
        for position, head in enumerate(order, 1):
            cumulative += weights[head]
            missing -= required[head]
            current = conjugate(cumulative, constants["F"]) if missing == 0 else ZERO
            exact_delta = previous - current
            delta = D(chain["deltas"][head])
            require(delta <= exact_delta + D(2e-12) * (1 + abs(exact_delta)), "unsafe chain increment")
            maximum_error = max(maximum_error, close(prefixes[position], current, "prefix value"))
            # Near cancellations can leave an increment much smaller than the
            # sum of weights; use the full chain scale for the closeness check.
            require(abs(delta - exact_delta) <= D(2e-8) * (1 + abs(shift) + sum(weights) + current + previous),
                    "chain increment differs beyond its numerical guard")
            ideal_value += exact_delta * transformed[head]
            if complement[head]:
                intercept += delta
                constant_exact += delta
                lower[start + head * L + l] -= delta
            else:
                lower[start + head * L + l] += delta
            previous = current
        if profile["kind"] == "nonnegative_savings_epi":
            total = sum(weights)
            slope = conjugate(total, constants["F"]) / total if total else ZERO
            old_chord -= slope * sum(weights[r] * design[start + r * L + l] for r in range(R))
    for field in ("numerical_margin", "conversion_error", "arithmetic_error", "increment_error_allowance", "intercept_correction"):
        require(math.isfinite(certificate[field]) and certificate[field] >= 0, "invalid numerical guard: " + field)
    correction = D(certificate["intercept_correction"])
    close(certificate["numerical_margin"], D(1e-8) * (1 + D(profile["constant"])), "numerical margin")
    close(correction, sum(D(certificate[k]) for k in ("numerical_margin", "conversion_error", "arithmetic_error")),
          "total intercept correction")
    rhs = D(row["rhs"])
    require(rhs <= constant_exact - correction + D(2e-12) * (1 + abs(constant_exact)), "unsafe row intercept")
    close(rhs, intercept - correction, "queue rhs", relative=2e-8)
    entries = row["coefficients"]
    require(all(type(j) is int and 0 <= j <= count and math.isfinite(c) for j, c in entries), "row coefficient indices")
    sparse = dict(entries)
    require(len(sparse) == len(entries) and sparse.get(count) == 1, "row duplicates/theta coefficient")
    for j, expected in enumerate(lower):
        maximum_error = max(maximum_error, close(sparse.get(j, 0), -expected, "queue coefficient", relative=2e-11))
    value = rhs - sum(D(c) * design[j] for j, c in entries if j != count)
    close(certificate["value_at_separation_point"], value, "separation value")
    close(row["violation"], value - theta, "separation violation")
    binary = all(min(abs(x), abs(x - 1)) <= D(1e-10) for x in design)
    output = dict(profile_id=profile["profile_id"], kind=profile["kind"], stage=row["stage"],
        maximum_reconstruction_error=maximum_error, recomputed_violation=float(value - theta),
        separation_point_binary=binary, ideal_chain_value=float(ideal_value),
        numerical_row_loss=float(ideal_value - value))
    if profile["kind"] == "nonnegative_savings_epi":
        old_chord -= D(certificate["numerical_margin"])
        output["same_profile_old_chord_value"] = float(old_chord)
        output["row_minus_same_profile_old_chord"] = float(value - old_chord)
    return output


def audit_fractional_stages(result, enabled, seconds):
    """Replay recorded LP/row/budget contracts; full matrices are static-reviewed.

    The saved optimal/feasible status is checked for consistency with every
    bound update. It is not mislabeled as an independent LP re-optimization.
    """
    summary, records = result["queue_fractional_summary"], result["queue_fractional_records"]
    require(summary["enabled"] is enabled and summary["calls"] == len(records), "fractional flag/count")
    require(summary["bound_scope"] == "optimality_preserving_master_relaxation" and
            summary["direct_cut_value_bound_update"] is False, "fractional global-bound scope")
    require(enabled or not records, "fractional stages active with flag off")
    for field, expected in (("lp_budget_fraction", .02), ("price_budget_fraction", .02),
                            ("call_limit_seconds", 2), ("cooldown_seconds", 5),
                            ("max_lp_solves", 3), ("max_new_cuts", 2),
                            ("marginal_relative_threshold", 1e-5)):
        require(summary[field] == expected, "fractional declaration differs: " + field)
    lp_spent, price_spent, stage_spent, previous_end, previous_donor = 0.0, 0.0, 0.0, None, 0
    total_rounds, total_cuts, maximum_overrun = 0, 0, 0.0
    for call_index, record in enumerate(records):
        require(record["stage"] in ("fractional_queue_root", "fractional_queue_integer"), "fractional stage name")
        require(record["candidate_id"].startswith("fractional:") and record["donor_candidate_id"], "fractional candidate/donor identity")
        require(type(record["donor_serial"]) is int and record["donor_serial"] > previous_donor,
                "fractional stage reuses a consumed donor")
        previous_donor = record["donor_serial"]
        values = [record[k] for k in ("started_elapsed_seconds", "previous_call_elapsed_seconds",
            "time_budget", "lp_budget", "price_budget", "lp_seconds", "price_seconds", "seconds")]
        require(all(math.isfinite(x) for x in values), "nonfinite fractional timing")
        started, previous, allowance, lp_budget, price_budget, lp_actual, price_actual, actual = values
        require(0 < allowance <= 2 and 0 < lp_budget <= 1 and 0 < price_budget <= 1,
                "fractional per-call allocation")
        close(allowance, D(lp_budget) + D(price_budget), "fractional reserved phase sum", relative=2e-12)
        require(lp_budget <= .02 * seconds - lp_spent + 1e-7 and
                price_budget <= .02 * seconds - price_spent + 1e-7,
                "fractional allocation exceeds dedicated remainder")
        require(started + allowance <= seconds + 1e-5 and started - previous >= 5 - 1e-7,
                "fractional global remainder/cooldown")
        if previous_end is not None:
            require(previous >= previous_end - 1e-6 and started - previous_end >= 5 - 1e-6,
                    "fractional cooldown does not follow actual previous finish")
        require(min(lp_actual, price_actual, actual) >= 0, "negative fractional time")
        close(actual, D(lp_actual) + D(price_actual), "fractional nonoverlapping phase accounting", relative=2e-9)
        maximum_overrun = max(maximum_overrun, actual - allowance)
        previous_end = started + actual
        diagnostics, rounds = record["diagnostics"], record["diagnostics"]["rounds"]
        require(diagnostics["scope"] == "optimality_preserving_master_relaxation" and
                diagnostics["bound_source"] == "validated_optimal_current_master_LP" and
                diagnostics["master_added_variables"] == 0 and
                diagnostics["ordinary_auxiliary_optimization_calls"] == 0,
                "fractional LP scope/ordinary-dual contract")
        require(0 <= len(rounds) <= 3 and diagnostics["input_master_variables"] > 0 and
                0 <= diagnostics["relaxed_binary_count"] <= diagnostics["input_master_variables"],
                "fractional LP/model dimensions")
        previous_rows, round_lp_spent = diagnostics["input_master_rows"], 0.0
        certificates, accepted_rounds, referenced_prices = [], [], set()
        local_prices = {p["local_price_index"]: p for p in result["queue_price_search_records"]
                        if p.get("fractional_call") == call_index}
        require(len(local_prices) == sum(p.get("fractional_call") == call_index for p in result["queue_price_search_records"]),
                "duplicate fractional local price index")
        for round_index, row in enumerate(rounds):
            require(row["rows_before"] == previous_rows, "fractional copied/current LP row count mismatch")
            require(math.isfinite(row["lp_time_budget"]) and 0 < row["lp_time_budget"] <= .3 and
                    row["lp_time_budget"] <= lp_budget - round_lp_spent + 1e-6,
                    "fractional LP request exceeds phase remainder")
            require(math.isfinite(row["lp_seconds"]) and row["lp_seconds"] >= 0, "fractional LP actual time")
            round_lp_spent += row["lp_seconds"]
            require(type(row["fractional_coordinates"]) is int and
                    0 <= row["fractional_coordinates"] <= diagnostics["relaxed_binary_count"], "fractional coordinate count")
            certificate = row["certified_lower"]
            if certificate is not None:
                require(row["optimal"] is True and row["has_solution"] is True and row["feasible"] is True,
                        "LP lower bound lacks optimal feasible status")
                require(math.isfinite(row["objective"]) and math.isfinite(certificate), "nonfinite certified LP objective")
                expected = D(row["objective"]) - D(1e-7) * (1 + abs(D(row["objective"])))
                require(D(certificate) <= expected + D(2e-12) * (1 + abs(expected)), "LP bound omitted safety margin")
                close(certificate, expected, "LP objective-minus-margin", relative=2e-12)
                certificates.append(certificate)
            if row["accepted"]:
                require(certificate is not None and len(accepted_rounds) < 2, "fractional cut without eligible LP / beyond cap")
                require(math.isfinite(row["theta"]) and math.isfinite(row["cut_violation"]), "fractional cut/theta nonfinite")
                threshold = D(1e-5) * (1 + abs(D(row["objective"])))
                close(row["acceptance_threshold"], threshold, "fractional acceptance threshold", relative=2e-12)
                require(D(row["cut_violation"]) > D(row["acceptance_threshold"]), "fractional row lacks marginal violation")
                require(math.isfinite(row["queue_separation_violation"]) and
                        D(row["queue_separation_violation"]) > D(row["acceptance_threshold"]),
                        "clamped fractional point lacks marginal violation")
                accepted_rounds.append(round_index)
            require(row["rows_after"] == row["rows_before"] + int(row["accepted"]), "fractional LP row update count")
            previous_rows = row["rows_after"]
            price_index = row.get("price_record_index")
            if price_index is not None:
                require(price_index in local_prices and price_index not in referenced_prices,
                        "fractional LP price reference missing/duplicate")
                referenced_prices.add(price_index)
                price = local_prices[price_index]
                require(price["candidate_id"] == record["candidate_id"] and price["stage"] == record["stage"]
                        and price["accepted"] is row["accepted"], "fractional price/LP acceptance identity")
        require(round_lp_spent <= lp_actual + 1e-6, "fractional LP time exceeds charged LP phase")
        require(referenced_prices == set(local_prices), "unlinked fractional price evaluation")
        close(price_actual, sum(D(p["seconds"]) for p in local_prices.values()), "fractional price time", relative=2e-9)
        accepted = [q for q in result["queue_polymatroid_cut_records"] if q.get("fractional_call") == call_index]
        require(len(accepted) == len(accepted_rounds), "fractional accepted row count differs")
        require([q["fractional_round"] for q in accepted] == accepted_rounds, "fractional returned cut order differs")
        for queue in accepted:
            row = rounds[queue["fractional_round"]]
            require(queue["candidate_id"] == record["candidate_id"] and queue["stage"] == record["stage"],
                    "fractional queue row identity")
            close(queue["theta"], D(row["theta"]), "fractional row LP theta")
            close(queue["violation"], D(row["queue_separation_violation"]), "fractional clamped-point violation")
            actual_design = [D(v) for v in row["actual_design"]]
            require(len(actual_design) == len(queue["design"]) and all(v.is_finite() for v in actual_design),
                    "fractional actual LP design dimensions/finite")
            for saved, value in zip(queue["design"], actual_design):
                close(saved, min(Decimal(1), max(ZERO, value)), "fractional actual/clamped design", relative=2e-13)
            theta_index = len(actual_design)
            actual_violation = D(queue["rhs"]) - sum(D(c) * (D(row["theta"]) if j == theta_index else actual_design[j])
                                                     for j, c in queue["coefficients"])
            close(row["cut_violation"], actual_violation, "fractional actual LP row violation", relative=2e-8)
        certified = record["certified_lower"]
        if certificates:
            close(certified, D(max(certificates)), "fractional stage certificate is maximum eligible LP bound", relative=2e-12)
        else:
            require(certified is None, "fractional stage certificate without an eligible LP")
        before, after, upper = (record[k] for k in ("global_lower_before", "global_lower_after", "global_upper"))
        require(all(math.isfinite(v) for v in (before, after, upper)), "fractional global bounds nonfinite")
        expected = before
        if certified is not None:
            require(certified <= upper + 1e-7 * (1 + abs(upper)), "fractional LP certificate contradicts UB")
            expected = max(before, min(upper, certified))
        close(after, D(expected), "fractional global LB uses only eligible LP certificate", relative=2e-12)
        lp_spent += lp_actual
        price_spent += price_actual
        stage_spent += actual
        total_rounds += len(rounds)
        total_cuts += len(accepted_rounds)
    for field, value in (("lp_seconds", lp_spent), ("price_seconds", price_spent), ("seconds", stage_spent)):
        close(summary[field], D(value), "fractional summary " + field, relative=2e-12)
    require(summary["donor_refreshes"] >= previous_donor, "fractional donor refresh counter")
    if enabled:
        close(result["queue_price_search_summary"]["seconds"], D(price_spent), "all treatment price time belongs to fractional stages", relative=2e-9)
    return dict(passed=True, calls=len(records), lp_solves=total_rounds, cuts=total_cuts,
                lp_seconds=lp_spent, price_seconds=price_spent, seconds=stage_spent,
                maximum_call_overrun_seconds=maximum_overrun,
                lp_feasibility_scope="recorded native feasibility/status plus static full-master-copy review; no independent solver invocation")


def audit_new_mechanisms(result, enabled, seconds=600.0, target_gap=.01, fractional_enabled=False):
    """Verify recorded allocations/accounting, not hypothetical solver hardness."""
    output = {"passed": True, "donor_full_dual_feasibility_required": False}
    for name, call_limit in (("queue_price_search", .5), ("target_master_refinement", 5.0)):
        records, summary = result[name + "_records"], result[name + "_summary"]
        require(summary["enabled"] is enabled and summary["calls"] == len(records), name + " count/flag")
        require(enabled or not records, name + " active with its flag off")
        require(summary["budget_fraction"] == .02 and summary["call_limit_seconds"] == call_limit,
                name + " budget declaration")
        elapsed, maximum_overrun = 0.0, 0.0
        for record in records:
            allocated, actual = record["time_budget"], record["seconds"]
            require(all(math.isfinite(v) for v in (allocated, actual)) and allocated > 0 and actual >= 0,
                    name + " invalid time record")
            require(allocated <= min(call_limit, .02 * seconds - elapsed) + 1e-7,
                    name + " exceeds remaining dedicated allocation")
            elapsed += actual
            maximum_overrun = max(maximum_overrun, actual - allocated)
        close(summary["seconds"], D(elapsed), name + " accumulated seconds", relative=2e-12)
        close(result["timings"].get(name, 0.0), D(elapsed), name + " timer", relative=2e-12)
        output[name] = dict(calls=len(records), seconds=elapsed,
            allocated_seconds=sum(r["time_budget"] for r in records),
            maximum_call_overrun_seconds=maximum_overrun,
            final_cumulative_overrun_seconds=max(0.0, elapsed - .02 * seconds))

    price_summary = result["queue_price_search_summary"]
    require(price_summary["max_evaluations"] == 7 and price_summary["auxiliary_dual_optimization"] is False,
            "price search evaluation/ordinary dual contract")
    statuses, evaluations, accepted = Counter(), 0, 0
    for record in result["queue_price_search_records"]:
        allowed_stages = ("fractional_queue_root", "fractional_queue_integer") if fractional_enabled else ("root_native_prices", "integer_native_prices")
        require(record["stage"] in allowed_stages, "price search at undeclared stage")
        diagnostics = record["diagnostics"]
        if diagnostics.get("status") == "price_search_error":
            require(fractional_enabled and record["accepted"] is False and diagnostics.get("error"),
                    "failed fractional price search cannot accept a row")
            statuses[diagnostics["status"]] += 1
            continue
        require(diagnostics["ordinary_auxiliary_optimization_calls"] == 0
                and diagnostics["master_added_variables"] == 0
                and diagnostics["direct_global_bound_update"] is False, "price search changes certificate scope")
        rows = diagnostics["evaluations"]
        require(diagnostics["max_evaluations"] == 7 and diagnostics["evaluation_count"] == len(rows)
                and len(rows) <= 7, "price evaluation count")
        require(0 <= diagnostics["complete_scans"] <= diagnostics["scan_attempts"] <= 7,
                "price complete-scan accounting")
        require(1 <= diagnostics["profile_count_before"] <= 8 and 1 <= diagnostics["profile_count_after"] <= 8,
                "price profile-pool cap")
        require(all(r["complete_domain"] is True and 0 <= r["line_weight"] <= 1
                    and math.isfinite(r["value"]) for r in rows), "price evaluation domain")
        before, after = diagnostics["best_before"], diagnostics["best_after"]
        if before is not None:
            require(after is not None and after >= before - 2e-9 * (1 + abs(before)),
                    "price search regressed its same-point incumbent row")
        if diagnostics.get("atomic_rollback"):
            require(diagnostics["profile_added"] is False and
                    diagnostics["profile_count_after"] == diagnostics["profile_count_before"],
                    "failed search mutated the retained pool")
            if before is not None:
                close(after, D(before), "price rollback value")
        if record["accepted"]:
            require(any(row["stage"] == record["stage"] and row["candidate_id"] == record["candidate_id"]
                        for row in result["queue_polymatroid_cut_records"]), "accepted search lacks a certified row")
        statuses[diagnostics["status"]] += 1
        evaluations += len(rows)
        accepted += bool(record["accepted"])
    output["queue_price_search"].update(statuses=dict(statuses), evaluations=evaluations, accepted=accepted)

    target_summary = result["target_master_refinement_summary"]
    require(target_summary["cooldown_seconds"] == 30 and target_summary["direct_candidate_bound_update"] is False,
            "target refinement cooldown/LB contract")
    masters = {row["solve"]: row for row in result["master_solves"]}
    previous_end = None
    for record in result["target_master_refinement_records"]:
        before, after, upper = (record[k] for k in ("global_lower_before", "global_lower_after", "global_upper"))
        require(all(math.isfinite(v) for v in (before, after, upper)) and upper > 0, "target finite bounds")
        outer = (upper - before) / max(1.0, abs(upper))
        require(target_gap < outer <= 1.25 * target_gap + 1e-12, "target outside near-gap eligibility")
        require(record["supported_cg_passes"] >= 3 and record["bound_stalled_passes"] >= 3,
                "target before maturity/stagnation")
        started, previous = record["started_elapsed_seconds"], record["previous_refinement_elapsed_seconds"]
        require(started - previous >= 30 - 1e-7 and started + record["time_budget"] <= seconds + 1e-5,
                "target violates cooldown/global remaining allocation")
        if previous_end is not None:
            # The caller stores the post-call wall clock after a few assignments.
            require(previous >= previous_end - 1e-7 and started - previous_end >= 30 - 1e-7,
                    "target cooldown does not follow previous actual finish")
        previous_end = started + record["seconds"]
        requested = D(upper) - D(target_gap) * max(Decimal(1), abs(D(upper)))
        require(D(record["target_bound"]) >= requested, "target bound rounded downward")
        close(record["target_bound"], requested, "target bound", relative=2e-12)
        close(record["bound_deficit"], D(record["target_bound"]) - D(before), "target deficit")
        master = masters[record["master_solve"]]
        candidate = master["candidate_objective"]
        require(candidate is not None and candidate > record["target_bound"], "target lacks finite candidate margin")
        close(record["candidate_margin"], D(candidate) - D(record["target_bound"]), "target candidate margin")
        require(record["previous_relative_gap_target"] == master["relative_gap_target"], "target changed preceding tolerance")
        requested_tolerance = .5 * record["candidate_margin"] / (1 + abs(candidate))
        require(0 <= record["relative_gap_target"] <=
                min(record["previous_relative_gap_target"], requested_tolerance) + 2e-15,
                "target tolerance is weaker than the declared request")
        bound = record["certified_bound"]
        expected = before
        if bound is not None:
            require(math.isfinite(bound) and bound <= upper + 1e-7 * (1 + abs(upper)), "target inconsistent best bound")
            expected = max(before, min(upper, bound))
        close(after, D(expected), "target LB must come only from certified best bound", relative=2e-12)
    output["fractional"] = audit_fractional_stages(result, fractional_enabled, seconds)
    return output



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
    """Remove only unsupported exactly idle task assignments; preserve all operations."""
    label, design, cost = point
    values, ids = model["values"], model["ids"]
    L, H = values["n_l"], values["n_h"]
    scope, groups = prior.assignment_groups(model, demand, assignment_scope=solution.get("assignment_scope"))
    require(len(solution["assignment"]) == len(groups), "Canonicalization assignment dimension")
    offset = L + H + H * len(ids["levels"])
    supported, nonzero = set(), set()
    for item in catalogue["missions"]:
        path = item["path"]
        second = None if path["second_od"] in (None, -1) else path["second_od"]
        key = (path["first_od"], path["b"][0], second)
        if all(solution["airline"][u][v] >= .5 for u, v, _ in path["arc_counts"]):
            supported.add(key)
        if solution["flow"][path["id"]] != 0:
            nonzero.add(key)
    normalized, changes, errors = list(design), [], []
    for q, tasks in enumerate(groups):
        for l in range(L):
            if not solution["assignment"][q][l]:
                continue
            missing = [(r, second) for r, second, rate in tasks if rate > 0 and (r, l, second) not in supported]
            if not missing:
                continue
            has_nonzero = any((r, l, second) in nonzero for r, second, rate in tasks)
            evidence = dict(entity=q, assignment_scope=scope, origin_base=l, missing_positive_tasks=missing,
                            rho=solution["rho"][l], has_nonzero_flow=has_nonzero)
            if solution["rho"][l] != 0 or has_nonzero:
                errors.append(dict(**evidence,
                    reason="Unsupported assignment cannot be removed while preserving the saved operations exactly"))
                continue
            normalized[offset + q*L + l] = 0
            changes.append(dict(**evidence,
                reason="Zero-availability and exactly zero-flow assignment removed; operations and cost unchanged"))
    return (label + ":canonical", normalized, cost), dict(point=label, passed=not errors,
        changed_assignments=changes, errors=errors, assignment_scope=scope,
        scope="Optional cost-free assignment canonicalization; no rounding of rho or flows")


def fresh_catalogue(folder, payload, instance):
    """Validate the new full-model export before using any zero-flow path."""
    from src.data.instances import to_planning_data
    from src.data.standard import load_standard
    from src.operations.validation import _physical, _segment

    folder, instance = Path(folder), Path(instance)
    result, solution = payload["result"], payload["solution"]
    require(result.get("method") == "full_path" and result.get("catalogue_complete") is True
            and result.get("support_complete") is True and result.get("construction_complete") is True,
            "Fresh full CPLEX did not finish the complete-domain model")
    export = result["complete_catalogue_export"]
    require(export["file"] == "full_catalogue.json", "Unexpected catalogue filename")
    path = folder / export["file"]
    digest = base.sha256(path)
    require(export["sha256"] == digest, "Complete catalogue export hash mismatch")
    require(base.read(folder / "artifact_audit.json")["artifact_sha256"].get(path.name) == digest,
            "Complete catalogue export was not covered by the native artifact audit")
    raw = base.read(path)
    require(raw["schema"] == export["schema"] == "mlnd-complete-catalogue/1"
            and raw["complete"] is True and export["complete"] is True,
            "Complete catalogue schema/state mismatch")
    for field in ("model_version", "demand_formulation", "max_orders", "assignment_scope", "flow_policy"):
        require(raw[field] == result[field], "Complete catalogue model identity differs: " + field)
    require(raw["scope"] == result["path_scope"], "Complete catalogue path scope differs")
    require(raw.get("first_leg_consistency") is False, "Complete catalogue first-leg policy differs")
    require(raw["counts"] == export["counts"] == result["catalogue"], "Complete catalogue counts disagree")
    for name in ("segments", "service", "empty"):
        require(len(raw[name]) == raw["counts"][name], "Missing complete catalogue records: " + name)
        require([p["id"] for p in raw[name]] == list(range(raw["counts"][name])),
                "Complete catalogue IDs are not dense and ordered: " + name)
    require(len(solution["flow"]) == raw["counts"]["service"]
            and len(solution["empty_flow"]) == raw["counts"]["empty"], "Complete flow vector dimensions")
    catalogue = dict(segments=raw["segments"],
        missions=[dict(path=p, flow=solution["flow"][p["id"]]) for p in raw["service"]],
        relocations=[dict(path=p, flow=solution["empty_flow"][p["id"]]) for p in raw["empty"]])
    data = to_planning_data(load_standard(instance, case="K" + str(result["max_orders"])))
    for segment in raw["segments"]:
        _segment(data, segment)
    for service in raw["service"]:
        _physical(data, service)
    for empty in raw["empty"]:
        _physical(data, empty, empty=True)
    coverage = prior._check_record_coverage(payload, catalogue)
    return catalogue, dict(source_directory=str(folder), source="fresh_full_cplex_complete_export",
        sha256=digest, catalogue_digest=prior.catalogue_digest(catalogue), counts=raw["counts"],
        independently_rebuilt_physical_paths=raw["counts"]["service"] + raw["counts"]["empty"],
        current_record_coverage=coverage, termination_reason=result.get("termination_reason"))


def same_model(payload, other, instance):
    digest = base.sha256(Path(instance) / "manifest.json")
    for current in (payload, other):
        require(current["result"]["input_manifest_sha256"] == digest, "Input manifest identity mismatch")
        require(current["input_reference"]["manifest"] == base.read(Path(instance) / "manifest.json"),
                "Saved input payload differs")
    for field in ("manifest", "model", "max_orders", "assignment_scope", "delivery_time_scope", "demand_formulation", "first_leg_consistency"):
        require(payload["input_reference"].get(field) == other["input_reference"].get(field),
                "Paired model identity differs: " + field)
    for field in ("single_rates", "pair_rates", "head_rates"):
        require(payload["preprocessed_demand"][field] == other["preprocessed_demand"][field],
                "Paired task demand differs: " + field)
    require(payload["result"]["catalogue"] == other["result"]["catalogue"], "Paired catalogue counts differ")


def task_rows(result, constants, catalogue, payload):
    """Retain the prior task certificate checks without its batch wrapper."""
    errors, rows, evidence = [], [], {}
    theta = constants["design_count"]
    meta = result.get("task_savings_metadata", {})
    if meta.get("enabled"):
        evidence["static"] = prior.audit_allocations(meta, constants, errors, "static")
        row = dict(rhs=meta["courier_cost"] - meta["numerical_margin"], coefficients=[[theta, 1.0],
            *[[constants["assignment"] + g, c] for g, c in enumerate(meta["chord_coefficient"]) if c]])
        excess = D(row["rhs"]) - constants["C0"] + sum(max(ZERO, exact - D(reported))
            for exact, reported in zip(constants["chord_coefficient"], meta["chord_coefficient"]))
        require(excess <= 0, "Static row exceeds the independent chord certificate")
        rows.append(("static", row))
    evidence["resource_dual"] = prior.audit_duals(result, constants, errors, "task_resource")
    rows.extend(("task_resource:" + str(i), row) for i, row in enumerate(result.get("task_resource_cut_records", [])))
    arc = result["task_airline_metadata"]
    require(arc.get("enabled") is True and len(arc["task_upper"]) == constants["N"] ** 2, "Task airline bounds missing")
    for reported, expected in zip(arc["task_upper"], constants["task_upper"]):
        require(-D(1e-10) <= D(reported) - expected <= D(1e-8) * (1 + abs(expected)), "Unsafe task airline bound")
    evidence["incumbent_flow"] = prior.flow_checks(payload["path_flows"], payload["solution"],
        payload["preprocessed_demand"], constants["task_upper"], constants, errors, "incumbent")
    require(not errors, "; ".join(errors))
    return rows, evidence


def audit_case(batch, jobs, state, artifacts):
    from src.experiments.baseline_benchmark_20261029 import record
    runs, points = [], []
    for job in jobs:
        identity = record(batch, job, state["jobs"][job["id"]])
        require(identity["validation_passed"], "Run identity/configuration does not match frozen protocol: " + job["id"])
        folder = batch / state["jobs"][job["id"]]["result_directory"]
        payload, design = load_verified(folder)
        runs.append((job, folder, payload))
        points.append((job["id"], design, payload["result"]["continuous_cost"]))
        for name in ("result.json", "solution.json", "path_flows.json", "input_reference.json", "preprocessed_demand.json",
                     "native_validation.json", "artifact_audit.json", "progress.jsonl"):
            artifacts[str((folder / name).relative_to(batch))] = base.sha256(folder / name)
    full = next(run for run in runs if run[0]["method"] == "full_cplex")
    instance = batch / full[0]["instance"]
    catalogue, provenance = fresh_catalogue(full[1], full[2], instance)
    artifacts[str((full[1] / "full_catalogue.json").relative_to(batch))] = provenance["sha256"]
    constants = input_constants(full[2]["input_reference"]["model"], full[2]["preprocessed_demand"], catalogue, instance,
        assignment_scope=full[2]["input_reference"].get("assignment_scope"), max_orders=full[2]["result"]["max_orders"])
    R, demand = constants["R"], full[2]["preprocessed_demand"]
    require(all(rate >= 0 for rate in demand["single_rates"])
            and all(rate >= 0 for row in demand["pair_rates"] for rate in row), "Negative task demand")
    for r in range(R):
        reconstructed = D(demand["single_rates"][r]) + sum((D(v) for v in demand["pair_rates"][r]), ZERO)
        reconstructed += sum((D(row[r]) for row in demand["pair_rates"]), ZERO)
        require(abs(constants["raw"][r] - reconstructed) <= D(1e-8) * (1 + sum(constants["raw"])), "OD demand not conserved")
        close(demand["head_rates"][r], D(demand["single_rates"][r]) + sum((D(v) for v in demand["pair_rates"][r]), ZERO), "Head rate")
    summaries, profile_cache = [], {}
    for job, folder, payload in runs:
        same_model(payload, full[2], instance)
        coverage = prior._check_record_coverage(payload, catalogue)
        result = payload["result"]
        gap = self_gap_summary(result)
        require(gap["self_gap_consistent"] and gap["bound_order_consistent"]
                and gap["target_gap_met"] == gap["computed_target_met"], "Reported bound/gap inconsistency")
        rows, task_evidence = task_rows(result, constants, catalogue, payload)
        task_check = cross_check_rows(rows, points)
        require(task_check["passed"], "; ".join(task_check["errors"]))
        summary = dict(job=job["id"], instance=job["instance"], method=job["method"],
            full_catalogue_provenance=provenance, record_coverage=coverage, task_certificate=task_evidence,
            task_crosspoint=task_check, gap=gap, ordinary_cuts=0, queue_cuts=0,
            ordinary_crosspoint_checks=0, queue_crosspoint_checks=0,
            ordinary_maximum_violation=0.0, queue_maximum_violation=0.0)
        if job["method"] == "benders_baseline":
            require(result.get("ordinary_auxiliary_dual_optimization") is False, "Ordinary native-dual contract missing")
            summary["mechanisms"] = audit_new_mechanisms(result, True, fractional_enabled=True)
            attempts, seen = [], set()
            for line in (folder / "progress.jsonl").open():
                event = json.loads(line)
                require(event["sequence"] not in seen, "Duplicate progress sequence")
                seen.add(event["sequence"])
                if event.get("event") == "cg_iteration":
                    attempts.extend(event.get("dual_attempts", []))
            require(all(a["method"] == "native_cplex_dual" and a.get("auxiliary_optimization_calls", 0) == 0
                        for a in attempts), "Ordinary auxiliary dual optimization appeared")
            for cut in result.get("benders_cuts", []):
                require(cut.get("condition") is None and all(cut.get(k) is True for k in
                        ("catalogue_complete", "pricing_complete", "support_complete")), "Ordinary cut complete-domain certificate")
                require(len(cut["coefficients"]) == constants["design_count"], "Ordinary cut dimensions")
                summary["ordinary_cuts"] += 1
                for label, design, cost in points:
                    violation = D(cut["intercept"]) + sum(D(c) * D(x) for c, x in zip(cut["coefficients"], design)) - D(cost)
                    summary["ordinary_crosspoint_checks"] += 1
                    summary["ordinary_maximum_violation"] = max(summary["ordinary_maximum_violation"], float(violation))
                    require(violation <= D(1e-7) * (1 + abs(D(cost))), "Ordinary cut excludes " + label)
            queue, qs = result["queue_polymatroid_cut_records"], result["queue_polymatroid_summary"]
            require(qs["enabled"] is True and qs["cuts"] == len(queue) and qs["direct_global_bound_update"] is False,
                    "Queue metadata/count differs")
            kinds = Counter()
            for cut in queue:
                require(all(cut.get(k) is True for k in ("accepted", "valid", "violated")), "Accepted queue row state")
                profile = cut["certificate"]["profile"]
                key = json.dumps(profile, sort_keys=True, separators=(",", ":"))
                if key not in profile_cache:
                    profile_cache[key] = validate_profile(profile, constants, demand, catalogue)
                replay_queue_row(cut, constants)
                kinds[profile["kind"]] += 1
                summary["queue_cuts"] += 1
                for label, design, cost in points:
                    value = D(cut["rhs"]) - sum(D(c) * D(design[j]) for j, c in cut["coefficients"] if j != len(design))
                    violation = value - D(cost)
                    summary["queue_crosspoint_checks"] += 1
                    summary["queue_maximum_violation"] = max(summary["queue_maximum_violation"], float(violation))
                    require(violation <= D(1e-7) * (1 + abs(D(cost))), "Queue cut excludes " + label)
            summary["queue_kinds"] = dict(kinds)
            airline = result.get("airline_resource_cut_records", [])
            require(all(c.get("valid") and c.get("max_dual_violation_after", math.inf) <= 0 for c in airline), "Airline dual metadata")
            check = cross_check_rows([(str(i), c) for i, c in enumerate(airline)], points)
            require(check["passed"], "; ".join(check["errors"]))
            summary["airline_crosspoint"] = check
            own = [p for p in points if p[0] == job["id"]]
            connectivity = [(str(i), c) for i, c in enumerate(result.get("airline_connectivity_metadata", {}).get("rows", []))]
            check = cross_check_rows(connectivity, own)
            require(check["passed"], "; ".join(check["errors"]))
            summary["connectivity_applied_crosspoint"] = check
            summary["connectivity_all_encoding_diagnostic"] = cross_check_rows(connectivity, points)
            fractional = [(str(i), c) for i, c in enumerate(result.get("fractional_path_support_records", []))]
            require(all(c.get("valid") for _, c in fractional), "Fractional support certificate state")
            canonical, evidence = [], []
            for current, point in zip(runs, points):
                normalized, checked = canonicalize_support_point(current[2]["input_reference"]["model"], demand,
                    catalogue, current[2]["solution"], point)
                require(checked["passed"], "Cannot canonicalize unsupported assignments without changing operations")
                canonical.append(normalized); evidence.append(checked)
            check = cross_check_rows(fractional, canonical)
            applied = cross_check_rows(fractional, own)
            require(check["passed"] and applied["passed"], "; ".join(check["errors"] + applied["errors"]))
            summary["fractional_support"] = dict(canonical=check, applied=applied, canonicalization=evidence,
                all_encoding_diagnostic=cross_check_rows(fractional, points))
        summary["passed"] = True
        summaries.append(summary)
    return summaries


def audit(batch, require_complete=False):
    from src.experiments.baseline_benchmark_20261029 import verify
    batch = Path(batch).resolve()
    manifest, state = verify(batch), base.read(batch / "state.json")
    pending = [j["id"] for j in manifest["jobs"] if state["jobs"][j["id"]]["status"] != "completed"]
    require(not require_complete or not pending, "All eight declared jobs must complete for the final audit")
    rows, errors, artifacts = [], [], {}
    with localcontext() as context:
        context.prec = 50
        for instance in dict.fromkeys(j["instance"] for j in manifest["jobs"]):
            jobs = [j for j in manifest["jobs"] if j["instance"] == instance]
            if any(j["id"] in pending for j in jobs):
                continue
            try:
                rows.extend(audit_case(batch, jobs, state, artifacts))
            except (ValueError, KeyError, TypeError, IndexError, ArithmeticError, OSError) as error:
                errors.append(instance + ": " + str(error))
    report = dict(schema="baseline-independent-cut-audit/1", passed=not errors, complete=not pending,
        errors=errors, pending_jobs=pending, rows=rows, solver_invoked=False,
        production_manifest_sha256=base.sha256(batch / "experiment_manifest.json"),
        binary_sha256=manifest["binary_sha256"], reviewer_sha256=base.sha256(__file__), artifact_sha256=artifacts,
        audit_dependencies_sha256={name: base.sha256(Path(__file__).with_name(name)) for name in ("task_cost_audit.py",)},
        mathematical_extraction_provenance_sha256=base.sha256(batch / "provenance/AUDIT_EXTRACTION.json"),
        numerical_scope="50-digit Decimal; unchanged directional tolerance and propagated conservative-rounding intervals",
        proof_scope="Complete fresh catalogue physical reconstruction and queue price/polymatroid replay; ordinary/airline finite crosspoints are necessary falsification checks, not universal proofs; connectivity/fractional support preserve canonical optima")
    output = batch / "analysis"
    output.mkdir(exist_ok=True)
    write_json(output / "CUT_AUDIT.json", report)
    lines = ["# 新 baseline 独立证据审计", "", f"通过：{report['passed']}；全部完成：{report['complete']}；求解器调用：无。", "",
        "完整目录来自本批新 full CPLEX，校验哈希、原始输入/任务率、全部 ID、每条路径物理量及 compact witness 对应关系。",
        "queue 数学重建与 exact/ordered-double demand 方向检查保持不变；跨点检查不替代通用有效性证明。", "",
        "| 作业 | 普通行 | queue 行 | 普通跨点 | queue 跨点 | 通过 |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) for k in ("job", "ordinary_cuts", "queue_cuts", "ordinary_crosspoint_checks", "queue_crosspoint_checks", "passed")) + " |")
    if errors:
        lines += ["", *["- " + error for error in errors]]
    (output / "CUT_AUDIT.md").write_text("\n".join(lines) + "\n")
    require(not errors, "Independent cut audit failed; inspect analysis/CUT_AUDIT.json")
    return dict(passed=True, complete=not pending, report=str(output / "CUT_AUDIT.md"), runs=len(rows))
