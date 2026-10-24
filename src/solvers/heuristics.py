"""Design proposals and independently checked cleanup of fixed path flows.

Design proposals are not primal solutions. The caller must solve and validate
their fixed-design recourse before using them as an incumbent or upper bound.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.model.locker_capacity import occupancy_limits
from src.operations.validation import build_operations
from .formulation import shapes


def _support(data, paths):
    if not paths.complete:
        raise ValueError("Heuristic designs require the complete path catalogue")
    double = {
        (*p.b, p.first_od, p.second_od)
        for p in paths.service_paths
        if p.mode == "double"
    }
    empty = {}
    for p in paths.empty_paths:
        cost = p.cost + data.fleet_cost * p.time
        empty[p.b] = min(empty.get(p.b, float("inf")), cost)
    offers = {}
    for p in paths.service_paths:
        if p.mode != "single" or data.demand[p.first_od] <= 0:
            continue
        reposition = 0.0 if p.b[0] == p.b[1] else empty.get(p.b[::-1], float("inf"))
        saving = (
            data.ground_cost[p.first_od]
            - p.cost
            - data.fleet_cost * p.time
            - reposition
        )
        if saving > 0:
            key = (p.first_od, *p.b)
            offers[key] = max(offers.get(key, 0.0), float(saving))
    return double, offers


def _compatible(data, double, assignment, od, pair):
    if data.effective_delta == 0:
        return True
    members = np.flatnonzero((assignment[:, *pair] > 0.5) & (data.demand > 0)).tolist()
    for other in [*members, od]:
        for first, second in ((od, other), (other, od)):
            if data.kappa[first, second] > 0 and (*pair, first, second) not in double:
                return False
    return True


def _open_design(data, assignment):
    design = {name: np.zeros(shape) for name, shape in shapes(data).items()}
    design["assignment"][:] = assignment
    if assignment.any():
        # The SOCP may select any supported route; cleanup removes unused assets.
        design["facility"][:] = 1
        design["airline"][:] = data.allowed_links
        for i in range(data.n_h):
            design["capacity_level"][i, np.argmax(data.slot_counts[i])] = 1
    return design


def initial_design(data, paths):
    """Greedily assign profitable single-path candidates to compatible pools.

    The score is demand times operating savings, including mission busy time
    and a reverse empty mission for cross-base proposals. Opening costs and
    idle inventory are decided by subsequent recourse and cleanup, so the score
    is only a proposal heuristic. K2 checks both ordered directions and AA.
    """
    double, offers = _support(data, paths)
    assignment = np.zeros((data.n_od, data.n_l, data.n_l))
    ranked = sorted(
        offers, key=lambda k: (-data.demand[k[0]] * offers[k], -data.demand[k[0]], k)
    )
    for od, departure, arrival in ranked:
        pair = (departure, arrival)
        if not assignment[od].any() and _compatible(data, double, assignment, od, pair):
            assignment[od, *pair] = 1
    return _open_design(data, assignment)


def design_neighbors(data, paths, incumbent, max_neighbors=3):
    """Return at most a few compatible one-OD additions for SOCP evaluation.

    Existing assignments are retained. This bounded proposal generator performs
    no optimization and makes no claim that a proposal improves the incumbent.
    """
    if (
        isinstance(max_neighbors, bool)
        or int(max_neighbors) != max_neighbors
        or max_neighbors < 0
    ):
        raise ValueError("max_neighbors must be a nonnegative integer")
    double, offers = _support(data, paths)
    assignment = np.asarray(incumbent["_solution"]["assignment"], dtype=float)
    if (
        assignment.shape != (data.n_od, data.n_l, data.n_l)
        or not np.isin(assignment, (0, 1)).all()
    ):
        raise ValueError("Incumbent assignments must have the binary design shape")
    ranked = sorted(offers, key=lambda k: (-data.demand[k[0]], -offers[k], k))
    neighbors = []
    for od, departure, arrival in ranked:
        if len(neighbors) >= max_neighbors:
            break
        pair = (departure, arrival)
        if assignment[od].any() or not _compatible(data, double, assignment, od, pair):
            continue
        proposal = assignment.copy()
        proposal[od, *pair] = 1
        neighbors.append(_open_design(data, proposal))
    return neighbors


def cleanup_incumbent(data, result):
    """Reduce fixed asset costs without changing any assignment or path flow.

    Every nonzero flow protects all its physical nodes and links, including
    arbitrarily small flows and transit-only lockers. The fleet, rho and idle
    inventory remain unchanged. Invalid input is rejected; a candidate failing
    the independent final check falls back to an unchanged independent copy.
    """
    operations = build_operations(data, result)
    if not operations["summary"]["checks"]["passed"]:
        raise ValueError("Cleanup requires an independently valid incumbent")
    candidate = deepcopy(result)
    sol = candidate["_solution"]
    used_nodes, used_links = set(), set()
    for records in (result.get("_missions", []), result.get("_relocations", [])):
        for record in records:
            if record["flow"] == 0:
                continue
            for leg in record["path"]["legs"]:
                used_nodes.update(leg)
                used_links.update(zip(leg, leg[1:]))
    for u, v in np.ndindex(sol["airline"].shape):
        if (u, v) not in used_links:
            sol["airline"][u, v] = 0
    thresholds = occupancy_limits(data.slot_counts, data.slot_epsilon[:, None])
    for row in operations["tables"]["lockers"]:
        i, load = row["locker"], row["mean_occupancy"]
        if data.n_l + i not in used_nodes and load == 0:
            sol["capacity_level"][i] = 0
            sol["facility"][data.n_l + i] = 0
        elif sol["facility"][data.n_l + i] > 0.5:
            levels = np.flatnonzero(thresholds[i] >= load)
            if len(levels):
                level = min(
                    levels,
                    key=lambda s: (data.slot_cost[i, s], data.slot_counts[i, s], s),
                )
                old_cost = float(data.slot_cost[i] @ sol["capacity_level"][i])
                if data.slot_cost[i, level] <= old_cost:
                    sol["capacity_level"][i] = 0
                    sol["capacity_level"][i, level] = 1
    for base in range(data.n_l):
        assigned = (
            sol["assignment"][:, base, :].any() or sol["assignment"][:, :, base].any()
        )
        if (
            base not in used_nodes
            and not assigned
            and sol["rho"][base] == 0
            and sol["idle"][base] == 0
        ):
            sol["facility"][base] = 0
    costs = {
        row["component"]: row["cny_per_hour"] for row in operations["tables"]["costs"]
    }
    opening = float(
        data.base_cost @ sol["facility"][: data.n_l]
        + np.sum(data.slot_cost * sol["capacity_level"])
        + np.sum(data.link_cost * sol["airline"])
    )
    continuous = sum(
        costs[key] for key in ("fleet", "service", "repositioning", "direct_courier")
    )
    candidate.update(
        objective=opening + continuous, opening_cost=opening, continuous_cost=continuous
    )
    if candidate.get("bound") is not None and candidate.get("gap") is not None:
        candidate["gap"] = max(0.0, candidate["objective"] - candidate["bound"]) / max(
            1.0, abs(candidate["objective"])
        )
    verified = build_operations(data, candidate)["summary"]["checks"]["passed"]
    if not verified or candidate["objective"] > result["objective"] + 1e-8:
        return deepcopy(result)
    candidate["independently_validated"] = True
    candidate["cleanup"] = dict(
        fixed_flows=True, saving=float(result["objective"] - candidate["objective"])
    )
    return candidate
