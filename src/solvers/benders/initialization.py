"""Starting designs and incumbent construction for integer-master Benders."""

import numpy as np
from src.model.network import network, levels, facility_costs, incidence


def fix_master(data, x, z, fixed_facility=None, fixed_capacity_level=None):
    """Apply optional fixed strategic decisions, rejecting inconsistent inputs."""
    L, H = data["n_l"], data["n_h"]
    _, capacity, _ = levels(data)
    facility = None if fixed_facility is None else np.asarray(fixed_facility, float)
    cabinets = (
        None
        if fixed_capacity_level is None
        else np.asarray(fixed_capacity_level, float)
    )
    if facility is not None:
        if facility.shape != (network(data)["N"],) or np.any(
            (facility != 0) & (facility != 1)
        ):
            raise ValueError("Invalid fixed facilities")
        x.LB = facility
        x.UB = facility
    if cabinets is not None:
        if (
            cabinets.shape != capacity.shape
            or np.any((cabinets != 0) & (cabinets != 1))
            or np.any(cabinets.sum(axis=1) > 1)
        ):
            raise ValueError("Invalid fixed capacity levels")
        if facility is not None and np.any(cabinets.sum(axis=1) != facility[L : L + H]):
            raise ValueError("Fixed levels do not match facilities")
        z.LB = cabinets
        z.UB = cabinets
    return facility, cabinets


def baseline(data, fixed_facility=None, fixed_capacity_level=None):
    """The all-ground plan is feasible for any consistent fixed facilities."""
    L, H, D = data["n_l"], data["n_h"], data["n_od"]
    net = network(data)
    sizes, capacity, level_cost = levels(data)
    facility = np.zeros(net["N"]) if fixed_facility is None else fixed_facility.copy()
    cabinets = np.zeros_like(capacity)
    if fixed_capacity_level is not None:
        cabinets = fixed_capacity_level.copy()
        facility[L : L + H] = cabinets.sum(axis=1)
    else:
        for i in range(H):
            if facility[L + i]:
                cabinets[i, np.argmin(level_cost[i])] = 1
    ground = float(data["demand"] @ data["ground"])
    solution = dict(
        facility=facility,
        capacity_level=cabinets,
        hub_count=cabinets @ sizes,
        assignment=np.zeros((D, L, L)),
        airline=np.zeros(net["count"]),
        rho=np.zeros(L),
        idle=np.zeros(L),
        empty=np.zeros((L, L)),
        fleet=0.0,
    )
    return dict(
        objective=ground
        + facility_costs(data) @ facility
        + float((level_cost * cabinets).sum()),
        recourse=ground,
        solution=solution,
        missions=[],
        relocations=[],
        hub_load=np.zeros(H),
        airline_load=np.zeros(net["count"]),
        served=0.0,
        primal_residual=0.0,
    )


def initial_designs(data, prices, facility, capacity_level):
    """Two optional starts: same-base return, then unrestricted base pairs."""
    L, H, D = data["n_l"], data["n_h"], data["n_od"]
    net = network(data)
    _, capacity, _ = levels(data)
    f = np.ones(net["N"]) if facility is None else facility.copy()
    z = np.zeros_like(capacity)
    if capacity_level is None:
        for i in range(H):
            if f[L + i]:
                z[i, np.argmax(capacity[i])] = 1
    else:
        z = capacity_level.copy()
        f[L : L + H] = z.sum(axis=1)
    y = f[net["uv"][:, 0]] * f[net["uv"][:, 1]]
    previous = None
    for same_base in (True, False):
        costs = prices["cost"].copy()
        costs[:, f[:L] == 0, :] = np.inf
        costs[:, :, f[:L] == 0] = np.inf
        if same_base:
            costs[:, ~np.eye(L, dtype=bool)] = np.inf
        flat = costs.reshape(D, L * L)
        chosen = flat.argmin(axis=1)
        use = flat[np.arange(D), chosen] < 0
        assignment = np.zeros((D, L, L))
        assignment[np.arange(D)[use], chosen[use] // L, chosen[use] % L] = 1
        if previous is None or not np.array_equal(assignment, previous):
            yield f, z, assignment, y
        previous = assignment


def incumbent(
    data,
    answer,
    facility,
    cabinets,
    assignment,
    airlines,
    *,
    cleanup=True,
    fixed_facility=None,
    fixed_capacity_level=None,
):
    """Build a feasible result; optional cleanup closes unused infrastructure."""
    if answer["primal_residual"] > 1e-5:
        return None
    L, H = data["n_l"], data["n_h"]
    net = network(data)
    sizes, capacity, level_cost = levels(data)
    f, z, y = facility.copy(), cabinets.copy(), airlines.copy()
    if cleanup:
        y[:] = 0
        used = set()
        for path in answer["missions"] + answer["relocations"]:
            for edge, _ in incidence(net, path["trace"]):
                y[edge] = 1
                used.update(net["uv"][edge])
        if fixed_capacity_level is None:
            z[:] = 0
            for i, load in enumerate(answer["hub_load"]):
                if L + i in used or (
                    fixed_facility is not None and fixed_facility[L + i]
                ):
                    eligible = np.flatnonzero(capacity[i] >= load - 1e-7)
                    if not len(eligible):
                        return None
                    z[i, eligible[np.argmin(level_cost[i, eligible])]] = 1
            if fixed_facility is None:
                f[L : L + H] = z.sum(axis=1)
        if fixed_facility is None:
            for k in range(L + H, net["N"]):
                f[k] = int(k in used)
            assigned = assignment.sum(axis=(0, 2)) + assignment.sum(axis=(0, 1))
            for l in range(L):
                f[l] = int(l in used or assigned[l] > 0.5 or answer["rho"][l] > 0)
    if np.any(y > f[net["uv"][:, 0]]) or np.any(y > f[net["uv"][:, 1]]):
        return None
    opening = facility_costs(data) @ f + (level_cost * z).sum() + net["fees"] @ y
    return dict(
        objective=float(answer["objective"] + opening),
        recourse=answer["objective"],
        solution=dict(
            facility=f,
            capacity_level=z,
            hub_count=z @ sizes,
            assignment=assignment.copy(),
            airline=y,
            **{k: answer[k] for k in ("rho", "idle", "empty", "fleet")},
        ),
        **{
            k: answer[k]
            for k in (
                "missions",
                "relocations",
                "hub_load",
                "airline_load",
                "served",
                "primal_residual",
            )
        },
    )


def load_initial(data, result, best, fixed_facility=None, fixed_capacity_level=None):
    """Validate and restore a saved incumbent and its certified global bound."""
    sol = {
        k: np.asarray(v, float).copy() if k != "fleet" else float(v)
        for k, v in result["_solution"].items()
    }
    if any(
        np.shape(sol[k]) != np.shape(value) for k, value in best["solution"].items()
    ):
        raise ValueError("Initial result dimensions do not match the instance")
    for key, expected in (
        ("facility", fixed_facility),
        ("capacity_level", fixed_capacity_level),
    ):
        if expected is not None and not np.array_equal(sol[key], expected):
            raise ValueError(f"Initial result violates fixed {key}")
    if (
        not np.isfinite(result["objective"])
        or float(result.get("primal_residual", 0)) > 1e-5
    ):
        raise ValueError("Initial result must have a finite feasible incumbent")
    if float(result["objective"]) < best["objective"]:
        net = network(data)
        _, _, costs = levels(data)
        opening = (
            facility_costs(data) @ sol["facility"]
            + (costs * sol["capacity_level"]).sum()
        )
        opening += net["fees"] @ sol["airline"]
        best = dict(
            objective=float(result["objective"]),
            recourse=float(result["objective"] - opening),
            solution=sol,
            missions=result["_missions"],
            relocations=result["_relocations"],
            hub_load=np.asarray(result["hub_load"]),
            airline_load=np.asarray(result["airline_load"]),
            served=float(result["served"]),
            primal_residual=float(result.get("primal_residual", 0)),
        )
    bound = float(result.get("bound", 0))
    if not np.isfinite(bound) or bound > best["objective"] + 1e-5:
        raise ValueError("Invalid initial global lower bound")
    return best, max(0.0, bound)
