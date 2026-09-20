"""Common feasible solution packaging and independent validation."""

import numpy as np
from src.model.network import facility_costs, incidence, levels, network
from src.operations.validation import build_operations


def build_result(data, solution, missions, relocations, global_lower_bound):
    """Build and independently validate a feasible incumbent with its global bound."""
    if not np.isfinite(global_lower_bound):
        raise ValueError("A finite full-model lower bound is required")
    net = network(data)
    sizes, _, levelcost = levels(data)
    sol = {
        key: np.asarray(solution[key], float).copy()
        for key in (
            "facility",
            "capacity_level",
            "assignment",
            "airline",
            "rho",
            "idle",
        )
    }
    sol["fleet"] = float(solution["fleet"])
    L, H = data["n_l"], data["n_h"]
    sol["hub_count"] = sol["capacity_level"] @ sizes
    sol["empty"] = np.zeros((L, L))
    hubload = np.zeros(H)
    linkload = np.zeros(net["count"])
    for service, records in ((True, missions), (False, relocations)):
        for row in records:
            tr, flow = row["trace"], row["flow"]
            for edge, count in incidence(net, tr):
                linkload[edge] += count * flow
            if service:
                hubload[tr["i"]] += flow
                hubload[tr["j"]] += flow
            else:
                sol["empty"][tr["l"], tr["m"]] += flow
    opening = float(
        facility_costs(data) @ sol["facility"]
        + np.sum(levelcost * sol["capacity_level"])
        + net["fees"] @ sol["airline"]
    )
    c0 = float(np.asarray(data["demand"]) @ np.asarray(data["ground"]))
    operating = c0 + float(data["fleet_cost"]) * sol["fleet"]
    operating += sum(
        (row["trace"]["operating_cost"] - data["ground"][row["od"]]) * row["flow"]
        for row in missions
    )
    operating += sum(
        row["trace"]["operating_cost"] * row["flow"] for row in relocations
    )
    objective = opening + operating
    if global_lower_bound > objective + 1e-5:
        raise ValueError(
            "Supplied full-model lower bound exceeds the feasible objective"
        )
    result = dict(
        name=data.get("name", "path_instance"),
        model_version="path_cqn_v1",
        method="path_incumbent",
        objective=objective,
        bound=float(global_lower_bound),
        gap=max(0.0, objective - global_lower_bound) / max(1.0, abs(objective)),
        fleet=sol["fleet"],
        served=float(sum(row["flow"] for row in missions)),
        demand=float(np.sum(data["demand"])),
        n_od=data["n_od"],
        rho=sol["rho"].tolist(),
        cabinet_total=float(sol["hub_count"].sum()),
        cabinet_counts=sol["hub_count"].tolist(),
        hub_load=hubload.tolist(),
        airline_load=linkload.tolist(),
        opened=dict(
            L=int(round(sol["facility"][:L].sum())),
            H=int(round(sol["facility"][L : L + H].sum())),
            E=int(round(sol["facility"][L + H :].sum())),
        ),
        open_airlines=int(round(sol["airline"][net["physical"]].sum())),
        open_connectors=int(round(sol["airline"][~net["physical"]].sum())),
        airline_fixed_cost=float(net["fees"] @ sol["airline"]),
        all_ground_cost=c0,
        provenance=data.get("provenance", {}),
        _solution=sol,
        _missions=missions,
        _relocations=relocations,
        errors=[],
    )
    checks = build_operations(data, result)["summary"]["checks"]
    result["validation"] = checks
    result["primal_residual"] = max(
        (
            float(v)
            for k, v in checks.items()
            if k != "passed" and "cost" not in k and "objective" not in k
        ),
        default=0.0,
    )
    if not checks["passed"]:
        raise ValueError(
            f"Independent full-model feasibility validation failed: {checks}"
        )
    return result
