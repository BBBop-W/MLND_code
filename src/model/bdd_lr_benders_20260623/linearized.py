"""Shared cone-free queue relaxations for candidate and bound MILPs."""

from __future__ import annotations

from typing import Dict

from src import config


def remove_all_quadratic_constraints(wrapper) -> Dict[str, int]:
    """Materialize, remove, and verify every quadratic constraint."""

    model = wrapper.model
    model.update()
    before = int(model.NumQConstrs)
    constraints = model.getQConstrs()
    if constraints:
        model.remove(constraints)
        model.update()
    after = int(model.NumQConstrs)
    if after != 0:
        raise RuntimeError(
            f"Expected a cone-free MILP, found {after} QConstrs"
        )
    return {"qconstrs_before": before, "qconstrs_after": after}


def add_queue_supporting_lines(
    wrapper,
    *,
    prefix: str,
) -> int:
    """Add globally valid supporting-line lower bounds for queue lengths."""

    model = wrapper.model
    xi = wrapper.vars["xi"]
    rho_l = wrapper.vars["rho_l"]
    rho_s = wrapper.vars["rho_s"]
    lambd = wrapper.vars["lambda"]
    z = wrapper.vars["z"]
    added = 0

    utilization_grid = (0.0, 0.25, 0.50, 0.70, 0.85, 0.93)
    for node in wrapper.L:
        for grid_index, rho_ref in enumerate(utilization_grid):
            derivative = 1.0 / ((1.0 - rho_ref) ** 2)
            value = rho_ref / (1.0 - rho_ref)
            intercept = value - derivative * rho_ref
            model.addConstr(
                xi[node] >= derivative * rho_l[node] + intercept,
                name=f"{prefix}_L_{node}_{grid_index}",
            )
            added += 1
    for node in wrapper.S:
        for grid_index, rho_ref in enumerate(utilization_grid):
            derivative = 1.0 / ((1.0 - rho_ref) ** 2)
            value = rho_ref / (1.0 - rho_ref)
            intercept = value - derivative * rho_ref
            model.addConstr(
                xi[node] >= derivative * rho_s[node] + intercept,
                name=f"{prefix}_S_{node}_{grid_index}",
            )
            added += 1

    total_demand = max(
        1.0, sum(float(value) for value in wrapper.inst.h.values())
    )
    hub_fraction_grid = (0.0, 0.25, 0.50, 0.70, 0.85, 0.93)
    for hub in wrapper.H:
        for capacity_index, capacity in enumerate(config.MU_HUB):
            mu_value = float(capacity)
            for grid_index, fraction in enumerate(hub_fraction_grid):
                lambda_ref = fraction * mu_value
                denominator = max(1e-6, mu_value - lambda_ref)
                derivative = mu_value / (denominator * denominator)
                value = lambda_ref / denominator
                intercept = value - derivative * lambda_ref
                # Incoming generated flow is bounded by the route-flow big-M.
                big_m = derivative * total_demand + abs(intercept) + 100.0
                model.addConstr(
                    xi[hub]
                    >= derivative * lambd[hub]
                    + intercept
                    - big_m * (1.0 - z[hub, capacity_index]),
                    name=(
                        f"{prefix}_H_{hub}_"
                        f"{capacity_index}_{grid_index}"
                    ),
                )
                added += 1
    model.update()
    return added
