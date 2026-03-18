"""Fixed-binary consistency test for monolithic/primal/dual subproblems.

Workflow:
1) Solve the copied monolithic model once to get a binary incumbent (x_hat, u_hat).
2) Rebuild the copied monolithic model and fix x/u to that incumbent.
3) Solve copied primal and copied dual subproblems under the same fixed binaries.
4) Compare objective values and report primal-dual / primal-reference gaps.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Tuple

from gurobipy import GRB

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.data_processor.loader import DataLoader
from src.model.benders_model import IndexSets

from subproblem_test.benders_dual import DualSubproblem
from subproblem_test.benders_primal import PrimalSubproblem
from subproblem_test.gurobi_model import MLNDModel

XHat = Dict[str, int]
UHat = Dict[Tuple[str, int], int]


def solve_master_and_get_binaries(inst) -> tuple[XHat, UHat, float]:
    model = MLNDModel(inst)
    model.build()
    model.model.update()
    model.model.setParam("OutputFlag", 0)
    model.model.optimize()

    if model.model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Initial monolithic solve failed, status={model.model.Status}")

    data = IndexSets(inst)
    x_hat = {
        node: int(round(model.model.getVarByName(f"x[{node}]").X))
        for node in data.N + data.L_v
    }
    u_hat = {
        (loc, level): int(round(model.model.getVarByName(f"u[{loc},{level}]").X))
        for loc in data.L
        for level in data.S
    }
    return x_hat, u_hat, model.model.ObjVal


def solve_fixed_binary_monolithic(inst, x_hat: XHat, u_hat: UHat) -> tuple[int, float | None]:
    fixed = MLNDModel(inst)
    fixed.build()
    fixed.model.update()
    fixed.model.setParam("OutputFlag", 0)

    data = IndexSets(inst)
    for node in data.N + data.L_v:
        v = fixed.model.getVarByName(f"x[{node}]")
        value = x_hat[node]
        v.lb = value
        v.ub = value

    for loc in data.L:
        for level in data.S:
            v = fixed.model.getVarByName(f"u[{loc},{level}]")
            value = u_hat[(loc, level)]
            v.lb = value
            v.ub = value

    fixed.model.optimize()
    if fixed.model.SolCount > 0:
        return fixed.model.Status, fixed.model.ObjVal
    return fixed.model.Status, None


def main() -> None:
    print("=" * 80)
    print("FIXED-BINARY CONSISTENCY CHECK")
    print("=" * 80)

    inst = DataLoader().load_instance(
        "data/instances/Meituan/locations.csv",
        "data/instances/Meituan/orders.csv",
    )
    print(f"Loaded instance: |L_hat|={len(inst.L_hat)}, |P|={len(inst.P)}, |D|={len(inst.D)}, |E|={len(inst.E)}")

    # x_hat, u_hat, master_obj = solve_master_and_get_binaries(inst)
    # print(f"Monolithic best objective (free binaries): {master_obj:.8f}")
    # print(f"Active x nodes: {sum(x_hat.values())}")

    data = IndexSets(inst)

    x_hat: XHat = {
        "L1": 0, "L2": 0, "L3": 0, "L4": 0, "L5": 0, "L6": 0, "L7": 1, "L8": 1, "L9": 0, "L10": 1,
        "P1": 1, "P2": 1, "P3": 1, "P4": 1, "P5": 1, "P6": 1, "P7": 1, "P8": 1, "P9": 1, "P10": 1,
        "D1": 0, "D2": 0, "D3": 0, "D4": 1, "D5": 0, "D6": 0, "D7": 0, "D8": 0, "D9": 0, "D10": 0,
        "E1": 0, "E2": 0, "E3": 1, "E4": 0, "E5": 1,
        "V_D1": 0, "V_D2": 0, "V_D3": 0, "V_D4": 0, "V_D5": 0, "V_D6": 0, "V_D7": 0, "V_D8": 0,
        "V_D9": 0, "V_D10": 0,
    }

    # Build a full u_hat. Use your known entries first, and a temporary default for L8.
    # If you have the exact L8 level from the problematic incumbent, replace it below.
    selected_levels = {
        "L7": 18,
        "L8": 3,
        "L10": 11,
    }

    

    u_hat: UHat = {(l, s): 0 for l in data.L for s in data.S}
    for l in data.L:
        if x_hat.get(l, 0) == 1:
            s = selected_levels.get(l, 0)
            u_hat[l, s] = 1


    fixed_status, fixed_obj = solve_fixed_binary_monolithic(inst, x_hat, u_hat)

    
    primal = PrimalSubproblem(inst, data)
    primal_status, primal_obj, _ = primal.solve(x_hat, u_hat)

    dual = DualSubproblem(inst, data)
    dual_status, dual_obj, _ = dual.solve(x_hat, u_hat)

    print("\n" + "-" * 80)
    print("RESULTS UNDER SAME FIXED (x_hat, u_hat)")
    print("-" * 80)
    print(f"Fixed monolithic: status={fixed_status}, obj={fixed_obj}")
    print(f"Primal subproblem: status={primal_status}, obj={primal_obj}")
    print(f"Dual subproblem: status={dual_status}, obj={dual_obj}")

    print("\n" + "-" * 80)
    print("GAPS")
    print("-" * 80)

    if fixed_obj is not None and primal_obj is not None:
        p_ref_gap = primal_obj - fixed_obj
        print(f"Primal - fixed-monolithic gap: {p_ref_gap:.8e}")
    else:
        print("Primal - fixed-monolithic gap: unavailable")

    if primal_obj is not None and dual_obj is not None:
        p_d_gap = primal_obj - dual_obj
        rel = abs(p_d_gap) / max(1.0, abs(primal_obj), abs(dual_obj))
        print(f"Primal - dual gap: {p_d_gap:.8e} (relative {rel:.8e})")
    else:
        print("Primal - dual gap: unavailable")

    tol = 1e-5
    pass_status = (
        fixed_obj is not None
        and primal_obj is not None
        and dual_obj is not None
        and abs(primal_obj - dual_obj) <= tol * max(1.0, abs(primal_obj), abs(dual_obj))
        and abs(primal_obj - fixed_obj) <= tol * max(1.0, abs(primal_obj), abs(fixed_obj))
    )

    print("\n" + "-" * 80)
    print("VERDICT")
    print("-" * 80)
    if pass_status:
        print("PASS: fixed-monolithic, primal subproblem, and dual subproblem match within tolerance.")
    else:
        print("FAIL: at least one mismatch detected. Check statuses and gaps above.")


if __name__ == "__main__":
    
    main()
