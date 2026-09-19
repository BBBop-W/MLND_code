"""Run status and an independently checked numerical summary."""

import platform
import sys
import numpy as np


def runtime():
    import gurobipy

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "gurobi": list(gurobipy.gurobi.version()),
    }


def classify(result, checks):
    if not checks.get("passed", False):
        return "validation_failed", False
    if result.get("errors"):
        return "solver_error", False
    if result.get("status") in (3, 4, 5, 12):
        return "solver_failure", False
    if result.get("objective") is None or not np.isfinite(result["objective"]):
        return "no_finite_incumbent", False
    if result.get("served", 0) <= 1e-8:
        return "feasible_baseline_only", True
    if result.get("termination_reason") == "gap_limit":
        return "gap_certified", True
    return "feasible_limit", True


def summarize(data, result, checks, output):
    status, passed = classify(result, checks)
    fields = (
        "objective",
        "bound",
        "gap",
        "rho",
        "fleet",
        "served",
        "demand",
        "opened",
        "cabinet_total",
        "open_airlines",
        "open_connectors",
        "wall",
        "budget",
        "method",
        "termination_reason",
        "stats",
        "errors",
    )
    return dict(
        case=output.name,
        status=status,
        passed=passed,
        candidates={"bases": data["n_l"], "lockers": data["n_h"], "swaps": data["n_k"]},
        n_od=data["n_od"],
        **{k: result.get(k) for k in fields},
        solver_status=result.get("status"),
        validation=checks,
        output=str(output.resolve()),
    )
