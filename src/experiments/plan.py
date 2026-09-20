"""Build an experiment matrix without loading instances or creating a solver."""

from pathlib import Path
from src.config import load_config
from src.solvers.methods import METHODS, get_method


def build_plan(hubs, methods=None, *, config=None, options=None, output=None):
    config = load_config(config, overrides={"solver": options or {}})
    sizes = list(dict.fromkeys(hubs))
    if not sizes or any(
        isinstance(h, bool) or not isinstance(h, int) or h < 2 for h in sizes
    ):
        raise ValueError("At least one integer H size of two or more is required")
    names = list(dict.fromkeys(methods or METHODS))
    descriptions = {name: get_method(name).describe() for name in names}
    cases = []
    for size in sizes:
        for name in names:
            case = f"H{size:03d}_{name}"
            cases.append(
                {
                    "case": case,
                    "hubs": size,
                    "method": name,
                    "output": str(Path(output) / case) if output else None,
                }
            )
    return {
        "status": "planned_only",
        "optimization_started": False,
        "solver": config["solver"],
        "demand": {
            "mode": "fixed"
            if config["data"]["demand_hubs"] is not None
            else "instance",
            "source_hubs": config["data"]["demand_hubs"],
            "scale": config["data"]["demand_scale"],
        },
        "methods": descriptions,
        "cases": cases,
    }
