"""Read persisted run state without starting optimization."""

import json
import platform
import sys
from pathlib import Path
import numpy as np


def runtime():
    import gurobipy

    return dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        numpy=np.__version__,
        gurobi=list(gurobipy.gurobi.version()),
    )


def read_status(path):
    path = Path(path)
    result = {}
    for name in ("manifest", "summary", "progress", "checkpoint"):
        p = path / (name + ".json")
        if p.is_file():
            result[name] = json.loads(p.read_text())
    if not result:
        raise ValueError(f"No saved run state: {path}")
    return result
