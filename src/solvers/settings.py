"""Shared solver precision and reproducibility settings."""

import math
import gurobipy as gp

DEFAULT_SEED = 43
DEFAULT_MIP_GAP = 1e-4
DEFAULT_MIP_GAP_ABS = 1e-6


def mip_bound(model):
    """Preserve a finite search bound even before an incumbent is available."""
    try:
        value = float(model.ObjBound)
    except (AttributeError, gp.GurobiError):
        return None
    return value if math.isfinite(value) and abs(value) < 1e90 else None


def settings(model, seconds, threads=1, seed=DEFAULT_SEED):
    """Apply continuous-conic settings; ``None`` leaves time unrestricted."""
    model.Params.OutputFlag = 0
    model.Params.Threads = threads
    model.Params.Seed = seed
    if seconds is not None:
        model.Params.TimeLimit = max(0.01, seconds)
    model.Params.NonConvex = 0
    model.Params.FeasibilityTol = 1e-8
    model.Params.OptimalityTol = 1e-8
    model.Params.BarQCPConvTol = 1e-9
