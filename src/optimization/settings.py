"""Shared solver precision and reproducibility settings."""

DEFAULT_SEED = 43
DEFAULT_MIP_GAP = 1e-4
DEFAULT_MIP_GAP_ABS = 1e-6


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
