"""Validation and immutable references for a fixed strategic design."""

from dataclasses import dataclass
import numpy as np
from .network import network, vector, levels


@dataclass(frozen=True)
class FixedDesign:
    """Strategic coefficients consumed by continuous recourse and its cuts."""

    net: dict
    facility: np.ndarray
    capacity_level: np.ndarray
    assignment: np.ndarray
    airlines: np.ndarray
    capacity: np.ndarray
    capacity_rhs: np.ndarray
    selected: np.ndarray
    origins: np.ndarray
    sources: np.ndarray
    returns: np.ndarray
    rates: np.ndarray
    base_caps: np.ndarray
    baseline_cost: float


def fixed_design(data, facility, capacity_level, assignment, airlines):
    """Validate a binary design and prepare its exact recourse right sides."""
    L, H, K, D = (data[k] for k in ("n_l", "n_h", "n_k", "n_od"))
    net = network(data)
    f = np.asarray(facility, float)
    zlevel = np.asarray(capacity_level, float)
    B = np.asarray(assignment, float)
    sizes, capacity, level_cost = levels(data)
    y = vector(data, airlines, net)
    if (
        f.shape != (L + H + K,)
        or zlevel.shape != capacity.shape
        or B.shape != (D, L, L)
    ):
        raise ValueError("Design dimensions")
    if not np.isfinite(f).all() or np.any((f != 0) & (f != 1)):
        raise ValueError("Binary facilities required")
    if (
        not np.isfinite(zlevel).all()
        or np.any((zlevel != 0) & (zlevel != 1))
        or np.any(zlevel.sum(axis=1) != f[L : L + H])
    ):
        raise ValueError("One capacity level per open locker required")
    capacity_rhs = (capacity * zlevel).sum(axis=1)
    if (
        np.max(np.abs(B - np.rint(B)), initial=0) > 1e-8
        or np.any(B < 0)
        or np.any(B.sum(axis=(1, 2)) > 1 + 1e-8)
    ):
        raise ValueError("Unique binary OD assignment required")
    if (
        np.any(y < 0)
        or np.any(y > 1)
        or np.max(np.abs(y - np.rint(y)), initial=0) > 1e-8
    ):
        raise ValueError("Binary links required")
    if np.any(y > f[net["uv"][:, 0]] + 1e-8) or np.any(y > f[net["uv"][:, 1]] + 1e-8):
        raise ValueError("Closed airline endpoint")
    selected = np.argwhere(B > 0.5)
    T = len(selected)
    os = selected[:, 0] if T else np.empty(0, int)
    ls = selected[:, 1] if T else np.empty(0, int)
    ms = selected[:, 2] if T else np.empty(0, int)
    if T and (np.any(f[ls] < 0.5) or np.any(f[ms] < 0.5)):
        raise ValueError("Closed assigned base")
    rate = data["demand"][os]
    caps = f[:L]
    c0 = float(np.asarray(data["demand"]) @ np.asarray(data["ground"]))
    return FixedDesign(
        net,
        f,
        zlevel,
        B,
        y,
        capacity,
        capacity_rhs,
        selected,
        os,
        ls,
        ms,
        rate,
        caps,
        c0,
    )
