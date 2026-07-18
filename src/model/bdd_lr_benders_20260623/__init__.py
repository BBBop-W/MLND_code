"""Isolated 2026-06-23 Regime-II LR/BD/BDD implementation.

This package deliberately does not expose or wrap the legacy
``StrictLagrangianBendersSolver`` entry point.  It reuses only the stable
instance helpers and fixed-design continuous SOCP oracle from ``src.model.Benders``.
"""

from .solver import (
    ConicBendersSolver20260623,
    LROnlySolver20260623,
    SolveResult20260623,
)

__all__ = [
    "ConicBendersSolver20260623",
    "LROnlySolver20260623",
    "SolveResult20260623",
]
