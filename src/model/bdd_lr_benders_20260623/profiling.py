"""Wall/solver-time accounting for LR and decomposition phases."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

from .types import PhaseTiming


class PhaseProfiler:
    """Accumulate wall and solver runtime without double-counting Python time."""

    def __init__(self) -> None:
        self._timings: Dict[str, PhaseTiming] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            timing = self._timings.setdefault(name, PhaseTiming(phase=name))
            timing.wall_time += elapsed
            timing.calls += 1

    def add_solver_time(self, name: str, runtime: Optional[float]) -> None:
        if runtime is None:
            return
        timing = self._timings.setdefault(name, PhaseTiming(phase=name))
        timing.solver_time += max(0.0, float(runtime))

    def rows(self) -> List[Dict[str, object]]:
        return [
            self._timings[name].as_dict()
            for name in sorted(self._timings)
        ]

    def totals(self) -> Dict[str, float]:
        wall = sum(item.wall_time for item in self._timings.values())
        solver = sum(item.solver_time for item in self._timings.values())
        non_solver = sum(
            item.non_solver_time for item in self._timings.values()
        )
        return {
            "profiled_wall_time": wall,
            "profiled_solver_time": solver,
            "profiled_non_solver_time": non_solver,
            "profiled_non_solver_fraction": (
                non_solver / wall if wall > 1e-12 else 0.0
            ),
        }

    def cpp_acceleration_decision(
        self,
        *,
        total_lr_time: float,
        aggregate_threshold: float = 0.20,
        hotspot_threshold: float = 0.10,
    ) -> Dict[str, object]:
        """Apply the goal's evidence gate for a C++ implementation."""

        denominator = max(1e-12, float(total_lr_time))
        non_solver_total = sum(
            item.non_solver_time for item in self._timings.values()
        )
        hotspots = [
            {
                "phase": item.phase,
                "non_solver_time": item.non_solver_time,
                "fraction_of_lr": item.non_solver_time / denominator,
            }
            for item in self._timings.values()
            if item.non_solver_time / denominator >= hotspot_threshold
        ]
        aggregate_fraction = non_solver_total / denominator
        return {
            "requires_cpp_candidate_review": bool(
                aggregate_fraction >= aggregate_threshold or hotspots
            ),
            "non_solver_fraction_of_lr": aggregate_fraction,
            "aggregate_threshold": aggregate_threshold,
            "hotspot_threshold": hotspot_threshold,
            "hotspots": sorted(
                hotspots,
                key=lambda row: float(row["non_solver_time"]),
                reverse=True,
            ),
            "decision_note": (
                "C++ migration is not automatic: a hotspot must also be "
                "mechanically portable and repeat on H=40/80/100."
            ),
        }
