"""Raw-order generation options independent of the optimization model."""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    """Raw-data preprocessing options shared by the TSL and Nanshan adapters."""

    source: str
    raw_path: str
    output_dir: str
    candidate_count: int = 120
    num_bases: int = 8
    num_swaps: int = 13
    candidate_hour_start: int = 8
    candidate_hour_end: int = 24
    demand_hour_start: int = 12
    demand_hour_end: int = 13
    map_half_size_km: float = 10.0
    od_grid_km: float = 0.5
    site_grid_km: float = 0.25
    seed: int = 42
    dates: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.source not in {"tsl", "nanshan"}:
            raise ValueError("source must be 'tsl' or 'nanshan'")
        if not 1 <= self.candidate_count:
            raise ValueError("candidate_count must be positive")
        if self.num_bases < 1 or self.num_swaps < 0:
            raise ValueError("num_bases must be positive and num_swaps nonnegative")
        if not 0 <= self.seed <= 2**32 - 3:
            raise ValueError("data seed must be in [0, 4294967293]")
        spatial_values = (self.map_half_size_km, self.od_grid_km, self.site_grid_km)
        if any(not math.isfinite(value) or value <= 0 for value in spatial_values):
            raise ValueError("map and grid sizes must be positive")
        _window_hours(self.candidate_hour_start, self.candidate_hour_end)
        _window_hours(self.demand_hour_start, self.demand_hour_end)


def _window_hours(start: int, end: int) -> int:
    if not 0 <= start <= 23 or not 1 <= end <= 24:
        raise ValueError(
            "hour windows use integer hours with start in [0,23], end in [1,24]"
        )
    if start == end:
        raise ValueError("start and end must differ; use 0 24 for a full day")
    duration = end - start if end > start else 24 - start + end
    if duration <= 0:
        raise ValueError("hour window must have positive duration")
    return duration


def to_dict(value: Any) -> dict[str, Any]:
    """Convert a configuration dataclass to a JSON-ready dictionary."""

    return asdict(value)
