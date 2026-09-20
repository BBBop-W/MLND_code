"""Geographic scale-family settings and intermediate records."""

from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any
from .generation_config import DataConfig
from .instance_io import ROOT

DEFAULT_NANSHAN_PATH = ROOT / "Meituan-Shenzhen/深圳市南山区运单抽样_已脱敏.csv"
CANONICAL_FILES = (
    "metadata.json",
    "sites.csv",
    "od_nodes.csv",
    "od_demand.csv",
    "periods.csv",
)
NESTED_SCALE_SCHEMA_VERSION = 2


@dataclass(frozen=True, order=True)
class NanshanScaleLevel:
    """One concentric square crop and its calibrated candidate targets."""

    half_size_km: float
    lockers: int
    bases: int
    swaps: int

    def validate(self) -> None:
        if not math.isfinite(self.half_size_km) or self.half_size_km <= 0:
            raise ValueError("crop half-size must be positive and finite")
        if self.lockers <= 0 or self.bases <= 0 or self.swaps < 0:
            raise ValueError("H/L targets must be positive and E must be nonnegative")


DEFAULT_NANSHAN_LEVELS = (
    NanshanScaleLevel(3.0, 11, 2, 2),
    NanshanScaleLevel(4.5, 24, 2, 3),
    NanshanScaleLevel(6.0, 43, 3, 5),
    NanshanScaleLevel(7.5, 68, 5, 7),
    NanshanScaleLevel(8.5, 87, 6, 9),
    NanshanScaleLevel(10.0, 120, 8, 13),
)

AREA_REGULAR_MAIN_COUNTS = (
    (20, 3, 5),
    (40, 6, 10),
    (60, 9, 15),
    (80, 12, 20),
    (100, 15, 25),
    (120, 18, 30),
)
AREA_REGULAR_SUPPLEMENTARY_COUNTS = ((10, 2, 3),)
AREA_REGULAR_LEVELS = tuple(
    NanshanScaleLevel(10.0 * math.sqrt(hubs / 120.0), hubs, bases, swaps)
    for hubs, bases, swaps in (
        *AREA_REGULAR_SUPPLEMENTARY_COUNTS,
        *AREA_REGULAR_MAIN_COUNTS,
    )
)


@dataclass(frozen=True)
class NanshanScaleBuildConfig:
    """Configuration shared by every member of a scale family."""

    output_root: Path = Path("data/instances/nanshan_nested_h120")
    raw_path: Path = Path(DEFAULT_NANSHAN_PATH)
    levels: tuple[NanshanScaleLevel, ...] = DEFAULT_NANSHAN_LEVELS
    candidate_hour_start: int = 8
    candidate_hour_end: int = 24
    demand_hour_start: int = 12
    demand_hour_end: int = 13
    od_grid_km: float = 0.5
    site_grid_km: float = 0.25
    dates: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7")
    seed: int = 42

    def validate(self) -> None:
        if not self.levels:
            raise ValueError("levels cannot be empty")
        for level in self.levels:
            level.validate()
        if tuple(sorted(self.levels)) != self.levels:
            raise ValueError("levels must be strictly increasing by crop half-size")
        radii = [level.half_size_km for level in self.levels]
        if len(radii) != len(set(radii)):
            raise ValueError("crop half-sizes must be distinct")
        for field in ("lockers", "bases", "swaps"):
            values = [getattr(level, field) for level in self.levels]
            if any(smaller > larger for smaller, larger in zip(values, values[1:])):
                raise ValueError(
                    f"{field} targets must be nondecreasing with crop size"
                )
        if not self.dates:
            raise ValueError("dates cannot be empty for a reproducible scale family")
        DataConfig(
            source="nanshan",
            raw_path=str(self.raw_path),
            output_dir=str(self.output_root),
            candidate_count=1,
            num_bases=1,
            num_swaps=0,
            candidate_hour_start=self.candidate_hour_start,
            candidate_hour_end=self.candidate_hour_end,
            demand_hour_start=self.demand_hour_start,
            demand_hour_end=self.demand_hour_end,
            map_half_size_km=self.levels[-1].half_size_km,
            od_grid_km=self.od_grid_km,
            site_grid_km=self.site_grid_km,
            dates=self.dates,
            seed=self.seed,
        ).validate()


@dataclass
class _GeneratedLevel:
    level: NanshanScaleLevel
    directory: Path
    metadata: dict[str, Any]
    sites: list[dict[str, str]]
    nodes: list[dict[str, str]]
    demand: list[dict[str, str]]
    periods: list[dict[str, str]]


@dataclass
class _RawOrderAudit:
    candidate_tokens: list[str]
    demand_tokens: list[str]
