"""Configuration for the OD-aware mixed-delivery network model.

All operational rates use hours, distances use kilometres, and costs use CNY
per hour (fixed items) or CNY per order/flight (flow-dependent items).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    """Raw-data preprocessing options shared by the TSL and Nanshan adapters."""

    source: str
    raw_path: str
    output_dir: str
    candidate_count: int = 34
    num_bases: int = 5
    num_swaps: int = 8
    candidate_hour_start: int = 8
    candidate_hour_end: int = 24
    demand_hour_start: int = 12
    demand_hour_end: int = 13
    map_half_size_km: float = 8.0
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


@dataclass(frozen=True)
class ChoiceConfig:
    """Scenario parameters for the uncalibrated two-alternative MNL layer.

    With the defaults, the utility difference is

        U_locker - U_home
        = -1 + 0.06 * (30 - 20) + 0.35 * 1 - 1.2 * walk_km.

    This gives a locker share of about 34% at 0.5 km. These values are a
    transparent baseline scenario, not empirical estimates.
    """

    locker_asc: float = -1.0
    delivery_time_coefficient: float = 0.06
    price_coefficient: float = 0.35
    walk_distance_coefficient: float = 1.2
    home_eta_minutes: float = 30.0
    locker_eta_minutes: float = 20.0
    locker_discount_cny: float = 1.0
    max_walk_km: float | None = 1.5

    def validate(self) -> None:
        values = {
            "locker_asc": self.locker_asc,
            "delivery_time_coefficient": self.delivery_time_coefficient,
            "price_coefficient": self.price_coefficient,
            "walk_distance_coefficient": self.walk_distance_coefficient,
            "home_eta_minutes": self.home_eta_minutes,
            "locker_eta_minutes": self.locker_eta_minutes,
            "locker_discount_cny": self.locker_discount_cny,
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("choice parameters must be finite")
        if self.delivery_time_coefficient < 0 or self.price_coefficient < 0:
            raise ValueError("time and price coefficients must be nonnegative")
        if self.walk_distance_coefficient < 0:
            raise ValueError("walk_distance_coefficient must be nonnegative")
        if self.home_eta_minutes <= 0 or self.locker_eta_minutes <= 0:
            raise ValueError("delivery-time attributes must be positive")
        if self.locker_discount_cny < 0:
            raise ValueError("locker_discount_cny must be nonnegative")
        if self.max_walk_km is not None and (
            not math.isfinite(self.max_walk_km) or self.max_walk_km <= 0
        ):
            raise ValueError("max_walk_km must be positive or None")


@dataclass(frozen=True)
class ModelConfig:
    """Physical, queueing, cost, and sparsification parameters."""

    choice: ChoiceConfig = field(default_factory=ChoiceConfig)
    availability_levels: tuple[float, ...] = (0.50, 0.65, 0.80, 0.90, 0.95)
    # Transparent Nanshan planning levels.  They are scenario assumptions,
    # not enterprise measurements, and correspond to mean handling times of
    # 2.0, 1.5, and 1.2 minutes per drone movement.
    locker_service_rates: tuple[float, ...] = (30.0, 40.0, 50.0)
    locker_capacity_costs: tuple[float, ...] = (0.5, 1.0, 2.0)
    # The confirmed 10 km value is treated as the usable range after any
    # operating reserve. It applies to a direct movement and separately to
    # both legs of a one-swap movement.
    battery_range_km: float = 10.0
    drone_speed_mps: float = 20.0
    swap_service_time_hours: float = 3.0 / 60.0
    max_locker_time_hours: float = 5.0 / 60.0
    max_virtual_wait_hours: float = 5.0 / 60.0
    base_cost_per_hour: float = 14.0
    locker_fixed_cost_per_hour: float = 1.0
    swap_cost_per_hour: float = 3.0
    corridor_cost_per_km_hour: float = 0.02
    connection_cost_per_km_hour: float = 0.05
    flight_cost_per_km: float = 0.20
    drone_cost_per_hour: float = 5.0
    # Literature-based professional crowd-courier tariff: CNY 6.4 includes
    # the first 3 road kilometres, then CNY 1.8 per additional kilometre.
    rider_initial_charge_cny: float = 6.40
    rider_included_distance_km: float = 3.0
    rider_incremental_cost_per_km: float = 1.80
    road_distance_factor: float = 1.25
    # The paper defines direct courier delivery (f_od) separately from the
    # first/last-mile feeder movements (f_oi and f_jd).  This factor provides a
    # compact scenario parameterization of that distinction while preserving
    # the observed direct-courier tariff above.  A value below one represents
    # contracted or batched short-haul feeder service.  The 0.30 Nanshan
    # planning value is a transparent scenario assumption used consistently
    # across MX, HD, and LO; it is not an enterprise estimate.
    feeder_rider_cost_factor: float = 0.30
    # Fulfillment cost charged to a successful locker-pickup order.  This is
    # distinct from choice.locker_discount_cny, which affects customer utility.
    locker_pickup_cost_per_order: float = 0.30
    # None retains the complete paper index set. Finite top-k values are an
    # explicit computational screening approximation for large instances.
    display_origin_choices: int | None = None
    general_corridor_choices: int | None = None

    @property
    def drone_speed_kmph(self) -> float:
        """Flight speed in the kilometre/hour unit used by route times."""

        return 3.6 * self.drone_speed_mps

    @property
    def usable_battery_time_hours(self) -> float:
        """Equivalent value of ``(1-epsilon) * Omega / omega`` in the paper."""

        return self.battery_range_km / self.drone_speed_kmph

    def validate(self) -> None:
        self.choice.validate()
        if not self.availability_levels:
            raise ValueError("availability_levels cannot be empty")
        if tuple(sorted(self.availability_levels)) != self.availability_levels:
            raise ValueError("availability_levels must be strictly increasing")
        if any(
            not math.isfinite(rho) or rho <= 0 or rho >= 1
            for rho in self.availability_levels
        ):
            raise ValueError("every availability level must be in (0, 1)")
        if len(set(self.availability_levels)) != len(self.availability_levels):
            raise ValueError("availability_levels must be distinct")
        if len(self.locker_service_rates) != len(self.locker_capacity_costs):
            raise ValueError(
                "locker service-rate and cost levels must have equal length"
            )
        if not self.locker_service_rates:
            raise ValueError("at least one locker service-rate level is required")
        if any(not math.isfinite(mu) or mu <= 0 for mu in self.locker_service_rates):
            raise ValueError("locker service rates must be positive")
        positive_parameters = {
            "battery_range_km": self.battery_range_km,
            "drone_speed_mps": self.drone_speed_mps,
            "swap_service_time_hours": self.swap_service_time_hours,
            "max_locker_time_hours": self.max_locker_time_hours,
            "max_virtual_wait_hours": self.max_virtual_wait_hours,
            "rider_included_distance_km": self.rider_included_distance_km,
            "road_distance_factor": self.road_distance_factor,
        }
        if any(
            not math.isfinite(value) or value <= 0
            for value in positive_parameters.values()
        ):
            raise ValueError(
                "ranges, speeds, service times, and road factor must be positive"
            )
        minimum_stable_rate = 1.0 / self.max_locker_time_hours
        if any(mu + 1e-9 < minimum_stable_rate for mu in self.locker_service_rates):
            raise ValueError(
                "every locker service-rate level must satisfy "
                "mu >= 1 / max_locker_time_hours"
            )
        nonnegative_costs = (
            *self.locker_capacity_costs,
            self.base_cost_per_hour,
            self.locker_fixed_cost_per_hour,
            self.swap_cost_per_hour,
            self.corridor_cost_per_km_hour,
            self.connection_cost_per_km_hour,
            self.flight_cost_per_km,
            self.rider_initial_charge_cny,
            self.rider_incremental_cost_per_km,
            self.locker_pickup_cost_per_order,
        )
        if any(not math.isfinite(value) or value < 0 for value in nonnegative_costs):
            raise ValueError("all cost parameters must be finite and nonnegative")
        if not math.isfinite(self.drone_cost_per_hour) or self.drone_cost_per_hour <= 0:
            raise ValueError("drone_cost_per_hour must be finite and strictly positive")
        if (
            not math.isfinite(self.feeder_rider_cost_factor)
            or self.feeder_rider_cost_factor <= 0
        ):
            raise ValueError(
                "feeder_rider_cost_factor must be finite and strictly positive"
            )
        if self.display_origin_choices is not None and self.display_origin_choices < 1:
            raise ValueError("display_origin_choices must be positive or None")
        if (
            self.general_corridor_choices is not None
            and self.general_corridor_choices < 1
        ):
            raise ValueError("general_corridor_choices must be positive or None")


@dataclass(frozen=True)
class SolverConfig:
    time_limit: float = 300.0
    mip_gap: float = 0.01
    threads: int = 0
    output_flag: int = 1
    seed: int = 42
    network_start_seconds: float = 20.0
    no_rel_heur_seconds: float = 10.0
    use_valid_inequalities: bool = False
    use_user_cuts: bool = True
    max_root_cut_rounds: int = 3
    # Two native root-cut passes prevent large screened H120 models from
    # spending the entire time budget in automatic root separation.
    root_cut_passes: int = 2
    max_cuts_per_round: int = 250
    cut_violation_tolerance: float = 1e-6
    use_local_branching: bool = True
    local_branching_radius: int = 12
    root_relaxation_method: int = 2

    def validate(self) -> None:
        if not math.isfinite(self.time_limit) or self.time_limit <= 0:
            raise ValueError("time_limit must be positive and finite")
        if not math.isfinite(self.mip_gap) or not 0 <= self.mip_gap <= 1:
            raise ValueError("mip_gap must be in [0, 1]")
        if self.threads < 0:
            raise ValueError("threads must be nonnegative")
        if self.output_flag not in {0, 1}:
            raise ValueError("output_flag must be 0 or 1")
        if not 0 <= self.seed <= 2_000_000_000:
            raise ValueError("seed must be in [0, 2000000000]")
        if (
            not math.isfinite(self.network_start_seconds)
            or self.network_start_seconds < 0
        ):
            raise ValueError("network_start_seconds must be finite and nonnegative")
        if not math.isfinite(self.no_rel_heur_seconds) or self.no_rel_heur_seconds < 0:
            raise ValueError("no_rel_heur_seconds must be finite and nonnegative")
        if self.max_root_cut_rounds < 0:
            raise ValueError("max_root_cut_rounds must be nonnegative")
        if self.root_cut_passes < -1:
            raise ValueError("root_cut_passes must be -1 or nonnegative")
        if self.max_cuts_per_round < 0:
            raise ValueError("max_cuts_per_round must be nonnegative")
        if (
            not math.isfinite(self.cut_violation_tolerance)
            or self.cut_violation_tolerance <= 0
        ):
            raise ValueError("cut_violation_tolerance must be positive and finite")
        if self.local_branching_radius < 1:
            raise ValueError("local_branching_radius must be positive")
        if self.root_relaxation_method not in {-1, 0, 1, 2, 3, 4, 5}:
            raise ValueError(
                "root_relaxation_method must be a valid Gurobi Method value"
            )


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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSL_PATH = str(REPO_ROOT / "Meituan-TSL" / "all_waybill_info_meituan_0322.csv")
DEFAULT_NANSHAN_PATH = str(
    REPO_ROOT / "Meituan-Shenzhen" / "深圳市南山区运单抽样_已脱敏.csv"
)
