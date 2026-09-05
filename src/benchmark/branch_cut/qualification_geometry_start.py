"""Qualification-only, cost-blind partial MIP-start planning.

The planner in this module reads only physical topology, coordinates, route
energy data, queue capacities, and the model's already-derived safe flow
bounds.  It never reads objective coefficients, rider costs, savings, profits,
or any A--E heuristic/cut implementation.  A plan contains positive hints for
only ``x``, ``y``, ``z``, physical ``psi``, and one canonical ``varphi`` route
per used physical arc.  Every unlisted model variable must remain undefined
when the adapter installs the plan.

No solver is imported here.  The production adapter owns the narrow Gurobi
operation that installs a validated plan as a partial start.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from src.data_processor.instance import DIRECT_ROUTE


GEOMETRY_START_PLAN_VERSION = "qualification-geometry-start-v2-system-energy"
GUROBI_MAX_START_NODE_LIMIT = 2_000_000_000
SMALL_SUPPORT_SCHEDULE = (12, 8, 15)
MEDIUM_SUPPORT_SCHEDULE = (26, 17, 34)
LARGE_SUPPORT_SCHEDULE = (55, 50, 60)
_EXPECTED_H_BY_SCALE = {"small": 23, "medium": 51, "large": 90}


class GeometryStartPlanningError(ValueError):
    """A planner contract or internal-consistency failure.

    This base error is infrastructure evidence.  Only the explicit
    :class:`GeometryStartInfeasibilityError` subclass may be interpreted as a
    normal failure to construct the registered support for one candidate.
    """


class GeometryStartInfeasibilityError(GeometryStartPlanningError):
    """An expected candidate-specific topology, demand, or capacity precheck."""


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} fields are inconsistent")


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a list of nonempty strings")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class GeometryStartPolicy:
    """Frozen controls for the optional qualification fallback.

    The three schedules and plan version are deliberately present in the
    serialized policy (and therefore the run ID/journal identity), but are not
    user-tunable.  This prevents outcome-dependent support-size searches.
    """

    enabled: bool = False
    per_attempt_seconds: float = 10.0
    start_node_limit: int = 500
    plan_version: str = GEOMETRY_START_PLAN_VERSION
    active_tolerance_multiplier: float = 2.0
    required_service_multiplier: float = 1.05
    small_support_schedule: tuple[int, ...] = SMALL_SUPPORT_SCHEDULE
    medium_support_schedule: tuple[int, ...] = MEDIUM_SUPPORT_SCHEDULE
    large_support_schedule: tuple[int, ...] = LARGE_SUPPORT_SCHEDULE

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("geometry-start enabled must be boolean")
        if (
            isinstance(self.per_attempt_seconds, bool)
            or not isinstance(self.per_attempt_seconds, (int, float))
            or not math.isfinite(float(self.per_attempt_seconds))
            or float(self.per_attempt_seconds) <= 0.0
        ):
            raise ValueError(
                "geometry-start per_attempt_seconds must be positive and finite"
            )
        if (
            isinstance(self.start_node_limit, bool)
            or not isinstance(self.start_node_limit, int)
            or self.start_node_limit < 0
            or self.start_node_limit > GUROBI_MAX_START_NODE_LIMIT
        ):
            raise ValueError(
                "geometry-start start_node_limit must be an integer in "
                f"[0, {GUROBI_MAX_START_NODE_LIMIT}]"
            )
        if self.plan_version != GEOMETRY_START_PLAN_VERSION:
            raise ValueError("unsupported geometry-start plan version")
        if self.active_tolerance_multiplier != 2.0:
            raise ValueError("geometry-start active tolerance multiplier is frozen")
        if self.required_service_multiplier != 1.05:
            raise ValueError("geometry-start required service multiplier is frozen")
        if tuple(self.small_support_schedule) != SMALL_SUPPORT_SCHEDULE:
            raise ValueError("small geometry-start schedule is pre-registered")
        if tuple(self.medium_support_schedule) != MEDIUM_SUPPORT_SCHEDULE:
            raise ValueError("medium geometry-start schedule is pre-registered")
        if tuple(self.large_support_schedule) != LARGE_SUPPORT_SCHEDULE:
            raise ValueError("large geometry-start schedule is pre-registered")

    def schedule_for(self, scale: str, candidate_h: int) -> tuple[int, ...]:
        if scale not in _EXPECTED_H_BY_SCALE:
            raise GeometryStartPlanningError(
                f"geometry start does not support scale {scale!r}"
            )
        expected_h = _EXPECTED_H_BY_SCALE[scale]
        if candidate_h != expected_h:
            raise GeometryStartPlanningError(
                "geometry start requires the frozen candidate-H count "
                f"{expected_h} for {scale}, got {candidate_h}"
            )
        return {
            "small": self.small_support_schedule,
            "medium": self.medium_support_schedule,
            "large": self.large_support_schedule,
        }[scale]

    def to_payload(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "per_attempt_seconds": self.per_attempt_seconds,
            "start_node_limit": self.start_node_limit,
            "plan_version": self.plan_version,
            "active_tolerance_multiplier": self.active_tolerance_multiplier,
            "required_service_multiplier": self.required_service_multiplier,
            "small_support_schedule": list(self.small_support_schedule),
            "medium_support_schedule": list(self.medium_support_schedule),
            "large_support_schedule": list(self.large_support_schedule),
        }

    @classmethod
    def from_payload(cls, payload: object) -> GeometryStartPolicy:
        if not isinstance(payload, Mapping):
            raise ValueError("geometry_start_policy must be an object")
        expected = {
            "enabled",
            "per_attempt_seconds",
            "start_node_limit",
            "plan_version",
            "active_tolerance_multiplier",
            "required_service_multiplier",
            "small_support_schedule",
            "medium_support_schedule",
            "large_support_schedule",
        }
        _require_exact_keys(payload, expected, label="geometry_start_policy")
        try:
            return cls(
                enabled=payload["enabled"],
                per_attempt_seconds=payload["per_attempt_seconds"],
                start_node_limit=payload["start_node_limit"],
                plan_version=payload["plan_version"],
                active_tolerance_multiplier=payload[
                    "active_tolerance_multiplier"
                ],
                required_service_multiplier=payload[
                    "required_service_multiplier"
                ],
                small_support_schedule=tuple(payload["small_support_schedule"]),
                medium_support_schedule=tuple(payload["medium_support_schedule"]),
                large_support_schedule=tuple(payload["large_support_schedule"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid geometry_start_policy: {error}") from error


@dataclass(frozen=True, slots=True)
class GeometryCycle:
    """One canonical base--H--H--base physical cycle."""

    base: str
    origin_h: str
    destination_h: str
    route_options: tuple[str, str, str]
    route_distances: tuple[float, float, float]
    route_times: tuple[float, float, float]

    def __post_init__(self) -> None:
        if any(not isinstance(node, str) or not node for node in self.audit_descriptor):
            raise ValueError("geometry cycle identifiers must be nonempty strings")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (*self.route_distances, *self.route_times)
        ):
            raise ValueError("geometry cycle route data must be positive and finite")

    @property
    def arcs(self) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
        return (
            (self.base, self.origin_h),
            (self.origin_h, self.destination_h),
            (self.destination_h, self.base),
        )

    @property
    def geometry_key(self) -> tuple[object, ...]:
        return (
            math.fsum(self.route_distances),
            math.fsum(self.route_times),
            self.base,
            self.origin_h,
            self.destination_h,
            self.route_options,
        )

    @property
    def audit_descriptor(self) -> tuple[str, ...]:
        return (
            self.base,
            self.origin_h,
            self.destination_h,
            *self.route_options,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "base": self.base,
            "origin_h": self.origin_h,
            "destination_h": self.destination_h,
            "route_options": list(self.route_options),
            "route_distances": list(self.route_distances),
            "route_times": list(self.route_times),
        }


@dataclass(frozen=True, slots=True)
class GeometryCapacityChoice:
    locker: str
    capacity_index: int
    service_rate: float
    incident_flow: float
    outgoing_route_time: float
    queue_population: float
    queue_slack: float
    population_upper_slack: float

    def __post_init__(self) -> None:
        if not self.locker:
            raise ValueError("geometry capacity locker must be nonempty")
        if self.capacity_index < 0:
            raise ValueError("geometry capacity index must be non-negative")
        for name in (
            "service_rate",
            "incident_flow",
            "outgoing_route_time",
            "queue_population",
            "queue_slack",
            "population_upper_slack",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"geometry capacity {name} must be finite")

    def to_payload(self) -> dict[str, object]:
        return {
            "locker": self.locker,
            "capacity_index": self.capacity_index,
            "service_rate": self.service_rate,
            "incident_flow": self.incident_flow,
            "outgoing_route_time": self.outgoing_route_time,
            "queue_population": self.queue_population,
            "queue_slack": self.queue_slack,
            "population_upper_slack": self.population_upper_slack,
        }


@dataclass(frozen=True, slots=True)
class GeometryStartPlan:
    """A fully prechecked positive-only partial-start plan."""

    plan_version: str
    scale: str
    target_support_size: int
    support_order: tuple[str, ...]
    cycles: tuple[GeometryCycle, ...]
    availability_index: int
    availability_probability: float
    offered_flow_per_cycle: float
    flow_per_cycle: float
    planned_service_per_hour: float
    open_nodes: tuple[str, ...]
    activated_y_arcs: tuple[tuple[str, str], ...]
    physical_arc_flows: tuple[tuple[str, str, float], ...]
    route_flows: tuple[tuple[str, str, str, float], ...]
    capacity_choices: tuple[GeometryCapacityChoice, ...]
    chi_flows: tuple[tuple[str, str, str, str, int, float], ...]
    w_flows: tuple[tuple[str, str, str, int, float], ...]
    delta_levels: tuple[tuple[str, int], ...]
    xi_values: tuple[tuple[str, float], ...]
    queue_v1_values: tuple[tuple[str, float], ...]
    queue_v2_values: tuple[tuple[str, float], ...]
    queue_t_values: tuple[tuple[str, float], ...]
    gamma_value: int
    precheck_metrics: tuple[tuple[str, float], ...]
    plan_sha256: str

    def __post_init__(self) -> None:
        if self.plan_version != GEOMETRY_START_PLAN_VERSION:
            raise ValueError("unsupported geometry-start plan version")
        if len(self.support_order) != self.target_support_size:
            raise ValueError("geometry support size does not match support_order")
        if len(set(self.support_order)) != len(self.support_order):
            raise ValueError("geometry support_order contains duplicates")
        if len(self.cycles) != self.target_support_size - 1:
            raise ValueError("geometry support must use exactly k-1 cycles")
        if not math.isfinite(self.flow_per_cycle) or self.flow_per_cycle <= 0.0:
            raise ValueError("geometry cycle flow must be positive and finite")
        if (
            not math.isfinite(self.offered_flow_per_cycle)
            or self.offered_flow_per_cycle <= 0.0
        ):
            raise ValueError("geometry offered cycle flow must be positive and finite")
        if not 0.0 < self.availability_probability < 1.0:
            raise ValueError("geometry availability probability must lie in (0, 1)")
        if self.plan_sha256 != _canonical_sha256(self.identity_payload()):
            raise ValueError("geometry plan SHA256 does not match its contents")

    @property
    def family_start_counts(self) -> tuple[tuple[str, int], ...]:
        # chi/w/delta/queue/xi/gamma values above are an audited feasible
        # completion used only by the strict prechecks.  The actual partial
        # start leaves every one of those families undefined so Gurobi can
        # complete or move continuous flow while respecting the two existing
        # qualification rows.
        return tuple(
            sorted(
                (
                    ("varphi", len(self.route_flows)),
                    ("psi", len(self.physical_arc_flows)),
                    ("x", len(self.open_nodes)),
                    ("y", len(self.activated_y_arcs)),
                    ("z", len(self.capacity_choices)),
                )
            )
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "plan_version": self.plan_version,
            "scale": self.scale,
            "target_support_size": self.target_support_size,
            "support_order": list(self.support_order),
            "cycles": [cycle.to_payload() for cycle in self.cycles],
            "availability_index": self.availability_index,
            "availability_probability": self.availability_probability,
            "offered_flow_per_cycle": self.offered_flow_per_cycle,
            "flow_per_cycle": self.flow_per_cycle,
            "planned_service_per_hour": self.planned_service_per_hour,
            "open_nodes": list(self.open_nodes),
            "activated_y_arcs": [list(arc) for arc in self.activated_y_arcs],
            "physical_arc_flows": [list(item) for item in self.physical_arc_flows],
            "route_flows": [list(item) for item in self.route_flows],
            "capacity_choices": [
                choice.to_payload() for choice in self.capacity_choices
            ],
            "chi_flows": [list(item) for item in self.chi_flows],
            "w_flows": [list(item) for item in self.w_flows],
            "delta_levels": [list(item) for item in self.delta_levels],
            "xi_values": [list(item) for item in self.xi_values],
            "queue_v1_values": [list(item) for item in self.queue_v1_values],
            "queue_v2_values": [list(item) for item in self.queue_v2_values],
            "queue_t_values": [list(item) for item in self.queue_t_values],
            "gamma_value": self.gamma_value,
            "precheck_metrics": {
                name: value for name, value in self.precheck_metrics
            },
        }


@dataclass(frozen=True, slots=True)
class GeometryStartPlanAudit:
    plan_version: str
    target_support_size: int
    plan_sha256: str | None
    support_order: tuple[str, ...]
    cycle_descriptors: tuple[tuple[str, ...], ...]
    availability_index: int | None
    availability_probability: float | None
    offered_flow_per_cycle: float | None
    flow_per_cycle: float | None
    planned_service_per_hour: float | None
    family_start_counts: tuple[tuple[str, int], ...]
    prechecks_passed: bool
    precheck_metrics: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.plan_version != GEOMETRY_START_PLAN_VERSION:
            raise ValueError("unsupported geometry plan audit version")
        if (
            isinstance(self.target_support_size, bool)
            or not isinstance(self.target_support_size, int)
            or self.target_support_size < 2
        ):
            raise ValueError("geometry plan target support must be an integer >= 2")
        if self.plan_sha256 is not None and (
            len(self.plan_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.plan_sha256
            )
        ):
            raise ValueError("geometry plan audit has an invalid SHA256")
        if len(set(self.support_order)) != len(self.support_order):
            raise ValueError("geometry plan audit support order has duplicates")
        if any(len(item) != 6 for item in self.cycle_descriptors):
            raise ValueError("geometry cycle audit descriptors must have six fields")
        if type(self.prechecks_passed) is not bool:
            raise ValueError("geometry prechecks_passed must be boolean")
        for name, value in self.precheck_metrics:
            if not name or not math.isfinite(float(value)):
                raise ValueError("geometry precheck metrics must be named and finite")
        if len({name for name, _value in self.precheck_metrics}) != len(
            self.precheck_metrics
        ):
            raise ValueError("geometry precheck metric names must be unique")
        allowed_families = {"x", "y", "z", "psi", "varphi"}
        for family, count in self.family_start_counts:
            if (
                family not in allowed_families
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("geometry plan family counts are invalid")
        if len({name for name, _count in self.family_start_counts}) != len(
            self.family_start_counts
        ):
            raise ValueError("geometry plan family names must be unique")
        numeric = (
            self.availability_probability,
            self.offered_flow_per_cycle,
            self.flow_per_cycle,
            self.planned_service_per_hour,
        )
        if self.prechecks_passed:
            if self.plan_sha256 is None or len(self.support_order) != (
                self.target_support_size
            ):
                raise ValueError("successful geometry plan audit is incomplete")
            if self.availability_index is None or self.availability_index < 0:
                raise ValueError("geometry plan availability index is invalid")
            if any(
                value is None
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in numeric
            ):
                raise ValueError("successful geometry plan numeric audit is invalid")
        elif any(
            (
                self.plan_sha256 is not None,
                bool(self.support_order),
                bool(self.cycle_descriptors),
                self.availability_index is not None,
                any(value is not None for value in numeric),
                bool(self.family_start_counts),
                bool(self.precheck_metrics),
            )
        ):
            raise ValueError("failed geometry plan audit must not claim a plan")

    @classmethod
    def from_plan(cls, plan: GeometryStartPlan) -> GeometryStartPlanAudit:
        return cls(
            plan_version=plan.plan_version,
            target_support_size=plan.target_support_size,
            plan_sha256=plan.plan_sha256,
            support_order=plan.support_order,
            cycle_descriptors=tuple(
                cycle.audit_descriptor for cycle in plan.cycles
            ),
            availability_index=plan.availability_index,
            availability_probability=plan.availability_probability,
            offered_flow_per_cycle=plan.offered_flow_per_cycle,
            flow_per_cycle=plan.flow_per_cycle,
            planned_service_per_hour=plan.planned_service_per_hour,
            family_start_counts=plan.family_start_counts,
            prechecks_passed=True,
            precheck_metrics=plan.precheck_metrics,
        )

    @classmethod
    def planning_failure(
        cls,
        *,
        plan_version: str,
        target_support_size: int,
    ) -> GeometryStartPlanAudit:
        return cls(
            plan_version=plan_version,
            target_support_size=target_support_size,
            plan_sha256=None,
            support_order=(),
            cycle_descriptors=(),
            availability_index=None,
            availability_probability=None,
            offered_flow_per_cycle=None,
            flow_per_cycle=None,
            planned_service_per_hour=None,
            family_start_counts=(),
            prechecks_passed=False,
            precheck_metrics=(),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "plan_version": self.plan_version,
            "target_support_size": self.target_support_size,
            "plan_sha256": self.plan_sha256,
            "support_order": list(self.support_order),
            "cycle_descriptors": [list(item) for item in self.cycle_descriptors],
            "availability_index": self.availability_index,
            "availability_probability": self.availability_probability,
            "offered_flow_per_cycle": self.offered_flow_per_cycle,
            "flow_per_cycle": self.flow_per_cycle,
            "planned_service_per_hour": self.planned_service_per_hour,
            "family_start_counts": {
                name: count for name, count in self.family_start_counts
            },
            "prechecks_passed": self.prechecks_passed,
            "precheck_metrics": {
                name: value for name, value in self.precheck_metrics
            },
        }

    @classmethod
    def from_payload(cls, payload: object) -> GeometryStartPlanAudit:
        if not isinstance(payload, Mapping):
            raise ValueError("geometry plan audit must be an object")
        expected = {
            "plan_version",
            "target_support_size",
            "plan_sha256",
            "support_order",
            "cycle_descriptors",
            "availability_index",
            "availability_probability",
            "offered_flow_per_cycle",
            "flow_per_cycle",
            "planned_service_per_hour",
            "family_start_counts",
            "prechecks_passed",
            "precheck_metrics",
        }
        _require_exact_keys(payload, expected, label="geometry plan audit")
        descriptors = payload["cycle_descriptors"]
        family_counts = payload["family_start_counts"]
        metrics = payload["precheck_metrics"]
        if not isinstance(descriptors, list):
            raise ValueError("cycle_descriptors must be a list")
        if not isinstance(family_counts, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError("geometry plan count/metric fields must be objects")
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            for name, count in family_counts.items()
        ):
            raise ValueError("geometry plan family counts have invalid types")
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            for name, value in metrics.items()
        ):
            raise ValueError("geometry plan precheck metrics have invalid types")
        plan_sha256 = payload["plan_sha256"]
        if plan_sha256 is not None and (
            not isinstance(plan_sha256, str)
            or len(plan_sha256) != 64
            or any(character not in "0123456789abcdef" for character in plan_sha256)
        ):
            raise ValueError("geometry plan audit has an invalid SHA256")
        return cls(
            plan_version=payload["plan_version"],
            target_support_size=payload["target_support_size"],
            plan_sha256=plan_sha256,
            support_order=_string_tuple(payload["support_order"], label="support_order"),
            cycle_descriptors=tuple(
                _string_tuple(item, label="cycle descriptor")
                for item in descriptors
            ),
            availability_index=payload["availability_index"],
            availability_probability=payload["availability_probability"],
            offered_flow_per_cycle=payload["offered_flow_per_cycle"],
            flow_per_cycle=payload["flow_per_cycle"],
            planned_service_per_hour=payload["planned_service_per_hour"],
            family_start_counts=tuple(
                sorted((name, count) for name, count in family_counts.items())
            ),
            prechecks_passed=payload["prechecks_passed"],
            precheck_metrics=tuple(
                sorted((name, float(value)) for name, value in metrics.items())
            ),
        )


@dataclass(frozen=True, slots=True)
class GeometryStartInstallAudit:
    reset_performed: bool
    undefined_variable_count: int
    family_start_counts: tuple[tuple[str, int], ...]
    # Both before/after fingerprints are read with NumStart=0.  Gurobi is
    # expected to report a different fingerprint while the MIP start exists;
    # that intermediate value is evidence only and is not a safety failure.
    fingerprint_before: int | None
    fingerprint_with_start: int | None
    fingerprint_after: int | None
    fingerprint_unchanged: bool
    constraint_counts_before: tuple[int, int, int]
    constraint_counts_after: tuple[int, int, int]
    hard_constraints_added: bool
    bounds_modified: bool = False
    objective_modified: bool = False

    def __post_init__(self) -> None:
        for name in (
            "reset_performed",
            "fingerprint_unchanged",
            "hard_constraints_added",
            "bounds_modified",
            "objective_modified",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"geometry install {name} must be boolean")
        if (
            isinstance(self.undefined_variable_count, bool)
            or not isinstance(self.undefined_variable_count, int)
            or self.undefined_variable_count < 0
        ):
            raise ValueError("geometry undefined variable count is invalid")
        for fingerprint in (
            self.fingerprint_before,
            self.fingerprint_with_start,
            self.fingerprint_after,
        ):
            if fingerprint is not None and (
                isinstance(fingerprint, bool) or not isinstance(fingerprint, int)
            ):
                raise ValueError("geometry model fingerprints must be integers")
        if self.fingerprint_unchanged != bool(
            self.fingerprint_before is not None
            and self.fingerprint_after is not None
            and self.fingerprint_before == self.fingerprint_after
        ):
            raise ValueError("geometry fingerprint audit is inconsistent")
        for counts in (self.constraint_counts_before, self.constraint_counts_after):
            if len(counts) != 3 or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in counts
            ):
                raise ValueError("geometry constraint counts are invalid")
        if self.hard_constraints_added != (
            self.constraint_counts_before != self.constraint_counts_after
        ):
            raise ValueError("geometry hard-constraint audit is inconsistent")
        allowed_families = {"x", "y", "z", "psi", "varphi"}
        if any(
            family not in allowed_families
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for family, count in self.family_start_counts
        ):
            raise ValueError("geometry install family counts are invalid")
        if len({name for name, _count in self.family_start_counts}) != len(
            self.family_start_counts
        ):
            raise ValueError("geometry install family names must be unique")

    @property
    def safety_preserved(self) -> bool:
        """Whether the positive-only start left the mathematical model intact."""

        return bool(
            self.reset_performed
            and self.fingerprint_before is not None
            and self.fingerprint_unchanged
            and not self.hard_constraints_added
            and not self.bounds_modified
            and not self.objective_modified
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "reset_performed": self.reset_performed,
            "undefined_variable_count": self.undefined_variable_count,
            "family_start_counts": {
                name: count for name, count in self.family_start_counts
            },
            "fingerprint_before": self.fingerprint_before,
            "fingerprint_with_start": self.fingerprint_with_start,
            "fingerprint_after": self.fingerprint_after,
            "fingerprint_unchanged": self.fingerprint_unchanged,
            "constraint_counts_before": list(self.constraint_counts_before),
            "constraint_counts_after": list(self.constraint_counts_after),
            "hard_constraints_added": self.hard_constraints_added,
            "bounds_modified": self.bounds_modified,
            "objective_modified": self.objective_modified,
        }

    @classmethod
    def from_payload(cls, payload: object) -> GeometryStartInstallAudit:
        if not isinstance(payload, Mapping):
            raise ValueError("geometry install audit must be an object")
        expected = {
            "reset_performed",
            "undefined_variable_count",
            "family_start_counts",
            "fingerprint_before",
            "fingerprint_with_start",
            "fingerprint_after",
            "fingerprint_unchanged",
            "constraint_counts_before",
            "constraint_counts_after",
            "hard_constraints_added",
            "bounds_modified",
            "objective_modified",
        }
        _require_exact_keys(payload, expected, label="geometry install audit")
        counts = payload["family_start_counts"]
        before = payload["constraint_counts_before"]
        after = payload["constraint_counts_after"]
        if not isinstance(counts, Mapping):
            raise ValueError("geometry install family counts must be an object")
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            for name, count in counts.items()
        ):
            raise ValueError("geometry install family counts have invalid types")
        if not isinstance(before, list) or not isinstance(after, list):
            raise ValueError("geometry constraint counts must be lists")
        return cls(
            reset_performed=payload["reset_performed"],
            undefined_variable_count=payload["undefined_variable_count"],
            family_start_counts=tuple(
                sorted((name, count) for name, count in counts.items())
            ),
            fingerprint_before=payload["fingerprint_before"],
            fingerprint_with_start=payload["fingerprint_with_start"],
            fingerprint_after=payload["fingerprint_after"],
            fingerprint_unchanged=payload["fingerprint_unchanged"],
            constraint_counts_before=tuple(before),
            constraint_counts_after=tuple(after),
            hard_constraints_added=payload["hard_constraints_added"],
            bounds_modified=payload["bounds_modified"],
            objective_modified=payload["objective_modified"],
        )


@dataclass(frozen=True, slots=True)
class GeometryStartAttemptAudit:
    """One registered support attempt, including safety and solve evidence."""

    plan: GeometryStartPlanAudit
    per_attempt_seconds: float
    start_node_limit: int
    install: GeometryStartInstallAudit | None = None
    solve_status: str = "NOT_RUN"
    solve_seconds: float = 0.0
    full_witness_found: bool = False
    cleanup_performed: bool = False
    cleanup_succeeded: bool | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.per_attempt_seconds, bool)
            or not isinstance(self.per_attempt_seconds, (int, float))
            or not math.isfinite(float(self.per_attempt_seconds))
            or float(self.per_attempt_seconds) <= 0.0
        ):
            raise ValueError("geometry attempt budget must be positive and finite")
        if (
            isinstance(self.start_node_limit, bool)
            or not isinstance(self.start_node_limit, int)
            or self.start_node_limit < 0
            or self.start_node_limit > GUROBI_MAX_START_NODE_LIMIT
        ):
            raise ValueError("geometry attempt StartNodeLimit is invalid")
        if not isinstance(self.solve_status, str) or not self.solve_status:
            raise ValueError("geometry attempt solve status must be nonempty")
        if (
            isinstance(self.solve_seconds, bool)
            or not isinstance(self.solve_seconds, (int, float))
            or not math.isfinite(float(self.solve_seconds))
            or float(self.solve_seconds) < 0.0
        ):
            raise ValueError("geometry attempt solve time must be finite and nonnegative")
        if type(self.full_witness_found) is not bool:
            raise ValueError("geometry attempt witness flag must be boolean")
        if type(self.cleanup_performed) is not bool:
            raise ValueError("geometry attempt cleanup flag must be boolean")
        if self.cleanup_succeeded is not None and type(
            self.cleanup_succeeded
        ) is not bool:
            raise ValueError("geometry attempt cleanup result must be boolean or null")
        if self.cleanup_performed != (self.cleanup_succeeded is not None):
            raise ValueError("geometry attempt cleanup audit is inconsistent")
        if self.error is not None and (
            not isinstance(self.error, str) or not self.error
        ):
            raise ValueError("geometry attempt error must be nonempty or null")
        if self.install is not None and not self.plan.prechecks_passed:
            raise ValueError("geometry install requires a prechecked plan")
        if self.install is not None and (
            self.install.family_start_counts != self.plan.family_start_counts
        ):
            raise ValueError("geometry install counts do not match plan audit")
        if not self.plan.prechecks_passed:
            if any(
                (
                    self.install is not None,
                    self.solve_status != "NOT_RUN",
                    float(self.solve_seconds) != 0.0,
                    self.full_witness_found,
                    self.cleanup_performed,
                )
            ):
                raise ValueError(
                    "failed geometry precheck cannot install, solve, witness, or clean"
                )
            if self.error is None:
                raise ValueError("failed geometry precheck requires an error audit")
            return
        if self.install is None:
            if any(
                (
                    self.solve_status != "NOT_RUN",
                    float(self.solve_seconds) != 0.0,
                    self.full_witness_found,
                    self.cleanup_performed,
                )
            ):
                raise ValueError(
                    "uninstalled geometry attempt cannot solve, witness, or clean"
                )
            if self.error is None:
                raise ValueError("uninstalled geometry attempt requires an error audit")
            return
        if self.full_witness_found:
            if not self.install.safety_preserved:
                raise ValueError("successful geometry witness requires a safe install")
            if self.solve_status == "NOT_RUN":
                raise ValueError("successful geometry witness requires a solve")
            if self.error is not None:
                raise ValueError("successful geometry witness cannot contain an error")
            if self.cleanup_performed:
                raise ValueError("successful geometry witness cannot be cleaned up")
        else:
            if not self.cleanup_performed:
                raise ValueError(
                    "installed unsuccessful geometry attempt requires cleanup"
                )
            if self.solve_status == "NOT_RUN" and self.error is None:
                raise ValueError("unsolved installed geometry attempt requires an error")

    def to_payload(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_payload(),
            "per_attempt_seconds": self.per_attempt_seconds,
            "start_node_limit": self.start_node_limit,
            "install": self.install.to_payload() if self.install is not None else None,
            "solve_status": self.solve_status,
            "solve_seconds": self.solve_seconds,
            "full_witness_found": self.full_witness_found,
            "cleanup_performed": self.cleanup_performed,
            "cleanup_succeeded": self.cleanup_succeeded,
            "error": self.error,
        }

    @classmethod
    def from_payload(cls, payload: object) -> GeometryStartAttemptAudit:
        if not isinstance(payload, Mapping):
            raise ValueError("geometry start attempt must be an object")
        expected = {
            "plan",
            "per_attempt_seconds",
            "start_node_limit",
            "install",
            "solve_status",
            "solve_seconds",
            "full_witness_found",
            "cleanup_performed",
            "cleanup_succeeded",
            "error",
        }
        _require_exact_keys(payload, expected, label="geometry start attempt")
        install_payload = payload["install"]
        return cls(
            plan=GeometryStartPlanAudit.from_payload(payload["plan"]),
            per_attempt_seconds=payload["per_attempt_seconds"],
            start_node_limit=payload["start_node_limit"],
            install=(
                GeometryStartInstallAudit.from_payload(install_payload)
                if install_payload is not None
                else None
            ),
            solve_status=payload["solve_status"],
            solve_seconds=payload["solve_seconds"],
            full_witness_found=payload["full_witness_found"],
            cleanup_performed=payload["cleanup_performed"],
            cleanup_succeeded=payload["cleanup_succeeded"],
            error=payload["error"],
        )


def _canonical_route_option(
    instance: object,
    arc: tuple[str, str],
) -> tuple[str, float, float] | None:
    candidates: list[tuple[float, float, str]] = []
    for option in instance.route_options.get(arc, ()):
        route_time = float(instance.route_time[arc[0], arc[1], option])
        route_distance = float(instance.route_distance[arc[0], arc[1], option])
        # ``route_options`` already contains only physically admissible direct
        # and one-swap routes.  Under formulation v3, a base departure and each
        # selected swap event contribute battery supply to one system-wide
        # balance, so rejecting an option merely because its aggregate route
        # time exceeds one battery would reintroduce the obsolete per-node
        # energy model.  Time, distance and identifier are deterministic,
        # cost-blind tie-breakers.
        candidates.append((route_time, route_distance, str(option)))
    if not candidates:
        return None
    route_time, distance, option = min(candidates)
    return option, distance, route_time


def _canonical_cycle(
    instance: object,
    corridor: tuple[str, str],
) -> GeometryCycle | None:
    origin_h, destination_h = corridor
    candidates: list[GeometryCycle] = []
    for base in sorted(instance.L):
        arcs = ((base, origin_h), corridor, (destination_h, base))
        if any(arc not in instance.A_F for arc in arcs):
            continue
        options = tuple(_canonical_route_option(instance, arc) for arc in arcs)
        if any(item is None for item in options):
            continue
        concrete = tuple(item for item in options if item is not None)
        candidates.append(
            GeometryCycle(
                base=base,
                origin_h=origin_h,
                destination_h=destination_h,
                route_options=tuple(item[0] for item in concrete),
                route_distances=tuple(item[1] for item in concrete),
                route_times=tuple(item[2] for item in concrete),
            )
        )
    return min(candidates, key=lambda cycle: cycle.geometry_key) if candidates else None


def _largest_component(
    nodes: Sequence[str],
    adjacency: Mapping[str, set[str]],
) -> tuple[str, ...]:
    remaining = set(nodes)
    components: list[tuple[str, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            remaining.discard(node)
            stack.extend(sorted(adjacency.get(node, set()) - component, reverse=True))
        components.append(tuple(sorted(component)))
    if not components:
        raise GeometryStartInfeasibilityError("serviceability graph is empty")
    return sorted(components, key=lambda item: (-len(item), item))[0]


def _distance(instance: object, first: str, second: str) -> float:
    first_xy = instance.coords[first]
    second_xy = instance.coords[second]
    return math.hypot(first_xy[0] - second_xy[0], first_xy[1] - second_xy[1])


def _support_tree(
    instance: object,
    component: tuple[str, ...],
    pair_cycles: Mapping[frozenset[str], tuple[GeometryCycle, ...]],
    target_support_size: int,
) -> tuple[tuple[str, ...], tuple[GeometryCycle, ...]]:
    if target_support_size < 2:
        raise GeometryStartPlanningError("geometry support size must be at least two")
    if len(component) < target_support_size:
        raise GeometryStartInfeasibilityError(
            "largest serviceable H component is too small: "
            f"{len(component)} < {target_support_size}"
        )
    centroid_x = math.fsum(instance.coords[node][0] for node in component) / len(
        component
    )
    centroid_y = math.fsum(instance.coords[node][1] for node in component) / len(
        component
    )
    anchor = min(
        component,
        key=lambda node: (
            math.hypot(
                instance.coords[node][0] - centroid_x,
                instance.coords[node][1] - centroid_y,
            ),
            node,
        ),
    )
    selected = [anchor]
    selected_set = {anchor}
    cycles: list[GeometryCycle] = []
    while len(selected) < target_support_size:
        frontier = [
            node
            for node in component
            if node not in selected_set
            and any(frozenset((node, parent)) in pair_cycles for parent in selected)
        ]
        if not frontier:
            raise GeometryStartPlanningError(
                "connected farthest-first traversal exhausted its frontier"
            )
        child = min(
            frontier,
            key=lambda node: (
                -min(_distance(instance, node, parent) for parent in selected),
                node,
            ),
        )
        parent_cycle_candidates: list[tuple[tuple[object, ...], str, GeometryCycle]] = []
        for parent in selected:
            for cycle in pair_cycles.get(frozenset((child, parent)), ()):
                parent_cycle_candidates.append((cycle.geometry_key, parent, cycle))
        if not parent_cycle_candidates:
            raise GeometryStartPlanningError(
                f"no canonical cycle connects frontier node {child!r}"
            )
        _key, _parent, chosen_cycle = min(parent_cycle_candidates)
        selected.append(child)
        selected_set.add(child)
        cycles.append(chosen_cycle)
    return tuple(selected), tuple(cycles)


def _allocate_chi_without_costs(
    wrapper: object,
    cycles: Sequence[GeometryCycle],
    *,
    availability_index: int,
    offered_flow_per_cycle: float,
) -> tuple[tuple[str, str, str, str, int, float], ...]:
    """Allocate cycle offers against OD capacities in deterministic graph order.

    Scarcer corridors are filled first, then OD keys are visited
    lexicographically.  This is a feasibility precheck, not an objective-based
    ranking.  A failure is conservative: the plan is rejected rather than
    weakening any OD bound.
    """

    remaining_market = {
        market: float(volume) for market, volume in sorted(wrapper.inst.h_od.items())
    }
    eligible_by_cycle: dict[int, tuple[tuple[str, str, str, str, int], ...]] = {}
    for cycle_index, cycle in enumerate(cycles):
        keys = tuple(
            sorted(
                key
                for key in wrapper.chi_by_corridor_level.get(
                    (cycle.origin_h, cycle.destination_h, availability_index),
                    (),
                )
            )
        )
        if not keys:
            raise GeometryStartInfeasibilityError(
                "selected current-mode corridor has no chi key at the common "
                f"availability level: {(cycle.origin_h, cycle.destination_h)!r}"
            )
        eligible_by_cycle[cycle_index] = keys

    allocations: dict[tuple[str, str, str, str, int], float] = {}
    order = sorted(
        range(len(cycles)),
        key=lambda index: (
            len(eligible_by_cycle[index]),
            cycles[index].audit_descriptor,
        ),
    )
    for cycle_index in order:
        residual = offered_flow_per_cycle
        for key in eligible_by_cycle[cycle_index]:
            market = (key[0], key[1])
            available = remaining_market.get(market, 0.0)
            amount = min(available, residual)
            if amount <= 0.0:
                continue
            allocations[key] = allocations.get(key, 0.0) + amount
            remaining_market[market] = available - amount
            residual -= amount
            if residual <= 1e-9:
                break
        if residual > 1e-9:
            cycle = cycles[cycle_index]
            raise GeometryStartInfeasibilityError(
                "OD demand cannot support the registered cycle offer on "
                f"{(cycle.origin_h, cycle.destination_h)!r}; residual={residual}"
            )

    market_totals: dict[tuple[str, str], float] = {}
    corridor_totals: dict[tuple[str, str], float] = {}
    for key, amount in allocations.items():
        market = (key[0], key[1])
        corridor = (key[2], key[3])
        market_totals[market] = market_totals.get(market, 0.0) + amount
        corridor_totals[corridor] = corridor_totals.get(corridor, 0.0) + amount
    for market, amount in market_totals.items():
        upper = float(wrapper.inst.h_od[market])
        if amount > upper + 1e-9:
            raise GeometryStartPlanningError(
                f"planned chi violates h_od for {market!r}: {amount} > {upper}"
            )
    for cycle in cycles:
        corridor = (cycle.origin_h, cycle.destination_h)
        amount = corridor_totals.get(corridor, 0.0)
        if not math.isclose(
            amount,
            offered_flow_per_cycle,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise GeometryStartPlanningError(
                f"planned chi does not match cycle offer on {corridor!r}"
            )
    return tuple((*key, amount) for key, amount in sorted(allocations.items()))


def build_geometry_start_plan(
    wrapper: object,
    *,
    scale: str,
    target_support_size: int,
    required_service_per_hour: float,
    active_h_tolerance_per_hour: float,
    active_tolerance_multiplier: float = 2.0,
    required_service_multiplier: float = 1.05,
) -> GeometryStartPlan:
    """Build one deterministic topology-only support and validate its flows."""

    if not math.isfinite(required_service_per_hour) or required_service_per_hour <= 0:
        raise GeometryStartPlanningError("required service must be positive and finite")
    if (
        not math.isfinite(active_h_tolerance_per_hour)
        or active_h_tolerance_per_hour <= 0
    ):
        raise GeometryStartPlanningError(
            "active-H tolerance must be positive and finite"
        )
    if active_tolerance_multiplier != 2.0:
        raise GeometryStartPlanningError(
            "active-H tolerance multiplier must equal the frozen value 2.0"
        )
    if required_service_multiplier != 1.05:
        raise GeometryStartPlanningError(
            "required-service multiplier must equal the frozen value 1.05"
        )
    instance = wrapper.inst
    if getattr(wrapper, "mode", None) == "LO":
        # LO couples every selected display to an all-demand allocation.  A
        # small topology seed is not generally completable under that equality,
        # so the first version refuses LO instead of supplying a misleading
        # partial start.
        raise GeometryStartPlanningError(
            "qualification geometry start is fail-closed for LO mode"
        )
    if getattr(wrapper, "mode", None) not in {"MX", "HD"}:
        raise GeometryStartPlanningError("unsupported qualification model mode")
    lockers = tuple(sorted(instance.H))
    if target_support_size > len(lockers):
        raise GeometryStartPlanningError("geometry support exceeds candidate H count")

    directed_cycles: list[GeometryCycle] = []
    for corridor, chi_keys in sorted(wrapper.chi_by_corridor.items()):
        if not chi_keys:
            continue
        cycle = _canonical_cycle(instance, corridor)
        if cycle is not None:
            directed_cycles.append(cycle)
    if not directed_cycles:
        raise GeometryStartInfeasibilityError(
            "no current-mode corridor has an energy-feasible swap cycle"
        )

    pair_cycles_lists: dict[frozenset[str], list[GeometryCycle]] = {}
    adjacency = {locker: set() for locker in lockers}
    for cycle in directed_cycles:
        pair = frozenset((cycle.origin_h, cycle.destination_h))
        if len(pair) != 2:
            continue
        pair_cycles_lists.setdefault(pair, []).append(cycle)
        adjacency[cycle.origin_h].add(cycle.destination_h)
        adjacency[cycle.destination_h].add(cycle.origin_h)
    pair_cycles = {
        pair: tuple(sorted(cycles, key=lambda item: item.geometry_key))
        for pair, cycles in pair_cycles_lists.items()
    }
    serviceable_nodes = tuple(
        locker for locker in lockers if adjacency.get(locker)
    )
    component = _largest_component(serviceable_nodes, adjacency)
    support_order, cycles = _support_tree(
        instance,
        component,
        pair_cycles,
        target_support_size,
    )

    cycle_count = len(cycles)
    availability_index, availability_probability = max(
        enumerate(instance.config.availability_levels),
        key=lambda item: (float(item[1]), -item[0]),
    )
    availability_probability = float(availability_probability)
    flow_per_cycle = max(
        active_tolerance_multiplier * active_h_tolerance_per_hour,
        required_service_multiplier * required_service_per_hour / cycle_count,
    )
    offered_flow_per_cycle = flow_per_cycle / availability_probability
    chi_flows = _allocate_chi_without_costs(
        wrapper,
        cycles,
        availability_index=availability_index,
        offered_flow_per_cycle=offered_flow_per_cycle,
    )
    physical_flows: dict[tuple[str, str], float] = {}
    route_flows: dict[tuple[str, str, str], float] = {}
    activated_y: set[tuple[str, str]] = set()
    open_nodes = set(support_order)
    for cycle in cycles:
        open_nodes.add(cycle.base)
        for arc, option in zip(cycle.arcs, cycle.route_options, strict=True):
            physical_flows[arc] = physical_flows.get(arc, 0.0) + flow_per_cycle
            route_key = (arc[0], arc[1], option)
            route_flows[route_key] = route_flows.get(route_key, 0.0) + flow_per_cycle
            activated_y.add(arc)
            if option != DIRECT_ROUTE:
                activated_y.add((arc[0], option))
                activated_y.add((option, arc[1]))
                open_nodes.add(option)

    tolerance = 1e-9
    max_psi_utilization = 0.0
    for arc, flow in sorted(physical_flows.items()):
        upper = float(instance.M_psi.get(arc, -1.0))
        if upper <= 0.0 or flow > upper + tolerance:
            raise GeometryStartInfeasibilityError(
                f"physical flow violates M_psi on {arc!r}: {flow} > {upper}"
            )
        max_psi_utilization = max(max_psi_utilization, flow / upper)

    max_offer_utilization = 0.0
    offered_by_loaded: dict[tuple[str, str], float] = {}
    available_w_keys = set(wrapper.w_keys)
    for cycle in cycles:
        loaded = (cycle.origin_h, cycle.destination_h)
        offered_by_loaded[loaded] = (
            offered_by_loaded.get(loaded, 0.0) + offered_flow_per_cycle
        )
        w_key = (
            cycle.base,
            cycle.origin_h,
            cycle.destination_h,
            availability_index,
        )
        if w_key not in available_w_keys:
            raise GeometryStartInfeasibilityError(
                f"canonical cycle has no matching w key: {w_key!r}"
            )
    for loaded, offered in sorted(offered_by_loaded.items()):
        offer_upper = float(wrapper.M_offer.get(loaded, -1.0))
        if offer_upper <= 0.0 or offered > offer_upper + tolerance:
            raise GeometryStartInfeasibilityError(
                f"planned w violates M_offer on {loaded!r}: "
                f"{offered} > {offer_upper}"
            )
        max_offer_utilization = max(
            max_offer_utilization,
            offered / offer_upper,
        )

    usable = float(instance.config.usable_battery_time_hours)
    max_wait = float(instance.config.max_locker_time_hours)
    service_rates = tuple(float(value) for value in instance.config.locker_service_rates)
    capacities: list[GeometryCapacityChoice] = []
    minimum_queue_slack = math.inf
    minimum_population_upper_slack = math.inf
    for locker in support_order:
        incoming = math.fsum(
            flow for (u, v), flow in physical_flows.items() if v == locker
        )
        outgoing = math.fsum(
            flow for (u, v), flow in physical_flows.items() if u == locker
        )
        if not math.isclose(incoming, outgoing, rel_tol=1e-12, abs_tol=1e-9):
            raise GeometryStartPlanningError(
                f"planned physical flow is unbalanced at {locker!r}"
            )
        if 2.0 * incoming <= active_h_tolerance_per_hour + tolerance:
            raise GeometryStartPlanningError(
                f"planned H flow is not active at {locker!r}"
            )
        outgoing_route_time = math.fsum(
            float(instance.route_time[key]) * flow
            for key, flow in route_flows.items()
            if key[0] == locker
        )
        # The largest registered service rate is the deterministic, cost-blind
        # choice.  Smaller levels are never compared by their capacity cost.
        capacity_index, service_rate = max(
            enumerate(service_rates),
            key=lambda item: (item[1], -item[0]),
        )
        chosen: GeometryCapacityChoice | None = None
        denominator = service_rate - incoming
        if denominator > 0.0:
            queue_population = incoming / denominator
            queue_slack = denominator - 1.0 / max_wait
            population_upper = float(instance.M_xi.get(locker, -1.0))
            population_upper_slack = population_upper - queue_population
            if (
                queue_slack >= -tolerance
                and population_upper_slack >= -tolerance
            ):
                chosen = GeometryCapacityChoice(
                    locker=locker,
                    capacity_index=capacity_index,
                    service_rate=service_rate,
                    incident_flow=incoming,
                    outgoing_route_time=outgoing_route_time,
                    queue_population=queue_population,
                    queue_slack=queue_slack,
                    population_upper_slack=population_upper_slack,
                )
        if chosen is None:
            raise GeometryStartInfeasibilityError(
                f"no capacity level passes queue/population prechecks at {locker!r}"
            )
        capacities.append(chosen)
        minimum_queue_slack = min(minimum_queue_slack, chosen.queue_slack)
        minimum_population_upper_slack = min(
            minimum_population_upper_slack,
            chosen.population_upper_slack,
        )

    queue_population_total = math.fsum(
        choice.queue_population for choice in capacities
    )
    route_population_total = math.fsum(
        float(instance.route_time[key]) * flow
        for key, flow in route_flows.items()
    )
    base_departure_arcs = set(getattr(instance, "A_LH", ()))
    if not base_departure_arcs:
        base_departure_arcs = {
            (u, v)
            for u, v in physical_flows
            if u in set(instance.L) and v in set(instance.H)
        }
    base_battery_departure_flow = math.fsum(
        flow
        for (u, v, _option), flow in route_flows.items()
        if (u, v) in base_departure_arcs
    )
    swap_event_flow = math.fsum(
        flow
        for (_u, _v, option), flow in route_flows.items()
        if option != DIRECT_ROUTE
    )
    system_energy_use = queue_population_total + route_population_total
    system_energy_supply = usable * (
        base_battery_departure_flow + swap_event_flow
    )
    system_energy_slack = system_energy_supply - system_energy_use
    if system_energy_slack < -tolerance:
        raise GeometryStartInfeasibilityError(
            "planned system energy precheck fails: "
            f"use={system_energy_use}, supply={system_energy_supply}"
        )

    w_flows = tuple(
        (
            cycle.base,
            cycle.origin_h,
            cycle.destination_h,
            availability_index,
            offered_flow_per_cycle,
        )
        for cycle in cycles
    )
    delta_levels = tuple(
        (base, availability_index)
        for base in sorted({cycle.base for cycle in cycles})
    )

    # Recompute every linking equality from the aggregated sparse values.  This
    # catches accidental overwrite when cycles share a base leg or H node.
    offered_by_corridor: dict[tuple[str, str], float] = {}
    source_realized: dict[tuple[str, str], float] = {}
    for base, origin_h, destination_h, _r, offered in w_flows:
        corridor = (origin_h, destination_h)
        offered_by_corridor[corridor] = (
            offered_by_corridor.get(corridor, 0.0) + offered
        )
        source = (base, origin_h)
        source_realized[source] = (
            source_realized.get(source, 0.0)
            + availability_probability * offered
        )
    chi_by_corridor: dict[tuple[str, str], float] = {}
    for _o, _d, origin_h, destination_h, _r, offered in chi_flows:
        corridor = (origin_h, destination_h)
        chi_by_corridor[corridor] = chi_by_corridor.get(corridor, 0.0) + offered
    for corridor, offered in offered_by_corridor.items():
        if not math.isclose(
            chi_by_corridor.get(corridor, 0.0),
            offered,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise GeometryStartPlanningError(
                f"chi/w reliability match fails on {corridor!r}"
            )
        loaded_realized = availability_probability * offered
        if not math.isclose(
            physical_flows.get(corridor, 0.0),
            loaded_realized,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise GeometryStartPlanningError(
                f"loaded-flow identity fails on {corridor!r}"
            )
    for source, realized in source_realized.items():
        if not math.isclose(
            physical_flows.get(source, 0.0),
            realized,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise GeometryStartPlanningError(
                f"source-flow identity fails on {source!r}"
            )
    for node in sorted(set(instance.L) | set(instance.H)):
        incoming = math.fsum(
            flow for (_u, v), flow in physical_flows.items() if v == node
        )
        outgoing = math.fsum(
            flow for (u, _v), flow in physical_flows.items() if u == node
        )
        if not math.isclose(incoming, outgoing, rel_tol=1e-12, abs_tol=1e-9):
            raise GeometryStartPlanningError(
                f"aggregated physical flow is unbalanced at {node!r}"
            )
    for arc, flow in physical_flows.items():
        route_total = math.fsum(
            amount
            for (u, v, _option), amount in route_flows.items()
            if (u, v) == arc
        )
        if not math.isclose(flow, route_total, rel_tol=1e-12, abs_tol=1e-9):
            raise GeometryStartPlanningError(
                f"route decomposition fails on {arc!r}"
            )
    if any(arc not in instance.A_Y for arc in activated_y):
        raise GeometryStartPlanningError("planned y arc is absent from A_Y")

    xi_values = tuple(
        (choice.locker, choice.queue_population) for choice in capacities
    )
    queue_v1_values = tuple(
        (choice.locker, 2.0 * math.sqrt(choice.service_rate))
        for choice in capacities
    )
    queue_v2_values = tuple(
        (
            choice.locker,
            choice.queue_population
            + choice.incident_flow
            - choice.service_rate
            + 1.0,
        )
        for choice in capacities
    )
    queue_t_values = tuple(
        (
            choice.locker,
            choice.queue_population
            - choice.incident_flow
            + choice.service_rate
            + 1.0,
        )
        for choice in capacities
    )
    transit_population = route_population_total
    fleet_lower_bound = (
        len(delta_levels)
        * availability_probability
        / (1.0 - availability_probability)
        + math.fsum(value for _locker, value in xi_values)
        + transit_population
    )
    gamma_value = math.ceil(fleet_lower_bound - 1e-12)
    if gamma_value < 0:
        raise GeometryStartPlanningError("planned fleet size is negative")

    planned_service = cycle_count * flow_per_cycle
    if (
        planned_service + tolerance
        < required_service_multiplier * required_service_per_hour
    ):
        raise GeometryStartPlanningError("planned service margin is insufficient")
    metrics = tuple(
        sorted(
            {
                "largest_component_h": float(len(component)),
                "base_battery_departure_flow": base_battery_departure_flow,
                "max_m_offer_utilization": max_offer_utilization,
                "max_m_psi_utilization": max_psi_utilization,
                "minimum_population_upper_slack": (
                    minimum_population_upper_slack
                ),
                "minimum_queue_slack": minimum_queue_slack,
                "offered_flow_per_cycle": offered_flow_per_cycle,
                "planned_active_h": float(len(support_order)),
                "planned_service_margin": planned_service
                - required_service_per_hour,
                "queue_population_total": queue_population_total,
                "route_population_total": route_population_total,
                "swap_event_flow": swap_event_flow,
                "system_energy_slack": system_energy_slack,
                "system_energy_supply": system_energy_supply,
                "system_energy_use": system_energy_use,
            }.items()
        )
    )
    identity = {
        "plan_version": GEOMETRY_START_PLAN_VERSION,
        "scale": scale,
        "target_support_size": target_support_size,
        "support_order": list(support_order),
        "cycles": [cycle.to_payload() for cycle in cycles],
        "availability_index": availability_index,
        "availability_probability": availability_probability,
        "offered_flow_per_cycle": offered_flow_per_cycle,
        "flow_per_cycle": flow_per_cycle,
        "planned_service_per_hour": planned_service,
        "open_nodes": sorted(open_nodes),
        "activated_y_arcs": [list(arc) for arc in sorted(activated_y)],
        "physical_arc_flows": [
            [u, v, flow] for (u, v), flow in sorted(physical_flows.items())
        ],
        "route_flows": [
            [u, v, option, flow]
            for (u, v, option), flow in sorted(route_flows.items())
        ],
        "capacity_choices": [choice.to_payload() for choice in capacities],
        "chi_flows": [list(item) for item in chi_flows],
        "w_flows": [list(item) for item in w_flows],
        "delta_levels": [list(item) for item in delta_levels],
        "xi_values": [list(item) for item in xi_values],
        "queue_v1_values": [list(item) for item in queue_v1_values],
        "queue_v2_values": [list(item) for item in queue_v2_values],
        "queue_t_values": [list(item) for item in queue_t_values],
        "gamma_value": gamma_value,
        "precheck_metrics": {name: value for name, value in metrics},
    }
    return GeometryStartPlan(
        plan_version=GEOMETRY_START_PLAN_VERSION,
        scale=scale,
        target_support_size=target_support_size,
        support_order=support_order,
        cycles=cycles,
        availability_index=availability_index,
        availability_probability=availability_probability,
        offered_flow_per_cycle=offered_flow_per_cycle,
        flow_per_cycle=flow_per_cycle,
        planned_service_per_hour=planned_service,
        open_nodes=tuple(sorted(open_nodes)),
        activated_y_arcs=tuple(sorted(activated_y)),
        physical_arc_flows=tuple(
            (u, v, flow) for (u, v), flow in sorted(physical_flows.items())
        ),
        route_flows=tuple(
            (u, v, option, flow)
            for (u, v, option), flow in sorted(route_flows.items())
        ),
        capacity_choices=tuple(capacities),
        chi_flows=chi_flows,
        w_flows=w_flows,
        delta_levels=delta_levels,
        xi_values=xi_values,
        queue_v1_values=queue_v1_values,
        queue_v2_values=queue_v2_values,
        queue_t_values=queue_t_values,
        gamma_value=gamma_value,
        precheck_metrics=metrics,
        plan_sha256=_canonical_sha256(identity),
    )


__all__ = [
    "GEOMETRY_START_PLAN_VERSION",
    "GUROBI_MAX_START_NODE_LIMIT",
    "GeometryStartAttemptAudit",
    "GeometryStartInstallAudit",
    "GeometryStartInfeasibilityError",
    "GeometryStartPlan",
    "GeometryStartPlanAudit",
    "GeometryStartPlanningError",
    "GeometryStartPolicy",
    "build_geometry_start_plan",
]
