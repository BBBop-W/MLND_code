"""Demand, nesting and source-integrity checks for scale generation."""

from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import Counter
from typing import Mapping, Sequence
from .generator import NanshanAdapter
from .scale_config import (
    CANONICAL_FILES,
    NanshanScaleLevel,
    NanshanScaleBuildConfig,
    _GeneratedLevel,
    _RawOrderAudit,
)
from .geometry import (
    _project_order,
    _both_endpoints_inside,
    _order_token,
    _in_hour_window,
)


def _read_generated_level(directory: Path, level: NanshanScaleLevel) -> _GeneratedLevel:
    missing = [name for name in CANONICAL_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"canonical instance {directory} is missing: {', '.join(missing)}"
        )
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    return _GeneratedLevel(
        level=level,
        directory=directory,
        metadata=metadata,
        sites=_read_csv(directory / "sites.csv"),
        nodes=_read_csv(directory / "od_nodes.csv"),
        demand=_read_csv(directory / "od_demand.csv"),
        periods=_read_csv(directory / "periods.csv"),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header")
        return [dict(row) for row in reader]


def _validate_generated_level(
    instance: _GeneratedLevel,
    config: NanshanScaleBuildConfig,
) -> None:
    metadata = instance.metadata
    level = instance.level
    if metadata.get("source") != "nanshan":
        raise ValueError(f"{instance.directory} is not a Nanshan instance")
    if metadata.get("coordinate_system") != "local_km":
        raise ValueError(f"{instance.directory} does not use local_km coordinates")
    data_config = metadata.get("data_config")
    if not isinstance(data_config, Mapping):
        raise ValueError(f"{instance.directory} metadata is missing data_config")
    expected_config: dict[str, object] = {
        "candidate_count": level.lockers,
        "num_bases": level.bases,
        "num_swaps": level.swaps,
        "candidate_hour_start": config.candidate_hour_start,
        "candidate_hour_end": config.candidate_hour_end,
        "demand_hour_start": config.demand_hour_start,
        "demand_hour_end": config.demand_hour_end,
        "map_half_size_km": level.half_size_km,
        "od_grid_km": config.od_grid_km,
        "site_grid_km": config.site_grid_km,
        "seed": config.seed,
        "dates": list(sorted(set(config.dates))),
    }
    mismatches = [
        f"{field}: expected {expected!r}, got {data_config.get(field)!r}"
        for field, expected in expected_config.items()
        if data_config.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            f"{instance.directory} preprocessing contract mismatch: "
            + "; ".join(mismatches)
        )

    actual_counts = _site_counts(instance.sites)
    expected_counts = {"H": level.lockers, "L": level.bases, "E": level.swaps}
    if actual_counts != expected_counts:
        raise ValueError(
            f"{instance.directory} facility counts mismatch: "
            f"expected={expected_counts}, actual={actual_counts}"
        )
    _validate_rows_inside_crop(instance)
    _validate_demand_totals(instance)
    _validate_site_activity(instance)


def _validate_rows_inside_crop(instance: _GeneratedLevel) -> None:
    radius = instance.level.half_size_km
    for artifact, rows in (
        ("sites.csv", instance.sites),
        ("od_nodes.csv", instance.nodes),
    ):
        for row in rows:
            x = _finite_float(row.get("x_km"), f"{artifact}.x_km")
            y = _finite_float(row.get("y_km"), f"{artifact}.y_km")
            if abs(x) > radius + 1e-8 or abs(y) > radius + 1e-8:
                raise ValueError(
                    f"{artifact} contains a point outside the requested crop"
                )


def _validate_demand_totals(instance: _GeneratedLevel) -> None:
    if len(instance.periods) != 1:
        raise ValueError("nested Nanshan builder requires exactly one period row")
    order_count = sum(
        _nonnegative_int(row.get("order_count"), "OD order_count")
        for row in instance.demand
    )
    total_rate = sum(
        _nonnegative_float(row.get("lambda_per_hour"), "OD lambda_per_hour")
        for row in instance.demand
    )
    for row in instance.demand:
        exposure = _nonnegative_float(row.get("exposure_hours"), "OD exposure_hours")
        if exposure <= 0:
            raise ValueError("OD exposure_hours must be positive")
        count = _nonnegative_int(row.get("order_count"), "OD order_count")
        rate = _nonnegative_float(row.get("lambda_per_hour"), "OD lambda_per_hour")
        if not math.isclose(rate, count / exposure, abs_tol=5e-10):
            raise ValueError("OD rate is inconsistent with order count and exposure")
    comparisons = {
        "period order count": (
            _nonnegative_int(
                instance.periods[0].get("order_count"), "period order_count"
            ),
            order_count,
        ),
        "metadata demand order count": (
            _nonnegative_int(
                instance.metadata.get("demand_orders_in_area"),
                "metadata demand_orders_in_area",
            ),
            order_count,
        ),
    }
    for label, (actual, expected) in comparisons.items():
        if actual != expected:
            raise ValueError(f"{label} does not match canonical OD demand")
    period_rate = _nonnegative_float(
        instance.periods[0].get("total_rate_per_hour"),
        "period total_rate_per_hour",
    )
    metadata_rate = _nonnegative_float(
        instance.metadata.get("total_demand_per_hour"),
        "metadata total_demand_per_hour",
    )
    if not math.isclose(period_rate, total_rate, abs_tol=1e-5) or not math.isclose(
        metadata_rate,
        total_rate,
        abs_tol=1e-5,
    ):
        raise ValueError("period or metadata rate does not match canonical OD demand")


def _validate_site_activity(instance: _GeneratedLevel) -> None:
    exposure = _nonnegative_float(
        instance.metadata.get("candidate_exposure_hours"),
        "candidate exposure",
    )
    candidate_orders = _nonnegative_int(
        instance.metadata.get("candidate_orders_in_area"),
        "candidate orders",
    )
    if exposure <= 0:
        raise ValueError("candidate exposure must be positive")
    activity = {kind: 0.0 for kind in ("H", "L", "E")}
    for row in instance.sites:
        kind = row.get("type")
        if kind not in activity:
            raise ValueError(f"unknown facility type {kind!r}")
        activity[kind] += _nonnegative_float(row.get("activity_rate"), "activity rate")
    expected = {
        "H": 2.0 * candidate_orders / exposure,
        "L": candidate_orders / exposure,
        "E": candidate_orders / exposure if instance.level.swaps else 0.0,
    }
    for kind, expected_rate in expected.items():
        if not math.isclose(activity[kind], expected_rate, abs_tol=2e-6):
            raise ValueError(
                f"{kind} activity does not account for all retained candidate orders"
            )


def _validate_common_provenance(
    generated: Sequence[_GeneratedLevel],
    config: NanshanScaleBuildConfig,
) -> None:
    full = generated[-1]
    invariant_fields = (
        "raw_file_sha256",
        "projection_center_lon",
        "projection_center_lat",
        "selected_days",
        "candidate_exposure_hours",
        "demand_exposure_hours",
        "date_basis",
        "hour_basis",
    )
    for instance in generated[:-1]:
        changed = [
            field
            for field in invariant_fields
            if instance.metadata.get(field) != full.metadata.get(field)
        ]
        if changed:
            raise ValueError(
                f"{instance.directory} differs from the full parent on: {', '.join(changed)}"
            )
    expected_days = list(sorted(set(config.dates)))
    if full.metadata.get("selected_days") != expected_days:
        raise ValueError("generated selected days do not match the requested dates")


def _audit_raw_orders(
    raw_path: Path,
    config: NanshanScaleBuildConfig,
    center: tuple[float, float],
    selected_days: tuple[str, ...],
) -> dict[float, _RawOrderAudit]:
    audits = {
        level.half_size_km: _RawOrderAudit(candidate_tokens=[], demand_tokens=[])
        for level in config.levels
    }
    selected = set(selected_days)
    for order in NanshanAdapter().iter_orders(str(raw_path)):
        if order.day_id not in selected:
            continue
        coordinates = _project_order(order, center)
        token = _order_token(order)
        for level in config.levels:
            if not _both_endpoints_inside(coordinates, level.half_size_km):
                continue
            audit = audits[level.half_size_km]
            if _in_hour_window(
                order.hour,
                config.candidate_hour_start,
                config.candidate_hour_end,
            ):
                audit.candidate_tokens.append(token)
            if _in_hour_window(
                order.hour,
                config.demand_hour_start,
                config.demand_hour_end,
            ):
                audit.demand_tokens.append(token)
    return audits


def _validate_raw_order_audits(
    generated: Sequence[_GeneratedLevel],
    audits: Mapping[float, _RawOrderAudit],
) -> None:
    previous_candidate: Counter[str] = Counter()
    previous_demand: Counter[str] = Counter()
    for instance in generated:
        audit = audits[instance.level.half_size_km]
        candidate = Counter(audit.candidate_tokens)
        demand = Counter(audit.demand_tokens)
        if not previous_candidate <= candidate or not previous_demand <= demand:
            raise ValueError(
                "raw order multisets are not nested across geographic crops"
            )
        previous_candidate = candidate
        previous_demand = demand
        expected_counts = {
            "candidate_orders_in_area": len(audit.candidate_tokens),
            "demand_orders_in_area": len(audit.demand_tokens),
        }
        for field, expected in expected_counts.items():
            actual = _nonnegative_int(instance.metadata.get(field), f"metadata {field}")
            if actual != expected:
                raise ValueError(
                    f"{instance.directory} {field}={actual} but raw audit found {expected}"
                )


def _site_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    return {
        kind: sum(row.get("type") == kind for row in rows) for kind in ("H", "L", "E")
    }


def _multiset_hash(tokens: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for token in sorted(tokens):
        payload = token.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _canonical_hashes(directory: Path) -> dict[str, str]:
    return {name: _sha256_file(directory / name) for name in CANONICAL_FILES}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _nonnegative_float(value: object, field: str) -> float:
    number = _finite_float(value, field)
    if number < 0:
        raise ValueError(f"{field} must be nonnegative")
    return number


def _nonnegative_int(value: object, field: str) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a nonnegative integer") from exc
    if number < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return number
