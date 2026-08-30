"""Independent semantic validation for schema-5 MLND result packages.

The solver writes convenient aggregates to ``summary.json``.  Those aggregates
are useful for reporting, but they are not sufficient evidence on their own.
This module therefore re-reads the row-level CSV artifacts, reconstructs the
principal accounting identities, and verifies the completion manifest before a
run is admitted to an experiment table.

Run from the repository root with::

    python -m src.utils.result_validator results/my_run

The validator always writes ``validation.json`` and
``validation_checks.csv``.  Its command-line exit status is non-zero whenever
any required check fails.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION


VALIDATION_SCHEMA_VERSION = 1
VALIDATOR_ID = "mlnd-schema5-independent-validator-v1"
SUPPORTED_MODES = ("HD", "MX", "LO")
_VALIDATOR_OUTPUTS = {"validation.json", "validation_checks.csv"}
_COST_COMPONENTS = (
    "facility",
    "capacity",
    "connection",
    "route_operation",
    "fleet",
    "direct_courier_fallback",
    "first_mile_courier",
    "last_mile_courier",
    "locker_handling",
)
_MODE_CSV_FILES = (
    "od_service.csv",
    "od_corridor_reliability.csv",
    "displayed_locker_service.csv",
    "corridor_flows.csv",
    "energy_balance.csv",
)


@dataclass(frozen=True)
class ValidationCheck:
    """One machine-readable validation assertion."""

    check_id: str
    category: str
    status: str
    severity: str
    mode: str
    observed: Any
    expected: Any
    tolerance: str
    details: str


class _Recorder:
    def __init__(self, *, abs_tol: float, rel_tol: float, flow_abs_tol: float):
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.flow_abs_tol = flow_abs_tol
        self.checks: list[ValidationCheck] = []

    def add(
        self,
        condition: bool,
        check_id: str,
        category: str,
        *,
        mode: str = "",
        observed: Any = "",
        expected: Any = "",
        tolerance: str = "",
        details: str = "",
        severity: str = "error",
    ) -> bool:
        self.checks.append(
            ValidationCheck(
                check_id=check_id,
                category=category,
                status="PASS" if condition else ("WARN" if severity == "warning" else "FAIL"),
                severity=severity,
                mode=mode,
                observed=observed,
                expected=expected,
                tolerance=tolerance,
                details=details,
            )
        )
        return condition

    def close(
        self,
        left: float,
        right: float,
        check_id: str,
        category: str,
        *,
        mode: str = "",
        details: str = "",
        flow: bool = False,
    ) -> bool:
        abs_tol = self.flow_abs_tol if flow else self.abs_tol
        condition = math.isclose(left, right, rel_tol=self.rel_tol, abs_tol=abs_tol)
        return self.add(
            condition,
            check_id,
            category,
            mode=mode,
            observed=left,
            expected=right,
            tolerance=f"abs={abs_tol:g}; rel={self.rel_tol:g}",
            details=details,
        )

    @property
    def valid(self) -> bool:
        return not any(check.status == "FAIL" for check in self.checks)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if len(fields) != len(set(fields)):
            raise ValueError(f"duplicate CSV columns: {path}")
        return fields, list(reader)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} is boolean, not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} is not finite: {value!r}")
    return number


def _cell(row: Mapping[str, str], key: str, context: str) -> float:
    if key not in row or row[key].strip() == "":
        raise ValueError(f"{context} is missing numeric column {key!r}")
    return _finite(row[key], f"{context}.{key}")


def _optional_cell(row: Mapping[str, str], key: str, context: str) -> float | None:
    if key not in row or row[key].strip() == "":
        return None
    return _finite(row[key], f"{context}.{key}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _required_columns(
    fields: Sequence[str], required: Iterable[str], context: str
) -> None:
    missing = sorted(set(required) - set(fields))
    if missing:
        raise ValueError(f"{context} is missing columns: {', '.join(missing)}")


def _bool_text(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"{context} is not boolean: {value!r}")


def _run_signature_payload(run_config: Mapping[str, Any]) -> dict[str, Any]:
    instance = _mapping(run_config.get("instance"), "run_config.instance")
    return {
        "schema_version": run_config.get("schema_version"),
        "formulation_id": run_config.get("formulation_id"),
        "formulation_source": run_config.get("formulation_source"),
        "implementation": run_config.get("implementation"),
        "instance_dir": instance.get("directory"),
        "canonical_file_sha256": instance.get("canonical_file_sha256"),
        "demand_scale": run_config.get("demand_scale"),
        "modes": run_config.get("modes"),
        "cross_mode_start_policy": run_config.get("cross_mode_start_policy"),
        "warm_start_from": run_config.get("warm_start_from"),
        "file_warm_start_policy": run_config.get("file_warm_start_policy"),
        "screening_is_exact": run_config.get("screening_is_exact"),
        "model": run_config.get("model"),
        "model_derived": run_config.get("model_derived"),
        "solver": run_config.get("solver"),
    }


def _validate_contract_and_signature(
    run_config: Mapping[str, Any], recorder: _Recorder
) -> list[str]:
    recorder.add(
        type(run_config.get("schema_version")) is int
        and run_config.get("schema_version") == RESULT_SCHEMA_VERSION,
        "run_schema_version",
        "contract",
        observed=run_config.get("schema_version"),
        expected=RESULT_SCHEMA_VERSION,
    )
    recorder.add(
        run_config.get("formulation_id") == FORMULATION_ID,
        "run_formulation_id",
        "contract",
        observed=run_config.get("formulation_id"),
        expected=FORMULATION_ID,
    )
    signature = run_config.get("run_signature_sha256")
    recorder.add(
        _is_sha256(signature),
        "run_signature_format",
        "contract",
        observed=signature,
        expected="64 lowercase hexadecimal characters",
    )
    if _is_sha256(signature):
        reconstructed = _payload_sha256(_run_signature_payload(run_config))
        recorder.add(
            signature == reconstructed,
            "run_signature_recomputed",
            "contract",
            observed=signature,
            expected=reconstructed,
            details="Recomputed from the exact configuration fields sealed by main.py.",
        )

    modes_raw = run_config.get("modes")
    modes = list(modes_raw) if isinstance(modes_raw, list) else []
    recorder.add(
        bool(modes)
        and all(isinstance(mode, str) and mode in SUPPORTED_MODES for mode in modes)
        and len(modes) == len(set(modes)),
        "run_modes",
        "contract",
        observed=modes_raw,
        expected="a non-empty unique subset of HD, MX, LO",
    )

    formulation_source = run_config.get("formulation_source")
    implementation = run_config.get("implementation")
    source_hash = formulation_source.get("sha256") if isinstance(formulation_source, Mapping) else None
    implementation_hash = implementation.get("sha256") if isinstance(implementation, Mapping) else None
    recorder.add(
        _is_sha256(source_hash),
        "formulation_source_hash_format",
        "provenance",
        observed=source_hash,
        expected="sha256",
    )
    recorder.add(
        _is_sha256(implementation_hash),
        "implementation_hash_format",
        "provenance",
        observed=implementation_hash,
        expected="sha256",
    )
    return modes


def _validate_instance_hashes(
    run_config: Mapping[str, Any], recorder: _Recorder
) -> None:
    try:
        instance = _mapping(run_config.get("instance"), "run_config.instance")
        hashes = _mapping(
            instance.get("canonical_file_sha256"),
            "run_config.instance.canonical_file_sha256",
        )
        directory_value = instance.get("directory")
        if not isinstance(directory_value, str) or not directory_value:
            raise ValueError("run_config.instance.directory must be a non-empty path")
        formats_ok = bool(hashes) and all(
            isinstance(name, str) and _is_sha256(digest)
            for name, digest in hashes.items()
        )
        recorder.add(
            formats_ok,
            "canonical_hash_format",
            "provenance",
            observed=len(hashes),
            expected="one valid sha256 per canonical instance file",
        )
        directory = Path(directory_value)
        if not directory.is_dir():
            recorder.add(
                False,
                "canonical_files_available",
                "provenance",
                observed=str(directory),
                expected="accessible instance directory",
                severity="warning",
                details="The sealed hashes remain auditable, but local bytes were unavailable.",
            )
            return
        mismatches: list[str] = []
        for name, expected_hash in hashes.items():
            path = directory / name
            if not path.is_file() or _sha256_file(path) != expected_hash:
                mismatches.append(name)
        recorder.add(
            not mismatches,
            "canonical_file_hashes",
            "provenance",
            observed=mismatches,
            expected="all recorded canonical hashes match local bytes",
        )
    except (OSError, ValueError) as exc:
        recorder.add(
            False,
            "canonical_file_hashes",
            "provenance",
            observed=str(exc),
            expected="valid canonical instance provenance",
        )


def _validate_completion_manifest(
    root: Path,
    completion: Mapping[str, Any],
    modes: Sequence[str],
    recorder: _Recorder,
) -> None:
    recorder.add(
        completion.get("schema_version") == 1 and completion.get("completed") is True,
        "completion_marker",
        "artifacts",
        observed={
            "schema_version": completion.get("schema_version"),
            "completed": completion.get("completed"),
        },
        expected={"schema_version": 1, "completed": True},
    )
    recorder.add(
        completion.get("modes") == list(modes),
        "completion_modes",
        "artifacts",
        observed=completion.get("modes"),
        expected=list(modes),
    )
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, list):
        recorder.add(
            False,
            "artifact_manifest_shape",
            "artifacts",
            observed=type(artifacts).__name__,
            expected="list",
        )
        return
    recorder.add(
        type(completion.get("artifact_count")) is int
        and completion.get("artifact_count") == len(artifacts),
        "artifact_count",
        "artifacts",
        observed=completion.get("artifact_count"),
        expected=len(artifacts),
    )

    seen: set[str] = set()
    malformed: list[str] = []
    mismatched: list[str] = []
    for index, record in enumerate(artifacts):
        if not isinstance(record, Mapping):
            malformed.append(f"index:{index}")
            continue
        relative = record.get("path")
        digest = record.get("sha256")
        size = record.get("size_bytes")
        if not isinstance(relative, str):
            malformed.append(f"index:{index}")
            continue
        pure = PurePosixPath(relative)
        safe = not pure.is_absolute() and ".." not in pure.parts and relative not in seen
        if not safe or not _is_sha256(digest) or type(size) is not int or size < 0:
            malformed.append(relative)
            continue
        seen.add(relative)
        path = root / relative
        if not path.is_file() or path.stat().st_size != size or _sha256_file(path) != digest:
            mismatched.append(relative)
    recorder.add(
        not malformed,
        "artifact_manifest_records",
        "artifacts",
        observed=malformed,
        expected="unique safe paths with size and sha256",
    )
    recorder.add(
        not mismatched,
        "artifact_hashes",
        "artifacts",
        observed=mismatched,
        expected="all listed artifact sizes and hashes match",
    )
    required = {"run_config.json", "mode_comparison.csv"}
    for mode in modes:
        required.add(f"{mode}/summary.json")
        required.update(f"{mode}/{name}" for name in _MODE_CSV_FILES)
    recorder.add(
        required <= seen,
        "core_artifacts_sealed",
        "artifacts",
        observed=sorted(required - seen),
        expected="no unsealed core result artifacts",
    )


def _summary_number(summary: Mapping[str, Any], key: str, context: str) -> float:
    if key not in summary:
        raise ValueError(f"{context} is missing {key!r}")
    return _finite(summary[key], f"{context}.{key}")


def _validate_summary_contract(
    summary: Mapping[str, Any],
    run_config: Mapping[str, Any],
    mode: str,
    recorder: _Recorder,
) -> None:
    recorder.add(
        type(summary.get("schema_version")) is int
        and summary.get("schema_version") == RESULT_SCHEMA_VERSION,
        "summary_schema_version",
        "contract",
        mode=mode,
        observed=summary.get("schema_version"),
        expected=RESULT_SCHEMA_VERSION,
    )
    recorder.add(
        summary.get("formulation_id") == FORMULATION_ID,
        "summary_formulation_id",
        "contract",
        mode=mode,
        observed=summary.get("formulation_id"),
        expected=FORMULATION_ID,
    )
    recorder.add(
        summary.get("mode") == mode,
        "summary_mode",
        "contract",
        mode=mode,
        observed=summary.get("mode"),
        expected=mode,
    )
    recorder.add(
        summary.get("run_signature_sha256")
        == run_config.get("run_signature_sha256"),
        "summary_run_signature",
        "contract",
        mode=mode,
        observed=summary.get("run_signature_sha256"),
        expected=run_config.get("run_signature_sha256"),
    )


def _validate_objective(
    summary: Mapping[str, Any], mode: str, recorder: _Recorder
) -> None:
    objective = _summary_number(summary, "objective", f"{mode} summary")
    costs = _mapping(summary.get("costs_cny_per_hour"), f"{mode}.costs")
    missing = [name for name in _COST_COMPONENTS if name not in costs]
    recorder.add(
        not missing,
        "cost_components_present",
        "objective",
        mode=mode,
        observed=missing,
        expected=list(_COST_COMPONENTS),
    )
    if missing:
        return
    component_total = sum(
        _finite(costs[name], f"{mode}.costs.{name}") for name in _COST_COMPONENTS
    )
    recorder.close(
        objective,
        component_total,
        "objective_from_cost_components",
        "objective",
        mode=mode,
    )
    optimized_total = _finite(
        costs.get("optimized_total"), f"{mode}.costs.optimized_total"
    )
    recorder.close(
        optimized_total,
        component_total,
        "optimized_total_from_cost_components",
        "objective",
        mode=mode,
    )
    reported_error = _finite(
        costs.get("objective_reconciliation_error"),
        f"{mode}.costs.objective_reconciliation_error",
    )
    recorder.close(
        reported_error,
        objective - component_total,
        "objective_error_field",
        "objective",
        mode=mode,
    )

    paper_terms = _mapping(
        summary.get("paper_objective_terms_cny_per_hour"),
        f"{mode}.paper_objective_terms",
    )
    paper_total = sum(
        _finite(paper_terms.get(name), f"{mode}.paper.{name}")
        for name in (
            "courier_baseline",
            "locker_pickup_adjustment",
            "realized_drone_adjustment",
            "network_design_and_operation",
        )
    )
    reconstructed = _finite(
        paper_terms.get("reconstructed_total"), f"{mode}.paper.reconstructed_total"
    )
    recorder.close(
        reconstructed,
        paper_total,
        "paper_objective_term_sum",
        "objective",
        mode=mode,
    )
    recorder.close(
        objective,
        reconstructed,
        "paper_objective_reconciliation",
        "objective",
        mode=mode,
    )


def _validate_gap(
    summary: Mapping[str, Any],
    completion: Mapping[str, Any],
    mode: str,
    recorder: _Recorder,
) -> None:
    objective = _summary_number(summary, "objective", f"{mode} summary")
    bound = _summary_number(summary, "objective_bound", f"{mode} summary")
    gap = _summary_number(summary, "mip_gap", f"{mode} summary")
    if abs(objective) <= recorder.abs_tol:
        expected_gap = 0.0 if abs(objective - bound) <= recorder.abs_tol else math.inf
    else:
        expected_gap = abs(objective - bound) / abs(objective)
    recorder.add(
        math.isfinite(expected_gap),
        "gap_defined",
        "solver",
        mode=mode,
        observed={"objective": objective, "bound": bound},
        expected="finite minimization relative gap",
    )
    if math.isfinite(expected_gap):
        recorder.close(
            gap,
            expected_gap,
            "gap_mathematical_identity",
            "solver",
            mode=mode,
        )
    recorder.add(
        bound <= objective + max(recorder.abs_tol, recorder.rel_tol * abs(objective)),
        "minimization_bound_direction",
        "solver",
        mode=mode,
        observed=bound,
        expected=f"<= incumbent {objective}",
    )

    solver = _mapping(summary.get("solver"), f"{mode}.solver")
    for field, top_key in (
        ("objective_cost_per_hour", "objective"),
        ("objective_bound", "objective_bound"),
        ("mip_gap", "mip_gap"),
        ("runtime_seconds", "runtime_seconds"),
    ):
        recorder.close(
            _finite(solver.get(field), f"{mode}.solver.{field}"),
            _summary_number(summary, top_key, f"{mode} summary"),
            f"solver_{field}_matches_summary",
            "solver",
            mode=mode,
        )
    requested = _summary_number(summary, "requested_mip_gap", f"{mode} summary")
    recorder.close(
        _finite(solver.get("requested_mip_gap"), f"{mode}.solver.requested_mip_gap"),
        requested,
        "solver_requested_gap_matches_summary",
        "solver",
        mode=mode,
    )
    if summary.get("status_name") == "OPTIMAL":
        recorder.add(
            gap <= requested + max(recorder.abs_tol, recorder.rel_tol * requested),
            "optimal_status_meets_requested_gap",
            "solver",
            mode=mode,
            observed=gap,
            expected=f"<= {requested}",
        )

    mode_status = completion.get("mode_status")
    status_record = mode_status.get(mode) if isinstance(mode_status, Mapping) else None
    if not isinstance(status_record, Mapping):
        recorder.add(
            False,
            "completion_mode_status",
            "solver",
            mode=mode,
            observed=status_record,
            expected="mode status mapping",
        )
        return
    for field, summary_key in (
        ("objective", "objective"),
        ("objective_bound", "objective_bound"),
        ("mip_gap", "mip_gap"),
        ("runtime_seconds", "runtime_seconds"),
    ):
        recorder.close(
            _finite(status_record.get(field), f"completion.{mode}.{field}"),
            _summary_number(summary, summary_key, f"{mode} summary"),
            f"completion_{field}_matches_summary",
            "solver",
            mode=mode,
        )
    recorder.add(
        status_record.get("status") == summary.get("status_name"),
        "completion_status_name_matches_summary",
        "solver",
        mode=mode,
        observed=status_record.get("status"),
        expected=summary.get("status_name"),
    )


def _validate_screening(
    run_config: Mapping[str, Any],
    summary: Mapping[str, Any],
    completion: Mapping[str, Any],
    mode: str,
    recorder: _Recorder,
) -> bool:
    screening = run_config.get("screening_is_exact")
    declared = screening.get(mode) if isinstance(screening, Mapping) else None
    recorder.add(
        isinstance(declared, bool),
        "screening_declaration_type",
        "scope",
        mode=mode,
        observed=declared,
        expected="boolean",
    )
    instance = summary.get("instance")
    instance_value = instance.get("screening_is_exact") if isinstance(instance, Mapping) else None
    completion_status = completion.get("mode_status")
    completion_record = (
        completion_status.get(mode) if isinstance(completion_status, Mapping) else None
    )
    completion_value = (
        completion_record.get("screening_is_exact")
        if isinstance(completion_record, Mapping)
        else None
    )
    values = [summary.get("screening_is_exact"), instance_value, completion_value]
    recorder.add(
        isinstance(declared, bool) and all(value is declared for value in values),
        "screening_declaration_consistency",
        "scope",
        mode=mode,
        observed=values,
        expected=declared,
        details=(
            "False means the solver gap certifies only the screened model, not the "
            "unscreened formulation."
        ),
    )
    return bool(declared)


def _validate_od_csv(
    path: Path,
    summary: Mapping[str, Any],
    mode: str,
    recorder: _Recorder,
) -> dict[tuple[str, str], dict[str, float]]:
    fields, rows = _read_csv(path)
    required = {
        "origin",
        "destination",
        "demand_per_hour",
        "drone_offered_per_hour",
        "drone_realized_per_hour",
        "offered_not_realized_per_hour",
        "not_offered_per_hour",
        "locker_pickup_per_hour",
        "assisted_home_per_hour",
        "direct_courier_fallback_per_hour",
        "service_conservation_error",
        "drone_service_rate",
        "direct_rider_unit_cost",
        "all_rider_baseline_cost_per_hour",
        "direct_courier_fallback_cost_per_hour",
        "first_mile_courier_cost_per_hour",
        "last_mile_courier_cost_per_hour",
        "locker_handling_cost_per_hour",
    }
    _required_columns(fields, required, str(path))
    parsed: dict[tuple[str, str], dict[str, float]] = {}
    max_service_error = 0.0
    max_offered_split_error = 0.0
    max_realized_split_error = 0.0
    max_fallback_error = 0.0
    max_reported_error_error = 0.0
    max_rate_error = 0.0
    minimum_flow = 0.0
    totals: defaultdict[str, float] = defaultdict(float)
    cost_columns = {
        "direct_courier_fallback": "direct_courier_fallback_cost_per_hour",
        "first_mile_courier": "first_mile_courier_cost_per_hour",
        "last_mile_courier": "last_mile_courier_cost_per_hour",
        "locker_handling": "locker_handling_cost_per_hour",
    }
    for index, row in enumerate(rows, start=2):
        context = f"{path.name}:{index}"
        key = (row["origin"], row["destination"])
        if key in parsed:
            raise ValueError(f"duplicate OD row {key!r} in {path}")
        values = {
            name: _cell(row, name, context)
            for name in (
                "demand_per_hour",
                "drone_offered_per_hour",
                "drone_realized_per_hour",
                "offered_not_realized_per_hour",
                "not_offered_per_hour",
                "locker_pickup_per_hour",
                "assisted_home_per_hour",
                "direct_courier_fallback_per_hour",
                "service_conservation_error",
                "drone_service_rate",
                "direct_rider_unit_cost",
                "all_rider_baseline_cost_per_hour",
                *cost_columns.values(),
            )
        }
        parsed[key] = values
        demand = values["demand_per_hour"]
        offered = values["drone_offered_per_hour"]
        realized = values["drone_realized_per_hour"]
        pickup = values["locker_pickup_per_hour"]
        home = values["assisted_home_per_hour"]
        fallback = values["direct_courier_fallback_per_hour"]
        service_error = pickup + home + fallback - demand
        max_service_error = max(max_service_error, abs(service_error))
        max_offered_split_error = max(
            max_offered_split_error,
            abs(offered + values["not_offered_per_hour"] - demand),
        )
        max_realized_split_error = max(
            max_realized_split_error,
            abs(realized + values["offered_not_realized_per_hour"] - offered),
            abs(pickup + home - realized),
        )
        max_fallback_error = max(max_fallback_error, abs(fallback - (demand - realized)))
        max_reported_error_error = max(
            max_reported_error_error,
            abs(values["service_conservation_error"] - service_error),
        )
        expected_rate = realized / demand if demand > recorder.abs_tol else 0.0
        max_rate_error = max(max_rate_error, abs(values["drone_service_rate"] - expected_rate))
        minimum_flow = min(
            minimum_flow,
            demand,
            offered,
            realized,
            pickup,
            home,
            fallback,
            values["offered_not_realized_per_hour"],
            values["not_offered_per_hour"],
        )
        unit_cost = values["direct_rider_unit_cost"]
        max_reported_error_error = max(
            max_reported_error_error,
            abs(values["all_rider_baseline_cost_per_hour"] - unit_cost * demand),
            abs(values["direct_courier_fallback_cost_per_hour"] - unit_cost * fallback),
        )
        totals["demand"] += demand
        totals["offered"] += offered
        totals["realized"] += realized
        totals["pickup"] += pickup
        totals["home"] += home
        totals["fallback"] += fallback
        totals["baseline"] += values["all_rider_baseline_cost_per_hour"]
        for cost_name, column in cost_columns.items():
            totals[f"cost:{cost_name}"] += values[column]

    instance = _mapping(summary.get("instance"), f"{mode}.instance")
    recorder.add(
        len(rows) == int(_finite(instance.get("markets"), f"{mode}.instance.markets")),
        "od_row_count",
        "od_service",
        mode=mode,
        observed=len(rows),
        expected=instance.get("markets"),
    )
    for check_id, value in (
        ("od_service_conservation", max_service_error),
        ("od_offered_split", max_offered_split_error),
        ("od_realized_split", max_realized_split_error),
        ("od_fallback_identity", max_fallback_error),
        ("od_reported_errors", max_reported_error_error),
        ("od_service_rate", max_rate_error),
    ):
        recorder.close(
            value,
            0.0,
            check_id,
            "od_service",
            mode=mode,
            flow=True,
        )
    recorder.add(
        minimum_flow >= -recorder.flow_abs_tol,
        "od_nonnegative_flows",
        "od_service",
        mode=mode,
        observed=minimum_flow,
        expected=f">= {-recorder.flow_abs_tol:g}",
    )

    funnel = _mapping(
        summary.get("demand_funnel_orders_per_hour"), f"{mode}.demand_funnel"
    )
    for field, total_key in (
        ("total_demand", "demand"),
        ("total_drone_offered", "offered"),
        ("total_drone_realized", "realized"),
        ("locker_pickup", "pickup"),
        ("assisted_home", "home"),
        ("direct_courier_fallback", "fallback"),
    ):
        recorder.close(
            _finite(funnel.get(field), f"{mode}.demand_funnel.{field}"),
            totals[total_key],
            f"funnel_{field}_from_od_csv",
            "od_service",
            mode=mode,
            flow=True,
        )
    baseline = _mapping(summary.get("all_rider_baseline"), f"{mode}.baseline")
    recorder.close(
        _finite(baseline.get("cost_per_hour"), f"{mode}.baseline.cost_per_hour"),
        totals["baseline"],
        "all_rider_baseline_from_od_csv",
        "objective",
        mode=mode,
        flow=True,
    )
    costs = _mapping(summary.get("costs_cny_per_hour"), f"{mode}.costs")
    for cost_name in cost_columns:
        recorder.close(
            _finite(costs.get(cost_name), f"{mode}.costs.{cost_name}"),
            totals[f"cost:{cost_name}"],
            f"cost_{cost_name}_from_od_csv",
            "objective",
            mode=mode,
            flow=True,
        )
    reconciliation = _mapping(summary.get("reconciliation"), f"{mode}.reconciliation")
    recorder.close(
        _finite(
            reconciliation.get("maximum_od_service_conservation_error"),
            f"{mode}.reconciliation.maximum_od_service_conservation_error",
        ),
        max_service_error,
        "summary_max_od_error_from_csv",
        "od_service",
        mode=mode,
        flow=True,
    )
    return parsed


def _validate_chi_csv(
    path: Path,
    mode: str,
    recorder: _Recorder,
) -> tuple[
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str, str], float],
]:
    fields, rows = _read_csv(path)
    required = {
        "origin",
        "destination",
        "origin_locker",
        "destination_locker",
        "rho_level",
        "rho",
        "chi_offered_per_hour",
        "realized_per_hour",
        "displayed_for_destination",
        "conditional_locker_share_eta",
        "locker_pickup_per_hour",
        "assisted_home_per_hour",
    }
    _required_columns(fields, required, str(path))
    od: defaultdict[tuple[str, str], defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    corridor: defaultdict[tuple[str, str], defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    successful: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    seen: set[tuple[str, str, str, str, int]] = set()
    max_realization_error = 0.0
    max_service_error = 0.0
    max_mode_error = 0.0
    minimum_flow = 0.0
    for index, row in enumerate(rows, start=2):
        context = f"{path.name}:{index}"
        rho_level_float = _cell(row, "rho_level", context)
        rho_level = int(round(rho_level_float))
        if abs(rho_level_float - rho_level) > recorder.abs_tol:
            raise ValueError(f"{context}.rho_level is not integral")
        key = (
            row["origin"],
            row["destination"],
            row["origin_locker"],
            row["destination_locker"],
            rho_level,
        )
        if key in seen:
            raise ValueError(f"duplicate chi row {key!r} in {path}")
        seen.add(key)
        rho = _cell(row, "rho", context)
        offered = _cell(row, "chi_offered_per_hour", context)
        realized = _cell(row, "realized_per_hour", context)
        displayed = _bool_text(row["displayed_for_destination"], f"{context}.displayed")
        eta = _optional_cell(row, "conditional_locker_share_eta", context)
        pickup = _cell(row, "locker_pickup_per_hour", context)
        home = _cell(row, "assisted_home_per_hour", context)
        max_realization_error = max(max_realization_error, abs(realized - rho * offered))
        max_service_error = max(max_service_error, abs(pickup + home - realized))
        if mode == "MX":
            if eta is None:
                raise ValueError(f"{context} requires eta in MX mode")
            expected_pickup = eta * realized if displayed else 0.0
        elif mode == "LO":
            expected_pickup = realized
        else:
            expected_pickup = 0.0
        max_mode_error = max(max_mode_error, abs(pickup - expected_pickup))
        minimum_flow = min(minimum_flow, offered, realized, pickup, home, rho)
        od_key = (row["origin"], row["destination"])
        corridor_key = (row["origin_locker"], row["destination_locker"])
        locker_key = (row["origin"], row["destination"], row["destination_locker"])
        for bucket in (od[od_key], corridor[corridor_key]):
            bucket["offered"] += offered
            bucket["realized"] += realized
            bucket["pickup"] += pickup
            bucket["home"] += home
        successful[locker_key] += realized
    for check_id, value in (
        ("chi_realized_equals_rho_times_offer", max_realization_error),
        ("chi_service_split", max_service_error),
        ("chi_mode_service_rule", max_mode_error),
    ):
        recorder.close(
            value,
            0.0,
            check_id,
            "chi_aggregation",
            mode=mode,
            flow=True,
        )
    recorder.add(
        minimum_flow >= -recorder.flow_abs_tol,
        "chi_nonnegative",
        "chi_aggregation",
        mode=mode,
        observed=minimum_flow,
        expected=f">= {-recorder.flow_abs_tol:g}",
    )
    return (
        {key: dict(value) for key, value in od.items()},
        {key: dict(value) for key, value in corridor.items()},
        dict(successful),
    )


def _compare_od_and_chi(
    od_rows: Mapping[tuple[str, str], Mapping[str, float]],
    chi_od: Mapping[tuple[str, str], Mapping[str, float]],
    mode: str,
    recorder: _Recorder,
) -> None:
    max_errors: defaultdict[str, float] = defaultdict(float)
    for key, od in od_rows.items():
        chi = chi_od.get(key, {})
        for od_field, chi_field in (
            ("drone_offered_per_hour", "offered"),
            ("drone_realized_per_hour", "realized"),
            ("locker_pickup_per_hour", "pickup"),
            ("assisted_home_per_hour", "home"),
        ):
            max_errors[od_field] = max(
                max_errors[od_field], abs(od[od_field] - float(chi.get(chi_field, 0.0)))
            )
    extra = sorted(set(chi_od) - set(od_rows))
    recorder.add(
        not extra,
        "chi_od_keys_exist",
        "chi_aggregation",
        mode=mode,
        observed=extra[:10],
        expected="all chi OD keys exist in od_service.csv",
    )
    for field, error in max_errors.items():
        recorder.close(
            error,
            0.0,
            f"chi_to_od_{field}",
            "chi_aggregation",
            mode=mode,
            flow=True,
        )


def _validate_corridor_csv(
    path: Path,
    chi_corridor: Mapping[tuple[str, str], Mapping[str, float]],
    summary: Mapping[str, Any],
    mode: str,
    recorder: _Recorder,
) -> None:
    fields, rows = _read_csv(path)
    required = {
        "origin_locker",
        "destination_locker",
        "od_offered_per_hour",
        "source_allocated_offered_per_hour",
        "offered_reconciliation_error",
        "realized_from_chi_per_hour",
        "loaded_psi_per_hour",
        "realized_reconciliation_error",
    }
    _required_columns(fields, required, str(path))
    seen: set[tuple[str, str]] = set()
    max_chi_offered_error = 0.0
    max_chi_realized_error = 0.0
    max_offered_identity_error = 0.0
    max_realized_identity_error = 0.0
    max_offered_error = 0.0
    max_realized_error = 0.0
    for index, row in enumerate(rows, start=2):
        context = f"{path.name}:{index}"
        key = (row["origin_locker"], row["destination_locker"])
        if key in seen:
            raise ValueError(f"duplicate corridor row {key!r} in {path}")
        seen.add(key)
        offered = _cell(row, "od_offered_per_hour", context)
        source = _cell(row, "source_allocated_offered_per_hour", context)
        offered_error = _cell(row, "offered_reconciliation_error", context)
        realized = _cell(row, "realized_from_chi_per_hour", context)
        loaded = _cell(row, "loaded_psi_per_hour", context)
        realized_error = _cell(row, "realized_reconciliation_error", context)
        chi = chi_corridor.get(key, {})
        max_chi_offered_error = max(
            max_chi_offered_error, abs(offered - float(chi.get("offered", 0.0)))
        )
        max_chi_realized_error = max(
            max_chi_realized_error, abs(realized - float(chi.get("realized", 0.0)))
        )
        max_offered_identity_error = max(
            max_offered_identity_error, abs(offered_error - (offered - source))
        )
        max_realized_identity_error = max(
            max_realized_identity_error, abs(realized_error - (realized - loaded))
        )
        max_offered_error = max(max_offered_error, abs(offered_error))
        max_realized_error = max(max_realized_error, abs(realized_error))
    missing = sorted(set(chi_corridor) - seen)
    recorder.add(
        not missing,
        "chi_corridors_exist",
        "chi_aggregation",
        mode=mode,
        observed=missing[:10],
        expected="all positive chi corridors exist in corridor_flows.csv",
    )
    for check_id, value in (
        ("corridor_offered_from_chi", max_chi_offered_error),
        ("corridor_realized_from_chi", max_chi_realized_error),
        ("corridor_offered_error_field", max_offered_identity_error),
        ("corridor_realized_error_field", max_realized_identity_error),
        ("corridor_offered_conservation", max_offered_error),
        ("corridor_realized_conservation", max_realized_error),
    ):
        recorder.close(
            value,
            0.0,
            check_id,
            "chi_aggregation",
            mode=mode,
            flow=True,
        )
    reconciliation = _mapping(summary.get("reconciliation"), f"{mode}.reconciliation")
    recorder.close(
        _finite(
            reconciliation.get("maximum_corridor_offered_flow_error"),
            f"{mode}.reconciliation.maximum_corridor_offered_flow_error",
        ),
        max_offered_error,
        "summary_max_corridor_offered_error",
        "chi_aggregation",
        mode=mode,
        flow=True,
    )
    recorder.close(
        _finite(
            reconciliation.get("maximum_corridor_realized_flow_error"),
            f"{mode}.reconciliation.maximum_corridor_realized_flow_error",
        ),
        max_realized_error,
        "summary_max_corridor_realized_error",
        "chi_aggregation",
        mode=mode,
        flow=True,
    )


def _validate_mx_display_csv(
    path: Path,
    successful: Mapping[tuple[str, str, str], float],
    od_rows: Mapping[tuple[str, str], Mapping[str, float]],
    summary: Mapping[str, Any],
    mode: str,
    recorder: _Recorder,
) -> None:
    fields, rows = _read_csv(path)
    required = {
        "origin",
        "destination",
        "locker",
        "alpha",
        "conditional_locker_share_eta",
        "successful_to_locker_per_hour",
        "epsilon_per_hour",
        "alpha_times_successful_per_hour",
        "epsilon_product_error",
        "locker_pickup_per_hour",
    }
    _required_columns(fields, required, str(path))
    if mode != "MX":
        return
    seen: set[tuple[str, str, str]] = set()
    pickup_by_od: defaultdict[tuple[str, str], float] = defaultdict(float)
    max_success_error = 0.0
    max_epsilon_error = 0.0
    max_qualified_error = 0.0
    max_pickup_error = 0.0
    max_reported_error = 0.0
    max_alpha_integrality = 0.0
    min_probability = 0.0
    max_probability = 1.0
    for index, row in enumerate(rows, start=2):
        context = f"{path.name}:{index}"
        key = (row["origin"], row["destination"], row["locker"])
        if key in seen:
            raise ValueError(f"duplicate displayed-locker row {key!r} in {path}")
        seen.add(key)
        alpha = _cell(row, "alpha", context)
        eta = _cell(row, "conditional_locker_share_eta", context)
        success = _cell(row, "successful_to_locker_per_hour", context)
        epsilon = _cell(row, "epsilon_per_hour", context)
        qualified = _cell(row, "alpha_times_successful_per_hour", context)
        product_error = _cell(row, "epsilon_product_error", context)
        pickup = _cell(row, "locker_pickup_per_hour", context)
        expected_success = float(successful.get(key, 0.0))
        expected_product = alpha * success
        max_success_error = max(max_success_error, abs(success - expected_success))
        max_epsilon_error = max(max_epsilon_error, abs(epsilon - expected_product))
        max_qualified_error = max(max_qualified_error, abs(qualified - expected_product))
        max_pickup_error = max(max_pickup_error, abs(pickup - eta * epsilon))
        max_reported_error = max(max_reported_error, abs(product_error - (epsilon - expected_product)))
        max_alpha_integrality = max(max_alpha_integrality, min(abs(alpha), abs(alpha - 1.0)))
        min_probability = min(min_probability, alpha, eta)
        max_probability = max(max_probability, alpha, eta)
        pickup_by_od[(row["origin"], row["destination"])] += pickup
    for check_id, value in (
        ("mx_successful_flow_from_chi", max_success_error),
        ("mx_epsilon_equals_alpha_times_success", max_epsilon_error),
        ("mx_reported_alpha_times_success", max_qualified_error),
        ("mx_pickup_equals_eta_times_epsilon", max_pickup_error),
        ("mx_epsilon_error_field", max_reported_error),
        ("mx_alpha_binary", max_alpha_integrality),
    ):
        recorder.close(
            value,
            0.0,
            check_id,
            "mx_choice",
            mode=mode,
            flow=True,
        )
    recorder.add(
        min_probability >= -recorder.flow_abs_tol
        and max_probability <= 1.0 + recorder.flow_abs_tol,
        "mx_alpha_eta_probability_bounds",
        "mx_choice",
        mode=mode,
        observed=[min_probability, max_probability],
        expected="within [0, 1]",
    )
    max_od_pickup_error = max(
        (
            abs(row["locker_pickup_per_hour"] - pickup_by_od.get(key, 0.0))
            for key, row in od_rows.items()
        ),
        default=0.0,
    )
    recorder.close(
        max_od_pickup_error,
        0.0,
        "mx_pickup_aggregates_to_od",
        "mx_choice",
        mode=mode,
        flow=True,
    )
    reconciliation = _mapping(summary.get("reconciliation"), f"{mode}.reconciliation")
    recorder.close(
        _finite(
            reconciliation.get("maximum_epsilon_product_error"),
            f"{mode}.reconciliation.maximum_epsilon_product_error",
        ),
        max_epsilon_error,
        "summary_max_epsilon_error",
        "mx_choice",
        mode=mode,
        flow=True,
    )


def _validate_energy_csv(
    path: Path,
    summary: Mapping[str, Any],
    mode: str,
    recorder: _Recorder,
) -> float:
    fields, rows = _read_csv(path)
    required = {
        "locker_handling_population_drones",
        "route_transit_and_handling_population_drones",
        "energy_use_equivalent_drones",
        "base_full_battery_departures_per_hour",
        "swap_events_per_hour",
        "battery_supply_events_per_hour",
        "usable_battery_time_hours",
        "energy_supply_equivalent_drones",
        "energy_slack_equivalent_drones",
        "energy_violation_equivalent_drones",
    }
    _required_columns(fields, required, str(path))
    recorder.add(
        bool(rows),
        "energy_rows_present",
        "energy",
        mode=mode,
        observed=len(rows),
        expected=">= 1",
    )
    max_use_error = 0.0
    max_event_error = 0.0
    max_supply_error = 0.0
    max_slack_error = 0.0
    max_violation_error = 0.0
    max_violation = 0.0
    total_use = 0.0
    total_supply = 0.0
    total_departures = 0.0
    total_swaps = 0.0
    total_events = 0.0
    for index, row in enumerate(rows, start=2):
        context = f"{path.name}:{index}"
        handling = _cell(row, "locker_handling_population_drones", context)
        route = _cell(row, "route_transit_and_handling_population_drones", context)
        use = _cell(row, "energy_use_equivalent_drones", context)
        departures = _cell(row, "base_full_battery_departures_per_hour", context)
        swaps = _cell(row, "swap_events_per_hour", context)
        events = _cell(row, "battery_supply_events_per_hour", context)
        endurance = _cell(row, "usable_battery_time_hours", context)
        supply = _cell(row, "energy_supply_equivalent_drones", context)
        slack = _cell(row, "energy_slack_equivalent_drones", context)
        violation = _cell(row, "energy_violation_equivalent_drones", context)
        expected_use = handling + route
        expected_events = departures + swaps
        expected_supply = endurance * events
        expected_slack = expected_supply - expected_use
        expected_violation = max(0.0, expected_use - expected_supply)
        max_use_error = max(max_use_error, abs(use - expected_use))
        max_event_error = max(max_event_error, abs(events - expected_events))
        max_supply_error = max(max_supply_error, abs(supply - expected_supply))
        max_slack_error = max(max_slack_error, abs(slack - expected_slack))
        max_violation_error = max(max_violation_error, abs(violation - expected_violation))
        max_violation = max(max_violation, expected_violation)
        total_use += use
        total_supply += supply
        total_departures += departures
        total_swaps += swaps
        total_events += events
    for check_id, value in (
        ("energy_use_identity", max_use_error),
        ("energy_event_identity", max_event_error),
        ("energy_supply_identity", max_supply_error),
        ("energy_slack_identity", max_slack_error),
        ("energy_violation_identity", max_violation_error),
    ):
        recorder.close(
            value,
            0.0,
            check_id,
            "energy",
            mode=mode,
            flow=True,
        )
    recorder.add(
        max_violation <= recorder.flow_abs_tol,
        "energy_maximum_violation_feasible",
        "energy",
        mode=mode,
        observed=max_violation,
        expected=f"<= {recorder.flow_abs_tol:g}",
    )
    energy = _mapping(summary.get("energy_balance"), f"{mode}.energy_balance")
    for field, expected in (
        ("total_energy_use_equivalent_drones", total_use),
        ("total_energy_supply_equivalent_drones", total_supply),
        ("base_full_battery_departures_per_hour", total_departures),
        ("swap_events_per_hour", total_swaps),
        ("battery_supply_events_per_hour", total_events),
        ("maximum_constraint_violation", max_violation),
    ):
        recorder.close(
            _finite(energy.get(field), f"{mode}.energy_balance.{field}"),
            expected,
            f"summary_energy_{field}",
            "energy",
            mode=mode,
            flow=True,
        )
    reconciliation = _mapping(summary.get("reconciliation"), f"{mode}.reconciliation")
    recorder.close(
        _finite(
            reconciliation.get("maximum_energy_constraint_violation"),
            f"{mode}.reconciliation.maximum_energy_constraint_violation",
        ),
        max_violation,
        "summary_max_energy_violation",
        "energy",
        mode=mode,
        flow=True,
    )
    return max_violation


def _validate_mode_comparison(
    path: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    run_config: Mapping[str, Any],
    recorder: _Recorder,
) -> None:
    fields, rows = _read_csv(path)
    _required_columns(
        fields,
        {
            "mode",
            "run_signature_sha256",
            "status_name",
            "objective",
            "courier_only_baseline",
            "cost_saving",
            "runtime_seconds",
            "mip_gap",
            "requested_mip_gap",
            "screening_is_exact",
        },
        str(path),
    )
    by_mode: dict[str, dict[str, str]] = {}
    for row in rows:
        mode = row["mode"]
        if mode in by_mode:
            raise ValueError(f"duplicate mode {mode!r} in {path}")
        by_mode[mode] = row
    recorder.add(
        set(by_mode) == set(summaries),
        "mode_comparison_modes",
        "cross_artifact",
        observed=sorted(by_mode),
        expected=sorted(summaries),
    )
    for mode, summary in summaries.items():
        row = by_mode.get(mode)
        if row is None:
            continue
        recorder.add(
            row["run_signature_sha256"] == run_config.get("run_signature_sha256"),
            "mode_comparison_signature",
            "cross_artifact",
            mode=mode,
            observed=row["run_signature_sha256"],
            expected=run_config.get("run_signature_sha256"),
        )
        recorder.add(
            row["status_name"] == summary.get("status_name"),
            "mode_comparison_status",
            "cross_artifact",
            mode=mode,
            observed=row["status_name"],
            expected=summary.get("status_name"),
        )
        for field in (
            "objective",
            "courier_only_baseline",
            "cost_saving",
            "runtime_seconds",
            "mip_gap",
            "requested_mip_gap",
        ):
            recorder.close(
                _finite(row[field], f"mode_comparison.{mode}.{field}"),
                _summary_number(summary, field, f"{mode} summary"),
                f"mode_comparison_{field}",
                "cross_artifact",
                mode=mode,
            )
        recorder.add(
            _bool_text(row["screening_is_exact"], f"mode_comparison.{mode}.screening")
            is summary.get("screening_is_exact"),
            "mode_comparison_screening",
            "cross_artifact",
            mode=mode,
            observed=row["screening_is_exact"],
            expected=summary.get("screening_is_exact"),
        )


def _validate_mode(
    root: Path,
    mode: str,
    summary: Mapping[str, Any],
    run_config: Mapping[str, Any],
    completion: Mapping[str, Any],
    recorder: _Recorder,
) -> tuple[bool, float | None]:
    _validate_summary_contract(summary, run_config, mode, recorder)
    _validate_objective(summary, mode, recorder)
    _validate_gap(summary, completion, mode, recorder)
    exact = _validate_screening(run_config, summary, completion, mode, recorder)
    mode_root = root / mode
    od_rows = _validate_od_csv(mode_root / "od_service.csv", summary, mode, recorder)
    chi_od, chi_corridor, successful = _validate_chi_csv(
        mode_root / "od_corridor_reliability.csv", mode, recorder
    )
    _compare_od_and_chi(od_rows, chi_od, mode, recorder)
    _validate_corridor_csv(
        mode_root / "corridor_flows.csv",
        chi_corridor,
        summary,
        mode,
        recorder,
    )
    _validate_mx_display_csv(
        mode_root / "displayed_locker_service.csv",
        successful,
        od_rows,
        summary,
        mode,
        recorder,
    )
    max_energy_violation = _validate_energy_csv(
        mode_root / "energy_balance.csv", summary, mode, recorder
    )
    return exact, max_energy_violation


def _write_validation_outputs(root: Path, payload: Mapping[str, Any]) -> None:
    (root / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    fieldnames = (
        "check_id",
        "category",
        "status",
        "severity",
        "mode",
        "observed",
        "expected",
        "tolerance",
        "details",
    )
    with (root / "validation_checks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for check in payload["checks"]:
            row = dict(check)
            for field in ("observed", "expected"):
                if isinstance(row[field], (dict, list)):
                    row[field] = json.dumps(
                        row[field], ensure_ascii=False, sort_keys=True
                    )
            writer.writerow(row)


def validate_run(
    run_root: str | Path,
    *,
    abs_tol: float = 2e-6,
    rel_tol: float = 2e-6,
    flow_abs_tol: float = 3e-5,
) -> dict[str, Any]:
    """Validate one completed schema-5 run and write its evidence package.

    ``valid`` is false if any required assertion fails.  An unavailable external
    instance directory is recorded as a warning because the completion manifest
    can still seal a portable result package; when the directory is available,
    every canonical input hash is checked.
    """

    if abs_tol <= 0 or rel_tol < 0 or flow_abs_tol <= 0:
        raise ValueError("validation tolerances must be positive")
    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"result root does not exist: {root}")
    recorder = _Recorder(abs_tol=abs_tol, rel_tol=rel_tol, flow_abs_tol=flow_abs_tol)
    summaries: dict[str, dict[str, Any]] = {}
    screening: dict[str, bool] = {}
    energy_violations: dict[str, float | None] = {}
    modes: list[str] = []

    try:
        run_config = _read_json(root / "run_config.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        recorder.add(
            False,
            "run_config_readable",
            "package",
            observed=str(exc),
            expected="readable run_config.json object",
        )
        run_config = {}
    else:
        recorder.add(True, "run_config_readable", "package")
        try:
            modes = _validate_contract_and_signature(run_config, recorder)
        except ValueError as exc:
            recorder.add(
                False,
                "run_contract_parse",
                "contract",
                observed=str(exc),
                expected="complete current run contract",
            )
        _validate_instance_hashes(run_config, recorder)

    try:
        completion = _read_json(root / "run_complete.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        recorder.add(
            False,
            "run_complete_readable",
            "package",
            observed=str(exc),
            expected="readable run_complete.json object",
        )
        completion = {}
    else:
        recorder.add(True, "run_complete_readable", "package")
        _validate_completion_manifest(root, completion, modes, recorder)

    for mode in modes:
        try:
            summary = _read_json(root / mode / "summary.json")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            recorder.add(
                False,
                "mode_summary_readable",
                "package",
                mode=mode,
                observed=str(exc),
                expected="readable summary.json object",
            )
            continue
        summaries[mode] = summary
        recorder.add(True, "mode_summary_readable", "package", mode=mode)
        try:
            exact, max_energy = _validate_mode(
                root,
                mode,
                summary,
                run_config,
                completion,
                recorder,
            )
            screening[mode] = exact
            energy_violations[mode] = max_energy
        except (OSError, ValueError, KeyError) as exc:
            recorder.add(
                False,
                "mode_semantic_validation",
                "package",
                mode=mode,
                observed=str(exc),
                expected="complete, parseable, internally consistent mode artifacts",
            )

    try:
        _validate_mode_comparison(
            root / "mode_comparison.csv", summaries, run_config, recorder
        )
    except (OSError, ValueError, KeyError) as exc:
        recorder.add(
            False,
            "mode_comparison_validation",
            "cross_artifact",
            observed=str(exc),
            expected="mode_comparison.csv consistent with mode summaries",
        )

    exact_all = bool(screening) and all(screening.values())
    checks = [asdict(check) for check in recorder.checks]
    payload: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "validator_id": VALIDATOR_ID,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root),
        "result_schema_version": run_config.get("schema_version"),
        "formulation_id": run_config.get("formulation_id"),
        "run_signature_sha256": run_config.get("run_signature_sha256"),
        "modes": modes,
        "valid": recorder.valid,
        "screening_is_exact": screening,
        "interpretation_scope": (
            "unscreened_formulation" if exact_all else "screened_candidate_model"
        ),
        "eligible_for_unscreened_exactness_claim": exact_all and recorder.valid,
        "maximum_energy_violation_by_mode": energy_violations,
        "tolerances": {
            "absolute": abs_tol,
            "relative": rel_tol,
            "flow_absolute": flow_abs_tol,
        },
        "check_counts": {
            "pass": sum(check.status == "PASS" for check in recorder.checks),
            "warn": sum(check.status == "WARN" for check in recorder.checks),
            "fail": sum(check.status == "FAIL" for check in recorder.checks),
        },
        "checks": checks,
    }
    _write_validation_outputs(root, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently validate one completed schema-5 MLND result package."
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--abs-tol", type=float, default=2e-6)
    parser.add_argument("--rel-tol", type=float, default=2e-6)
    parser.add_argument("--flow-abs-tol", type=float, default=3e-5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = validate_run(
        args.run_root,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        flow_abs_tol=args.flow_abs_tol,
    )
    counts = payload["check_counts"]
    print(
        f"MLND validation {'PASSED' if payload['valid'] else 'FAILED'}: "
        f"{counts['pass']} pass, {counts['warn']} warning, {counts['fail']} fail"
    )
    print(Path(payload["run_root"]) / "validation.json")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
