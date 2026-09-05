"""Build a deterministic, provenance-bound ledger for paper experiments.

An experiment root may contain runs at any depth.  A run is a directory with
``run_config.json`` and ``run_complete.json``; every mode declared in the run
configuration must also contain ``<mode>/summary.json``.  Partially written
runs are rejected instead of silently disappearing from a paper table.

The command writes one mode-level ledger, one run/scenario-level summary, and a
manifest that hashes both generated tables and every JSON source used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.model_contract import (
    require_result_contract, require_same_result_schema,
    require_validation_binding, runtime_contract,
)


RUN_LEDGER_FIELDS = (
    "run_path",
    "experiment_id",
    "scenario_id",
    "mode",
    "schema_version",
    "runtime_contract",
    "formulation_id",
    "run_signature_sha256",
    "git_head",
    "git_branch",
    "implementation_sha256",
    "input_sha256",
    "scale_family_id",
    "run_config_sha256",
    "run_complete_sha256",
    "summary_sha256",
    "validation_sha256",
    "crop_side_km",
    "crop_area_km2",
    "demand_orders_raw",
    "demand_per_hour",
    "demand_scale",
    "candidate_bases",
    "candidate_lockers",
    "candidate_swap_stations",
    "od_markets",
    "od_nodes",
    "screening_is_exact",
    "model_parameters_sha256",
    "solver_parameters_sha256",
    "solver_seed",
    "solver_threads",
    "solver_time_limit_seconds",
    "solver_requested_mip_gap",
    "solver_method",
    "solver_miqcp_method",
    "status_code",
    "status_name",
    "objective_cny_per_hour",
    "objective_bound_cny_per_hour",
    "mip_gap_fraction",
    "mip_gap_percent",
    "runtime_seconds",
    "work_units",
    "optimized_total_cost_cny_per_hour",
    "facility_cost_cny_per_hour",
    "capacity_cost_cny_per_hour",
    "connection_cost_cny_per_hour",
    "route_operation_cost_cny_per_hour",
    "fleet_cost_cny_per_hour",
    "first_mile_courier_cost_cny_per_hour",
    "last_mile_courier_cost_cny_per_hour",
    "direct_courier_fallback_cost_cny_per_hour",
    "locker_handling_cost_cny_per_hour",
    "courier_only_baseline_cny_per_hour",
    "cost_saving_cny_per_hour",
    "cost_saving_percent",
    "total_demand_per_hour",
    "drone_offered_per_hour",
    "drone_realized_per_hour",
    "drone_service_fraction",
    "locker_pickup_per_hour",
    "assisted_home_per_hour",
    "direct_courier_fallback_per_hour",
    "realization_fraction_of_offered",
    "open_bases",
    "open_lockers",
    "open_swap_stations",
    "active_connections",
    "active_virtual_states",
    "fleet_size",
    "transit_drones",
    "queue_maximum_utilization",
    "queue_weighted_mean_handling_minutes",
    "energy_use_equivalent_drones",
    "energy_supply_equivalent_drones",
    "energy_maximum_constraint_violation",
    "validation_present",
    "validation_pass",
)

_SCENARIO_COMMON_FIELDS = (
    "run_path",
    "experiment_id",
    "scenario_id",
    "run_signature_sha256",
    "git_head",
    "implementation_sha256",
    "input_sha256",
    "scale_family_id",
    "crop_side_km",
    "crop_area_km2",
    "demand_orders_raw",
    "demand_per_hour",
    "candidate_bases",
    "candidate_lockers",
    "candidate_swap_stations",
    "od_markets",
    "model_parameters_sha256",
    "solver_parameters_sha256",
    "mode_count",
    "modes",
    "all_validations_pass",
    "mx_gap_within_one_percent",
)

_SCENARIO_MODE_SUFFIXES = (
    "status_name",
    "objective_cny_per_hour",
    "objective_bound_cny_per_hour",
    "mip_gap_fraction",
    "runtime_seconds",
    "cost_saving_percent",
    "drone_service_fraction",
    "validation_pass",
)

_VALID_MODES = ("HD", "MX", "LO")


@dataclass(frozen=True)
class LedgerOutputs:
    """Paths and record counts produced by :func:`build_paper_experiment_ledger`."""

    run_ledger: Path
    scenario_summary: Path
    manifest: Path
    run_count: int
    row_count: int


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool | None:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pass", "passed", "true", "ok", "valid", "success"}:
            return True
        if normalized in {"fail", "failed", "false", "invalid", "error"}:
            return False
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"top-level JSON value must be an object: {path}")
    return payload


def _require_contract(payload: Mapping[str, Any], path: Path) -> None:
    require_result_contract(payload, path)


def _required_sha256(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{path}: {field} must be a 64-character sha256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{path}: {field} must be hexadecimal sha256 text") from exc
    if value != value.lower():
        raise ValueError(f"{path}: {field} must be lowercase sha256 text")
    return value


def _configured_modes(run_config: Mapping[str, Any], path: Path) -> tuple[str, ...]:
    raw = run_config.get("modes")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: modes must be a nonempty array")
    if any(not isinstance(mode, str) or mode not in _VALID_MODES for mode in raw):
        raise ValueError(f"{path}: modes must contain only HD, MX, and LO")
    modes = tuple(raw)
    if len(set(modes)) != len(modes):
        raise ValueError(f"{path}: modes must not contain duplicates")
    return modes


def _input_sha256(run_config: Mapping[str, Any]) -> str:
    declared = _nested(run_config, "instance", "sha256")
    if isinstance(declared, str) and len(declared) == 64:
        return declared
    canonical = _nested(run_config, "instance", "canonical_file_sha256")
    if not isinstance(canonical, Mapping) or not canonical:
        raise ValueError("run_config.instance.canonical_file_sha256 is required")
    return _payload_sha256(canonical)


def _validation_pass(validation: Mapping[str, Any]) -> bool | None:
    direct = _first(
        validation.get("valid"),
        validation.get("validation_pass"),
        validation.get("passed"),
        validation.get("overall_pass"),
        validation.get("status"),
    )
    parsed = _bool(direct)
    if parsed is not None:
        return parsed
    checks = validation.get("checks")
    if isinstance(checks, list) and checks:
        values = [
            _bool(_first(check.get("passed"), check.get("status")))
            for check in checks
            if isinstance(check, Mapping)
        ]
        if values and all(value is not None for value in values):
            return all(values)
    return None


def _completion_is_sealed(
    run_dir: Path,
    completion: Mapping[str, Any],
    configured_modes: tuple[str, ...],
) -> None:
    if completion.get("completed") is not True:
        raise ValueError(f"incomplete run: {run_dir}/run_complete.json is not sealed")
    completed_modes = completion.get("modes")
    if completed_modes is not None and completed_modes != list(configured_modes):
        raise ValueError(f"incomplete run: {run_dir} completion modes disagree")


def _row_for_mode(
    *,
    root: Path,
    run_dir: Path,
    run_config: Mapping[str, Any],
    completion_path: Path,
    summary_path: Path,
    validation_path: Path,
    mode: str,
) -> dict[str, Any]:
    summary = _load_object(summary_path)
    _require_contract(summary, summary_path)
    require_same_result_schema(run_config, summary, summary_path)
    run_signature = _required_sha256(
        run_config.get("run_signature_sha256"), "run_signature_sha256", run_dir
    )
    if summary.get("run_signature_sha256") != run_signature:
        raise ValueError(f"{summary_path}: run signature disagrees with run_config")
    if summary.get("mode") != mode:
        raise ValueError(f"{summary_path}: summary mode disagrees with directory")

    metadata = _mapping(_nested(run_config, "instance", "metadata"))
    nested_scale = _mapping(metadata.get("nested_scale"))
    data_config = _mapping(metadata.get("data_config"))
    model = _mapping(run_config.get("model"))
    solver = _mapping(run_config.get("solver"))
    costs = _mapping(summary.get("costs_cny_per_hour"))
    funnel = _mapping(summary.get("demand_funnel_orders_per_hour"))
    facilities = _mapping(summary.get("facilities"))
    candidates = _mapping(facilities.get("candidates"))
    opened = _mapping(facilities.get("open"))
    instance = _mapping(summary.get("instance"))
    fleet = _mapping(summary.get("fleet"))
    queue = _mapping(summary.get("locker_queue"))
    energy = _mapping(summary.get("energy_balance"))
    summary_solver = _mapping(summary.get("solver"))
    search = _mapping(summary_solver.get("search_statistics"))

    validation: Mapping[str, Any] = {}
    if validation_path.is_file():
        validation = _load_object(validation_path)
        require_validation_binding(validation, run_config, validation_path)

    relative = run_dir.relative_to(root).as_posix()
    experiment_id = run_config.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        experiment_id = root.name
    scenario_id = _first(run_config.get("scenario_id"), run_config.get("scenario"))
    if not isinstance(scenario_id, str) or not scenario_id:
        scenario_id = relative if relative != "." else run_dir.name

    configured_screening = run_config.get("screening_is_exact")
    if isinstance(configured_screening, Mapping):
        configured_screening = configured_screening.get(mode)
    screening = _bool(_first(summary.get("screening_is_exact"), configured_screening))

    gap = _finite(_first(summary.get("mip_gap"), summary_solver.get("mip_gap")))
    row = {
        "run_path": relative,
        "experiment_id": experiment_id,
        "scenario_id": scenario_id,
        "mode": mode,
        "schema_version": summary.get("schema_version"),
        "runtime_contract": runtime_contract(summary["schema_version"]),
        "formulation_id": summary.get("formulation_id"),
        "run_signature_sha256": run_signature,
        "git_head": _nested(run_config, "git", "head"),
        "git_branch": _nested(run_config, "git", "branch"),
        "implementation_sha256": _nested(run_config, "implementation", "sha256"),
        "input_sha256": _input_sha256(run_config),
        "scale_family_id": nested_scale.get("family_id"),
        "run_config_sha256": _sha256_file(run_dir / "run_config.json"),
        "run_complete_sha256": _sha256_file(completion_path),
        "summary_sha256": _sha256_file(summary_path),
        "validation_sha256": _sha256_file(validation_path) if validation else None,
        "crop_side_km": _first(nested_scale.get("crop_side_km"), data_config.get("map_side_km")),
        "crop_area_km2": nested_scale.get("crop_area_km2"),
        "demand_orders_raw": _first(nested_scale.get("demand_order_count"), metadata.get("demand_orders_in_area")),
        "demand_per_hour": _first(metadata.get("total_demand_per_hour"), funnel.get("total_demand")),
        "demand_scale": run_config.get("demand_scale"),
        "candidate_bases": _first(instance.get("candidate_bases"), candidates.get("bases"), data_config.get("num_bases")),
        "candidate_lockers": _first(instance.get("candidate_lockers"), candidates.get("lockers"), data_config.get("candidate_count")),
        "candidate_swap_stations": _first(instance.get("candidate_swap_stations"), candidates.get("swap_stations"), data_config.get("num_swaps")),
        "od_markets": _first(instance.get("markets"), metadata.get("od_market_count")),
        "od_nodes": _first(instance.get("od_nodes"), (metadata.get("origin_node_count") or 0) + (metadata.get("destination_node_count") or 0)),
        "screening_is_exact": screening,
        "model_parameters_sha256": _payload_sha256(model),
        "solver_parameters_sha256": _payload_sha256(solver),
        "solver_seed": solver.get("seed"),
        "solver_threads": solver.get("threads"),
        "solver_time_limit_seconds": solver.get("time_limit"),
        "solver_requested_mip_gap": solver.get("mip_gap"),
        "solver_method": _first(solver.get("method"), solver.get("root_relaxation_method")),
        "solver_miqcp_method": solver.get("miqcp_method"),
        "status_code": _first(summary.get("status"), summary_solver.get("status")),
        "status_name": _first(summary.get("status_name"), summary_solver.get("status_name")),
        "objective_cny_per_hour": _first(summary.get("objective"), summary_solver.get("objective_cost_per_hour")),
        "objective_bound_cny_per_hour": _first(summary.get("objective_bound"), summary_solver.get("objective_bound")),
        "mip_gap_fraction": gap,
        "mip_gap_percent": 100.0 * gap if gap is not None else None,
        "runtime_seconds": _first(summary.get("runtime_seconds"), summary_solver.get("runtime_seconds")),
        "work_units": _first(search.get("work_units"), summary_solver.get("work_units")),
        "optimized_total_cost_cny_per_hour": costs.get("optimized_total"),
        "facility_cost_cny_per_hour": costs.get("facility"),
        "capacity_cost_cny_per_hour": costs.get("capacity"),
        "connection_cost_cny_per_hour": costs.get("connection"),
        "route_operation_cost_cny_per_hour": costs.get("route_operation"),
        "fleet_cost_cny_per_hour": costs.get("fleet"),
        "first_mile_courier_cost_cny_per_hour": costs.get("first_mile_courier"),
        "last_mile_courier_cost_cny_per_hour": costs.get("last_mile_courier"),
        "direct_courier_fallback_cost_cny_per_hour": costs.get("direct_courier_fallback"),
        "locker_handling_cost_cny_per_hour": costs.get("locker_handling"),
        "courier_only_baseline_cny_per_hour": summary.get("courier_only_baseline"),
        "cost_saving_cny_per_hour": summary.get("cost_saving"),
        "cost_saving_percent": summary.get("cost_saving_percent"),
        "total_demand_per_hour": funnel.get("total_demand"),
        "drone_offered_per_hour": funnel.get("total_drone_offered"),
        "drone_realized_per_hour": funnel.get("total_drone_realized"),
        "drone_service_fraction": _first(funnel.get("drone_service_fraction_of_demand"), summary.get("drone_served_rate")),
        "locker_pickup_per_hour": funnel.get("locker_pickup"),
        "assisted_home_per_hour": funnel.get("assisted_home"),
        "direct_courier_fallback_per_hour": funnel.get("direct_courier_fallback"),
        "realization_fraction_of_offered": funnel.get("realization_fraction_of_offered"),
        "open_bases": _first(opened.get("bases"), summary.get("open_bases")),
        "open_lockers": _first(opened.get("lockers"), summary.get("open_lockers")),
        "open_swap_stations": _first(opened.get("swap_stations"), summary.get("open_swap_stations")),
        "active_connections": facilities.get("active_connections"),
        "active_virtual_states": _first(summary.get("active_virtual_state_count"), fleet.get("virtual_idle_drones")),
        "fleet_size": _first(summary.get("fleet_size"), fleet.get("gamma_drones")),
        "transit_drones": fleet.get("transit_drones"),
        "queue_maximum_utilization": queue.get("maximum_utilization"),
        "queue_weighted_mean_handling_minutes": queue.get("workload_weighted_mean_handling_formula_minutes"),
        "energy_use_equivalent_drones": energy.get("total_energy_use_equivalent_drones"),
        "energy_supply_equivalent_drones": energy.get("total_energy_supply_equivalent_drones"),
        "energy_maximum_constraint_violation": energy.get("maximum_constraint_violation"),
        "validation_present": bool(validation),
        "validation_pass": _validation_pass(validation) if validation else None,
    }
    return row


def collect_paper_experiment_ledger(experiment_root: str | Path) -> list[dict[str, Any]]:
    """Collect one validated ledger row per configured mode.

    Discovery uses the union of ``run_config.json`` and ``run_complete.json``
    parents so that an interrupted run cannot be skipped accidentally.
    """

    root = Path(experiment_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"experiment root is not a directory: {root}")
    config_dirs = {path.parent for path in root.rglob("run_config.json")}
    complete_dirs = {path.parent for path in root.rglob("run_complete.json")}
    run_dirs = sorted(config_dirs | complete_dirs, key=lambda path: path.relative_to(root).as_posix())
    if not run_dirs:
        raise ValueError(f"no experiment runs found below {root}")

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        config_path = run_dir / "run_config.json"
        completion_path = run_dir / "run_complete.json"
        if not config_path.is_file() or not completion_path.is_file():
            raise ValueError(f"incomplete run: {run_dir} requires run_config.json and run_complete.json")
        run_config = _load_object(config_path)
        completion = _load_object(completion_path)
        _require_contract(run_config, config_path)
        modes = _configured_modes(run_config, config_path)
        _completion_is_sealed(run_dir, completion, modes)
        declared = set(modes)
        discovered = {path.parent.name for path in run_dir.glob("*/summary.json")}
        extra = sorted(discovered - declared)
        if extra:
            raise ValueError(f"{run_dir}: undeclared summary modes: {', '.join(extra)}")
        for mode in modes:
            summary_path = run_dir / mode / "summary.json"
            if not summary_path.is_file():
                raise ValueError(f"incomplete run: missing {summary_path}")
            rows.append(
                _row_for_mode(
                    root=root,
                    run_dir=run_dir,
                    run_config=run_config,
                    completion_path=completion_path,
                    summary_path=summary_path,
                    validation_path=run_dir / "validation.json",
                    mode=mode,
                )
            )
    return sorted(rows, key=lambda row: (str(row["run_path"]), _VALID_MODES.index(str(row["mode"]))))


def _scenario_fields(modes: Sequence[str]) -> tuple[str, ...]:
    mode_fields = tuple(
        f"{mode.lower()}_{suffix}"
        for mode in modes
        for suffix in _SCENARIO_MODE_SUFFIXES
    )
    return (*_SCENARIO_COMMON_FIELDS, *mode_fields)


def summarize_scenarios(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pivot the mode ledger to one deterministic row per run/scenario."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run_path"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for run_path in sorted(grouped):
        group = sorted(grouped[run_path], key=lambda row: _VALID_MODES.index(str(row["mode"])))
        first = group[0]
        validations = [row.get("validation_pass") for row in group]
        summary = {field: first.get(field) for field in _SCENARIO_COMMON_FIELDS if field not in {"mode_count", "modes", "all_validations_pass", "mx_gap_within_one_percent"}}
        summary["mode_count"] = len(group)
        summary["modes"] = ",".join(str(row["mode"]) for row in group)
        summary["all_validations_pass"] = all(validations) if all(value is not None for value in validations) else None
        mx = next((row for row in group if row["mode"] == "MX"), None)
        mx_gap = _finite(mx.get("mip_gap_fraction")) if mx is not None else None
        summary["mx_gap_within_one_percent"] = mx_gap <= 0.01 if mx_gap is not None else None
        for row in group:
            prefix = str(row["mode"]).lower()
            for suffix in _SCENARIO_MODE_SUFFIXES:
                summary[f"{prefix}_{suffix}"] = row.get(suffix)
        summaries.append(summary)
    return summaries


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    return value


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def build_paper_experiment_ledger(
    experiment_root: str | Path,
    output_dir: str | Path | None = None,
) -> LedgerOutputs:
    """Validate all runs and write the three deterministic ledger artifacts."""

    root = Path(experiment_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir else root
    destination.mkdir(parents=True, exist_ok=True)
    rows = collect_paper_experiment_ledger(root)
    scenarios = summarize_scenarios(rows)
    modes = tuple(mode for mode in _VALID_MODES if any(row["mode"] == mode for row in rows))
    ledger_path = destination / "run_ledger.csv"
    scenario_path = destination / "scenario_summary.csv"
    manifest_path = destination / "ledger_manifest.json"
    _write_csv(ledger_path, RUN_LEDGER_FIELDS, rows)
    _write_csv(scenario_path, _scenario_fields(modes), scenarios)

    source_paths = sorted(
        {
            path
            for run_dir in {root / str(row["run_path"]) for row in rows}
            for path in (
                run_dir / "run_config.json",
                run_dir / "run_complete.json",
                *(run_dir / str(row["mode"]) / "summary.json" for row in rows if root / str(row["run_path"]) == run_dir),
                run_dir / "validation.json",
            )
            if path.is_file()
        },
        key=lambda path: path.relative_to(root).as_posix(),
    )
    manifest = {
        "schema_version": 1,
        "experiment_root_name": root.name,
        "run_count": len(scenarios),
        "mode_row_count": len(rows),
        "runs": [
            {
                "run_path": scenario["run_path"],
                "run_signature_sha256": scenario["run_signature_sha256"],
                "modes": scenario["modes"].split(","),
            }
            for scenario in scenarios
        ],
        "source_files": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in source_paths
        ],
        "outputs": {
            "run_ledger.csv": _sha256_file(ledger_path),
            "scenario_summary.csv": _sha256_file(scenario_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return LedgerOutputs(
        run_ledger=ledger_path,
        scenario_summary=scenario_path,
        manifest=manifest_path,
        run_count=len(scenarios),
        row_count=len(rows),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", help="root containing completed experiment runs")
    parser.add_argument("--output-dir", help="destination (defaults to experiment root)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = build_paper_experiment_ledger(args.experiment_root, args.output_dir)
    print(f"Collected {outputs.row_count} mode rows from {outputs.run_count} runs")
    print(outputs.run_ledger)
    print(outputs.scenario_summary)
    print(outputs.manifest)


if __name__ == "__main__":
    main()
