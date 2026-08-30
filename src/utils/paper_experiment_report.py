"""Build the paper-experiment report from an explicit scenario registry.

The module is deliberately a thin, read-only reporting layer.  It consumes a
sealed paper-experiment ledger, an explicit role registry, and the selected
three-mode result package.  It does not discover a paper role from a directory
name and never changes an instance, solver configuration, or result artifact.

``scenario_config.json`` schema (version 1)
------------------------------------------------

Paths in ``run_path`` are relative to ``experiment_root``.  ``instance_dir``
may be absolute or repository-relative.  A role item has the following shape::

    {
      "id": "main_h24",
      "label": "H24 baseline",
      "run_path": "formal/main/H24_baseline_audit_900s",
      "instance_dir": "data/instances/nanshan_paper_20260831/H024_L02_E03_R04p50",
      "modes": ["HD", "MX", "LO"],
      "x_value": 24,
      "group": "baseline",
      "role": "selected"
    }

The recommended top-level document is::

    {
      "schema_version": 1,
      "artifact_status": "final_evidence_registry",
      "experiment_root": "results/paper_experiments_20260831",
      "implementation_sha256_required": "<64-character solver implementation hash>",
      "repository_commit_required": "<selected-main provenance commit>",
      "instance_family": {
        "family_id": "<nested geographic crop family>",
        "required_solve_demand_scale": 1.0
      },
      "baseline": {
        "formulation_id": "<current formulation id>",
        "model": {"...": "the complete selected-main run_config.model"},
        "solver": {
          "target_mip_gap_fraction": 0.01,
          "threads": 4,
          "solver_seed": 43,
          "root_method": 2,
          "miqcp_method": -1
        }
      },
      "ledger": {
        "run_ledger": "ledger/run_ledger.csv",
        "scenario_summary": "ledger/scenario_summary.csv",
        "manifest": "ledger/ledger_manifest.json"
      },
      "roles": {
        "main": {"...": "one selected three-mode item"},
        "scale": [{"...": "one item per crop; x_value is candidate H"}],
        "algorithm": [{"...": "one item per matched method"}],
        "exactness": [
          {"role": "screened", "...": "screened H11"},
          {"role": "complete", "...": "complete-index H11"}
        ],
        "sensitivity": [
          {"group": "rider_cost", "x_value": 0.8, "...": "low"}
        ]
      },
      "choice_curve": {
        "distances_km": [0, 0.5, 1.0, 1.5],
        "discounts_cny": [0, 1, 2]
      },
      "basemap_cache": "results/nanshan_current_report_schema5/basemap_cache"
    }

For compatibility with small hand-written registries, ``selected_main``,
``scale``, ``algorithm``, ``exactness``, and ``sensitivity`` may instead be
placed at the top level.  ``results_dir`` is accepted as an alias for
``run_path``.  The registry is allowed to contain planned scenarios that do
not yet occur in the ledger; these are rendered as ``pending`` and excluded
from quantitative conclusions.  ``planning_only_not_a_result`` creates a
progress report, while ``final_evidence_registry`` is the only final-report
status.  A completed scenario is evidence-ready only when its ledger rows are
present, every requested mode has an independent ``validation_pass=true``
record, all sealed source hashes still match, and the instance/model/solver
role contract matches the real ``run_config.json``.  If no role remains
evidence-ready, generation fails before writing a deliverable artifact.

Outputs
-------

The writer creates a Chinese Markdown technical report, CSV evidence tables,
PNG/SVG research figures, a canonical portable-report ``artifact.json``, and
``report_manifest.json`` with hashes.  Portable HTML is an interactive
supplement without the static Nanshan maps; Markdown plus PNG/SVG is the
complete geographic report.  Solver gaps for screened runs certify only the
explicitly screened formulation.  All monetary outcomes are modeled fulfilment
cost or cost saving in CNY/hour, never revenue or profit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlnd-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mlnd-xdg-cache")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION
from src.utils import nanshan_geographic_report as geo


MODE_NAMES = {"HD": "到家配送", "MX": "混合配送", "LO": "机柜自取"}
ROLE_NAMES = {
    "main": "主算例",
    "scale": "规模实验",
    "algorithm": "算法对照",
    "exactness": "完整/筛选对照",
    "sensitivity": "敏感性实验",
}
PALETTE = {
    "blue": "#2563EB",
    "blue_open": "#DBEAFE",
    "gold": "#D97706",
    "gold_open": "#FEF3C7",
    "olive": "#4D7C0F",
    "pink": "#BE185D",
    "ink": "#172033",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "paper": "#FFFFFF",
}
_ONE_PERCENT = 0.01
_CANONICAL_INSTANCE_FILES = (
    "metadata.json",
    "sites.csv",
    "od_nodes.csv",
    "od_demand.csv",
    "periods.csv",
)
_REGISTRY_STATUS = {
    "planning_only_not_a_result": "progress",
    "final_evidence_registry": "final",
}
_REQUIRED_BASELINE_MODEL_KEYS = {"drone_speed_mps", "battery_range_km"}
_REQUIRED_BASELINE_SOLVER_KEYS = {
    "mip_gap",
    "threads",
    "seed",
    "root_relaxation_method",
    "miqcp_method",
}
_MAP_SOURCE_ARTIFACTS = ("incumbent.sol", "open_facilities.csv")


@dataclass(frozen=True)
class ScenarioRole:
    category: str
    scenario_id: str
    label: str
    run_path: str
    instance_dir: Path | None
    modes: tuple[str, ...]
    x_value: float | str | None
    group: str | None
    role: str | None
    declared_status: str | None
    contract: Mapping[str, Any]


@dataclass(frozen=True)
class RoleEvidence:
    role: ScenarioRole
    status: str
    note: str
    rows: tuple[dict[str, Any], ...]

    @property
    def ready(self) -> bool:
        return self.status in {"ready", "pilot"}


@dataclass(frozen=True)
class ReportOutputs:
    markdown: Path
    artifact: Path
    manifest: Path
    figures: tuple[Path, ...]
    tables: tuple[Path, ...]


@dataclass(frozen=True)
class MapRender:
    png: Path
    svg: Path
    status: str
    note: str


@dataclass(frozen=True)
class ProvenanceSource:
    path: Path
    locator: str


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"top-level JSON value must be an object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read CSV table from {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status", "note"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite(value)
    if number is None or abs(number - round(number)) > 1e-7:
        return None
    return int(round(number))


def _boolean(value: Any) -> bool | None:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "pass", "passed", "yes", "1"}:
            return True
        if normalized in {"false", "fail", "failed", "no", "0"}:
            return False
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    preferred = (_repo_root() / path).resolve()
    if preferred.exists() or base is None:
        return preferred
    return (base / path).resolve()


def _normalise_run_path(value: Any, experiment_root: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every configured paper role requires run_path/results_dir")
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            return path.resolve().relative_to(experiment_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"role results path {path} is outside experiment_root {experiment_root}"
            ) from exc
    repo_candidate = (_repo_root() / path).resolve()
    try:
        return repo_candidate.relative_to(experiment_root).as_posix()
    except ValueError:
        pass
    normalized = Path(os.path.normpath(value)).as_posix()
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"role run_path escapes experiment_root: {value!r}")
    return normalized.removeprefix("./")


def _ledger_paths(
    config: Mapping[str, Any],
    experiment_root: Path,
) -> tuple[Path, Path, Path]:
    ledger = config.get("ledger")
    ledger_map = ledger if isinstance(ledger, Mapping) else {}

    def resolve(key: str, default: str) -> Path:
        value = ledger_map.get(key, default)
        if not isinstance(value, str) or not value:
            raise ValueError(f"ledger.{key} must be a nonempty path")
        path = Path(value).expanduser()
        return (
            path.resolve() if path.is_absolute() else (experiment_root / path).resolve()
        )

    return (
        resolve("run_ledger", "ledger/run_ledger.csv"),
        resolve("scenario_summary", "ledger/scenario_summary.csv"),
        resolve("manifest", "ledger/ledger_manifest.json"),
    )


def _load_verified_ledger(
    config: Mapping[str, Any],
    experiment_root: Path,
) -> tuple[
    list[dict[str, str]], list[dict[str, str]], dict[str, Any], tuple[Path, ...]
]:
    experiment_root = experiment_root.resolve()
    ledger_path, scenario_path, manifest_path = _ledger_paths(config, experiment_root)
    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"ledger manifest schema_version must equal 1: {manifest_path}"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"ledger manifest has no outputs mapping: {manifest_path}")
    for path in (ledger_path, scenario_path):
        expected = outputs.get(path.name)
        if not isinstance(expected, str) or expected != _sha256(path):
            raise ValueError(f"ledger output hash mismatch: {path}")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError(
            f"ledger manifest source_files must be nonempty: {manifest_path}"
        )
    source_index: dict[str, Mapping[str, Any]] = {}
    for index, source in enumerate(source_files):
        if not isinstance(source, Mapping):
            raise ValueError(f"ledger source_files[{index}] must be an object")
        relative = source.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"ledger source_files[{index}].path must be nonempty")
        normalized = Path(os.path.normpath(relative))
        if (
            normalized.is_absolute()
            or normalized == Path("..")
            or ".." in normalized.parts
        ):
            raise ValueError(
                f"ledger source path escapes experiment_root: {relative!r}"
            )
        relative_key = normalized.as_posix()
        if relative_key in source_index:
            raise ValueError(f"duplicate ledger source path: {relative_key}")
        source_path = (experiment_root / normalized).resolve()
        try:
            source_path.relative_to(experiment_root)
        except ValueError as exc:
            raise ValueError(
                f"ledger source path escapes experiment_root: {relative!r}"
            ) from exc
        if not source_path.is_file():
            raise ValueError(f"ledger source file is missing: {source_path}")
        expected_size = _integer(source.get("size_bytes"))
        expected_hash = source.get("sha256")
        if expected_size != source_path.stat().st_size:
            raise ValueError(f"ledger source file size mismatch: {source_path}")
        if not isinstance(expected_hash, str) or expected_hash != _sha256(source_path):
            raise ValueError(f"ledger source file hash mismatch: {source_path}")
        source_index[relative_key] = source
    ledger_rows = _read_csv(ledger_path)
    scenario_rows = _read_csv(scenario_path)
    if not ledger_rows:
        raise ValueError(f"ledger has no mode rows: {ledger_path}")
    keys: set[tuple[str, str]] = set()
    for index, row in enumerate(ledger_rows, start=2):
        schema_version = _integer(row.get("schema_version"))
        formulation_id = row.get("formulation_id")
        if schema_version != RESULT_SCHEMA_VERSION or formulation_id != FORMULATION_ID:
            raise ValueError(
                f"incompatible ledger row {ledger_path}:{index}; expected schema "
                f"{RESULT_SCHEMA_VERSION} and formulation {FORMULATION_ID!r}"
            )
        run_path = row.get("run_path")
        mode = row.get("mode")
        if not isinstance(run_path, str) or not run_path or mode not in MODE_NAMES:
            raise ValueError(f"invalid run_path/mode in {ledger_path}:{index}")
        key = (run_path, mode)
        if key in keys:
            raise ValueError(f"duplicate ledger run/mode row: {key}")
        keys.add(key)
    run_paths = {run_path for run_path, _mode in keys}
    scenario_paths = {row.get("run_path") for row in scenario_rows}
    if scenario_paths != run_paths:
        raise ValueError("scenario_summary run paths disagree with run_ledger")
    if _integer(manifest.get("mode_row_count")) != len(ledger_rows):
        raise ValueError("ledger manifest mode_row_count disagrees with run_ledger")
    if _integer(manifest.get("run_count")) != len(run_paths):
        raise ValueError("ledger manifest run_count disagrees with run_ledger")
    manifest_runs = manifest.get("runs")
    if not isinstance(manifest_runs, list) or len(manifest_runs) != len(run_paths):
        raise ValueError("ledger manifest runs disagree with run_count")
    manifest_run_map: dict[str, Mapping[str, Any]] = {}
    for item in manifest_runs:
        if not isinstance(item, Mapping) or not isinstance(item.get("run_path"), str):
            raise ValueError("ledger manifest runs must contain run_path objects")
        run_path = str(item["run_path"])
        if run_path in manifest_run_map:
            raise ValueError(f"duplicate ledger manifest run: {run_path}")
        manifest_run_map[run_path] = item
    if set(manifest_run_map) != run_paths:
        raise ValueError("ledger manifest runs disagree with run_ledger paths")
    rows_by_run: dict[str, list[Mapping[str, str]]] = {}
    for row in ledger_rows:
        rows_by_run.setdefault(str(row["run_path"]), []).append(row)
    scenario_by_run = {str(row["run_path"]): row for row in scenario_rows}
    mode_order = {"HD": 0, "MX": 1, "LO": 2}
    for run_path, rows in rows_by_run.items():
        manifest_run = manifest_run_map[run_path]
        signatures = {row.get("run_signature_sha256") for row in rows}
        if (
            len(signatures) != 1
            or manifest_run.get("run_signature_sha256") not in signatures
        ):
            raise ValueError(f"ledger manifest run signature mismatch: {run_path}")
        ledger_modes = sorted((str(row["mode"]) for row in rows), key=mode_order.get)
        if manifest_run.get("modes") != ledger_modes:
            raise ValueError(f"ledger manifest run modes mismatch: {run_path}")
        scenario = scenario_by_run[run_path]
        scenario_modes = str(scenario.get("modes", "")).split(",")
        if (
            scenario_modes != ledger_modes
            or _integer(scenario.get("mode_count")) != len(ledger_modes)
            or scenario.get("run_signature_sha256")
            != manifest_run.get("run_signature_sha256")
        ):
            raise ValueError(
                f"scenario_summary run modes/signature mismatch: {run_path}"
            )
        common_sources = {
            "run_config_sha256": f"{run_path}/run_config.json",
            "run_complete_sha256": f"{run_path}/run_complete.json",
            "validation_sha256": f"{run_path}/validation.json",
        }
        for field, relative in common_sources.items():
            expected_hash = rows[0].get(field)
            source = source_index.get(relative)
            if (
                not expected_hash
                or source is None
                or source.get("sha256") != expected_hash
            ):
                raise ValueError(
                    f"ledger {field} disagrees with source manifest: {run_path}"
                )
        for row in rows:
            relative = f"{run_path}/{row['mode']}/summary.json"
            source = source_index.get(relative)
            if source is None or source.get("sha256") != row.get("summary_sha256"):
                raise ValueError(
                    f"ledger summary hash disagrees with source manifest: {relative}"
                )
    return (
        ledger_rows,
        scenario_rows,
        manifest,
        (ledger_path, scenario_path, manifest_path),
    )


def _role_items(config: Mapping[str, Any], category: str) -> list[Mapping[str, Any]]:
    roles = config.get("roles")
    role_map = roles if isinstance(roles, Mapping) else {}
    if category == "main":
        raw = role_map.get("main", config.get("selected_main"))
    else:
        raw = role_map.get(category, config.get(category))
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [raw]
    if isinstance(raw, list) and all(isinstance(item, Mapping) for item in raw):
        return list(raw)
    raise ValueError(f"role {category!r} must be an object or an array of objects")


def _resolve_instance_dir(value: Any, experiment_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("instance_dir must be a path string")
    return _resolve_repo_path(value, base=experiment_root)


def _parse_roles(
    config: Mapping[str, Any], experiment_root: Path
) -> list[ScenarioRole]:
    parsed: list[ScenarioRole] = []
    for category in ROLE_NAMES:
        for index, item in enumerate(_role_items(config, category), start=1):
            scenario_id = item.get("scenario_id", item.get("id"))
            if not isinstance(scenario_id, str) or not scenario_id:
                raise ValueError(f"{category}[{index}] requires id/scenario_id")
            label = item.get("label", scenario_id)
            if not isinstance(label, str) or not label:
                raise ValueError(f"{category}[{index}] label must be nonempty")
            run_path = _normalise_run_path(
                item.get("run_path", item.get("results_dir")), experiment_root
            )
            modes_raw = item.get("modes", [])
            if (
                not isinstance(modes_raw, list)
                or not modes_raw
                or any(mode not in MODE_NAMES for mode in modes_raw)
            ):
                raise ValueError(
                    f"{category}[{index}] modes must be a nonempty HD/MX/LO array"
                )
            if len(set(modes_raw)) != len(modes_raw):
                raise ValueError(
                    f"{category}[{index}] modes must not contain duplicates"
                )
            scale = item.get("scale")
            scale_map = scale if isinstance(scale, Mapping) else {}
            x_value: float | str | None = item.get(
                "x_value", item.get("factor", scale_map.get("H"))
            )
            if isinstance(x_value, bool) or not isinstance(
                x_value, (int, float, str, type(None))
            ):
                raise ValueError(f"{category}[{index}] x_value is invalid")
            parsed.append(
                ScenarioRole(
                    category=category,
                    scenario_id=scenario_id,
                    label=label,
                    run_path=run_path,
                    instance_dir=_resolve_instance_dir(
                        item.get("instance_dir"), experiment_root
                    ),
                    modes=tuple(modes_raw),
                    x_value=(
                        float(x_value) if isinstance(x_value, (int, float)) else x_value
                    ),
                    group=str(item["group"]) if item.get("group") is not None else None,
                    role=str(item["role"]) if item.get("role") is not None else None,
                    declared_status=(
                        str(item["status"]) if item.get("status") is not None else None
                    ),
                    contract=dict(item),
                )
            )
    mains = [role for role in parsed if role.category == "main"]
    selected = [role for role in mains if role.role in {"selected", "selected_main"}]
    if mains and len(selected) != 1:
        raise ValueError("main roles require exactly one explicit role='selected'")
    selected_alias = config.get("selected_main")
    if selected_alias is not None:
        if not isinstance(selected_alias, Mapping):
            raise ValueError("selected_main must be an object when provided")
        alias_id = selected_alias.get("scenario_id", selected_alias.get("id"))
        alias_path = _normalise_run_path(
            selected_alias.get("run_path", selected_alias.get("results_dir")),
            experiment_root,
        )
        if not selected or (
            alias_id != selected[0].scenario_id or alias_path != selected[0].run_path
        ):
            raise ValueError("selected_main alias disagrees with role='selected'")
    return parsed


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _contract_equal(actual: Any, expected: Any) -> bool:
    if expected == "all":
        return actual is None
    if (isinstance(expected, (int, float)) and not isinstance(expected, bool)) or (
        isinstance(actual, (int, float)) and not isinstance(actual, bool)
    ):
        actual_number = _finite(actual)
        expected_number = _finite(expected)
        return (
            actual_number is not None
            and expected_number is not None
            and math.isclose(
                actual_number, expected_number, rel_tol=1e-10, abs_tol=1e-10
            )
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _contract_equal(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            _contract_equal(actual[key], expected[key]) for key in actual
        )
    return actual == expected


_MODEL_OVERRIDE_PATHS: dict[str, tuple[str, ...]] = {
    "locker_discount_cny": ("choice", "locker_discount_cny"),
    "rider_initial_charge_cny": ("rider_base_cost",),
    "rider_extra_cost_cny_per_km": ("rider_incremental_cost_per_km",),
    "locker_capacity_costs_cny_per_hour": ("locker_capacity_costs",),
    "base_cost_cny_per_hour": ("base_cost_per_hour",),
    "locker_fixed_cost_cny_per_hour": ("locker_fixed_cost_per_hour",),
    "swap_cost_cny_per_hour": ("swap_cost_per_hour",),
    "corridor_cost_cny_per_km_hour": ("corridor_cost_per_km_hour",),
    "connection_cost_cny_per_km_hour": ("connection_cost_per_km_hour",),
    "flight_cost_cny_per_km": ("flight_cost_per_km",),
    "drone_cost_cny_per_hour": ("drone_cost_per_hour",),
}


_SOLVER_OVERRIDE_KEYS = {
    "time_limit_seconds": "time_limit",
    "time_limit_seconds_per_mode": "time_limit",
    "target_mip_gap_fraction": "mip_gap",
    "solver_seed": "seed",
    "root_method": "root_relaxation_method",
}


def _set_nested(payload: dict[str, Any], path: Sequence[str], value: Any) -> None:
    current = payload
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = None if value == "all" else value


def _expected_model(
    baseline: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    expected = json.loads(json.dumps(baseline))
    for key, value in overrides.items():
        mapped_path = _MODEL_OVERRIDE_PATHS.get(key, (key,))
        path = (key,) if key in baseline else mapped_path
        _set_nested(expected, path, value)
    return expected


def _normalise_solver_contract(
    contract: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    """Map registry-facing solver names to their run_config keys."""

    normalized: dict[str, Any] = {}
    declared_names: dict[str, str] = {}
    for declared_key, value in contract.items():
        actual_key = _SOLVER_OVERRIDE_KEYS.get(declared_key, declared_key)
        if actual_key in normalized and not _contract_equal(normalized[actual_key], value):
            raise ValueError(
                f"{context} maps both {declared_names[actual_key]!r} and "
                f"{declared_key!r} to {actual_key!r} with different values"
            )
        normalized[actual_key] = value
        declared_names[actual_key] = declared_key
    return normalized


def _solver_contract_errors(
    solver: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> list[str]:
    errors: list[str] = []
    for actual_key, value in expected.items():
        if not _contract_equal(solver.get(actual_key), value):
            errors.append(
                f"{context}：{actual_key} 应为 {value!r}，"
                f"实际为 {solver.get(actual_key)!r}"
            )
    return errors


def _canonical_instance_contract(
    instance_dir: Path,
    run_config: Mapping[str, Any],
    cache: dict[Path, dict[str, str]],
) -> tuple[dict[str, str] | None, list[str]]:
    errors: list[str] = []
    instance = _as_mapping(run_config.get("instance"))
    configured_directory = instance.get("directory")
    if not isinstance(configured_directory, str):
        errors.append("run_config.instance.directory 缺失")
    else:
        if Path(configured_directory).expanduser().resolve() != instance_dir.resolve():
            errors.append("instance_dir 与 run_config.instance.directory 不一致")
    declared = instance.get("canonical_file_sha256")
    if not isinstance(declared, Mapping) or set(declared) != set(
        _CANONICAL_INSTANCE_FILES
    ):
        errors.append("run_config 未声明恰好五个 canonical instance hashes")
        return None, errors
    if instance_dir not in cache:
        actual: dict[str, str] = {}
        for filename in _CANONICAL_INSTANCE_FILES:
            path = instance_dir / filename
            if not path.is_file():
                errors.append(f"实例规范文件缺失：{filename}")
                continue
            actual[filename] = _sha256(path)
        cache[instance_dir] = actual
    actual = cache[instance_dir]
    for filename in _CANONICAL_INSTANCE_FILES:
        if actual.get(filename) != declared.get(filename):
            errors.append(f"实例规范文件哈希不匹配：{filename}")
    return dict(declared), errors


def _validate_solver_overrides(
    solver: Mapping[str, Any], overrides: Mapping[str, Any]
) -> list[str]:
    expected = _normalise_solver_contract(overrides, context="solver_overrides")
    return _solver_contract_errors(solver, expected, context="solver override 未生效")


def _enforce_role_contracts(
    config: Mapping[str, Any],
    evidence: Sequence[RoleEvidence],
    ledger_rows: Sequence[Mapping[str, str]],
    experiment_root: Path,
) -> tuple[list[RoleEvidence], dict[str, dict[str, Any]]]:
    """Cross-check registry labels against the sealed run configurations."""

    required_commit = config.get("repository_commit_required")
    if required_commit is not None and (
        not isinstance(required_commit, str) or len(required_commit) != 40
    ):
        raise ValueError(
            "scenario_config repository_commit_required must be a 40-character hash"
        )
    configured_implementation = config.get("implementation_sha256_required")
    if configured_implementation is not None and (
        not isinstance(configured_implementation, str)
        or len(configured_implementation) != 64
    ):
        raise ValueError(
            "scenario_config implementation_sha256_required must be a 64-character hash"
        )
    family = _as_mapping(config.get("instance_family"))
    required_family = family.get("family_id")
    required_demand_scale = _finite(family.get("required_solve_demand_scale"))
    if not isinstance(required_family, str) or required_demand_scale is None:
        raise ValueError(
            "scenario_config.instance_family requires family_id and required_solve_demand_scale"
        )
    baseline_config = _as_mapping(config.get("baseline"))
    if baseline_config.get("formulation_id") != FORMULATION_ID:
        raise ValueError(
            "scenario_config baseline formulation_id disagrees with current model"
        )
    raw_baseline_model = baseline_config.get("model")
    raw_baseline_solver = baseline_config.get("solver")
    if not isinstance(raw_baseline_model, Mapping) or not raw_baseline_model:
        raise ValueError("scenario_config baseline.model must be a nonempty mapping")
    if not isinstance(raw_baseline_solver, Mapping) or not raw_baseline_solver:
        raise ValueError("scenario_config baseline.solver must be a nonempty mapping")
    baseline_model = dict(raw_baseline_model)
    missing_model_keys = _REQUIRED_BASELINE_MODEL_KEYS - set(baseline_model)
    if missing_model_keys:
        raise ValueError(
            "scenario_config baseline.model is missing required selected-main keys: "
            + ", ".join(sorted(missing_model_keys))
        )
    baseline_solver = _normalise_solver_contract(
        raw_baseline_solver, context="scenario_config baseline.solver"
    )
    missing_solver_keys = _REQUIRED_BASELINE_SOLVER_KEYS - set(baseline_solver)
    if missing_solver_keys:
        raise ValueError(
            "scenario_config baseline.solver is missing required selected-main keys: "
            + ", ".join(sorted(missing_solver_keys))
        )

    ledger_by_run: dict[str, list[Mapping[str, str]]] = {}
    for row in ledger_rows:
        ledger_by_run.setdefault(str(row["run_path"]), []).append(row)
    run_configs: dict[str, dict[str, Any]] = {}
    instance_hash_cache: dict[Path, dict[str, str]] = {}
    for item in evidence:
        if not item.rows:
            continue
        run_path = item.role.run_path
        if run_path not in run_configs:
            run_config_path = experiment_root / run_path / "run_config.json"
            run_config = _read_object(run_config_path)
            if (
                run_config.get("schema_version") != RESULT_SCHEMA_VERSION
                or run_config.get("formulation_id") != FORMULATION_ID
            ):
                raise ValueError(f"incompatible run_config: {run_config_path}")
            run_configs[run_path] = run_config

    selected_candidates = [
        item
        for item in evidence
        if item.role.category == "main"
        and item.role.role in {"selected", "selected_main"}
    ]
    if len(selected_candidates) != 1:
        raise ValueError("evidence requires exactly one selected main role")
    selected_item = selected_candidates[0]
    selected_config = run_configs.get(selected_item.role.run_path)
    baseline_implementation = (
        _nested_value(selected_config, ("implementation", "sha256"))
        if selected_config is not None
        else None
    )
    required_implementation = configured_implementation or baseline_implementation
    selected_input_hash = (
        _payload_sha256(
            _as_mapping(
                _nested_value(selected_config, ("instance", "canonical_file_sha256"))
            )
        )
        if selected_config is not None
        else None
    )

    revised: list[RoleEvidence] = []
    for item in evidence:
        if not item.rows or item.status in {"pending", "invalid"}:
            revised.append(item)
            continue
        role = item.role
        run_config = run_configs[role.run_path]
        errors: list[str] = []
        warnings: list[str] = []
        all_run_rows = ledger_by_run[role.run_path]
        configured_modes = run_config.get("modes")
        ledger_modes = [str(row["mode"]) for row in all_run_rows]
        if not isinstance(configured_modes, list) or set(configured_modes) != set(
            ledger_modes
        ):
            errors.append("run_config.modes 与完整账本模式不一致")
        if not set(role.modes).issubset(set(ledger_modes)):
            errors.append("角色 modes 不属于该 run 的实际模式")
        demand_scale = _finite(run_config.get("demand_scale"))
        if demand_scale is None or not math.isclose(
            demand_scale, required_demand_scale, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("run_config demand_scale 与实例族契约不一致")
        nested_scale = _as_mapping(
            _nested_value(run_config, ("instance", "metadata", "nested_scale"))
        )
        if nested_scale.get("family_id") != required_family:
            errors.append("run_config scale family 与注册表不一致")
        for row in item.rows:
            if not _contract_equal(row.get("demand_scale"), required_demand_scale):
                errors.append(f"{row.get('mode')} ledger demand_scale 不一致")
            if row.get("scale_family_id") != required_family:
                errors.append(f"{row.get('mode')} ledger scale_family_id 不一致")

        if role.instance_dir is None:
            errors.append("角色未显式声明 instance_dir")
            declared_hashes = None
        else:
            declared_hashes, hash_errors = _canonical_instance_contract(
                role.instance_dir, run_config, instance_hash_cache
            )
            errors.extend(hash_errors)
        if declared_hashes is not None:
            input_hash = _payload_sha256(declared_hashes)
            for row in item.rows:
                if row.get("input_sha256") != input_hash:
                    errors.append(f"{row.get('mode')} ledger input_sha256 不一致")
        else:
            input_hash = None

        model = _as_mapping(run_config.get("model"))
        solver = _as_mapping(run_config.get("solver"))
        actual_git_head = _nested_value(run_config, ("git", "head"))
        actual_implementation = _nested_value(run_config, ("implementation", "sha256"))
        for row in item.rows:
            mode = str(row.get("mode"))
            if row.get("git_head") != actual_git_head:
                errors.append(f"{mode} ledger git_head 与 run_config 不一致")
            if row.get("implementation_sha256") != actual_implementation:
                errors.append(
                    f"{mode} ledger implementation_sha256 与 run_config 不一致"
                )
            exact_map = _as_mapping(run_config.get("screening_is_exact"))
            if _boolean(row.get("screening_is_exact")) is not _boolean(
                exact_map.get(mode)
            ):
                errors.append(f"{mode} screening_is_exact 与 run_config 不一致")
            if row.get("model_parameters_sha256") != _payload_sha256(model):
                errors.append(f"{mode} ledger model_parameters_sha256 不一致")
            if row.get("solver_parameters_sha256") != _payload_sha256(solver):
                errors.append(f"{mode} ledger solver_parameters_sha256 不一致")

        actual_h = _integer(
            _nested_value(
                run_config,
                ("instance", "metadata", "nested_scale", "facility_actual_counts", "H"),
            )
        )
        for row in item.rows:
            if _integer(row.get("candidate_lockers")) != actual_h:
                errors.append(f"{row.get('mode')} ledger candidate_lockers 不一致")

        if role.category != "algorithm":
            if actual_implementation != required_implementation:
                errors.append("正式角色 implementation hash 与正式实现契约不一致")
            if (
                role.category == "main"
                and role.role in {"selected", "selected_main"}
                and required_commit is not None
                and actual_git_head != required_commit
            ):
                errors.append(
                    "selected main git commit 与 repository_commit_required 不一致"
                )
            elif (
                required_commit is not None
                and actual_git_head != required_commit
                and actual_implementation == required_implementation
            ):
                warnings.append(
                    "git_head 与 selected main 不同，但求解 implementation hash 完全一致"
                )

        declared_model_overrides = _as_mapping(role.contract.get("model_overrides"))
        declared_solver_overrides = _as_mapping(role.contract.get("solver_overrides"))
        errors.extend(_validate_solver_overrides(solver, declared_solver_overrides))

        if role.category == "main" and role.role in {"selected", "selected_main"}:
            if not _contract_equal(model, baseline_model):
                errors.append(
                    "selected main run_config.model 与 scenario_config baseline.model 不一致"
                )
            errors.extend(
                _solver_contract_errors(
                    solver,
                    baseline_solver,
                    context=(
                        "selected main run_config.solver 与 "
                        "scenario_config baseline.solver 不一致"
                    ),
                )
            )

        if role.category != "algorithm":
            expected_model = _expected_model(
                baseline_model, declared_model_overrides
            )
            if not _contract_equal(model, expected_model):
                errors.append(
                    "正式角色 run_config.model 未继承 baseline.model/显式 overrides"
                )
            # Per-role wall-clock budgets are allowed to differ.  They become a
            # checked contract only when the role explicitly declares one.
            expected_solver = {
                key: value
                for key, value in baseline_solver.items()
                if key != "time_limit"
            }
            expected_solver.update(
                _normalise_solver_contract(
                    declared_solver_overrides,
                    context=f"{role.scenario_id} solver_overrides",
                )
            )
            errors.extend(
                _solver_contract_errors(
                    solver,
                    expected_solver,
                    context=(
                        "正式角色 run_config.solver 未继承 "
                        "baseline.solver/显式 overrides"
                    ),
                )
            )

        if role.category == "scale":
            scale = _as_mapping(role.contract.get("scale"))
            if not scale or _integer(scale.get("H")) is None:
                errors.append("规模角色必须显式声明 scale.H")
            else:
                expected_h = _integer(scale.get("H"))
                if expected_h != actual_h or _integer(role.x_value) != actual_h:
                    errors.append("规模角色 H 与实际 candidate_lockers 不一致")
                facility_counts = _as_mapping(
                    nested_scale.get("facility_actual_counts")
                )
                for site_type in ("H", "L", "E"):
                    if _integer(scale.get(site_type)) != _integer(
                        facility_counts.get(site_type)
                    ):
                        errors.append(f"规模角色 {site_type} 数与实例不一致")
                for declared_key, actual_key in (
                    ("crop_side_km", "crop_side_km"),
                    ("crop_area_km2", "crop_area_km2"),
                ):
                    if not _contract_equal(
                        nested_scale.get(actual_key), scale.get(declared_key)
                    ):
                        errors.append(f"规模角色 {declared_key} 与实例不一致")
        elif role.category == "algorithm":
            required_algorithm_keys = {
                "miqcp_method",
                "seed",
                "threads",
                "time_limit_seconds",
            }
            if not required_algorithm_keys.issubset(declared_solver_overrides):
                errors.append(
                    "算法角色 solver_overrides 必须显式声明 miqcp_method/seed/threads/time_limit_seconds"
                )
            if baseline_model and not _contract_equal(model, baseline_model):
                errors.append("算法先导试验模型参数与 selected main 基线不一致")
            if selected_input_hash is not None and input_hash != selected_input_hash:
                errors.append("算法先导试验实例与 selected main 不一致")
            for row in item.rows:
                if not _contract_equal(
                    row.get("solver_miqcp_method"),
                    declared_solver_overrides.get("miqcp_method"),
                ):
                    errors.append("算法角色 MIQCPMethod 与 ledger 不一致")
                for field, key in (
                    ("solver_seed", "seed"),
                    ("solver_threads", "threads"),
                    ("solver_time_limit_seconds", "time_limit_seconds"),
                ):
                    if not _contract_equal(
                        row.get(field), declared_solver_overrides.get(key)
                    ):
                        errors.append(f"算法角色 {field} 与声明不一致")

        elif role.category == "exactness":
            if role.role not in {"screened", "complete"}:
                errors.append("exactness 角色必须声明 role=screened/complete")
            expected_exact = role.role == "complete"
            if any(
                _boolean(row.get("screening_is_exact")) is not expected_exact
                for row in item.rows
            ):
                errors.append("exactness 角色与 screening_is_exact 语义冲突")
            if expected_exact and {
                "display_origin_choices",
                "general_corridor_choices",
            } - set(declared_model_overrides):
                errors.append("complete-index 必须显式声明两个 all/all model overrides")
            if role.role == "screened" and isinstance(role.x_value, str):
                parts = role.x_value.split("/")
                if len(parts) == 2 and all(part.isdigit() for part in parts):
                    if not _contract_equal(
                        model.get("display_origin_choices"), int(parts[0])
                    ):
                        errors.append(
                            "screened display_origin_choices 与 x_value 不一致"
                        )
                    if not _contract_equal(
                        model.get("general_corridor_choices"), int(parts[1])
                    ):
                        errors.append(
                            "screened general_corridor_choices 与 x_value 不一致"
                        )
        elif role.category == "sensitivity":
            if not isinstance(role.group, str) or role.x_value is None:
                errors.append("敏感性角色必须声明 group 与 x_value/factor")

        if errors:
            revised.append(
                RoleEvidence(
                    role,
                    "invalid",
                    "角色契约失败：" + "；".join(dict.fromkeys(errors)),
                    item.rows,
                )
            )
        else:
            revised.append(
                RoleEvidence(
                    role,
                    item.status,
                    item.note
                    + ("；" + "；".join(dict.fromkeys(warnings)) if warnings else ""),
                    item.rows,
                )
            )

    selected_revised = next(
        item
        for item in revised
        if item.role.category == "main"
        and item.role.role in {"selected", "selected_main"}
    )
    selected_versions = {
        (row.get("git_head"), row.get("implementation_sha256"))
        for row in selected_revised.rows
    }
    final: list[RoleEvidence] = []
    for item in revised:
        if item.status != "pilot":
            final.append(item)
            continue
        pilot_versions = {
            (row.get("git_head"), row.get("implementation_sha256")) for row in item.rows
        }
        if pilot_versions != selected_versions:
            final.append(
                RoleEvidence(
                    item.role,
                    item.status,
                    item.note
                    + "；实现提交或实现哈希与 selected main 不一致，因此仅作开发期算法证据",
                    item.rows,
                )
            )
        else:
            final.append(item)
    return final, run_configs


def _evidence_for_roles(
    roles: Sequence[ScenarioRole], ledger_rows: Sequence[Mapping[str, str]]
) -> list[RoleEvidence]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for raw in ledger_rows:
        run_path = raw.get("run_path")
        if isinstance(run_path, str):
            by_run.setdefault(run_path, []).append(dict(raw))
    evidence: list[RoleEvidence] = []
    for role in roles:
        rows = sorted(
            by_run.get(role.run_path, []), key=lambda row: str(row.get("mode"))
        )
        if not rows:
            note = "账本中尚无完成记录"
            if role.declared_status:
                note += f"（配置状态：{role.declared_status}）"
            evidence.append(RoleEvidence(role, "pending", note, ()))
            continue
        present_modes = {str(row.get("mode")) for row in rows}
        requested_modes = set(role.modes)
        if requested_modes and not requested_modes.issubset(present_modes):
            missing = ",".join(sorted(requested_modes - present_modes))
            evidence.append(
                RoleEvidence(role, "pending", f"账本缺少模式：{missing}", tuple(rows))
            )
            continue
        selected_rows = [
            row
            for row in rows
            if not requested_modes or row.get("mode") in requested_modes
        ]
        failed = [
            str(row.get("mode"))
            for row in selected_rows
            if _boolean(row.get("validation_pass")) is not True
        ]
        if failed:
            evidence.append(
                RoleEvidence(
                    role,
                    "invalid",
                    "独立验证未通过或缺失：" + ",".join(failed),
                    tuple(selected_rows),
                )
            )
            continue
        status = "pilot" if role.category == "algorithm" else "ready"
        note = (
            "算法先导试验：已进入密封账本且独立验证通过；不与正式管理结果混为同一基线"
            if status == "pilot"
            else "已进入密封账本且独立验证通过"
        )
        evidence.append(RoleEvidence(role, status, note, tuple(selected_rows)))

    return evidence


def _mode_row(item: RoleEvidence, mode: str = "MX") -> dict[str, Any] | None:
    return next((row for row in item.rows if row.get("mode") == mode), None)


def _row_number(row: Mapping[str, Any] | None, key: str) -> float | None:
    return _finite(row.get(key)) if row is not None else None


def _role_table_rows(evidence: Sequence[RoleEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evidence:
        if not item.rows:
            rows.append(
                {
                    "实验类别": ROLE_NAMES[item.role.category],
                    "情景ID": item.role.scenario_id,
                    "标签": item.role.label,
                    "模式": ",".join(item.role.modes) or "—",
                    "横轴值": (
                        item.role.x_value if item.role.x_value is not None else "—"
                    ),
                    "分组": item.role.group or item.role.role or "—",
                    "证据状态": item.status,
                    "目标值CNY每小时": "",
                    "下界CNY每小时": "",
                    "Gap百分比": "",
                    "运行时间s": "",
                    "筛选是否精确": "",
                    "验证通过": "",
                    "说明": item.note,
                    "结果路径": item.role.run_path,
                }
            )
            continue
        for row in item.rows:
            gap = _row_number(row, "mip_gap_fraction")
            rows.append(
                {
                    "实验类别": ROLE_NAMES[item.role.category],
                    "情景ID": item.role.scenario_id,
                    "标签": item.role.label,
                    "模式": row.get("mode", ""),
                    "横轴值": (
                        item.role.x_value if item.role.x_value is not None else "—"
                    ),
                    "分组": item.role.group or item.role.role or "—",
                    "证据状态": item.status,
                    "目标值CNY每小时": row.get("objective_cny_per_hour", ""),
                    "下界CNY每小时": row.get("objective_bound_cny_per_hour", ""),
                    "Gap百分比": 100.0 * gap if gap is not None else "",
                    "运行时间s": row.get("runtime_seconds", ""),
                    "筛选是否精确": row.get("screening_is_exact", ""),
                    "验证通过": row.get("validation_pass", ""),
                    "说明": item.note,
                    "结果路径": item.role.run_path,
                }
            )
    return rows


def _style_axes(axis: plt.Axes) -> None:
    axis.set_facecolor(PALETTE["paper"])
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#94A3B8")
    axis.tick_params(colors=PALETTE["muted"], labelsize=8.5)
    axis.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, zorder=0)


def _save_figure(figure: plt.Figure, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    figure.savefig(png, dpi=230, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, svg


def _pending_figure(stem: Path, title: str, note: str) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(8.4, 4.5))
    axis.axis("off")
    figure.patch.set_facecolor("white")
    axis.text(
        0.04,
        0.80,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color=PALETTE["ink"],
    )
    axis.text(
        0.04,
        0.52,
        "PENDING",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=PALETTE["gold"],
    )
    axis.text(
        0.04,
        0.32,
        note,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=PALETTE["muted"],
        wrap=True,
    )
    return _save_figure(figure, stem)


def _plot_scale_capability(
    evidence: Sequence[RoleEvidence], stem: Path
) -> tuple[Path, Path]:
    items: list[tuple[float, float, float, bool]] = []
    for item in evidence:
        if item.role.category != "scale" or not item.ready:
            continue
        row = _mode_row(item, "MX")
        x = _finite(item.role.x_value)
        gap = _row_number(row, "mip_gap_fraction")
        runtime = _row_number(row, "runtime_seconds")
        exact = _boolean(row.get("screening_is_exact")) if row else None
        if x is not None and gap is not None and runtime is not None:
            items.append((x, gap, runtime, exact is True))
    if not items:
        return _pending_figure(stem, "MX 规模求解能力", "尚无通过独立验证的规模实验。")
    items.sort()
    x = np.array([item[0] for item in items])
    gap = np.array([max(item[1], 1e-6) for item in items])
    runtime = np.array([item[2] for item in items])
    exact = [item[3] for item in items]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.5))
    _style_axes(left)
    left.plot(x, gap * 100.0, color=PALETTE["blue"], marker="o", linewidth=2.0)
    left.axhline(1.0, color=PALETTE["gold"], linestyle="--", linewidth=1.4)
    left.set_yscale("log")
    left.set_xlabel("候选 H 数", color=PALETTE["ink"])
    left.set_ylabel("最终 Gap（% · 对数轴）", color=PALETTE["ink"])
    left.set_title("MX 最优性证书", loc="left", fontsize=11, color=PALETTE["ink"])
    for xi, yi, is_exact in zip(x, gap * 100.0, exact):
        left.annotate(
            f"{yi:.2f}%" + (" · 完整" if is_exact else " · 筛选"),
            (xi, yi),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
            color=PALETTE["muted"],
        )
    _style_axes(right)
    right.bar(
        x.astype(str), runtime, color=PALETTE["blue_open"], edgecolor=PALETTE["blue"]
    )
    right.set_xlabel("候选 H 数", color=PALETTE["ink"])
    right.set_ylabel("运行时间（s）", color=PALETTE["ink"])
    right.set_title(
        "同一规模实验预算下的实际用时", loc="left", fontsize=11, color=PALETTE["ink"]
    )
    figure.suptitle(
        "MX 规模求解能力",
        x=0.07,
        y=1.02,
        ha="left",
        fontsize=15,
        fontweight="semibold",
        color=PALETTE["ink"],
    )
    figure.text(
        0.07,
        -0.02,
        "虚线为 1% 目标；‘筛选’点的 Gap 只证实预设筛选模型。",
        color=PALETTE["muted"],
        fontsize=8,
    )
    figure.tight_layout()
    return _save_figure(figure, stem)


def _plot_algorithm(evidence: Sequence[RoleEvidence], stem: Path) -> tuple[Path, Path]:
    items: list[tuple[str, float, float]] = []
    for item in evidence:
        if item.role.category != "algorithm" or not item.ready:
            continue
        row = _mode_row(item, "MX")
        gap = _row_number(row, "mip_gap_fraction")
        runtime = _row_number(row, "runtime_seconds")
        if gap is not None and runtime is not None:
            items.append((item.role.label, gap, runtime))
    if not items:
        return _pending_figure(
            stem, "MIQCP 方法对照", "尚无通过独立验证的匹配预算方法实验。"
        )
    labels, gaps, runtimes = zip(*items)
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    _style_axes(axis)
    bars = axis.bar(
        np.arange(len(labels)),
        np.array(gaps) * 100.0,
        color=PALETTE["blue_open"],
        edgecolor=PALETTE["blue"],
        linewidth=1.2,
    )
    axis.axhline(1.0, color=PALETTE["gold"], linestyle="--", linewidth=1.4)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylabel("最终 Gap（%）", color=PALETTE["ink"])
    figure.suptitle(
        "MIQCP 方法对照",
        x=0.125,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="semibold",
        color=PALETTE["ink"],
    )
    figure.text(
        0.125,
        0.925,
        "相同实例、线程、种子与时限；柱上同时标注实际用时。",
        color=PALETTE["muted"],
        fontsize=8.5,
    )
    for bar, gap, runtime in zip(bars, gaps, runtimes):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{100 * gap:.2f}%\n{runtime:.0f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            color=PALETTE["ink"],
        )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    return _save_figure(figure, stem)


def _plot_exactness(evidence: Sequence[RoleEvidence], stem: Path) -> tuple[Path, Path]:
    items: list[tuple[str, float, bool]] = []
    for item in evidence:
        if item.role.category != "exactness" or not item.ready:
            continue
        row = _mode_row(item, "MX")
        gap = _row_number(row, "mip_gap_fraction")
        exact = _boolean(row.get("screening_is_exact")) if row else None
        if gap is not None:
            items.append((item.role.label, gap, exact is True))
    if not items:
        return _pending_figure(
            stem, "完整索引与筛选模型对照", "尚无通过独立验证的完整/筛选对照。"
        )
    labels = [item[0] for item in items]
    gaps = [100.0 * item[1] for item in items]
    colors = [PALETTE["blue"] if item[2] else PALETTE["blue_open"] for item in items]
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    _style_axes(axis)
    bars = axis.bar(
        np.arange(len(items)), gaps, color=colors, edgecolor=PALETTE["blue"]
    )
    axis.axhline(1.0, color=PALETTE["gold"], linestyle="--", linewidth=1.4)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylabel("最终 Gap（%）", color=PALETTE["ink"])
    figure.suptitle(
        "完整索引与筛选模型的求解证书",
        x=0.125,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="semibold",
        color=PALETTE["ink"],
    )
    figure.text(
        0.125,
        0.925,
        "实色为 complete-index；两种模型的目标值不被当作无条件可比的利润结果。",
        color=PALETTE["muted"],
        fontsize=8.5,
    )
    for bar, gap in zip(bars, gaps):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{gap:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color=PALETTE["ink"],
        )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    return _save_figure(figure, stem)


def _plot_sensitivity(
    evidence: Sequence[RoleEvidence], stem: Path
) -> tuple[Path, Path]:
    groups: dict[str, list[tuple[float, float, float | None, bool]]] = {}
    for item in evidence:
        if item.role.category != "sensitivity" or not item.ready:
            continue
        row = _mode_row(item, "MX")
        x = _finite(item.role.x_value)
        gap = _row_number(row, "mip_gap_fraction")
        saving = _row_number(row, "cost_saving_percent")
        if x is None or gap is None:
            continue
        decision_ready = gap <= _ONE_PERCENT
        groups.setdefault(item.role.group or "sensitivity", []).append(
            (x, gap, saving, decision_ready)
        )
    if not groups:
        return _pending_figure(stem, "敏感性实验", "尚无通过独立验证的敏感性结果。")
    names = sorted(groups)
    figure, axes = plt.subplots(
        len(names), 2, figsize=(11.4, 3.5 * len(names)), squeeze=False
    )
    for row_axes, name in zip(axes, names):
        values = sorted(groups[name])
        x = np.array([value[0] for value in values])
        gaps = np.array([100.0 * value[1] for value in values])
        savings = np.array(
            [np.nan if value[2] is None else value[2] for value in values]
        )
        ready = np.array([value[3] for value in values])
        left, right = row_axes
        _style_axes(left)
        left.plot(x, gaps, color=PALETTE["blue"], marker="o", linewidth=1.8)
        left.axhline(1.0, color=PALETTE["gold"], linestyle="--", linewidth=1.2)
        left.set_title(
            f"{name} · 求解 Gap", loc="left", fontsize=10.5, color=PALETTE["ink"]
        )
        left.set_xlabel("场景水平", color=PALETTE["ink"])
        left.set_ylabel("Gap（%）", color=PALETTE["ink"])
        _style_axes(right)
        if ready.any():
            right.plot(
                x[ready],
                savings[ready],
                color=PALETTE["olive"],
                marker="o",
                linewidth=1.8,
            )
        unresolved = ~ready
        if unresolved.any():
            right.scatter(
                x[unresolved],
                np.zeros(unresolved.sum()),
                color=PALETTE["gold"],
                marker="x",
                s=55,
                label="Gap>1%，不比较点估计",
            )
            right.legend(frameon=False, fontsize=7.5, loc="best")
        right.set_title(
            f"{name} · 可决策成本节约", loc="left", fontsize=10.5, color=PALETTE["ink"]
        )
        right.set_xlabel("场景水平", color=PALETTE["ink"])
        right.set_ylabel("相对全骑手基线的节约（%）", color=PALETTE["ink"])
    figure.suptitle(
        "敏感性结果与求解质量",
        x=0.06,
        y=1.01,
        ha="left",
        fontsize=15,
        fontweight="semibold",
        color=PALETTE["ink"],
    )
    figure.text(
        0.06,
        -0.005,
        "只有 Gap≤1% 且独立验证通过的情景才显示成本点估计；成本节约不是收入或利润。",
        color=PALETTE["muted"],
        fontsize=8,
    )
    figure.tight_layout()
    return _save_figure(figure, stem)


def _choice_parameters(
    config: Mapping[str, Any],
    selected: RoleEvidence | None,
    experiment_root: Path,
    scenario_path: Path,
) -> tuple[
    dict[str, float] | None,
    list[float],
    list[float],
    ProvenanceSource,
]:
    choice_config = config.get("choice_curve")
    curve = choice_config if isinstance(choice_config, Mapping) else {}
    raw_parameters = curve.get("parameters")
    source = ProvenanceSource(scenario_path, "choice_curve.parameters")
    parameters: Mapping[str, Any] | None = (
        raw_parameters if isinstance(raw_parameters, Mapping) else None
    )
    if parameters is None and selected is not None:
        run_config_path = experiment_root / selected.role.run_path / "run_config.json"
        if run_config_path.is_file():
            run_config = _read_object(run_config_path)
            if (
                run_config.get("schema_version") == RESULT_SCHEMA_VERSION
                and run_config.get("formulation_id") == FORMULATION_ID
            ):
                model = run_config.get("model")
                if isinstance(model, Mapping) and isinstance(
                    model.get("choice"), Mapping
                ):
                    parameters = model["choice"]
                    source = ProvenanceSource(run_config_path, "model.choice")
    required = (
        "locker_asc",
        "delivery_time_coefficient",
        "price_coefficient",
        "walk_distance_coefficient",
        "home_eta_minutes",
        "locker_eta_minutes",
        "locker_discount_cny",
        "max_walk_km",
    )
    parsed: dict[str, float] | None = None
    if parameters is not None:
        values = {key: _finite(parameters.get(key)) for key in required}
        if all(value is not None for value in values.values()):
            parsed = {
                key: float(value) for key, value in values.items() if value is not None
            }
    raw_distances = curve.get("distances_km")
    if isinstance(raw_distances, list) and raw_distances:
        distances = [
            value for raw in raw_distances if (value := _finite(raw)) is not None
        ]
    elif parsed is not None:
        distances = list(np.linspace(0.0, parsed["max_walk_km"], 31))
    else:
        distances = []
    raw_discounts = curve.get("discounts_cny")
    if isinstance(raw_discounts, list) and raw_discounts:
        discounts = [
            value for raw in raw_discounts if (value := _finite(raw)) is not None
        ]
    elif parsed is not None:
        discounts = [parsed["locker_discount_cny"]]
    else:
        discounts = []
    return parsed, sorted(set(distances)), sorted(set(discounts)), source


def _choice_curve_rows(
    parameters: Mapping[str, float] | None,
    distances: Sequence[float],
    discounts: Sequence[float],
) -> list[dict[str, Any]]:
    if parameters is None:
        return []
    rows: list[dict[str, Any]] = []
    for discount in discounts:
        for distance in distances:
            utility = (
                parameters["locker_asc"]
                + parameters["delivery_time_coefficient"]
                * (parameters["home_eta_minutes"] - parameters["locker_eta_minutes"])
                + parameters["price_coefficient"] * discount
                - parameters["walk_distance_coefficient"] * distance
            )
            share = 1.0 / (1.0 + math.exp(-utility))
            rows.append(
                {
                    "步行距离km": distance,
                    "机柜优惠CNY": discount,
                    "到柜选择比例": share,
                    "到柜选择百分比": 100.0 * share,
                    "是否在模型覆盖范围": distance <= parameters["max_walk_km"] + 1e-12,
                }
            )
    return rows


def _plot_choice_curve(
    rows: Sequence[Mapping[str, Any]], stem: Path
) -> tuple[Path, Path]:
    if not rows:
        return _pending_figure(
            stem, "步行距离与到柜选择比例", "缺少显式用户选择参数或距离/优惠网格。"
        )
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    _style_axes(axis)
    discounts = sorted({float(row["机柜优惠CNY"]) for row in rows})
    colors = [PALETTE["blue"], PALETTE["gold"], PALETTE["olive"], PALETTE["pink"]]
    linestyles = ["-", "--", "-.", ":"]
    for index, discount in enumerate(discounts):
        subset = [row for row in rows if float(row["机柜优惠CNY"]) == discount]
        subset.sort(key=lambda row: float(row["步行距离km"]))
        axis.plot(
            [float(row["步行距离km"]) for row in subset],
            [float(row["到柜选择百分比"]) for row in subset],
            label=f"优惠 {discount:g} CNY",
            color=colors[index % len(colors)],
            linestyle=linestyles[index % len(linestyles)],
            marker="o" if len(subset) <= 8 else None,
            linewidth=2.0,
        )
    axis.set_ylim(bottom=0)
    axis.set_xlabel("顾客到机柜的步行距离（km）", color=PALETTE["ink"])
    axis.set_ylabel("选择到柜取餐的订单比例（%）", color=PALETTE["ink"])
    figure.suptitle(
        "步行距离与到柜选择比例",
        x=0.125,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="semibold",
        color=PALETTE["ink"],
    )
    figure.text(
        0.125,
        0.925,
        "由当前未校准二项 Logit 场景参数计算；不是企业真实转化率预测。",
        color=PALETTE["muted"],
        fontsize=8.5,
    )
    axis.legend(frameon=False, fontsize=8.5, loc="best")
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    return _save_figure(figure, stem)


def _selected_main(evidence: Sequence[RoleEvidence]) -> RoleEvidence | None:
    selected = [
        item
        for item in evidence
        if item.role.category == "main"
        and item.role.role in {"selected", "selected_main"}
    ]
    if len(selected) > 1:
        raise ValueError("evidence contains more than one selected main role")
    return selected[0] if selected else None


def _verify_selected_map_artifacts(
    result_dir: Path, modes: Sequence[str]
) -> None:
    """Verify every result file the selected-map loader will consume."""

    completion_path = result_dir / "run_complete.json"
    completion = _read_object(completion_path)
    if completion.get("schema_version") != 1 or completion.get("completed") is not True:
        raise ValueError(f"selected map requires a sealed run_complete.json: {completion_path}")
    if completion.get("modes") != list(modes):
        raise ValueError(
            f"selected map modes disagree with run_complete.json: {completion_path}"
        )
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(
            f"selected map requires run_complete.json artifacts manifest: {completion_path}"
        )

    indexed: dict[str, list[Mapping[str, Any]]] = {}
    for record in artifacts:
        if not isinstance(record, Mapping):
            continue
        relative = record.get("path")
        if isinstance(relative, str):
            indexed.setdefault(relative, []).append(record)

    for mode in modes:
        for filename in _MAP_SOURCE_ARTIFACTS:
            relative = f"{mode}/{filename}"
            records = indexed.get(relative, [])
            if len(records) != 1:
                qualifier = "missing" if not records else "duplicated"
                raise ValueError(
                    f"selected map artifact manifest record is {qualifier}: {relative}"
                )
            record = records[0]
            expected_size = record.get("size_bytes")
            expected_hash = record.get("sha256")
            if type(expected_size) is not int or expected_size < 0:
                raise ValueError(
                    f"selected map artifact manifest has invalid size: {relative}"
                )
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(
                    f"selected map artifact manifest has invalid sha256: {relative}"
                )
            try:
                int(expected_hash, 16)
            except ValueError as exc:
                raise ValueError(
                    f"selected map artifact manifest has invalid sha256: {relative}"
                ) from exc
            artifact_path = (result_dir / mode / filename).resolve()
            try:
                artifact_path.relative_to(result_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"selected map artifact path escapes result directory: {relative}"
                ) from exc
            if not artifact_path.is_file():
                raise ValueError(f"selected map artifact is missing: {relative}")
            if artifact_path.stat().st_size != expected_size:
                raise ValueError(f"selected map artifact size mismatch: {relative}")
            if _sha256(artifact_path) != expected_hash:
                raise ValueError(f"selected map artifact hash mismatch: {relative}")


def _plot_selected_map(
    selected: RoleEvidence | None,
    experiment_root: Path,
    basemap_cache: Path | None,
    stem: Path,
) -> MapRender:
    if selected is None:
        paths = _pending_figure(
            stem, "南山主算例三模式网络", "scenario_config 未指定 selected main。"
        )
        return MapRender(*paths, "pending", "未指定主算例")
    if not selected.ready:
        paths = _pending_figure(stem, "南山主算例三模式网络", selected.note)
        return MapRender(*paths, "pending", selected.note)
    if selected.role.instance_dir is None:
        paths = _pending_figure(
            stem, "南山主算例三模式网络", "主算例未显式提供 instance_dir。"
        )
        return MapRender(*paths, "failed", "缺少 instance_dir")
    if set(selected.role.modes) != {"HD", "MX", "LO"}:
        paths = _pending_figure(
            stem, "南山主算例三模式网络", "主算例必须显式包含 HD、MX、LO。"
        )
        return MapRender(*paths, "failed", "主算例模式不完整")
    mx = _mode_row(selected, "MX")
    h = _integer(mx.get("candidate_lockers")) if mx else None
    if h is None:
        paths = _pending_figure(
            stem, "南山主算例三模式网络", "账本缺少 candidate_lockers。"
        )
        return MapRender(*paths, "failed", "缺少候选 H 数")
    result_dir = experiment_root / selected.role.run_path
    try:
        _verify_selected_map_artifacts(result_dir, selected.role.modes)
        case = geo._load_case(
            h,
            selected.role.instance_dir,
            result_dir,
            ("HD", "MX", "LO"),
        )
        bounds = geo._window_bounds(
            case.layout.metadata, case.layout.half_size_km, padding_km=0.8
        )
        cache = basemap_cache or (
            _repo_root() / "results/nanshan_current_report_schema5/basemap_cache"
        )
        basemap = geo.build_basemap(bounds, cache, zoom=geo.TILE_ZOOM)
        boundary = geo._load_nanshan_boundary(cache / "nanshan_R5664195.geojson")
        boundary_rings = geo._outer_rings(boundary)
        figure, axes = plt.subplots(1, 3, figsize=(18.2, 6.9))
        for axis, mode in zip(axes, ("HD", "MX", "LO")):
            geo._draw_case_panel(
                axis,
                case,
                mode,
                basemap,
                boundary_rings,
                compact=True,
            )
        figure.suptitle(
            f"南山 H{h} 主算例：三种服务方式的网络设计",
            x=0.035,
            y=0.992,
            ha="left",
            fontsize=16,
            fontweight="semibold",
            color=PALETTE["ink"],
        )
        figure.legend(
            handles=geo._legend_handles(),
            loc="lower center",
            ncol=6,
            frameon=False,
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.04),
        )
        figure.text(
            0.035,
            0.012,
            f"{geo._COORDINATE_NOTE} · {geo.BASEMAP_ATTRIBUTION}",
            fontsize=7.3,
            color=PALETTE["muted"],
        )
        figure.subplots_adjust(top=0.88, bottom=0.105, wspace=0.04)
        png, svg = _save_figure(figure, stem)
        return MapRender(png, svg, "ready", f"已生成 H{h} 三模式南山地图")
    except (OSError, ValueError, RuntimeError) as exc:
        paths = _pending_figure(
            stem, f"南山 H{h} 主算例三模式网络", f"地图生成失败：{exc}"
        )
        return MapRender(*paths, "failed", f"地图生成失败：{exc}")


def _largest_certified_scale(
    evidence: Sequence[RoleEvidence],
) -> tuple[int, float] | None:
    eligible: list[tuple[int, float]] = []
    for item in evidence:
        if item.role.category != "scale" or not item.ready:
            continue
        mx = _mode_row(item, "MX")
        h = _integer(item.role.x_value)
        gap = _row_number(mx, "mip_gap_fraction")
        if h is not None and gap is not None and gap <= _ONE_PERCENT:
            eligible.append((h, gap))
    return max(eligible) if eligible else None


def _fmt_number(value: Any, digits: int = 2) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:,.{digits}f}"


def _fmt_percent_fraction(value: Any, digits: int = 2) -> str:
    number = _finite(value)
    return "—" if number is None else f"{100.0 * number:.{digits}f}%"


def _markdown_report(
    output_dir: Path,
    evidence: Sequence[RoleEvidence],
    choice_rows: Sequence[Mapping[str, Any]],
    choice_source: ProvenanceSource,
    ledger_paths: Sequence[Path],
    map_render: MapRender,
    report_kind: str,
) -> Path:
    selected = _selected_main(evidence)
    mx = _mode_row(selected, "MX") if selected and selected.ready else None
    largest = _largest_certified_scale(evidence)
    formal_ready_count = sum(item.status == "ready" for item in evidence)
    pilot_count = sum(item.status == "pilot" for item in evidence)
    invalid_count = sum(item.status == "invalid" for item in evidence)
    pending_count = sum(item.status == "pending" for item in evidence)
    if mx is not None:
        h = _integer(mx.get("candidate_lockers"))
        gap = _row_number(mx, "mip_gap_fraction")
        main_sentence = (
            f"显式选定的 H{h} MX 主算例 Gap 为 {_fmt_percent_fraction(gap)}，"
            + (
                "达到 1% 目标。"
                if gap is not None and gap <= _ONE_PERCENT
                else "尚未达到 1% 目标。"
            )
        )
        screening = _boolean(mx.get("screening_is_exact"))
        certificate_sentence = (
            "该证书属于完整索引模型。"
            if screening is True
            else "该证书只属于预设筛选模型，不外推为未筛选完整模型的证书。"
        )
    else:
        main_sentence = "主算例尚未形成可用于论文结论的三模式验证结果。"
        certificate_sentence = "缺失或未通过验证的结果只列入 pending/invalid 清单。"
    if largest is None:
        scale_sentence = "当前账本尚无通过验证且 MX Gap≤1% 的规模点。"
    else:
        scale_sentence = f"在显式登记并已完成的规模点中，最大达标档为 H{largest[0]}（Gap {100 * largest[1]:.2f}%）。"

    report_title = (
        "MLND 南山论文数值实验报告"
        if report_kind == "final"
        else "MLND 南山数值实验进度报告"
    )
    registry_sentence = (
        "本报告由 final evidence registry 生成。"
        if report_kind == "final"
        else "当前注册表是 planning registry；本产物只汇报完成进度，不是最终论文证据包。"
    )
    try:
        choice_path = choice_source.path.relative_to(_repo_root()).as_posix()
    except ValueError:
        choice_path = str(choice_source.path)
    lines = [
        f"# {report_title}",
        "",
        "## 技术摘要",
        "",
        registry_sentence,
        f"{main_sentence}{certificate_sentence}",
        f"{scale_sentence}",
        f"本报告共读取 {len(evidence)} 个显式论文角色：{formal_ready_count} 个 ready、{pilot_count} 个 pilot、{pending_count} 个 pending、{invalid_count} 个 invalid。",
        "所有金额指标均为模型中的履约成本或相对全骑手反事实的成本节约（CNY/h），不是营业收入或利润。",
        "",
        "## 关键结果与图证",
        "",
        "### MX 规模求解能力",
        "",
        "![MX 规模求解能力](figures/scale_capability.png)",
        "",
        "图中 1% 线是求解质量目标；筛选模型和完整索引模型以账本字段 `screening_is_exact` 区分。",
        "",
        "### 同一南山算例下的三种服务方式",
        "",
        "![南山主算例三模式网络](figures/selected_three_modes_map.png)",
        "",
        f"地图状态：{map_render.status}；{map_render.note}",
        "",
        "### 求解方法与完整性检查",
        "",
        "![MIQCP 方法对照](figures/algorithm_comparison.png)",
        "",
        "![完整索引与筛选模型对照](figures/exactness_comparison.png)",
        "",
        "方法图只比较显式登记、匹配预算并通过独立验证的算法先导试验；若实现提交或实现哈希与主算例不同，只作为开发期方法证据，不与正式管理结果混为同一基线。完整/筛选图首先比较证书质量，不把两种索引集合下的目标值直接解释为利润差。",
        "",
        "### 敏感性与用户选择",
        "",
        "![敏感性结果](figures/sensitivity.png)",
        "",
        "敏感性情景只有在独立验证通过且 Gap≤1% 时才显示可比较的成本点估计；其余情景保留 Gap 信息并标为 unresolved。",
        "",
        "![用户选择距离曲线](figures/choice_distance_curve.png)",
        "",
        f"距离曲线由 `{choice_path}` 的 `{choice_source.locator}` 参数计算。它给出场景内到柜比例，不是由企业行为数据校准的真实转化率预测。",
        "",
        "## 范围、数据与指标定义",
        "",
        "- 论文角色只由 `scenario_config.json` 的显式路径确定；目录名称不参与推断。",
        "- `objective_cny_per_hour` 是模型履约成本，`objective_bound_cny_per_hour` 是同一最小化模型的有效下界。",
        "- `mip_gap_fraction` 来自密封结果并由独立验证器复算；Gap≤1% 才满足本轮 MX 目标。",
        "- `screening_is_exact=false` 时，Gap 只约束固定 top-k 候选集合下的筛选模型。",
        "- 规模变化必须来自地图裁剪；本报告不自行抽样、复制或缩放订单。",
        "",
        "## 实验设计与验证",
        "",
        "报告器核对 ledger manifest 中两个 CSV 及全部 source_files 的路径、大小和 SHA-256，并只把 `validation_pass=true` 且角色契约与真实 run_config 一致的请求模式标为 ready。选中算例还绑定五个规范实例文件的 SHA-256；主地图要求 HD、MX、LO 三种结果。所有图表都从本次账本动态计算，不沿用其他报告中的固化结论。",
        "",
        "## 局限、稳健性与未完成项",
        "",
    ]
    for item in (item for item in evidence if item.status == "pilot"):
        lines.append(
            f"- `{item.role.scenario_id}`（算法先导试验）：pilot；{item.note}。"
        )
    unresolved = [item for item in evidence if item.status in {"pending", "invalid"}]
    if unresolved:
        for item in unresolved:
            lines.append(
                f"- `{item.role.scenario_id}`（{ROLE_NAMES[item.role.category]}）：{item.status}；{item.note}。"
            )
    else:
        lines.append("- 所有非先导的显式登记角色均已进入密封账本并通过独立验证。")
    lines.extend(
        [
            "- 用户选择参数尚未用企业观测数据校准；优惠额当前影响效用，但不自动作为平台补贴成本计入目标。",
            "- 南山底图的坐标转换仅用于结构展示，不作为测绘级站点坐标。",
            "- Portable HTML 是不含静态南山地图的交互补充；包含全部地理图证的完整阅读版本是本 Markdown 与同目录 PNG/SVG。",
            "",
            "## 建议的下一步",
            "",
            "先完成 pending 角色并重新生成账本与本报告；只有 Gap≤1% 且独立验证通过的敏感性情景进入论文点估计比较。若主算例使用筛选，需保留 complete-index 小算例作为完整模型证书，且分别报告两类证书范围。",
            "",
            "## 进一步问题",
            "",
            "企业用户选择数据能否校准距离与优惠敏感度？当前候选规模之外是否需要更宽筛选集合的稳健性检查？这些问题会改变外部有效性，但不改变本报告对当前求解证书的陈述。",
            "",
            "## 证据文件",
            "",
        ]
    )
    for path in ledger_paths:
        try:
            relative = path.relative_to(_repo_root())
        except ValueError:
            relative = path
        lines.append(f"- `{relative}`（SHA-256 `{_sha256(path)}`）")
    if choice_rows:
        anchors = {}
        for row in choice_rows:
            distance = float(row["步行距离km"])
            if any(
                math.isclose(distance, target, abs_tol=1e-9)
                for target in (0.0, 0.5, 1.0, 1.5)
            ):
                anchors[(distance, float(row["机柜优惠CNY"]))] = float(
                    row["到柜选择百分比"]
                )
        if anchors:
            lines.extend(
                [
                    "",
                    "## 用户选择直观锚点",
                    "",
                    "|步行距离 km|优惠 CNY|到柜比例|",
                    "|---:|---:|---:|",
                ]
            )
            for (distance, discount), share in sorted(anchors.items()):
                lines.append(f"|{distance:g}|{discount:g}|{share:.2f}%|")
    path = output_dir / "paper_experiment_report_zh.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _artifact_dataset_rows(evidence: Sequence[RoleEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evidence:
        if not item.rows:
            rows.append(
                {
                    "category": item.role.category,
                    "scenario_id": item.role.scenario_id,
                    "label": item.role.label,
                    "run_path": item.role.run_path,
                    "mode": ",".join(item.role.modes),
                    "x_value": item.role.x_value,
                    "group": item.role.group or item.role.role,
                    "evidence_status": item.status,
                    "note": item.note,
                }
            )
            continue
        for raw in item.rows:
            row = {
                "category": item.role.category,
                "scenario_id": item.role.scenario_id,
                "label": item.role.label,
                "run_path": item.role.run_path,
                "mode": raw.get("mode"),
                "x_value": item.role.x_value,
                "group": item.role.group or item.role.role,
                "evidence_status": item.status,
                "note": item.note,
            }
            numeric_fields = (
                "candidate_lockers",
                "crop_side_km",
                "crop_area_km2",
                "demand_per_hour",
                "od_markets",
                "objective_cny_per_hour",
                "objective_bound_cny_per_hour",
                "mip_gap_fraction",
                "runtime_seconds",
                "cost_saving_percent",
                "drone_service_fraction",
                "open_bases",
                "open_lockers",
                "open_swap_stations",
                "fleet_size",
            )
            for field in numeric_fields:
                row[field] = _finite(raw.get(field))
            row["screening_is_exact"] = _boolean(raw.get("screening_is_exact"))
            row["validation_pass"] = _boolean(raw.get("validation_pass"))
            rows.append(row)
    return rows


def _portable_source_path(path: Path, output_dir: Path) -> str:
    """Return the actual source path relative to the portable artifact."""

    resolved = path.resolve()
    for base in (output_dir.resolve(), _repo_root().resolve()):
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return Path(os.path.relpath(resolved, output_dir.resolve())).as_posix()


def _write_artifact(
    output_dir: Path,
    evidence: Sequence[RoleEvidence],
    choice_rows: Sequence[Mapping[str, Any]],
    generated_at: str,
    *,
    scenario_path: Path,
    ledger_paths: Sequence[Path],
    choice_source: ProvenanceSource,
    result_snapshot_path: Path,
    choice_snapshot_path: Path,
    report_kind: str,
    map_render: MapRender,
) -> Path:
    result_rows = _artifact_dataset_rows(evidence)
    validated_rows = [
        row for row in result_rows if row["evidence_status"] in {"ready", "pilot"}
    ]
    ready_rows = [row for row in validated_rows if row["evidence_status"] == "ready"]
    selected = _selected_main(evidence)
    selected_mx = _mode_row(selected, "MX") if selected and selected.ready else None
    main_mx = (
        next(
            (
                row
                for row in ready_rows
                if selected is not None
                and row["category"] == "main"
                and row["scenario_id"] == selected.role.scenario_id
                and row["run_path"] == selected.role.run_path
                and row["mode"] == "MX"
            ),
            None,
        )
        if selected_mx is not None
        else None
    )
    headline = [
        {
            "selected_h": main_mx.get("candidate_lockers") if main_mx else None,
            "selected_mx_gap": main_mx.get("mip_gap_fraction") if main_mx else None,
            "selected_mx_runtime_seconds": (
                main_mx.get("runtime_seconds") if main_mx else None
            ),
            "ready_roles": sum(item.status == "ready" for item in evidence),
            "pilot_roles": sum(item.status == "pilot" for item in evidence),
            "pending_roles": sum(item.status == "pending" for item in evidence),
            "invalid_roles": sum(item.status == "invalid" for item in evidence),
        }
    ]
    choice_dataset = [
        {
            "walk_distance_km": float(row["步行距离km"]),
            "locker_discount_cny": float(row["机柜优惠CNY"]),
            "locker_choice_share": float(row["到柜选择比例"]),
            "within_model_coverage": bool(row["是否在模型覆盖范围"]),
        }
        for row in choice_rows
    ]
    evidence_path = _portable_source_path(result_snapshot_path, output_dir)
    choice_path = _portable_source_path(choice_snapshot_path, output_dir)
    registry_path = _portable_source_path(scenario_path, output_dir)
    choice_parameter_path = _portable_source_path(choice_source.path, output_dir)
    upstream = [scenario_path, *ledger_paths]
    upstream_text = "; ".join(
        f"{_portable_source_path(path, output_dir)} sha256={_sha256(path)}"
        for path in upstream
    )
    sources = [
        {
            "id": "evidence_snapshot",
            "label": "显式角色与密封账本连接快照",
            "path": evidence_path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": f"SELECT * FROM read_csv_auto('{evidence_path}', header=true)",
                "description": (
                    "该 CSV 是报告器按显式 run_path 连接 registry 与密封 ledger，"
                    f"再执行角色契约校验后的可复现快照。上游：{upstream_text}"
                ),
                "tables_used": [evidence_path, registry_path]
                + [_portable_source_path(path, output_dir) for path in ledger_paths],
                "filters": [
                    "ready 要求 validation_pass=true 且 run_config 角色契约全部一致",
                    "pilot 仅为算法开发证据",
                ],
                "metric_definitions": [
                    "Gap 为最小化模型 incumbent 与 bound 的相对差。",
                    "screening_is_exact=false 时只解释为筛选模型证书。",
                    "成本是 CNY/h 履约成本，不是收入或利润。",
                ],
            },
        },
        {
            "id": "scenario_registry",
            "label": "显式论文角色注册表",
            "path": registry_path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    f"SELECT * FROM read_json_auto('{registry_path}', "
                    "maximum_object_size=16777216)"
                ),
                "description": (
                    "角色完全由该配置路径决定，不从目录名称推断；"
                    f"sha256={_sha256(scenario_path)}。"
                ),
                "tables_used": [registry_path],
                "filters": [],
            },
        },
        {
            "id": "choice_snapshot",
            "label": "用户选择距离曲线快照",
            "path": choice_path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": f"SELECT * FROM read_csv_auto('{choice_path}', header=true)",
                "description": (
                    f"曲线由 {choice_parameter_path} 的 {choice_source.locator} 动态计算；"
                    f"参数源 sha256={_sha256(choice_source.path)}。"
                ),
                "tables_used": [choice_path, choice_parameter_path],
                "filters": [],
                "metric_definitions": [
                    "到柜比例来自未校准二项 Logit 场景，不是企业真实转化率预测。"
                ],
            },
        },
    ]
    cards = [
        {
            "id": "main_quality",
            "dataset": "headline",
            "sourceId": "evidence_snapshot",
            "description": "显式选中主算例的 MX 求解证书；空值表示 pending。",
            "metrics": [
                {"label": "候选 H", "field": "selected_h", "format": "number"},
                {"label": "MX Gap", "field": "selected_mx_gap", "format": "percent"},
                {
                    "label": "运行秒数",
                    "field": "selected_mx_runtime_seconds",
                    "format": "number",
                },
            ],
        },
        {
            "id": "evidence_state",
            "dataset": "headline",
            "sourceId": "evidence_snapshot",
            "description": "显式登记论文角色的证据状态。",
            "metrics": [
                {"label": "Ready", "field": "ready_roles", "format": "number"},
                {"label": "Pilot", "field": "pilot_roles", "format": "number"},
                {"label": "Pending", "field": "pending_roles", "format": "number"},
                {"label": "Invalid", "field": "invalid_roles", "format": "number"},
            ],
        },
    ]
    charts: list[dict[str, Any]] = []
    scale_ready = [
        row for row in ready_rows if row["category"] == "scale" and row["mode"] == "MX"
    ]
    if scale_ready:
        charts.append(
            {
                "id": "scale_gap",
                "title": "MX 规模实验 Gap",
                "subtitle": "候选 H 数与最终相对 Gap；1% 是本轮目标。",
                "intent": "uncertainty",
                "question": "随着显式地图裁剪规模增加，MX 求解证书如何变化？",
                "rationale": "按候选 H 排序的线点图展示离散规模下的证书变化。",
                "type": "line",
                "dataset": "scale_ready",
                "sourceId": "evidence_snapshot",
                "encodings": {
                    "x": {
                        "field": "x_value",
                        "type": "quantitative",
                        "label": "候选 H",
                    },
                    "y": {
                        "field": "mip_gap_fraction",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "Gap",
                    },
                    "tooltip": [
                        {
                            "field": "runtime_seconds",
                            "type": "quantitative",
                            "label": "运行时间",
                            "unit": "s",
                        },
                        {
                            "field": "screening_is_exact",
                            "type": "text",
                            "label": "完整索引",
                        },
                    ],
                },
                "referenceLines": [
                    {
                        "axis": "y",
                        "value": 0.01,
                        "label": "1% 目标",
                        "lineStyle": "dashed",
                        "color": "orange",
                    }
                ],
                "layout": "full",
            }
        )
    algorithm_pilot = [
        row
        for row in validated_rows
        if row["category"] == "algorithm" and row["mode"] == "MX"
    ]
    if algorithm_pilot:
        charts.append(
            {
                "id": "algorithm_gap",
                "title": "MIQCP 方法先导试验 Gap",
                "subtitle": "匹配预算的算法开发证据；不与正式管理结果视为同一实现基线。",
                "intent": "comparison",
                "question": "各显式登记 MIQCP 方法在匹配预算下的最终 Gap 如何？",
                "rationale": "方法是少量离散类别，柱形图支持直接比较。",
                "type": "bar",
                "dataset": "algorithm_pilot",
                "sourceId": "evidence_snapshot",
                "encodings": {
                    "x": {"field": "label", "type": "nominal", "label": "方法"},
                    "y": {
                        "field": "mip_gap_fraction",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "Gap",
                    },
                    "tooltip": [
                        {
                            "field": "runtime_seconds",
                            "type": "quantitative",
                            "label": "运行时间",
                            "unit": "s",
                        },
                        {"field": "note", "type": "text", "label": "证据范围"},
                    ],
                },
                "referenceLines": [
                    {
                        "axis": "y",
                        "value": 0.01,
                        "label": "1% 目标",
                        "lineStyle": "dashed",
                        "color": "orange",
                    }
                ],
                "layout": "full",
            }
        )
    exactness_ready = [
        row
        for row in ready_rows
        if row["category"] == "exactness" and row["mode"] == "MX"
    ]
    if exactness_ready:
        charts.append(
            {
                "id": "exactness_gap",
                "title": "完整索引与筛选模型 Gap",
                "subtitle": "证书范围由 screening_is_exact 区分；不把目标值解释为利润差。",
                "intent": "uncertainty",
                "question": "完整索引与筛选模型分别取得怎样的求解证书？",
                "rationale": "离散模型版本以同单位 Gap 柱形比较。",
                "type": "bar",
                "dataset": "exactness_ready",
                "sourceId": "evidence_snapshot",
                "encodings": {
                    "x": {"field": "label", "type": "nominal", "label": "索引版本"},
                    "y": {
                        "field": "mip_gap_fraction",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "Gap",
                    },
                    "tooltip": [
                        {
                            "field": "screening_is_exact",
                            "type": "text",
                            "label": "完整索引",
                        },
                        {
                            "field": "runtime_seconds",
                            "type": "quantitative",
                            "label": "运行时间",
                            "unit": "s",
                        },
                    ],
                },
                "referenceLines": [
                    {
                        "axis": "y",
                        "value": 0.01,
                        "label": "1% 目标",
                        "lineStyle": "dashed",
                        "color": "orange",
                    }
                ],
                "layout": "full",
            }
        )
    sensitivity_ready = [
        row
        for row in ready_rows
        if row["category"] == "sensitivity" and row["mode"] == "MX"
    ]
    if sensitivity_ready:
        charts.append(
            {
                "id": "sensitivity_gap",
                "title": "敏感性情景求解 Gap",
                "subtitle": "所有点均已独立验证；Gap>1% 的点不进入成本点估计排序。",
                "intent": "uncertainty",
                "question": "各敏感性情景是否达到 1% 的决策比较门槛？",
                "rationale": "场景水平有顺序，分组线点图突出各因素的求解质量。",
                "type": "line",
                "dataset": "sensitivity_ready",
                "sourceId": "evidence_snapshot",
                "encodings": {
                    "x": {
                        "field": "x_value",
                        "type": "quantitative",
                        "label": "场景水平",
                    },
                    "y": {
                        "field": "mip_gap_fraction",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "Gap",
                    },
                    "color": {
                        "field": "group",
                        "type": "nominal",
                        "label": "敏感性因素",
                    },
                    "tooltip": [
                        {
                            "field": "cost_saving_percent",
                            "type": "quantitative",
                            "label": "成本节约",
                            "unit": "%",
                        },
                        {
                            "field": "runtime_seconds",
                            "type": "quantitative",
                            "label": "运行时间",
                            "unit": "s",
                        },
                    ],
                },
                "referenceLines": [
                    {
                        "axis": "y",
                        "value": 0.01,
                        "label": "1% 门槛",
                        "lineStyle": "dashed",
                        "color": "orange",
                    }
                ],
                "legend": {"position": "bottom", "title": "敏感性因素"},
                "layout": "full",
            }
        )
    if choice_dataset:
        charts.append(
            {
                "id": "choice_distance",
                "title": "步行距离与到柜选择比例",
                "subtitle": "当前未校准二项 Logit 场景；不同机柜优惠水平。",
                "intent": "relationship",
                "question": "步行距离和优惠如何改变模型中的到柜比例？",
                "rationale": "距离为连续输入，优惠为有意义的情景分组。",
                "type": "line",
                "dataset": "choice_curve",
                "sourceId": "choice_snapshot",
                "encodings": {
                    "x": {
                        "field": "walk_distance_km",
                        "type": "quantitative",
                        "label": "步行距离 km",
                    },
                    "y": {
                        "field": "locker_choice_share",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "到柜比例",
                    },
                    "color": {
                        "field": "locker_discount_cny",
                        "type": "nominal",
                        "label": "优惠 CNY",
                    },
                },
                "legend": {"position": "bottom", "title": "机柜优惠"},
                "layout": "full",
            }
        )
    tables = [
        {
            "id": "role_status_table",
            "title": "显式论文角色与证据状态",
            "subtitle": "只有 ready 行进入定量结论；成本单位 CNY/h。",
            "dataset": "results",
            "sourceId": "evidence_snapshot",
            "defaultSort": {"field": "category", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "category", "label": "实验类别", "type": "text"},
                {"field": "scenario_id", "label": "情景", "type": "text"},
                {"field": "mode", "label": "模式", "type": "text"},
                {"field": "evidence_status", "label": "证据状态", "type": "text"},
                {
                    "field": "objective_cny_per_hour",
                    "label": "履约成本 CNY/h",
                    "format": "number",
                },
                {
                    "field": "objective_bound_cny_per_hour",
                    "label": "下界 CNY/h",
                    "format": "number",
                },
                {"field": "mip_gap_fraction", "label": "Gap", "format": "percent"},
                {"field": "runtime_seconds", "label": "时间 s", "format": "number"},
                {"field": "screening_is_exact", "label": "完整索引", "type": "text"},
                {"field": "note", "label": "说明", "type": "text"},
            ],
        }
    ]
    if not charts:
        raise ValueError(
            "no evidence-backed quantitative chart is available for the portable artifact"
        )
    report_title = (
        "MLND 南山论文数值实验报告"
        if report_kind == "final"
        else "MLND 南山数值实验进度报告"
    )
    blocks: list[dict[str, Any]] = [
        {
            "id": "title",
            "type": "markdown",
            "body": f"# {report_title}",
            "layout": "full",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "evidence_snapshot",
            "body": (
                "## 技术摘要\n\n"
                + (
                    "本产物来自 final evidence registry。"
                    if report_kind == "final"
                    else "本产物来自 planning registry，只表示实验进度，不是最终论文证据包。"
                )
                + "筛选运行的 Gap 仅证实筛选模型；"
                "金额是模型履约成本或成本节约，不是收入或利润。"
            ),
            "layout": "full",
        },
        {
            "id": "headline",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in cards],
            "layout": "full",
        },
        {
            "id": "findings",
            "type": "markdown",
            "sourceId": "evidence_snapshot",
            "body": "## 关键结果与证书\n\n图表和表格中的结论均由当前账本动态计算。",
            "layout": "full",
        },
        {
            "id": "geographic_scope",
            "type": "markdown",
            "body": (
                "## 地理图证\n\n"
                "本 portable HTML 是交互补充，不嵌入静态南山地图。"
                "包含三模式南山地图的完整阅读版本是同目录 Markdown 与 PNG/SVG。"
                f"当前地图状态：{map_render.status}；{map_render.note}。"
            ),
            "layout": "full",
        },
    ]
    for chart in charts:
        blocks.append(
            {
                "id": f"{chart['id']}_block",
                "type": "chart",
                "chartId": chart["id"],
                "layout": "full",
            }
        )
    blocks.extend(
        [
            {
                "id": "status_table",
                "type": "table",
                "tableId": "role_status_table",
                "layout": "full",
            },
            {
                "id": "methods",
                "type": "markdown",
                "sourceId": "scenario_registry",
                "body": (
                    "## 范围、方法与验证\n\n"
                    "run_path 显式连接论文角色与账本；validation_pass=true 才标记 ready。"
                    "Gap≤1% 是 MX 点估计进入决策比较的门槛。"
                ),
                "layout": "full",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "sourceId": "evidence_snapshot",
                "body": (
                    "## 局限与稳健性\n\n"
                    "未完成或验证失败的角色保持 pending/invalid。用户选择参数未用企业数据校准；"
                    "筛选模型和完整索引模型的证书范围分开报告。"
                ),
                "layout": "full",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": "## 建议的下一步\n\n补齐 pending 实验后重建 ledger，再运行同一报告命令；不得用未达 1% 的敏感性 incumbent 作精确排序。",
                "layout": "full",
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": "## 进一步问题\n\n用户行为参数能否由企业数据校准？更宽筛选集合是否改变主算例设计？",
                "layout": "full",
            },
        ]
    )
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": report_title,
            "description": (
                "显式情景注册、密封账本与角色契约校验下的数值实验"
                + ("技术报告。" if report_kind == "final" else "进度报告。")
            ),
            "generatedAt": generated_at,
            "blocks": blocks,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": (
                "ready"
                if report_kind == "final"
                and evidence
                and all(item.status in {"ready", "pilot"} for item in evidence)
                and map_render.status == "ready"
                else "partial"
            ),
            "datasets": {
                "headline": headline,
                "results": result_rows,
                "scale_ready": scale_ready,
                "algorithm_pilot": algorithm_pilot,
                "exactness_ready": exactness_ready,
                "sensitivity_ready": sensitivity_ready,
                "choice_curve": choice_dataset,
            },
        },
        "sources": sources,
    }
    if map_render.status != "ready":
        artifact["snapshot"]["accessIssues"] = [
            {
                "id": "selected_map_unavailable",
                "scope": "selected_map",
                "dataset": "selected_map",
                "message": map_render.note,
            }
        ]
    path = output_dir / "artifact.json"
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def write_paper_experiment_report(
    scenario_config: str | Path,
    output_dir: str | Path,
    *,
    basemap_cache: str | Path | None = None,
) -> ReportOutputs:
    """Generate all report artifacts from sealed, explicitly assigned evidence."""

    scenario_path = Path(scenario_config).expanduser().resolve()
    config = _read_object(scenario_path)
    if config.get("schema_version") != 1:
        raise ValueError("scenario_config schema_version must equal 1")
    registry_status = config.get("artifact_status")
    report_kind = _REGISTRY_STATUS.get(registry_status)
    if report_kind is None:
        allowed = ", ".join(sorted(_REGISTRY_STATUS))
        raise ValueError(f"scenario_config artifact_status must be one of: {allowed}")
    root_raw = config.get("experiment_root")
    if not isinstance(root_raw, str) or not root_raw:
        raise ValueError("scenario_config requires experiment_root")
    experiment_root = _resolve_repo_path(root_raw, base=scenario_path.parent)
    if not experiment_root.is_dir():
        raise ValueError(f"experiment_root is not a directory: {experiment_root}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial Unicode MS",
                "Lantinghei SC",
                "Songti SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
        }
    )

    ledger_rows, _scenario_rows, ledger_manifest, ledger_paths = _load_verified_ledger(
        config, experiment_root
    )
    roles = _parse_roles(config, experiment_root)
    evidence = _evidence_for_roles(roles, ledger_rows)
    evidence, _run_configs = _enforce_role_contracts(
        config, evidence, ledger_rows, experiment_root
    )
    if not any(item.status in {"ready", "pilot"} for item in evidence):
        failures = " | ".join(
            f"{item.role.scenario_id}: {item.note}"
            for item in evidence
            if item.status == "invalid"
        )
        raise ValueError(
            "no evidence-ready roles remain after ledger and role-contract validation"
            + (f": {failures}" if failures else "")
        )
    selected = _selected_main(evidence)

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    figures_dir = destination / "figures"
    tables_dir = destination / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    cache_value = (
        basemap_cache if basemap_cache is not None else config.get("basemap_cache")
    )
    cache_path = (
        _resolve_repo_path(cache_value, base=experiment_root)
        if isinstance(cache_value, (str, Path))
        else None
    )

    figure_paths: list[Path] = []
    for paths in (
        _plot_scale_capability(evidence, figures_dir / "scale_capability"),
        _plot_algorithm(evidence, figures_dir / "algorithm_comparison"),
        _plot_exactness(evidence, figures_dir / "exactness_comparison"),
        _plot_sensitivity(evidence, figures_dir / "sensitivity"),
    ):
        figure_paths.extend(paths)
    parameters, distances, discounts, choice_source = _choice_parameters(
        config, selected, experiment_root, scenario_path
    )
    choice_rows = _choice_curve_rows(parameters, distances, discounts)
    figure_paths.extend(
        _plot_choice_curve(choice_rows, figures_dir / "choice_distance_curve")
    )
    map_render = _plot_selected_map(
        selected,
        experiment_root,
        cache_path,
        figures_dir / "selected_three_modes_map",
    )
    figure_paths.extend((map_render.png, map_render.svg))

    role_rows = _role_table_rows(evidence)
    role_csv = tables_dir / "scenario_evidence_zh.csv"
    choice_csv = tables_dir / "choice_distance_curve_zh.csv"
    artifact_results_csv = tables_dir / "artifact_results.csv"
    _write_csv(role_csv, role_rows)
    _write_csv(choice_csv, choice_rows)
    _write_csv(artifact_results_csv, _artifact_dataset_rows(evidence))
    table_paths = [role_csv, choice_csv, artifact_results_csv]

    markdown = _markdown_report(
        destination,
        evidence,
        choice_rows,
        choice_source,
        ledger_paths,
        map_render,
        report_kind,
    )
    generated_at = datetime.now(UTC).isoformat()
    artifact = _write_artifact(
        destination,
        evidence,
        choice_rows,
        generated_at,
        scenario_path=scenario_path,
        ledger_paths=ledger_paths,
        choice_source=choice_source,
        result_snapshot_path=artifact_results_csv,
        choice_snapshot_path=choice_csv,
        report_kind=report_kind,
        map_render=map_render,
    )
    artifact_payload = _read_object(artifact)
    manifest_payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "result_contract": {
            "schema_version": RESULT_SCHEMA_VERSION,
            "formulation_id": FORMULATION_ID,
        },
        "inputs": {
            "registry_status": registry_status,
            "report_kind": report_kind,
            "scenario_config": str(scenario_path),
            "scenario_config_sha256": _sha256(scenario_path),
            "experiment_root": str(experiment_root),
            "ledger_manifest_sha256": _sha256(ledger_paths[2]),
            "ledger_run_csv_sha256": _sha256(ledger_paths[0]),
            "ledger_scenario_csv_sha256": _sha256(ledger_paths[1]),
            "ledger_run_count": ledger_manifest.get("run_count"),
            "ledger_mode_row_count": ledger_manifest.get("mode_row_count"),
            "choice_parameter_source": str(choice_source.path),
            "choice_parameter_locator": choice_source.locator,
            "choice_parameter_source_sha256": _sha256(choice_source.path),
        },
        "map": {
            "status": map_render.status,
            "note": map_render.note,
            "png": str(map_render.png),
            "svg": str(map_render.svg),
        },
        "map_status": map_render.status,
        "map_note": map_render.note,
        "artifact_snapshot_status": _nested_value(
            artifact_payload, ("snapshot", "status")
        ),
        "evidence_state": {
            "ready": sum(item.status == "ready" for item in evidence),
            "pilot": sum(item.status == "pilot" for item in evidence),
            "pending": sum(item.status == "pending" for item in evidence),
            "invalid": sum(item.status == "invalid" for item in evidence),
        },
        "outputs": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (markdown, artifact, *table_paths, *figure_paths)
        ],
        "interpretation_contract": {
            "screened_gap_scope": "prespecified screened formulation only",
            "money_scope": "modeled fulfillment cost or cost saving in CNY/hour; not revenue or profit",
            "sensitivity_point_estimate_gate": "validation_pass=true and mip_gap_fraction<=0.01",
            "role_assignment": "explicit scenario_config run_path only",
        },
    }
    manifest = destination / "report_manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ReportOutputs(
        markdown=markdown,
        artifact=artifact,
        manifest=manifest,
        figures=tuple(figure_paths),
        tables=tuple(table_paths),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_config", help="explicit scenario_config.json")
    parser.add_argument("--output-dir", required=True, help="report output directory")
    parser.add_argument(
        "--basemap-cache",
        help="optional existing OSM cache; overrides scenario_config.basemap_cache",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = write_paper_experiment_report(
        args.scenario_config,
        args.output_dir,
        basemap_cache=args.basemap_cache,
    )
    print(outputs.markdown)
    print(outputs.artifact)
    print(outputs.manifest)


if __name__ == "__main__":
    main()
