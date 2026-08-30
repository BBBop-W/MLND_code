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
      "experiment_root": "results/paper_experiments_20260831",
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
from quantitative conclusions.  A completed scenario is evidence-ready only
when its ledger rows are present and every requested mode has an independent
``validation_pass=true`` record.

Outputs
-------

The writer creates a Chinese Markdown technical report, CSV evidence tables,
PNG/SVG research figures, a canonical portable-report ``artifact.json``, and
``report_manifest.json`` with hashes.  Solver gaps for screened runs certify
only the explicitly screened formulation.  All monetary outcomes are modeled
fulfilment cost or cost saving in CNY/hour, never revenue or profit.
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
            if not isinstance(modes_raw, list) or any(
                mode not in MODE_NAMES for mode in modes_raw
            ):
                raise ValueError(f"{category}[{index}] modes must use HD/MX/LO")
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
                )
            )
    mains = [role for role in parsed if role.category == "main"]
    selected = [role for role in mains if role.role in {"selected", "selected_main"}]
    if len(mains) > 1 and len(selected) != 1:
        raise ValueError("multiple main roles require exactly one role='selected'")
    return parsed


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

    main = next(
        (
            item
            for item in evidence
            if item.role.category == "main" and item.status == "ready" and item.rows
        ),
        None,
    )
    if main is None:
        return evidence
    main_versions = {
        (row.get("git_head"), row.get("implementation_sha256")) for row in main.rows
    }
    revised: list[RoleEvidence] = []
    for item in evidence:
        if item.status != "pilot":
            revised.append(item)
            continue
        pilot_versions = {
            (row.get("git_head"), row.get("implementation_sha256")) for row in item.rows
        }
        comparable_versions = {
            version for version in main_versions | pilot_versions if any(version)
        }
        if comparable_versions and pilot_versions != main_versions:
            note = (
                item.note
                + "；实现提交或实现哈希与 selected main 不一致，因此仅作开发期算法证据"
            )
            revised.append(RoleEvidence(item.role, item.status, note, item.rows))
        else:
            revised.append(item)
    return revised


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
    config: Mapping[str, Any], selected: RoleEvidence | None, experiment_root: Path
) -> tuple[dict[str, float] | None, list[float], list[float], str]:
    choice_config = config.get("choice_curve")
    curve = choice_config if isinstance(choice_config, Mapping) else {}
    raw_parameters = curve.get("parameters")
    source = "scenario_config.choice_curve.parameters"
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
                    source = str(run_config_path)
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
    mains = [item for item in evidence if item.role.category == "main"]
    selected = [
        item for item in mains if item.role.role in {"selected", "selected_main"}
    ]
    return (selected or mains or [None])[0]


def _plot_selected_map(
    selected: RoleEvidence | None,
    experiment_root: Path,
    basemap_cache: Path | None,
    stem: Path,
) -> tuple[Path, Path, str]:
    if selected is None:
        paths = _pending_figure(
            stem, "南山主算例三模式网络", "scenario_config 未指定 selected main。"
        )
        return (*paths, "未指定主算例")
    if not selected.ready:
        paths = _pending_figure(stem, "南山主算例三模式网络", selected.note)
        return (*paths, selected.note)
    if selected.role.instance_dir is None:
        paths = _pending_figure(
            stem, "南山主算例三模式网络", "主算例未显式提供 instance_dir。"
        )
        return (*paths, "缺少 instance_dir")
    if set(selected.role.modes) != {"HD", "MX", "LO"}:
        paths = _pending_figure(
            stem, "南山主算例三模式网络", "主算例必须显式包含 HD、MX、LO。"
        )
        return (*paths, "主算例模式不完整")
    mx = _mode_row(selected, "MX")
    h = _integer(mx.get("candidate_lockers")) if mx else None
    if h is None:
        paths = _pending_figure(
            stem, "南山主算例三模式网络", "账本缺少 candidate_lockers。"
        )
        return (*paths, "缺少候选 H 数")
    result_dir = experiment_root / selected.role.run_path
    try:
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
        return png, svg, f"已生成 H{h} 三模式南山地图"
    except (OSError, ValueError, RuntimeError) as exc:
        paths = _pending_figure(
            stem, f"南山 H{h} 主算例三模式网络", f"地图生成失败：{exc}"
        )
        return (*paths, f"地图生成失败：{exc}")


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
    choice_source: str,
    ledger_paths: Sequence[Path],
    map_note: str,
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

    lines = [
        "# MLND 南山论文数值实验报告",
        "",
        "## 技术摘要",
        "",
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
        map_note,
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
        f"距离曲线由 `{choice_source}` 中的当前二项 Logit 参数计算。它给出场景内到柜比例，不是由企业行为数据校准的真实转化率预测。",
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
        "报告器核对 ledger manifest 中两个 CSV 的 SHA-256，并只把 `validation_pass=true` 的请求模式标为 ready。主地图还要求选中算例显式给出实例目录以及 HD、MX、LO 三种结果。所有图表都从本次账本动态计算，不沿用其他报告中的固化结论。",
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


def _write_artifact(
    output_dir: Path,
    evidence: Sequence[RoleEvidence],
    choice_rows: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> Path:
    result_rows = _artifact_dataset_rows(evidence)
    validated_rows = [
        row for row in result_rows if row["evidence_status"] in {"ready", "pilot"}
    ]
    ready_rows = [row for row in validated_rows if row["evidence_status"] == "ready"]
    main_mx = next(
        (
            row
            for row in ready_rows
            if row["category"] == "main" and row["mode"] == "MX"
        ),
        None,
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
    sources = [
        {
            "id": "sealed_ledger",
            "label": "密封论文实验账本",
            "path": "ledger/run_ledger.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    "SELECT * FROM read_csv_auto('ledger/run_ledger.csv', "
                    "header=true) WHERE lower(CAST(validation_pass AS VARCHAR)) "
                    "= 'true'"
                ),
                "description": "按 scenario_config 的显式 run_path 连接角色与已验证 ledger 行。",
                "tables_used": [
                    "run_ledger.csv",
                    "scenario_summary.csv",
                    "ledger_manifest.json",
                ],
                "filters": ["validation_pass=true 才标记 ready"],
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
            "path": "scenario_config.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    "SELECT * FROM read_json_auto('scenario_config.json', "
                    "maximum_object_size=16777216)"
                ),
                "description": "角色完全由配置路径决定，不从目录名称推断。",
                "tables_used": ["scenario_config.json"],
                "filters": [],
            },
        },
    ]
    cards = [
        {
            "id": "main_quality",
            "dataset": "headline",
            "sourceId": "sealed_ledger",
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
            "sourceId": "scenario_registry",
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
                "sourceId": "sealed_ledger",
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
                "sourceId": "sealed_ledger",
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
                "sourceId": "sealed_ledger",
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
                "sourceId": "sealed_ledger",
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
                "sourceId": "scenario_registry",
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
            "sourceId": "sealed_ledger",
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
    blocks: list[dict[str, Any]] = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# MLND 南山论文数值实验报告",
            "layout": "full",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "sealed_ledger",
            "body": (
                "## 技术摘要\n\n"
                "本报告只使用显式 scenario_config 与密封 ledger。筛选运行的 Gap 仅证实筛选模型；"
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
            "sourceId": "sealed_ledger",
            "body": "## 关键结果与证书\n\n图表和表格中的结论均由当前账本动态计算。",
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
                "sourceId": "sealed_ledger",
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
            "title": "MLND 南山论文数值实验报告",
            "description": "显式情景注册、密封账本与独立验证约束下的数值实验技术报告。",
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
                if evidence
                and all(item.status in {"ready", "pilot"} for item in evidence)
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
    root_raw = config.get("experiment_root")
    if not isinstance(root_raw, str) or not root_raw:
        raise ValueError("scenario_config requires experiment_root")
    experiment_root = _resolve_repo_path(root_raw, base=scenario_path.parent)
    if not experiment_root.is_dir():
        raise ValueError(f"experiment_root is not a directory: {experiment_root}")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    figures_dir = destination / "figures"
    tables_dir = destination / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

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
    selected = _selected_main(evidence)

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
        config, selected, experiment_root
    )
    choice_rows = _choice_curve_rows(parameters, distances, discounts)
    figure_paths.extend(
        _plot_choice_curve(choice_rows, figures_dir / "choice_distance_curve")
    )
    map_png, map_svg, map_note = _plot_selected_map(
        selected,
        experiment_root,
        cache_path,
        figures_dir / "selected_three_modes_map",
    )
    figure_paths.extend((map_png, map_svg))

    role_rows = _role_table_rows(evidence)
    role_csv = tables_dir / "scenario_evidence_zh.csv"
    choice_csv = tables_dir / "choice_distance_curve_zh.csv"
    _write_csv(role_csv, role_rows)
    _write_csv(choice_csv, choice_rows)
    table_paths = [role_csv, choice_csv]

    markdown = _markdown_report(
        destination,
        evidence,
        choice_rows,
        choice_source,
        ledger_paths,
        map_note,
    )
    generated_at = datetime.now(UTC).isoformat()
    artifact = _write_artifact(destination, evidence, choice_rows, generated_at)
    manifest_payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "result_contract": {
            "schema_version": RESULT_SCHEMA_VERSION,
            "formulation_id": FORMULATION_ID,
        },
        "inputs": {
            "scenario_config": str(scenario_path),
            "scenario_config_sha256": _sha256(scenario_path),
            "experiment_root": str(experiment_root),
            "ledger_manifest_sha256": _sha256(ledger_paths[2]),
            "ledger_run_count": ledger_manifest.get("run_count"),
            "ledger_mode_row_count": ledger_manifest.get("mode_row_count"),
        },
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
