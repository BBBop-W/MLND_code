from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION
from src.utils import paper_experiment_report as report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _ledger_row(
    run_path: str,
    *,
    mode: str = "MX",
    gap: float = 0.008,
    validation: bool = True,
    screening_is_exact: bool = False,
    h: int = 24,
) -> dict[str, object]:
    return {
        "run_path": run_path,
        "scenario_id": run_path.replace("/", "_"),
        "mode": mode,
        "schema_version": RESULT_SCHEMA_VERSION,
        "formulation_id": FORMULATION_ID,
        "candidate_lockers": h,
        "crop_side_km": 9,
        "crop_area_km2": 81,
        "demand_per_hour": 4906.857,
        "od_markets": 6445,
        "objective_cny_per_hour": 32000 + {"HD": 100, "MX": 0, "LO": 200}[mode],
        "objective_bound_cny_per_hour": 31744,
        "mip_gap_fraction": gap,
        "runtime_seconds": 220,
        "cost_saving_percent": 2.1,
        "drone_service_fraction": 0.06,
        "open_bases": 2,
        "open_lockers": 21,
        "open_swap_stations": 1,
        "fleet_size": 62,
        "screening_is_exact": str(screening_is_exact).lower(),
        "validation_pass": str(validation).lower(),
    }


def _write_ledger_bundle(root: Path, rows: list[dict[str, object]]) -> None:
    ledger_dir = root / "ledger"
    run_ledger = ledger_dir / "run_ledger.csv"
    scenario_summary = ledger_dir / "scenario_summary.csv"
    _write_csv(run_ledger, rows)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run_path"]), []).append(row)
    _write_csv(
        scenario_summary,
        [
            {
                "run_path": run_path,
                "scenario_id": group[0]["scenario_id"],
                "modes": ",".join(str(row["mode"]) for row in group),
            }
            for run_path, group in grouped.items()
        ],
    )
    manifest = {
        "schema_version": 1,
        "run_count": len({row["run_path"] for row in rows}),
        "mode_row_count": len(rows),
        "outputs": {
            "run_ledger.csv": _sha256(run_ledger),
            "scenario_summary.csv": _sha256(scenario_summary),
        },
    }
    (ledger_dir / "ledger_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _scenario_config(root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_root": str(root),
        "ledger": {
            "run_ledger": "ledger/run_ledger.csv",
            "scenario_summary": "ledger/scenario_summary.csv",
            "manifest": "ledger/ledger_manifest.json",
        },
        "roles": {
            "main": {
                "id": "main_h24",
                "label": "H24 正式主算例",
                "run_path": "main/h24",
                "modes": ["HD", "MX", "LO"],
            },
            "scale": [
                {
                    "id": "scale_h24",
                    "label": "H24",
                    "run_path": "scale/h24",
                    "modes": ["MX"],
                    "scale": {"H": 24},
                },
                {
                    "id": "scale_h43_pending",
                    "label": "H43",
                    "run_path": "scale/h43",
                    "modes": ["MX"],
                    "scale": {"H": 43},
                    "status": "planned",
                },
            ],
            "algorithm": [
                {
                    "id": "method_auto",
                    "label": "MIQCPMethod=-1",
                    "run_path": "algorithm/auto",
                    "modes": ["MX"],
                }
            ],
            "exactness": [
                {
                    "id": "h11_screened",
                    "label": "H11 screened",
                    "run_path": "exact/screened",
                    "modes": ["MX"],
                    "role": "screened",
                },
                {
                    "id": "h11_complete",
                    "label": "H11 complete-index",
                    "run_path": "exact/complete",
                    "modes": ["MX"],
                    "role": "complete",
                },
            ],
            "sensitivity": [
                {
                    "id": "rider_080",
                    "label": "Rider x0.8",
                    "run_path": "sensitivity/rider080",
                    "modes": ["MX"],
                    "group": "rider_cost",
                    "factor": 0.8,
                },
                {
                    "id": "rider_120",
                    "label": "Rider x1.2",
                    "run_path": "sensitivity/rider120",
                    "modes": ["MX"],
                    "group": "rider_cost",
                    "factor": 1.2,
                },
            ],
        },
        "choice_curve": {
            "distances_km": [0, 0.5, 1.0, 1.5],
            "discounts_cny": [0, 1, 2],
            "parameters": {
                "locker_asc": -1,
                "delivery_time_coefficient": 0.06,
                "price_coefficient": 0.35,
                "walk_distance_coefficient": 1.2,
                "home_eta_minutes": 30,
                "locker_eta_minutes": 20,
                "locker_discount_cny": 1,
                "max_walk_km": 1.5,
            },
        },
    }


def test_report_is_dynamic_and_keeps_pending_roles_out_of_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    rows = [
        *[_ledger_row("main/h24", mode=mode) for mode in ("HD", "MX", "LO")],
        _ledger_row("scale/h24", gap=0.009),
        _ledger_row("algorithm/auto", gap=0.021),
        _ledger_row("exact/screened", gap=0.007, screening_is_exact=False, h=11),
        _ledger_row("exact/complete", gap=0.009, screening_is_exact=True, h=11),
        _ledger_row("sensitivity/rider080", gap=0.008),
        _ledger_row("sensitivity/rider120", gap=0.013),
    ]
    _write_ledger_bundle(experiment_root, rows)
    config_path = tmp_path / "scenario_config.json"
    config_path.write_text(
        json.dumps(_scenario_config(experiment_root), ensure_ascii=False),
        encoding="utf-8",
    )

    def fake_map(selected, experiment_root, basemap_cache, stem):  # noqa: ANN001
        png, svg = report._pending_figure(stem, "测试地图", "地图接口已调用")
        return png, svg, "测试中以占位图替代 OSM 渲染。"

    monkeypatch.setattr(report, "_plot_selected_map", fake_map)
    output = tmp_path / "report"
    outputs = report.write_paper_experiment_report(config_path, output)

    markdown = outputs.markdown.read_text(encoding="utf-8")
    assert "H24 MX 主算例 Gap 为 0.80%" in markdown
    assert "最大达标档为 H24" in markdown
    assert "scale_h43_pending" in markdown
    assert "H120" not in markdown
    assert "不是营业收入或利润" in markdown
    assert "只属于预设筛选模型" in markdown

    artifact = json.loads(outputs.artifact.read_text(encoding="utf-8"))
    assert artifact["surface"] == "report"
    assert artifact["snapshot"]["status"] == "partial"
    result_rows = artifact["snapshot"]["datasets"]["results"]
    pending = next(
        row for row in result_rows if row["scenario_id"] == "scale_h43_pending"
    )
    assert pending["evidence_status"] == "pending"
    scale = artifact["snapshot"]["datasets"]["scale_ready"]
    assert [row["x_value"] for row in scale] == [24.0]

    manifest = json.loads(outputs.manifest.read_text(encoding="utf-8"))
    assert manifest["evidence_state"] == {
        "ready": 6,
        "pilot": 1,
        "pending": 1,
        "invalid": 0,
    }
    assert len(outputs.figures) == 12
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.figures)
    assert all(path.is_file() for path in outputs.tables)


def test_role_assignment_uses_exact_explicit_run_path(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    rows = [_ledger_row("formal/main/H24_finished")]
    _write_ledger_bundle(experiment_root, rows)
    config = {
        "schema_version": 1,
        "experiment_root": str(experiment_root),
        "roles": {
            "main": {
                "id": "selected",
                "label": "Selected H24",
                "run_path": "formal/main/H24_requested",
                "modes": ["MX"],
            }
        },
    }
    roles = report._parse_roles(config, experiment_root)
    evidence = report._evidence_for_roles(roles, rows)
    assert evidence[0].status == "pending"
    assert not evidence[0].rows


def test_ledger_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir()
    _write_ledger_bundle(experiment_root, [_ledger_row("main/h24")])
    ledger = experiment_root / "ledger/run_ledger.csv"
    ledger.write_text(
        ledger.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ledger output hash mismatch"):
        report._load_verified_ledger(
            {"ledger": {}},
            experiment_root,
        )


def test_choice_curve_matches_current_logit_definition() -> None:
    parameters = {
        "locker_asc": -1.0,
        "delivery_time_coefficient": 0.06,
        "price_coefficient": 0.35,
        "walk_distance_coefficient": 1.2,
        "home_eta_minutes": 30.0,
        "locker_eta_minutes": 20.0,
        "locker_discount_cny": 1.0,
        "max_walk_km": 1.5,
    }
    rows = report._choice_curve_rows(parameters, [0.0, 1.5], [1.0])
    assert rows[0]["到柜选择比例"] == pytest.approx(0.4875026035)
    assert rows[1]["到柜选择比例"] == pytest.approx(0.1358728970)
    assert all(row["是否在模型覆盖范围"] for row in rows)


def test_repo_relative_run_path_is_normalized_against_experiment_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    root = repo / "results/paper"
    root.mkdir(parents=True)
    monkeypatch.setattr(report, "_repo_root", lambda: repo)
    assert (
        report._normalise_run_path("results/paper/formal/main/H24", root)
        == "formal/main/H24"
    )
