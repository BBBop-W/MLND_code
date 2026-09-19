"""Generate a complete report for one independently validated saved solution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from .loader import load_case
from .metrics import audit
from .charts import figure, solver_stage_figure
from .text import report
from .maps import render_case_maps
from .io import _csv, _sha


def run(path, output=None, draw_maps=True):
    path = Path(path).resolve()
    out = Path(output).resolve() if output else path / "report"
    out.mkdir(parents=True, exist_ok=True)
    case = load_case(path)
    if case.get("pending") or case["row"]["validation"] != "通过":
        raise ValueError(f"Saved case has not passed checks: {path}")
    metrics = audit(case)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    for name in ("bases", "lockers", "costs", "fleet", "od_times", "paths"):
        _csv(out / f"{name}.csv", case["tables"][name])
    if metrics["site_costs"]["active"]:
        _csv(out / "site_costs.csv", metrics["site_costs"]["rows"])
    _csv(out / "swapping_stations.csv", metrics["swapping"]["stations"])
    for lang in ("cn", "en"):
        figure(case, metrics, out, lang)
        solver_stage_figure(metrics, out, lang)
    if draw_maps:
        render_case_maps(case, out)
    for lang in ("cn", "en"):
        report(case, metrics, out, lang)
    sources = [
        path / name
        for name in (
            "input_metadata.json",
            "input_arrays.npz",
            "result.json",
            "solution.npz",
            "path_flows.json.gz",
            "operations.json",
        )
    ]
    manifest = dict(
        source_files={str(p): _sha(p) for p in sources},
        generator=str(Path(__file__).resolve()),
        generator_sha256=_sha(__file__),
        optimization_rerun=False,
        independent_operations_rebuilt=True,
        geographic_renderer_sha256=_sha(Path(__file__).with_name("maps.py")),
        candidate_counts=metrics["candidates"],
        report_outputs=[str(out / "REPORT_CN.md"), str(out / "REPORT_EN.md")],
    )
    (out / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                output=str(out),
                case=metrics["case"],
                candidates=metrics["candidates"],
                objective=metrics["costs"]["total"],
                served=metrics["demand"]["served_per_hour"],
                reposition=metrics["repositioning"]["flow_per_hour"],
                passed=True,
            ),
            ensure_ascii=False,
        )
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-maps", action="store_true")
    args = parser.parse_args()
    run(args.case, args.output, not args.no_maps)
