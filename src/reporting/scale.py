"""Batch summaries and charts for saved experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from .loader import load_case
from .style import _style
from .io import _csv
from .scale_charts import _scale_charts
from .spatial import _spatial, _base_matrices
from .formatting import _n, _table


def build_report(root, output=None, max_edges=65):
    root = Path(root).resolve()
    out = Path(output).resolve() if output else root
    out.mkdir(parents=True, exist_ok=True)
    figdir, csvdir = out / "figures", out / "tables"
    figdir.mkdir(exist_ok=True)
    csvdir.mkdir(exist_ok=True)
    paths = sorted(
        {
            p.parent
            for pattern in ("*/result.json", "*/manifest.json")
            for p in root.glob(pattern)
        }
    )
    allcases = [load_case(path) for path in paths]
    cases = sorted(
        (c for c in allcases if not c["pending"]),
        key=lambda c: (c["row"]["H"], c["path"].name),
    )
    pending = [c for c in allcases if c["pending"]]
    _style()
    selections = {}
    if cases:
        _scale_charts(cases, figdir)
        for c in cases:
            selections[c["path"].name] = _spatial(c, figdir, csvdir, max_edges)
            _base_matrices(c, figdir)
        fields = list(dict.fromkeys(k for c in cases for k in c["row"]))
        _csv(out / "summary.csv", [c["row"] for c in cases], fields)
    rows = []
    for c in cases:
        r = c["row"]
        rows.append(
            [
                r["case"],
                r["L"],
                r["H"],
                r["E"],
                r["OD"],
                _n(r["demand"]),
                _n(r["served"]),
                f'{r["open_lockers"]}/{r["cabinets"]:.0f}',
                _n(r["fleet"]),
                _n(r["cost"]),
                _n(r["gap_pct"]),
                r["validation"],
            ]
        )
    lines = [
        "# Experiment summary",
        "",
        "All figures use each run's saved input and independently reconstructed operations.",
        "",
        _table(
            [
                "Case",
                "Bases",
                "Locker sites",
                "Swap sites",
                "OD pairs",
                "Demand/h",
                "Served/h",
                "Open sites/cabinets",
                "Fleet",
                "Cost (CNY/h)",
                "Gap (%)",
                "Validation",
            ],
            rows,
        ),
        "",
    ]
    if cases:
        for name in ("01_orders", "02_fleet", "03_costs", "04_cost_and_gap"):
            lines += [f"![{name}](figures/{name}.png)", ""]
    if pending:
        lines += [
            "## Pending cases",
            "",
            _table(
                ["Case", "Status"], [[c["path"].name, c["status"]] for c in pending]
            ),
            "",
        ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = dict(
        root=str(root),
        completed=len(cases),
        pending=len(pending),
        spatial_edge_filters=selections,
        verified=(
            all(c["row"]["validation"] == "通过" for c in cases) if cases else None
        ),
        source_case_paths=[str(c["path"]) for c in allcases],
        figures=sorted(p.name for p in figdir.glob("*.png")),
    )
    (out / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-edges", type=int, default=65)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_report(args.root, args.output, args.max_edges),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
