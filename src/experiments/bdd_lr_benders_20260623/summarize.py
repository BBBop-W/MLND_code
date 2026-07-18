"""Aggregate dated benchmark summaries into paper-ready CSV and LaTeX."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


FIELDS = (
    "H",
    "method",
    "variant",
    "obj",
    "validation_obj",
    "bound",
    "gap",
    "wall_time",
    "time_to_best",
    "lr_bound",
    "lr_wall_time",
    "global_linearized_bound",
    "global_linearized_wall_time",
    "iterations",
    "cuts",
    "topology_cuts",
    "swap_reachability_cuts",
    "master_qconstrs",
)


def load_rows(
    results_root: Path,
    *,
    include_invalidated: bool = False,
) -> list[dict]:
    rows = []
    for path in sorted(results_root.glob("H*/**/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        variant = str(data.get("variant") or "")
        if not include_invalidated and not variant.startswith("qfix_"):
            continue
        row = {field: data.get(field) for field in FIELDS}
        row["result_dir"] = str(path.parent.relative_to(PROJECT_ROOT))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [*FIELDS, "result_dir"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latex_value(value, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("_", r"\_")


def write_latex(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{tabular}{rrlrrrr}",
        r"\toprule",
        r"$H$ & Method & Variant & Obj. & Bound & Gap & Time (s) \\",
        r"\midrule",
    ]
    for row in rows:
        objective = (
            row["validation_obj"]
            if row["validation_obj"] is not None
            else row["obj"]
        )
        lines.append(
            " & ".join(
                (
                    latex_value(row["H"], 0),
                    latex_value(row["method"]),
                    latex_value(row["variant"]),
                    latex_value(objective),
                    latex_value(row["bound"]),
                    latex_value(row["gap"]),
                    latex_value(row["wall_time"], 2),
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str(
            PROJECT_ROOT / "results" / "bdd_lr_benders_20260623"
        ),
    )
    parser.add_argument("--variant", default=None)
    parser.add_argument("--include-invalidated", action="store_true")
    args = parser.parse_args()
    root = Path(args.results_root).resolve()
    rows = load_rows(
        root,
        include_invalidated=args.include_invalidated,
    )
    if args.variant is not None:
        rows = [row for row in rows if row["variant"] == args.variant]
    write_csv(root / "comparison_all.csv", rows)
    write_latex(root / "comparison_all.tex", rows)
    print(root / "comparison_all.csv")
    print(root / "comparison_all.tex")


if __name__ == "__main__":
    main()
