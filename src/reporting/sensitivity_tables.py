"""Generate the two sensitivity tables from independently audited results."""

import argparse
import json
from pathlib import Path


FACTORS = (
    ("demand", "Demand"),
    ("rider_cost", "Rider cost"),
    ("fleet_cost", "Drone fixed cost"),
    ("speed", "Flight speed"),
    ("battery", "Battery endurance"),
)
SCALES = (0.5, 0.75, 1.25, 1.5)
METRICS = (
    ("unit_cost", "Cost per order, CNY"),
    ("saving_pct", r"Cost saving, \%"),
    ("drone_served_share_pct", r"Drone-served share, \%"),
    ("fleet", "Fleet size"),
    ("opened", "Opened L/H/E"),
    ("gap_pct", r"MIP gap, \%"),
)


def value(row, metric):
    if metric == "opened":
        return "/".join(str(int(row[key])) for key in (
            "opened_bases", "opened_lockers", "opened_swaps"
        ))
    if metric == "unit_cost":
        number = row["total_cost_per_hour"] / row["demand_per_hour"]
    else:
        number = row[metric]
    text = f"{number:.1f}" if metric == "fleet" else f"{number:.2f}"
    if metric == "gap_pct" and row["above_gap_target"]:
        text += r"$^{\dagger}$"
    return text


def table_start(caption, label, columns):
    return [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begingroup",
        r"\fontsize{9.5}{11.5}\selectfont",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.13}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
    ]


def table_end(notes):
    return [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{5pt}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textit{Notes.} " + notes,
        r"\end{minipage}",
        r"\endgroup",
        r"\end{table}",
    ]


COMMON_NOTES = (
    "All cells report actual levels. Cost per order divides total hourly cost "
    "by total hourly demand. Cost saving is relative to the all-courier cost "
    "recomputed for each scenario. Fleet size is continuous. "
    "Opened L/H/E lists bases, locker locations, and swap stations; "
    "locker locations are distinct from cabinet units. "
    r"$\dagger$ marks runs that reached the one-hour limit above the 0.5\% gap target."
)


def generate(output):
    rows = json.loads((output / "all_results.json").read_text())
    assert len(rows) == 25 and all(row["validation_passed"] for row in rows)
    lookup = {(row["factor"], row["multiplier"]): row for row in rows}
    assert len(lookup) == len(rows)
    baseline = lookup[("baseline", 1.0)]

    parameter = table_start(
        "Sensitivity to demand, technology, and cost parameters",
        "tab:sensitivity_parameters",
        "@{}llrrrr@{}",
    )
    parameter += [
        r" & & \multicolumn{4}{c}{Parameter multiplier} \\",
        r"\cmidrule(l){3-6}",
        r"Parameter & Metric (baseline value) & 0.50 & 0.75 & 1.25 & 1.50 \\",
        r"\midrule",
    ]
    for index, (factor, title) in enumerate(FACTORS):
        for metric_index, (metric, label) in enumerate(METRICS):
            cells = [title if metric_index == 0 else "",
                     f"{label} ({value(baseline, metric)})"]
            cells += [value(lookup[(factor, scale)], metric) for scale in SCALES]
            parameter.append(" & ".join(cells) + r" \\")
        if index < len(FACTORS) - 1:
            parameter.append(r"\midrule")
    parameter += table_end(
        "Parentheses contain the common baseline values. " + COMMON_NOTES + " "
        "Drone fixed cost scales fleet acquisition only. Speed changes flight "
        "times and attainable range at fixed endurance; battery scaling changes "
        "endurance at fixed speed. Other parameters remain fixed."
    )

    density_rows = [lookup[("density", scale)] for scale in (0.5, 0.75)]
    density_rows += [baseline]
    density_rows += [lookup[("density", scale)] for scale in (1.25, 1.5)]
    density = table_start(
        "Sensitivity to candidate-site density",
        "tab:sensitivity_density",
        "@{}lrrrrr@{}",
    )
    density += [
        r" & \multicolumn{5}{c}{Candidate facilities (L/H/E)} \\",
        r"\cmidrule(l){2-6}",
        r"Metric & 2/10/4 & 3/15/6 & \textbf{4/20/8} & 5/25/10 & 6/30/12 \\",
        r"\midrule",
    ]
    for metric, label in METRICS:
        cells = [label]
        for index, row in enumerate(density_rows):
            cell = value(row, metric)
            cells.append(r"\textbf{" + cell + "}" if index == 2 else cell)
        density.append(" & ".join(cells) + r" \\")
    density += table_end(
        "The bold column is the common baseline. Candidate sets are nested "
        r"within the same 66.67~km$^2$ region, with the same 400 OD pairs "
        "and unchanged demand. " + COMMON_NOTES
    )
    tables = "\n".join(parameter) + "\n\n" + "\n".join(density) + "\n"
    (output / "sensitivity_tables.tex").write_text(
        "% Requires \\usepackage{booktabs}; generated from all_results.json.\n" + tables
    )
    preview = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\setlength{\parindent}{0pt}
\begin{document}
""" + "\n".join(parameter) + "\n\\clearpage\n" + "\n".join(density) + r"""
\end{document}
"""
    (output / "tables_preview.tex").write_text(preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.output.resolve())


if __name__ == "__main__":
    main()
