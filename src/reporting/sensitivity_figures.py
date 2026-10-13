"""Render publication figures from the audited, combined sensitivity results.

Example:
    python -m src.reporting.sensitivity_figures --report outputs/reports/REPORT
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


BLUE = "#163e63"
TEAL = "#087b86"
RUST = "#b25118"
INK = "#262d33"
GREY = "#7c8890"
GROUPS = ("demand", "rider_cost", "fleet_cost", "speed", "battery", "density")
TITLES = {
    "demand": "Order demand",
    "rider_cost": "Courier cost",
    "fleet_cost": "Drone fixed cost",
    "speed": "Flight speed",
    "battery": "Battery endurance",
    "density": "Candidate-site density",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "medium",
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#acb3b8",
        "axes.linewidth": 0.7,
        "axes.axisbelow": True,
        "grid.color": "#e3e7e9",
        "grid.linewidth": 0.65,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def _validate(rows: list[dict]) -> dict[str, list[dict]]:
    groups = {g: sorted((r for r in rows if r["group"] == g), key=lambda r: r["x"])
              for g in GROUPS}
    if len(rows) != 30 or any(len(group) != 5 for group in groups.values()):
        raise ValueError("Expected six factors with five points each.")
    for group, points in groups.items():
        expected = [10, 15, 20, 25, 30] if group == "density" else [.5, .75, 1, 1.25, 1.5]
        if [r["x"] for r in points] != expected:
            raise ValueError(f"Unexpected factor levels: {group}")
        if any(not r["validation_passed"] for r in points):
            raise ValueError(f"Unvalidated solution: {group}")
        if any(r["saving_upper_bound_pct"] < r["saving_pct"] - 1e-6 for r in points):
            raise ValueError(f"Inverted objective bounds: {group}")
    return groups


def _axes(ax, rows: list[dict], *, density: bool = False) -> None:
    ticks = [r["x"] for r in rows]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(x)) if density else f"{x:g}" for x in ticks])
    ax.set_xlim((8.5, 31.5) if density else (.43, 1.57))
    ax.grid(axis="y")
    ax.axvline(20 if density else 1, color="#c6cdd1", ls=(0, (3, 3)), lw=.8, zorder=0)
    ax.tick_params(length=3, width=.6, pad=5)


def _series(ax, rows: list[dict], key: str, color: str = BLUE,
            *, bounds: bool = False, marker: str = "o", baseline: bool = True) -> None:
    x = [r["x"] for r in rows]
    y = [r[key] for r in rows]
    ax.plot(x, y, color=color, lw=1.65, zorder=2)
    if bounds:
        upper = [r["saving_upper_bound_pct"] for r in rows]
        ax.vlines(x, y, upper, color=color, lw=1.2, alpha=.7, zorder=1)
        cap = .021 if rows[0]["group"] != "density" else .42
        ax.hlines(upper, [v - cap for v in x], [v + cap for v in x],
                  color=color, lw=1.2, alpha=.7, zorder=1)
    for r in rows:
        is_base = bool(r["shared_baseline"])
        ax.plot(r["x"], r[key], linestyle="none",
                marker="D" if is_base and baseline else marker,
                ms=6.2 if is_base and baseline else 5.8,
                mec=INK if is_base and baseline else color,
                mfc=INK if is_base and baseline else ("white" if r["above_gap_target"] else color),
                mew=1.3, zorder=4, clip_on=False)


def _status_legend(fig, *, y: float = .91, bounds: bool = False) -> None:
    handles = [
        Line2D([], [], color=BLUE, marker="o", lw=1.4, ms=5.5, label="Best feasible solution"),
        Line2D([], [], color=BLUE, marker="o", mfc="white", lw=0, ms=5.5,
               label="Gap > 0.5%"),
        Line2D([], [], color=INK, marker="D", lw=0, ms=5.5, label="Shared baseline"),
    ]
    if bounds:
        handles.append(Line2D([], [], color=BLUE, marker="_", lw=1.2,
                              label="Upper bound on savings"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.5, y),
               ncol=len(handles), columnspacing=1.8, handletextpad=.6)


def _save(fig, output: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=240,
                    bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def savings(groups: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.4), sharey=True)
    fig.subplots_adjust(left=.07, right=.985, bottom=.19, top=.82, wspace=.18, hspace=.46)
    fig.suptitle("Cost savings across sensitivity factors", x=.07, y=.982,
                 ha="left", fontsize=15, fontweight="medium")
    _status_legend(fig, y=.925, bounds=True)
    for idx, (group, ax) in enumerate(zip(GROUPS, axes.flat)):
        rows = groups[group]
        _axes(ax, rows, density=group == "density")
        _series(ax, rows, "saving_pct", bounds=True)
        ax.set_title(f"({chr(97 + idx)}) {TITLES[group]}", loc="left", pad=10)
        ax.set_ylim(0, 11)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_xlabel("Candidate lockers, H" if group == "density" else "Multiplier")
        if idx % 3 == 0:
            ax.set_ylabel("Savings (%)")
    fig.text(.07, .034,
             "Savings are relative to the all-courier cost in each case. Vertical ranges show solver-bound intervals.\n"
             "All cases retain the same area and 400 OD pairs.",
             fontsize=9, color="#58646d", linespacing=1.55)
    _save(fig, output, "savings_sensitivity")


def technology(groups: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.5), sharey="col")
    fig.subplots_adjust(left=.07, right=.985, bottom=.19, top=.83, wspace=.25, hspace=.46)
    fig.suptitle("Drone technology and network operations", x=.07, y=.982,
                 ha="left", fontsize=15, fontweight="medium")
    _status_legend(fig, y=.93)
    columns = [
        ("drone_served_share_pct", "Drone-served share", "Share of all orders (%)", (0, 40)),
        ("fleet", "Fleet size", "Drones", (0, 140)),
        ("service_using_swap_pct", "Service using swaps", "Share of drone orders (%)", (0, 45)),
    ]
    for row, (group, color) in enumerate((("speed", BLUE), ("battery", TEAL))):
        for col, (key, title, ylabel, ylim) in enumerate(columns):
            ax = axes[row, col]
            rows = groups[group]
            _axes(ax, rows)
            _series(ax, rows, key, color)
            ax.set_title(f"({chr(97 + 3 * row + col)}) {title}", loc="left", pad=10)
            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xlabel("Flight-speed multiplier" if group == "speed" else "Battery-endurance multiplier")
    fig.text(.07, .035,
             "Speed changes hold flight endurance fixed; attainable range therefore changes with speed. Battery cases hold speed fixed.\n"
             "Operational metrics describe the saved feasible solutions; open circles mark cases above the 0.5% gap target.",
             fontsize=9, color="#58646d", linespacing=1.55)
    _save(fig, output, "technology_sensitivity")


def density(groups: dict, output: Path) -> None:
    rows = groups["density"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8))
    fig.subplots_adjust(left=.09, right=.98, bottom=.19, top=.83, wspace=.28, hspace=.44)
    fig.suptitle("Candidate-site density and network operations", x=.09, y=.982,
                 ha="left", fontsize=15, fontweight="medium")
    _status_legend(fig, y=.93)
    for ax in axes.flat:
        _axes(ax, rows, density=True)
        ax.set_xlabel("Candidate lockers, H")
    specs = [
        (axes[0, 0], "saving_pct", "(a) Cost savings", "Savings (%)", (0, 8)),
        (axes[0, 1], "drone_served_share_pct", "(b) Drone-served share", "Share of all orders (%)", (0, 40)),
        (axes[1, 0], "fleet", "(c) Fleet size", "Drones", (0, 140)),
    ]
    for ax, key, title, ylabel, ylim in specs:
        _series(ax, rows, key, bounds=key == "saving_pct")
        ax.set_title(title, loc="left", pad=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax = axes[1, 1]
    facility = [
        ("opened_lockers", "Lockers", TEAL, "o"),
        ("opened_bases", "Bases", BLUE, "s"),
        ("opened_swaps", "Swaps", RUST, "^"),
    ]
    handles = []
    for key, label, color, marker in facility:
        _series(ax, rows, key, color, marker=marker, baseline=False)
        handles.append(Line2D([], [], color=color, lw=1.65, marker=marker, ms=5, label=label))
    ax.set_title("(d) Opened facilities", loc="left", pad=10)
    ax.set_ylabel("Sites")
    ax.set_ylim(0, 28)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.legend(handles=handles, ncol=3, loc="upper left", handlelength=1.3,
              columnspacing=.8, fontsize=8.8)
    fig.text(.09, .045,
             "The area and 400 OD pairs are fixed. Candidate counts: L = H/5 and E = 2H/5; sites are nested.\n"
             "In (a), vertical ranges show solver-bound intervals.",
             fontsize=9, color="#58646d", linespacing=1.55)
    _save(fig, output, "density_sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve()
    source = report / "plot_data.json"
    rows = json.loads(source.read_text())
    groups = _validate(rows)
    output = report / "figures"
    output.mkdir(parents=True, exist_ok=True)
    _style()
    savings(groups, output)
    technology(groups, output)
    density(groups, output)
    (output / "provenance.json").write_text(json.dumps({
        "source": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(rows),
        "unique_cases": len({r["job_id"] for r in rows}),
        "figures": ["savings_sensitivity", "technology_sensitivity", "density_sensitivity"],
        "interval_meaning": "Attained savings to savings implied by the solver objective lower bound; not confidence intervals.",
        "open_marker_meaning": "Saved solution has relative MIP gap above 0.5%.",
        "script": str(Path(__file__).resolve()),
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
