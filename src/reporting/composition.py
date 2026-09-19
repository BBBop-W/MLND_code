"""English composition figures from a saved, audited single-case report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .style import plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np

BLUE = "#246187"
LIGHT_BLUE = "#7babc7"
GOLD = "#b27a27"
OLIVE = "#587354"
RUST = "#8b2e16"
GRAY = "#747f8b"
PALE = "#e3e7eb"
INK = "#24323e"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def frame(title, subtitle, size=(11.6, 7.4)):
    fig = plt.figure(figsize=size, facecolor="white")
    fig.text(0.055, 0.945, title, fontsize=22, fontweight="bold", color=INK)
    fig.text(0.055, 0.900, subtitle, fontsize=12, color="#536371")
    return fig


def note(fig, text):
    fig.text(
        0.055,
        0.045,
        text,
        fontsize=10.2,
        color="#536371",
        va="bottom",
        linespacing=1.45,
    )


def clean(ax, which="x"):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c7ced4")
    ax.tick_params(color="#c7ced4", labelcolor=INK, length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis=which, color="#e7ebef", linewidth=0.7)


def composition(
    title, subtitle, labels, values, colors, center, unit, footer, center_unit=None
):
    total = sum(values)
    fig = frame(title, subtitle)
    ax = fig.add_axes([0.03, 0.23, 0.40, 0.55])
    ax.pie(
        values if total > 0 else [1.0],
        colors=colors if total > 0 else [PALE],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.30, "edgecolor": "white", "linewidth": 2},
    )
    ax.text(
        0,
        0.05,
        center,
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0,
        -0.18,
        center_unit or unit,
        ha="center",
        va="center",
        fontsize=12,
        color="#536371",
    )
    ax.set_aspect("equal")
    tx = fig.add_axes([0.46, 0.23, 0.49, 0.55])
    tx.set_axis_off()
    tx.text(0, 0.94, "Component", fontsize=11, color="#536371")
    tx.text(0.75, 0.94, unit.capitalize(), ha="right", fontsize=11, color="#536371")
    tx.text(1, 0.94, "Share", ha="right", fontsize=11, color="#536371")
    ys = np.linspace(0.76, 0.16, len(labels))
    for y, label, value, color in zip(ys, labels, values, colors):
        tx.add_patch(
            Rectangle(
                (0, y - 0.018), 0.025, 0.035, facecolor=color, transform=tx.transAxes
            )
        )
        tx.text(0.05, y, label, fontsize=12, va="center", color=INK)
        tx.text(
            0.75, y, f"{value:,.2f}", fontsize=12, va="center", ha="right", color=INK
        )
        tx.text(
            1,
            y,
            f"{100*value/total if total else 0:.2f}%",
            fontsize=12,
            va="center",
            ha="right",
            color=INK,
        )
    note(fig, footer)
    return fig


def run(report_dir):
    """Create English composition figures from one audited case-report directory."""
    report = Path(report_dir).resolve()
    out = report / "composition_en"
    out.mkdir(exist_ok=True)
    m = json.loads((report / "metrics.json").read_text())
    bases = read_csv(report / "bases.csv")
    lockers = read_csv(report / "lockers.csv")
    f, c, d = m["fleet"]["components"], m["costs"]["components"], m["demand"]
    t = {key: (0.0 if value is None else value) for key, value in m["times"].items()}
    for key in ("time", "mean_delivery", "terminal", "direct_same_served_OD"):
        t.setdefault(key, 0.0)
    case = m["case"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    checks = {
        "cost_sum_error": abs(sum(c.values()) - m["solver"]["objective"]),
        "fleet_sum_error": abs(sum(f.values()) - m["fleet"]["total"]),
        "demand_sum_error": abs(
            sum(d[k] for k in ("pickup", "home", "direct")) - d["total_per_hour"]
        ),
        "cabinet_sum_error": abs(
            sum(float(r["count"]) for r in lockers) - m["cabinet_total"]
        ),
        "service_departure_sum_error": abs(
            sum(float(r["service_out"]) for r in bases) - d["served_per_hour"]
        ),
    }
    assert all(v < 1e-6 for v in checks.values()), checks
    chart_rows = []
    manifest = []
    pdf = PdfPages(out / f"{case}_composition_figures.pdf")

    def record(chart, labels, values, denominator, unit, keys):
        for label, value, key in zip(labels, values, keys):
            chart_rows.append(
                dict(
                    chart=chart,
                    component=label,
                    value=value,
                    unit=unit,
                    denominator=denominator,
                    share=value / denominator if denominator else None,
                    source="metrics.json",
                    source_key=key,
                )
            )

    def save(fig, name, caption):
        fig.savefig(out / f"{name}.png", dpi=210)
        fig.savefig(out / f"{name}.pdf")
        pdf.savefig(fig)
        plt.close(fig)
        manifest.append(dict(name=name, caption=caption))

    # All cost components are exclusive. Facility amounts already include site setup.
    cost_spec = [
        ("Courier-only delivery", "direct_rider", BLUE),
        ("First-mile couriers", "first_rider", BLUE),
        ("Last-mile couriers", "terminal_rider", BLUE),
        ("Locker equipment + setup", "locker_fixed", OLIVE),
        ("Drone fleet", "fleet", GOLD),
        ("Service-mission flight", "service_flight", LIGHT_BLUE),
        ("Directed air routes", "link_fixed", OLIVE),
        ("Base equipment + setup", "base_fixed", OLIVE),
        ("Aerial waiting", "aerial_wait", LIGHT_BLUE),
        ("Swapping stations + setup", "swap_fixed", RUST),
        ("Empty repositioning flight", "empty_flight", RUST),
    ]
    labels, keys, colors = zip(*cost_spec)
    values = [c[k] for k in keys]
    total = m["costs"]["total"]
    assert abs(sum(values) - total) < 1e-6
    fig = frame(
        "Cost composition",
        f"{case} | Total cost: CNY {total:,.2f} per operating hour",
        (11.6, 8.0),
    )
    ax = fig.add_axes([0.30, 0.17, 0.48, 0.66])
    shares = 100 * np.array(values) / total if total else np.zeros(len(values))
    ax.barh(np.arange(len(values)), shares, color=colors, height=0.61)
    ax.set_yticks(np.arange(len(values)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(10.0, min(100.0, max(shares, default=0.0) * 1.08)))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Share of total cost")
    clean(ax)
    for i, (v, s) in enumerate(zip(values, shares)):
        ax.text(
            1.015,
            i,
            f"{s:5.2f}%  |  {v:,.2f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=11,
            color=INK,
        )
    fig.text(0.785, 0.845, "Share  |  CNY/h", fontsize=11, color="#536371")
    note(
        fig,
        "Fixed assets: five-year depreciation, 365 days/year and 12 operating hours/day.\n"
        f'Site setup is included in facility costs. Empty repositioning cost: CNY {c.get("empty_flight", 0.):,.2f}/h.',
    )
    record(
        "cost", labels, values, total, "CNY/h", [f"costs.components.{k}" for k in keys]
    )
    save(
        fig,
        "01_cost_composition",
        "Exclusive cost components; shares use the total system cost.",
    )

    flight_keys = ["pre_pickup_flight", "loaded_flight", "return_flight"]
    fleet_labels = [
        "Service flight",
        "Idle at bases",
        "Pickup / unloading",
        "Service battery swapping",
        "Aerial waiting",
        "Empty repositioning",
    ]
    fleet_values = [
        sum(f.get(k, 0) for k in flight_keys),
        f["idle"],
        f.get("pickup_handling", 0) + f.get("delivery_handling", 0),
        sum(f.get(k, 0) for k in ("pre_pickup_swap", "loaded_swap", "return_swap")),
        f.get("sync_wait", 0),
        f.get("empty_flight", 0) + f.get("empty_swap", 0),
    ]
    fleet_total = m["fleet"]["total"]
    slack = f.get("fleet_slack", 0.0)
    if slack > 1e-6:
        fleet_labels.append("Fleet slack")
        fleet_values.append(slack)
    assert abs(sum(fleet_values) - fleet_total) < 1e-6
    fig = composition(
        "Fleet occupancy",
        f"{case} | Mean number of drones in each state",
        fleet_labels,
        fleet_values,
        [BLUE, GRAY, GOLD, RUST, OLIVE, LIGHT_BLUE, PALE][: len(fleet_values)],
        f"{fleet_total:.2f}",
        "drones",
        f'Flight inventory: approach {f.get("pre_pickup_flight",0):.2f}, loaded {f.get("loaded_flight",0):.2f}, '
        f'mission return {f.get("return_flight",0):.2f} drones.\n'
        f'Empty repositioning: {f.get("empty_flight",0)+f.get("empty_swap",0):.2f} drones. Shares use the complete fleet, including idle drones.',
    )
    record(
        "fleet",
        fleet_labels,
        fleet_values,
        fleet_total,
        "drones",
        [
            "fleet.components: flight phases",
            "fleet.components.idle",
            "fleet.components: handling phases",
            "fleet.components: swap phases",
            "fleet.components.sync_wait",
            "fleet.components: empty phases",
            "fleet.components.fleet_slack",
        ][: len(fleet_labels)],
    )
    save(
        fig,
        "02_fleet_composition",
        "Mean fleet occupancy, including all service flights and idle inventory.",
    )

    demand_labels = ["Courier only", "Drone + last-mile courier", "Drone + self-pickup"]
    demand_values = [d["direct"], d["home"], d["pickup"]]
    fig = composition(
        "Delivery-mode shares",
        f'{case} | All potential demand: {d["total_per_hour"]:,.2f} orders per hour',
        demand_labels,
        demand_values,
        [GRAY, BLUE, GOLD],
        f'{100*d["served_share"]:.2f}%',
        "orders/h",
        f'Drone-assisted service: {d["served_per_hour"]:,.2f} orders/h. '
        f'Self-pickup among drone-assisted orders: {100*d["pickup"]/d["served_per_hour"] if d["served_per_hour"] else 0.:.2f}%.\n'
        "The three shares use all potential orders; the center shows the drone-assisted share.",
        center_unit="drone-assisted",
    )
    record(
        "demand",
        demand_labels,
        demand_values,
        d["total_per_hour"],
        "orders/h",
        ["demand.direct", "demand.home", "demand.pickup"],
    )
    save(
        fig,
        "03_delivery_modes",
        "Mutually exclusive delivery modes; denominator is all potential orders.",
    )

    fig = frame(
        "Facility selection and locker capacity",
        f'{case} | {m["cabinet_total"]:.0f} cabinets installed across {m["opened"]["lockers"]} opened locker sites',
    )
    ax = fig.add_axes([0.16, 0.23, 0.35, 0.53])
    candidate_keys = ["bases", "lockers", "swaps"]
    names = ["Drone bases", "Locker sites", "Swapping stations"]
    ratios = [
        100 * m["opened"][k] / m["candidates"][k] if m["candidates"][k] else 0.0
        for k in candidate_keys
    ]
    ax.barh(range(3), [100] * 3, color=PALE, height=0.50)
    ax.barh(range(3), ratios, color=[BLUE, OLIVE, RUST], height=0.50)
    ax.set_yticks(range(3), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Share of candidate locations opened")
    ax.set_title("(a) Facility opening", loc="left", fontsize=14, pad=20)
    for i, (k, p) in enumerate(zip(candidate_keys, ratios)):
        ax.text(
            p / 2,
            i,
            f'{m["opened"][k]}/{m["candidates"][k]}  ({p:.1f}%)',
            va="center",
            ha="center",
            color="white",
            fontweight="bold",
            fontsize=11,
        )
    clean(ax)
    ax = fig.add_axes([0.64, 0.23, 0.31, 0.53])
    x = np.arange(1, int(m["parameters"]["max_cabinets"]) + 1)
    counts = [m["locker_capacity"]["count_distribution"].get(str(i), 0) for i in x]
    ax.bar(x, counts, color=OLIVE, width=0.60)
    ax.set_xticks(x)
    ax.set_ylim(0, max(max(counts, default=0) * 1.35, 1.0))
    ax.set_ylabel("Opened locker sites")
    ax.set_xlabel("Cabinets per opened site")
    ax.set_title("(b) Cabinet configuration", loc="left", fontsize=14, pad=20)
    for i, n in zip(x, counts):
        ax.text(
            i,
            n + 0.6,
            f"{n}\n({100*n/sum(counts) if sum(counts) else 0.:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    clean(ax, "y")
    note(
        fig,
        f'Aggregate handling-capacity utilization: {100*m["locker_capacity"]["utilization"]:.2f}%. '
        f'{m["locker_capacity"]["at_least_95pct_sites"]}/{m["opened"]["lockers"]} opened sites are at least 95% utilized.\n'
        "Gray portions in panel (a) denote unopened candidates. Panel (b) shares use opened locker sites.",
    )
    for name, k in zip(names, candidate_keys):
        record(
            "facility_opening",
            [name],
            [m["opened"][k]],
            m["candidates"][k],
            "sites",
            [f"opened.{k}"],
        )
    record(
        "cabinet_configuration",
        [f"{i} cabinets" for i in x],
        counts,
        sum(counts),
        "sites",
        [f"locker_capacity.count_distribution.{i}" for i in x],
    )
    save(
        fig,
        "04_facility_selection",
        "Candidate opening shares and cabinet-count distribution among opened locker sites.",
    )

    mission_labels = [
        "Loaded flight",
        "Return flight",
        "Approach flight",
        "Pickup / unloading",
        "Battery swapping",
        "Aerial waiting",
    ]
    mission_vals = [
        t.get("loaded_flight", 0),
        t.get("return_flight", 0),
        t.get("pre_pickup_flight", 0),
        t.get("pickup_handling", 0) + t.get("delivery_handling", 0),
        sum(t.get(k, 0) for k in ("pre_pickup_swap", "loaded_swap", "return_swap")),
        t.get("sync_wait", 0),
    ]
    dispatch = m["parameters"]["terminal_dispatch_time"] * 60
    delivery_labels = [
        "Drone service to locker",
        "Last-mile rider dispatch",
        "Last-mile rider travel",
    ]
    delivery_vals = (
        [
            t.get("mean_delivery", 0) - t.get("terminal", 0),
            dispatch,
            t.get("terminal", 0) - dispatch,
        ]
        if d["served_per_hour"]
        else [0.0, 0.0, 0.0]
    )
    checks["mission_time_sum_error"] = abs(sum(mission_vals) - t.get("time", 0))
    checks["delivery_time_sum_error"] = abs(
        sum(delivery_vals) - t.get("mean_delivery", 0)
    )
    assert (
        checks["mission_time_sum_error"] < 1e-6
        and checks["delivery_time_sum_error"] < 1e-6
    )
    fig = frame(
        "Time composition",
        f"{case} | Flow-weighted means from merchant pickup",
        size=(12.6, 7.6),
    )
    specs = [
        (
            [0.20, 0.25, 0.25, 0.52],
            mission_labels,
            mission_vals,
            t.get("time", 0),
            [BLUE, LIGHT_BLUE, BLUE, GOLD, RUST, OLIVE],
            f'(a) Drone mission: {t["time"]:.2f} min',
        ),
        (
            [0.69, 0.25, 0.20, 0.52],
            delivery_labels,
            delivery_vals,
            t.get("mean_delivery", 0),
            [BLUE, GRAY, GOLD],
            f'(b) Modeled delivery: {t["mean_delivery"]:.2f} min',
        ),
    ]
    for pos, ls, vs, den, colors, title in specs:
        ax = fig.add_axes(pos)
        ss = 100 * np.array(vs) / den if den else np.zeros(len(vs))
        ax.barh(range(len(ls)), ss, color=colors, height=0.55)
        ax.set_yticks(range(len(ls)), ls)
        ax.invert_yaxis()
        ax.set_xlim(0, max(10.0, min(100.0, max(ss, default=0.0) * 1.1)))
        ax.set_xticks([0, 20, 40])
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel("Share of mean duration")
        ax.set_title(title, loc="left", fontsize=14, pad=20)
        clean(ax)
        for j, (v, s) in enumerate(zip(vs, ss)):
            ax.text(
                1.015,
                j,
                f"{v:.2f} min\n{s:.2f}%",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=10,
            )
    note(
        fig,
        "Panel (a) includes mission return; panel (b) ends at delivery and uses the common home-delivery terminal time.\n"
        "Customer walking is excluded. First-mile rider travel overlaps the drone approach and is not added again.\n"
        f'Matched courier-only delivery for the same served OD demand: {t["direct_same_served_OD"]:.2f} min.',
    )
    record(
        "mission_time",
        mission_labels,
        mission_vals,
        t.get("time", 0),
        "min",
        [
            "times.loaded_flight",
            "times.return_flight",
            "times.pre_pickup_flight",
            "times: handling phases",
            "times: swap phases",
            "times.sync_wait",
        ],
    )
    record(
        "delivery_time",
        delivery_labels,
        delivery_vals,
        t.get("mean_delivery", 0),
        "min",
        [
            "times.mean_delivery - times.terminal",
            "parameters.terminal_dispatch_time * 60",
            "times.terminal - dispatch",
        ],
    )
    save(
        fig,
        "05_time_composition",
        "Separate mission and delivery clocks; rider and drone approach travel overlap.",
    )

    active = [r for r in bases if int(r["open"]) == 1]
    names = [f'L{int(r["base"])+1}' for r in active]
    closed = [f'L{int(r["base"])+1}' for r in bases if int(r["open"]) == 0]
    subtitle = f"{case} | {len(active)} opened bases / {len(bases)} candidates"
    if closed:
        subtitle += "; closed: " + ", ".join(closed)
    fig = frame(
        "Operations at opened bases",
        subtitle,
        size=(12.6, max(7.4, 3.0 + 0.5 * len(active))),
    )
    specs = [
        (
            [0.10, 0.22, 0.23, 0.54],
            "Availability",
            100 * np.array([float(r["rho"]) for r in active]),
            BLUE,
            "Availability probability",
            100,
        ),
        (
            [0.42, 0.22, 0.23, 0.54],
            "Service departures",
            [float(r["service_out"]) for r in active],
            BLUE,
            "Orders per hour",
            max([float(r["service_out"]) * 1.25 for r in active] + [1.0]),
        ),
        (
            [0.74, 0.22, 0.20, 0.54],
            "Idle inventory",
            [float(r["idle"]) for r in active],
            GRAY,
            "Mean number of drones",
            max([float(r["idle"]) * 1.25 for r in active] + [1.0]),
        ),
    ]
    for panel, (pos, title, vals, color, xlabel, xmax) in enumerate(specs):
        ax = fig.add_axes(pos)
        ax.barh(range(len(active)), vals, color=color, height=0.58)
        ax.set_yticks(range(len(active)), names)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(f"({chr(97+panel)}) {title}", loc="left", fontsize=14, pad=20)
        ax.set_xlabel(xlabel)
        clean(ax)
        if panel == 0:
            ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        for j, v in enumerate(vals):
            value = f"{v:.1f}%" if panel == 0 else f"{v:,.2f}"
            ax.text(v + xmax * 0.018, j, value, va="center", fontsize=10)
    note(
        fig,
        f'Cross-base service: {m["repositioning"]["cross_base_service_per_hour"]:.2f} orders/h '
        f'({100*m["repositioning"]["cross_base_service_per_hour"]/d["served_per_hour"] if d["served_per_hour"] else 0.:.2f}% of drone-assisted orders).\n'
        f'Empty inter-base repositioning: {m["repositioning"]["flow_per_hour"]:.2f} missions/h. Mission return flights are included in fleet occupancy.',
    )
    for r, name in zip(active, names):
        for key, den, unit in [
            ("rho", 1, "probability"),
            ("service_out", d["served_per_hour"], "orders/h"),
            ("idle", f["idle"], "drones"),
        ]:
            chart_rows.append(
                dict(
                    chart="base_" + key,
                    component=name,
                    value=float(r[key]),
                    unit=unit,
                    denominator=den,
                    share=float(r[key]) / den if den else None,
                    source="bases.csv",
                    source_key=f'base={r["base"]}.{key}',
                )
            )
    save(
        fig,
        "06_base_operations",
        "Availability, service departures and idle inventory for each opened base.",
    )
    pdf.close()

    with (out / "chart_data.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(chart_rows[0]))
        writer.writeheader()
        writer.writerows(chart_rows)
    validation = {
        "case": case,
        "source_checks": checks,
        "passed": all(x < 1e-6 for x in checks.values()),
        "figures": manifest,
        "sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                report / "metrics.json",
                report / "bases.csv",
                report / "lockers.csv",
            ]
        },
        "full_model_gap": m["solver"]["gap"],
        "new_optimization_performed": False,
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2))
    lines = [
        f"# {case}: Composition figures",
        "",
        f'English figures for the saved solution with {m["candidates"]["bases"]} candidate bases, {m["candidates"]["lockers"]} candidate locker sites, and {m["candidates"]["swaps"]} candidate swapping stations.',
        f'This saved incumbent has a full-model optimality gap of {100*m["solver"]["gap"]:.2f}%. No new optimization was performed.',
        "",
        f"[All figures as a PDF]({case}_composition_figures.pdf)",
        "",
        "Fixed investments use five years, 365 days per year, and 12 operating hours per day. Facility totals include setup fees.",
        "",
        "## Figures",
        "",
    ]
    for item in manifest:
        name = item["name"]
        lines += [
            f'### {name[3:].replace("_"," ").capitalize()}',
            "",
            item["caption"],
            "",
            f'![{item["caption"]}]({name}.png)',
            "",
            f"[Vector PDF]({name}.pdf)",
            "",
        ]
    lines += [
        "## Evidence",
        "",
        "Values and denominator definitions are saved in `chart_data.csv`; arithmetic checks and source hashes are in `validation.json`.",
        "Sources: `../metrics.json`, `../bases.csv`, and `../lockers.csv`.",
        "Fleet values are mean occupancies. Delivery-time panel uses the common home-delivery terminal convention and excludes self-pickup walking.",
        "Percentages are computed from unrounded values; displayed totals can differ by rounding.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))
    return {"output": str(out), "figures": len(manifest), "validation": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.report), indent=2))


if __name__ == "__main__":
    main()
