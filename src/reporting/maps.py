"""Geographic facility, service-corridor and repositioning figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
from matplotlib import colors, patheffects
from matplotlib.lines import Line2D
from .style import plt, _style
from .basemap import CACHE
from .map_data import _prepare
from .io import _sha, _write_csv
from .loader import load_case

BASE_COLORS = [
    "#186b98",
    "#b04f42",
    "#a88220",
    "#7657a0",
    "#258675",
    "#cf793f",
    "#5d659b",
    "#8e6550",
]
SWAP_COLOR = "#8b2e16"


def _map(ax, view, *, small=False):
    ax.imshow(
        view["image"], extent=view["extent"], origin="upper", zorder=-5, rasterized=True
    )
    for ring in view["boundary"]:
        ax.plot(
            ring[:, 0],
            ring[:, 1],
            color="#788895",
            ls=(0, (5, 4)),
            lw=0.7,
            alpha=0.62,
            zorder=-2,
        )
    ax.set_xlim(view["bounds"][:2])
    ax.set_ylim(view["bounds"][2:])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    x0, x1, y0, y1 = view["bounds"]
    # Web Mercator lengths are converted to ground distance at map centre.
    lat = float(view["lonlat"][:, 1].mean())
    length = 2000 / math.cos(math.radians(lat))
    x, y = x0 + 0.04 * (x1 - x0), y0 + 0.038 * (y1 - y0)
    ax.plot(
        [x, x + length],
        [y, y],
        color="#263746",
        lw=3,
        solid_capstyle="butt",
        zorder=20,
        path_effects=[
            patheffects.Stroke(linewidth=5, foreground="white"),
            patheffects.Normal(),
        ],
    )
    ax.text(
        x + length / 2,
        y + 0.016 * (y1 - y0),
        "2 km",
        ha="center",
        fontsize=8 if small else 11,
        zorder=21,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
    )
    ax.annotate(
        "N",
        xy=(0.95, 0.94),
        xytext=(0.95, 0.865),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=10 if small else 12,
        color="#263746",
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#263746"),
        zorder=21,
    )
    ax.text(
        0.985,
        0.009,
        "© OpenStreetMap contributors",
        transform=ax.transAxes,
        ha="right",
        fontsize=6 if small else 8,
        color="#5c6670",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
        zorder=21,
    )


def _base_labels(ax, view, indices=None, color_by_base=False, small=False):
    c = view["case"]
    L = c["row"]["L"]
    for l in range(L):
        if not view["opened"][l] or indices is not None and l not in indices:
            continue
        x, y = view["xy"][l]
        color = BASE_COLORS[l % len(BASE_COLORS)] if color_by_base else "#203e56"
        ax.scatter(
            x,
            y,
            marker="s",
            s=85 if small else 150,
            c=color,
            edgecolors="white",
            linewidths=1.7,
            zorder=12,
        )
        ax.annotate(
            f"L{l+1}",
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9 if small else 12,
            color=color,
            weight="bold",
            zorder=15,
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=1.4),
        )


def _title(fig, title, subtitle, extra):
    fig.text(
        0.055, 0.968, title, fontsize=23, weight="bold", ha="left", color="#203747"
    )
    fig.text(0.055, 0.936, subtitle, fontsize=12, ha="left", color="#5d6c78")
    fig.text(0.055, 0.913, extra, fontsize=11, ha="left", color="#5d6c78")


def _foot(fig, text, language="zh"):
    fig.text(0.055, 0.024, text, fontsize=9, color="#687886", ha="left")
    note = (
        "Coordinates follow the historical plotting convention; basemap for geographic reference."
        if language == "en"
        else "坐标转换沿用旧图规则。底图仅用于位置展示。"
    )
    fig.text(0.055, 0.008, note, fontsize=8, color="#87939d", ha="left")


def _save(fig, out, name):
    paths = []
    for suffix in ("png", "pdf"):
        p = out / f"{name}.{suffix}"
        fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
        paths.append(str(p))
    plt.close(fig)
    return paths


def _facilities(view, out):
    c = view["case"]
    H, L = c["row"]["H"], c["row"]["L"]
    summary = c["summary"]
    en = view.get("language") == "en"
    fig = plt.figure(figsize=(9, 12))
    ax = fig.add_axes([0.055, 0.108, 0.89, 0.765])
    _map(ax, view)
    lockers = c["tables"]["lockers"]
    ids = np.array([L + r["locker"] for r in lockers if r["open"]], dtype=int)
    opened = [r for r in lockers if r["open"]]
    cmap = colors.LinearSegmentedColormap.from_list(
        "utilization", ["#d2e8ee", "#5595ad", "#124d73"]
    )
    m = ax.scatter(
        view["xy"][ids, 0],
        view["xy"][ids, 1],
        s=[32 + 26 * r["count"] for r in opened],
        c=[r["utilization"] for r in opened],
        cmap=cmap,
        vmin=0,
        vmax=1,
        edgecolors="white",
        linewidths=1,
        zorder=5,
    )
    closed = np.flatnonzero(~view["opened"])
    if len(closed):
        ax.scatter(
            view["xy"][closed, 0],
            view["xy"][closed, 1],
            marker="x",
            s=35,
            c="#a4acb3",
            lw=1.4,
            zorder=6,
        )
    e = np.flatnonzero(view["opened"][L + H :]) + L + H
    ax.scatter(
        view["xy"][e, 0],
        view["xy"][e, 1],
        marker="D",
        s=67,
        c=SWAP_COLOR,
        edgecolors="white",
        linewidths=1,
        zorder=9,
    )
    _base_labels(ax, view)
    if en:
        _title(
            fig,
            f"H{H} · Facilities and locker capacity",
            f"Nanshan, Shenzhen  |  {summary['opened']['bases']} bases · {summary['opened']['lockers']} locker sites · {summary['opened']['swaps']} swap stations",
            f"{summary['cabinet_total']:.0f} cabinets  |  {summary['served']:,.0f} orders/h  |  Fleet: {summary['fleet']:.1f} drones",
        )
    else:
        _title(
            fig,
            f"H{H} · 设施与柜容量",
            f"深圳南山  |  {summary['opened']['bases']} 个基地 · {summary['opened']['lockers']} 个柜点 · {summary['opened']['swaps']} 个换电站",
            f"安装 {summary['cabinet_total']:.0f} 个柜  |  每小时服务 {summary['served']:,.0f} 单  |  机队 {summary['fleet']:.1f} 架",
        )
    legend = [
        Line2D(
            [],
            [],
            ls="",
            marker="s",
            ms=8,
            mfc="#203e56",
            mec="white",
            label="Base L" if en else "基地 L",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="D",
            ms=7,
            mfc=SWAP_COLOR,
            mec="white",
            label="Swap station E" if en else "换电站 E",
        ),
    ]
    for count in (1, 3, 5):
        label = f"{count} cabinet{'s' if count > 1 else ''}" if en else f"{count} 个柜"
        legend.append(
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=math.sqrt(32 + 26 * count),
                mfc="#75a9b9",
                mec="white",
                label=label,
            )
        )
    if len(closed):
        legend.append(
            Line2D(
                [],
                [],
                ls="",
                marker="x",
                ms=7,
                color="#a4acb3",
                label="Unopened candidate" if en else "未开放候选位置",
            )
        )
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.065, 0.063),
        ncol=3 if len(closed) else 5,
        frameon=False,
        fontsize=9 if en else 10,
        handletextpad=0.5,
        columnspacing=1.3 if en else 1.5,
    )
    cax = fig.add_axes([0.30, 0.05, 0.38, 0.011])
    cb = fig.colorbar(m, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.ax.set_xticklabels(["0%", "50%", "100%"])
    cb.ax.tick_params(labelsize=9, length=0, pad=3)
    cb.outline.set_visible(False)
    fig.text(
        0.055,
        0.047,
        "Capacity utilization" if en else "柜处理能力利用率",
        fontsize=10,
        color="#5d6c78",
    )
    _foot(
        fig,
        (
            "Circle area: cabinet count; color: utilization. Squares: bases. Diamonds: swap stations."
            if en
            else "圆面积表示柜数量，颜色表示处理能力利用率；方形基地标签与菱形换电站标出实际位置。"
        ),
        view.get("language", "zh"),
    )
    return _save(fig, out, f"H{H}_facilities")


def _select_coverage(edges, target=0.9):
    rows = sorted(edges, key=lambda r: (-r["flow_per_hour"], r["base"], r["u"], r["v"]))
    total = sum(r["flow_per_hour"] for r in rows)
    selected = []
    shown = 0.0
    for r in rows:
        if shown >= target * total:
            break
        selected.append(r)
        shown += r["flow_per_hour"]
    return selected, dict(
        total_edges=len(rows),
        shown_edges=len(selected),
        total_arc_flow=total,
        shown_arc_flow=shown,
        coverage=shown / total if total else 1.0,
    )


def _lines(ax, view, edges, *, alpha=0.4, small=False):
    if not edges:
        return
    vmax = max(e["flow_per_hour"] for e in edges)
    # Draw thin/low-flow edges first, preventing one large path hiding all others.
    for e in sorted(edges, key=lambda r: r["flow_per_hour"]):
        line = view["xy"][[e["u"], e["v"]]]
        f = e["flow_per_hour"]
        width = (0.25 + 2.15 * math.sqrt(f / vmax)) * (0.72 if small else 1.0)
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=BASE_COLORS[e["base"] % len(BASE_COLORS)],
            lw=width,
            alpha=alpha,
            zorder=2,
            solid_capstyle="round",
        )


def _corridors(view, out, target):
    c = view["case"]
    H, L = c["row"]["H"], c["row"]["L"]
    en = view.get("language") == "en"
    edges = [
        e for e in view["edges"] if e["kind"] == "service" and e["phase"] == "loaded"
    ]
    shown, coverage = _select_coverage(edges, target)
    fig = plt.figure(figsize=(9, 12))
    ax = fig.add_axes([0.055, 0.10, 0.89, 0.785])
    _map(ax, view)
    _lines(ax, view, shown, alpha=0.41)
    ids = np.flatnonzero(view["opened"][L : L + H]) + L
    ax.scatter(
        view["xy"][ids, 0],
        view["xy"][ids, 1],
        s=15,
        c="white",
        edgecolors="#6d8999",
        linewidths=0.65,
        zorder=6,
    )
    ids = np.flatnonzero(view["opened"][L + H :]) + L + H
    ax.scatter(
        view["xy"][ids, 0],
        view["xy"][ids, 1],
        s=48,
        marker="D",
        c=SWAP_COLOR,
        edgecolors="white",
        lw=0.8,
        zorder=7,
    )
    _base_labels(ax, view, color_by_base=True)
    if en:
        _title(
            fig,
            f"H{H} · Loaded flight segments",
            "Nanshan, Shenzhen  |  Color: departure base; line width: segment flow",
            f"{coverage['shown_edges']:,} base–segment records shown, covering {coverage['coverage']:.1%} of loaded-segment flow",
        )
    else:
        _title(
            fig,
            f"H{H} · 带餐航段与服务基地",
            "深圳南山  |  颜色对应任务出发基地，线宽对应航段流量",
            f"显示 {coverage['shown_edges']:,} 条基地—航段记录，覆盖 {coverage['coverage']:.1%} 带餐航段流量",
        )
    legend = [
        Line2D([], [], color=BASE_COLORS[l % len(BASE_COLORS)], lw=3, label=f"L{l+1}")
        for l in range(L)
        if view["opened"][l]
    ]
    legend.append(
        Line2D(
            [],
            [],
            ls="",
            marker="D",
            ms=7,
            mfc=SWAP_COLOR,
            mec="white",
            label="Swap station E" if en else "换电站 E",
        )
    )
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.065, 0.046),
        ncol=5,
        frameon=False,
        fontsize=10,
        columnspacing=1.9,
    )
    _foot(
        fig,
        (
            "Actual locker-to-locker segments, including swap stops. Directed edges are in the full data table."
            if en
            else "绘制真实 H–H 航段及其换电中转段；同一航段可由多个基地任务使用。方向见完整边表。"
        ),
        view.get("language", "zh"),
    )
    paths = _save(fig, out, f"H{H}_service_corridors")
    # Individual base panels include every positive-flow loaded segment, without
    # inventing radial assignment lines or suppressing minor bases.
    facet_paths = []
    nrows = math.ceil(L / 2)
    fig, axs = plt.subplots(nrows, 2, figsize=(12, 7 * nrows), squeeze=False)
    fig.subplots_adjust(
        left=0.035, right=0.975, bottom=0.018, top=0.958, hspace=0.11, wspace=0.035
    )
    for l, ax in enumerate(axs.flat):
        if l >= L:
            ax.set_visible(False)
            continue
        _map(ax, view, small=True)
        base_edges = [e for e in edges if e["base"] == l]
        _lines(ax, view, base_edges, alpha=0.50, small=True)
        _base_labels(ax, view, indices={l}, color_by_base=True, small=True)
        served = c["tables"]["bases"][l]["service_out"]
        ax.set_title(
            f"L{l+1}  ·  {served:,.0f} {'orders/h' if en else '单/小时'}",
            loc="left",
            fontsize=15,
            pad=8,
            color=BASE_COLORS[l % len(BASE_COLORS)],
        )
    fig.suptitle(
        f"H{H} · {'All loaded flight segments by base' if en else '各基地的全部带餐航段'}",
        x=0.04,
        y=0.996,
        ha="left",
        fontsize=22,
        weight="bold",
        color="#203747",
    )
    fig.text(
        0.04,
        0.978,
        (
            "Every positive-flow loaded segment is shown for each base; map extent is identical across panels."
            if en
            else "各面板均显示该基地任务的 100% 正流量航段；地图范围一致。"
        ),
        fontsize=11,
        color="#607482",
    )
    fig.text(
        0.04,
        0.004,
        (
            "Coordinates follow the historical plotting convention. © OpenStreetMap contributors"
            if en
            else "坐标转换沿用旧图规则。© OpenStreetMap contributors"
        ),
        fontsize=9,
        color="#788997",
    )
    facet_paths += _save(fig, out, f"H{H}_base_panels")
    return paths + facet_paths, coverage


def _reposition_map(view, out):
    case = view["case"]
    H = case["row"]["H"]
    en = view["language"] == "en"
    edges = [
        e
        for e in view["edges"]
        if e["kind"] == "reposition" and e["flow_per_hour"] > 1e-4
    ]
    if not edges:
        return []
    fig = plt.figure(figsize=(9, 12))
    ax = fig.add_axes([0.055, 0.10, 0.89, 0.785])
    _map(ax, view)
    maxflow = max(e["flow_per_hour"] for e in edges)
    for e in edges:
        u, v = view["xy"][e["u"]], view["xy"][e["v"]]
        flow = e["flow_per_hour"]
        width = 0.9 + 3.0 * np.sqrt(flow / maxflow)
        if np.linalg.norm(v - u) < 1e-8:
            continue
        ax.annotate(
            "",
            xy=v,
            xytext=u,
            zorder=9,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=width,
                color="#c96b3e",
                alpha=0.82,
                mutation_scale=13,
                shrinkA=5,
                shrinkB=5,
                connectionstyle="arc3,rad=.055",
            ),
        )
        midpoint = (u + v) / 2
        ax.annotate(
            f"{flow:.1f}/h",
            midpoint,
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=10,
            color="#9d4d2a",
            zorder=12,
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=2),
        )
    L = case["row"]["L"]
    ids = np.flatnonzero(view["opened"][L + H :]) + L + H
    ax.scatter(
        view["xy"][ids, 0],
        view["xy"][ids, 1],
        s=70,
        marker="D",
        c=SWAP_COLOR,
        ec="white",
        zorder=11,
    )
    for i in ids:
        if any(e["u"] == i or e["v"] == i for e in edges):
            ax.annotate(
                view["labels"][i],
                view["xy"][i],
                xytext=(6, -12),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
            )
    _base_labels(ax, view)
    flow = case["summary"]["empty_flow"]
    _title(
        fig,
        f"H{H} · {'Empty repositioning' if en else '基地间空载重调度'}",
        (
            "Nanshan, Shenzhen  |  Arrows: direction; labels: physical-segment flow"
            if en
            else "深圳南山  |  箭头为飞行方向，数字为物理航段流量"
        ),
        f"{flow:,.2f} {'repositioning missions/h; segments above 0.0001/h are shown' if en else '架次/小时；显示流量超过0.0001/小时的重调度航段'}",
    )
    _foot(
        fig,
        (
            "Service-mission return flights are reported separately from inter-base repositioning."
            if en
            else "任务内返航与基地间重调度分别统计；图中仅包含重调度路径。"
        ),
        view["language"],
    )
    return _save(fig, out, f"H{H}_repositioning")


def render_case_maps(case, out, cache=CACHE):
    _style()
    view = _prepare(case, Path(cache))
    manifest = dict(
        provider="OpenStreetMap Standard",
        cache=str(cache),
        downloaded=False,
        coordinate_contract="Recorded local-km projection -> historical GCJ-02-to-WGS84 convention -> EPSG:3857",
        model_coordinates_modified=False,
        source_datum_confirmed_by_metadata=False,
        source_metadata=str(view["source_meta_path"]),
        source_metadata_sha256=_sha(view["source_meta_path"]),
        tile_sources=view["tile_sources"],
        boundary=view["boundary_source"],
        maps={},
    )
    _write_csv(
        out / "all_base_phase_edges.csv",
        [
            dict(**e, source=view["labels"][e["u"]], destination=view["labels"][e["v"]])
            for e in view["edges"]
        ],
    )
    _write_csv(
        out / "geographic_nodes.csv",
        [
            dict(
                node=view["labels"][i],
                open=int(view["opened"][i]),
                local_x_km=p[0],
                local_y_km=p[1],
                longitude=view["lonlat"][i, 0],
                latitude=view["lonlat"][i, 1],
            )
            for i, p in enumerate(view["local"])
        ],
    )
    empty_total = sum(
        e["flow_per_hour"] for e in view["edges"] if e["kind"] == "reposition"
    )
    empty_shown = sum(
        e["flow_per_hour"]
        for e in view["edges"]
        if e["kind"] == "reposition" and e["flow_per_hour"] > 1e-4
    )
    for lang in ("cn", "en"):
        _style()
        directory = out / f"maps_{lang}"
        directory.mkdir(exist_ok=True)
        view["language"] = "en" if lang == "en" else "zh"
        paths = _facilities(view, directory)
        more, coverage = _corridors(view, directory, 0.9)
        paths += more
        paths += _reposition_map(view, directory)
        manifest["maps"][lang] = dict(
            files=paths,
            service_coverage=coverage,
            reposition_display_threshold_per_hour=1e-4,
            reposition_arc_flow_coverage=(
                empty_shown / empty_total if empty_total else None
            ),
        )
    (out / "map_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


def run(root, output=None, coverage=0.90, cache=CACHE, language="en", hubs=None):
    """Render one case or each immediate saved case in an experiment directory."""
    if language not in {"zh", "en"}:
        raise ValueError("language must be zh or en")
    root = Path(root).resolve()
    out = Path(output).resolve() if output else root / "maps"
    out.mkdir(parents=True, exist_ok=True)
    paths = (
        [root]
        if (root / "result.json").is_file()
        else sorted(p.parent for p in root.glob("*/result.json"))
    )
    cases = [load_case(p) for p in paths]
    if hubs is not None:
        cases = [c for c in cases if c.get("row", {}).get("H") in set(hubs)]
    if not cases:
        raise ValueError("No matching saved cases found")
    _style()
    manifest = dict(language=language, cache=str(cache), downloaded=False, cases=[])
    for case in cases:
        if case.get("pending") or case["row"]["validation"] != "通过":
            raise ValueError(f"Case operations are not verified: {case['path']}")
        directory = out if len(cases) == 1 else out / case["path"].name
        directory.mkdir(parents=True, exist_ok=True)
        view = _prepare(case, Path(cache))
        view["language"] = language
        files = _facilities(view, directory)
        more, selected = _corridors(view, directory, coverage)
        files += more + _reposition_map(view, directory)
        _write_csv(directory / "all_base_phase_edges.csv", view["edges"])
        manifest["cases"].append(
            dict(
                case=str(case["path"]),
                files=files,
                coverage=selected,
                source_metadata=str(view["source_meta_path"]),
                tile_sources=view["tile_sources"],
            )
        )
    (out / "map_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    args = parser.parse_args()
    run(args.root, args.output, args.coverage, args.cache, args.language)
