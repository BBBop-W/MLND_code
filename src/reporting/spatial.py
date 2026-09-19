"""Directed flow tables, schematic networks and base-pair flow matrices."""

from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from .style import plt, _grid, _save
from .io import _csv
from .formatting import _n
from .scale_charts import COLORS


def aggregate_edges(segments):
    """Preserve direction and operating phase, including edges below plot cutoff."""
    weights = defaultdict(float)
    for segment in segments:
        phase = segment["phase"]
        if segment.get("kind") == "empty":
            phase = "empty"
        weights[(phase, int(segment["u_index"]), int(segment["v_index"]))] += float(
            segment["flow_per_hour"]
        )
    return [
        dict(phase=p, u=u, v=v, flow_per_hour=f)
        for (p, u, v), f in sorted(weights.items())
    ]


def _selected(edges, phases, max_edges=65, min_flow=1.0):
    rows = sorted(
        (e for e in edges if e["phase"] in phases and e["flow_per_hour"] >= min_flow),
        key=lambda e: (-e["flow_per_hour"], e["u"], e["v"]),
    )[:max_edges]
    total = sum(e["flow_per_hour"] for e in edges if e["phase"] in phases)
    shown = sum(e["flow_per_hour"] for e in rows)
    return rows, dict(
        min_flow=min_flow,
        maximum_edges=max_edges,
        actual_min_flow=min((e["flow_per_hour"] for e in rows), default=None),
        shown_arc_flow=shown,
        total_arc_flow=total,
        coverage=shown / total if total else 1.0,
    )


def _draw_edges(ax, edges, xy, color, maximum, alpha=0.65, dashed=False):
    if not edges:
        return
    lines = [[xy[e["u"]], xy[e["v"]]] for e in edges]
    widths = [0.5 + 3.5 * math.sqrt(e["flow_per_hour"] / maximum) for e in edges]
    ax.add_collection(
        LineCollection(
            lines,
            colors=color,
            linewidths=widths,
            alpha=alpha,
            linestyles="--" if dashed else "-",
            zorder=2,
        )
    )
    for edge, width in zip(edges, widths):
        a, b = xy[edge["u"]], xy[edge["v"]]
        if np.linalg.norm(b - a) < 1e-8:
            continue
        mid = a + 0.58 * (b - a)
        ax.annotate(
            "",
            xy=mid + 0.04 * (b - a),
            xytext=mid - 0.04 * (b - a),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=max(0.6, width * 0.65),
                mutation_scale=6 + width * 1.4,
                alpha=alpha,
            ),
            zorder=3,
        )


def _spatial(c, figdir, csvdir, max_edges):
    path = c["path"]
    if not (path / "input_arrays.npz").exists() or not (path / "solution.npz").exists():
        return None
    with np.load(path / "input_arrays.npz", allow_pickle=False) as arrays:
        bases, hubs, remote = arrays["bases"], arrays["hubs"], arrays["remote"]
        xy = np.concatenate((bases, hubs, remote))
    with np.load(path / "solution.npz", allow_pickle=False) as solution:
        facility = solution["facility"]
    edges = aggregate_edges(c["segments"])
    L, H = len(bases), len(hubs)

    def node(v):
        return f"L{v+1}" if v < L else (f"H{v-L+1}" if v < L + H else f"E{v-L-H+1}")

    csvrows = [
        dict(
            **e,
            source=node(e["u"]),
            destination=node(e["v"]),
            x_u=float(xy[e["u"], 0]),
            y_u=float(xy[e["u"], 1]),
            x_v=float(xy[e["v"], 0]),
            y_v=float(xy[e["v"], 1]),
        )
        for e in edges
    ]
    _csv(
        csvdir / f"{path.name}_all_edges.csv",
        csvrows,
        [
            "phase",
            "u",
            "v",
            "flow_per_hour",
            "source",
            "destination",
            "x_u",
            "y_u",
            "x_v",
            "y_v",
        ],
    )
    span = np.ptp(xy, axis=0)
    panel_height = min(10.0, max(4.2, 6.1 * span[1] / max(span[0], 1e-8)))
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 2 * panel_height + 2.7), layout="constrained"
    )
    selections = {}
    for ax, kind, label, phases, color in [
        (
            axes[0],
            "loaded",
            "带餐飞行：H → H（可经换电站）",
            {"loaded"},
            COLORS["loaded"],
        ),
        (
            axes[1],
            "connection",
            "接餐与返回飞行；橙色为基地间空载重调度",
            {"pre_pickup", "return"},
            COLORS["first"],
        ),
    ]:
        selected, info = _selected(edges, phases, max_edges=max_edges)
        selections[kind] = info
        maximum = max([e["flow_per_hour"] for e in selected] + [1.0])
        _draw_edges(ax, selected, xy, color, maximum)
        if kind == "connection":
            empty, ei = _selected(edges, {"empty"}, max_edges=max_edges, min_flow=0.01)
            selections["empty"] = ei
            _draw_edges(
                ax,
                empty,
                xy,
                COLORS["empty"],
                max([e["flow_per_hour"] for e in empty] + [1.0]),
                alpha=0.9,
                dashed=True,
            )
        closed = facility[L : L + H] < 0.5
        ax.scatter(
            hubs[closed, 0],
            hubs[closed, 1],
            s=15,
            facecolors="none",
            edgecolors="#c2c5ca",
            zorder=3,
        )
        for coords, idx, marker, color_node, size in [
            (hubs, L, "s", "#427f99", 32),
            (remote, L + H, "D", "#c08c33", 65),
            (bases, 0, "^", "#283c57", 130),
        ]:
            mask = facility[idx : idx + len(coords)] > 0.5
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=size,
                marker=marker,
                facecolor=color_node,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )
        for idx, point in enumerate(bases):
            if facility[idx] < 0.5:
                continue
            ax.annotate(
                f"L{idx+1}",
                point,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                zorder=6,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale_view()
        ax.margins(0.12)
        ax.set_xlabel("东西向位置（km）")
        ax.set_ylabel("南北向位置（km）")
        threshold = _n(
            (
                info["actual_min_flow"]
                if info["actual_min_flow"] is not None
                else info["min_flow"]
            ),
            1,
        )
        ax.set_title(
            label
            + f'\n显示前 {len(selected)} 条（≥{threshold} 架次/h）；占该类边流量 {info["coverage"]:.1%}',
            fontsize=13,
            loc="left",
            pad=10,
        )
        _grid(ax, "both")
    handles = [
        Line2D([], [], linestyle="", marker=m, color=co, label=la, markersize=s)
        for m, co, la, s in [
            ("^", "#283c57", "基地 L", 9),
            ("s", "#427f99", "开放柜点 H", 7),
            ("D", "#c08c33", "换电站 E", 7),
        ]
    ]
    fig.suptitle(f"H{H:03d} 空间网络 · 线宽表示流量", fontsize=18, x=0.08, ha="left")
    axes[0].legend(handles=handles, frameon=False, loc="upper right", fontsize=10)
    _save(fig, figdir, path.name + "_network")
    return selections


def _base_matrices(c, out):
    service = np.asarray(c["summary"]["service_flow_by_base_pair"], dtype=float)
    empty = np.asarray(c["summary"]["empty_flow_by_base_pair"], dtype=float)
    L = len(service)
    _csv(
        out.parent / "tables" / (c["path"].name + "_base_flows.csv"),
        [
            dict(
                source=f"L{i+1}",
                destination=f"L{j+1}",
                service_per_hour=float(service[i, j]),
                empty_per_hour=float(empty[i, j]),
            )
            for i in range(L)
            for j in range(L)
        ],
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, max(7.5, L * 0.62 * 2)), layout="constrained"
    )
    for ax, matrix, title, cmap in [
        (axes[0], service, "服务任务：出发基地 → 返回基地", "Blues"),
        (axes[1], empty, "空载重调度：起点基地 → 终点基地", "Oranges"),
    ]:
        vmax = max(float(matrix.max()), 1.0)
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax)
        for i in range(L):
            for j in range(L):
                value = f"{matrix[i,j]:.0f}" if L >= 6 else f"{matrix[i,j]:.1f}"
                ax.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > 0.55 * vmax else "#253540",
                    fontsize=12,
                )
        ax.set_xticks(range(L), [f"L{i+1}" for i in range(L)])
        ax.set_yticks(range(L), [f"L{i+1}" for i in range(L)])
        ax.set_xlabel("终点")
        ax.set_ylabel("起点")
        ax.set_title(
            title + ("\n格内显示整数，完整精度见 CSV" if L >= 6 else ""),
            loc="left",
            fontsize=13,
        )
        fig.colorbar(im, ax=ax, shrink=0.78, label="任务 / 小时")
    fig.suptitle(
        f'H{c["row"]["H"]:03d} 基地间流量 · 跨基地任务 {c["row"]["cross_base_pct"]:.1f}%',
        fontsize=16,
        x=0.1,
        ha="left",
    )
    _save(fig, out, c["path"].name + "_base_flows")
