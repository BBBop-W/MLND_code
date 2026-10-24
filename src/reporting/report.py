"""Model-specific reports and figures built from independently validated results."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from src.experiments.storage import load_seed, write_json, sha256
from src.operations.export import export_operations
from src.model.types import MODEL_VERSION
from .style import _style, _save, _grid, plt


def _coordinates(inputs):
    meta = inputs.get("metadata", {})
    bases = np.asarray(inputs.get("bases", meta.get("bases", [])), float)
    hubs = np.asarray(inputs.get("hubs", meta.get("hubs", [])), float)
    if (
        bases.ndim != 2
        or hubs.ndim != 2
        or bases.shape[1:] != (2,)
        or hubs.shape[1:] != (2,)
    ):
        return None
    return np.vstack((bases, hubs))


def _projection_metadata(value):
    if isinstance(value, dict):
        if "projection_center_lon" in value and "projection_center_lat" in value:
            return value
        for item in value.values():
            found = _projection_metadata(item)
            if found:
                return found
    return None


def network_figure(inputs, result, payload, output, *, cached_basemap=False):
    xy = _coordinates(inputs)
    if xy is None:
        return dict(generated=False, reason="No geographic coordinates in saved inputs")
    from src.data.instances import to_planning_data

    data = to_planning_data(inputs)
    if len(xy) != data.n_nodes:
        raise ValueError("Map coordinates disagree with candidate node counts")
    sources = []
    fig, ax = plt.subplots(figsize=(10, 8))
    label = "km"
    if cached_basemap:
        from .projection import local_km_to_plot_lonlat, _mercator
        from .basemap import _basemap, CACHE

        meta = _projection_metadata(inputs)
        if meta is None or meta.get("geographic_crs") not in ("GCJ02", "WGS84"):
            plt.close(fig)
            raise ValueError(
                "Cached basemap requires explicit projection center and geographic_crs (GCJ02/WGS84)"
            )
        xy = _mercator(
            [
                local_km_to_plot_lonlat(
                    x, y, meta, assume_gcj02=meta["geographic_crs"] == "GCJ02"
                )
                for x, y in xy
            ]
        )
        span = np.maximum(np.ptp(xy, axis=0), 100)
        low = xy.min(axis=0) - 0.1 * span
        high = xy.max(axis=0) + 0.1 * span
        image, extent, sources = _basemap((low[0], high[0], low[1], high[1]), CACHE)
        ax.imshow(image, extent=extent, origin="upper", zorder=0)
        ax.text(
            0.01,
            0.01,
            "© OpenStreetMap contributors",
            transform=ax.transAxes,
            fontsize=8,
        )
        label = "EPSG:3857 m"
    colors = {
        "first": "#2b6cb0",
        "transfer": "#d28b26",
        "second": "#23896e",
        "return": "#8a70aa",
        "empty": "#8a939e",
    }
    arcs = {}
    for empty, key in ((False, "_missions"), (True, "_relocations")):
        for record in result.get(key, []):
            flow = float(record["flow"])
            if flow <= 1e-8:
                continue
            legs = record["path"]["legs"]
            stages = (
                ["empty"]
                if empty
                else (
                    ["first", "first", "return"]
                    if len(legs) == 3
                    else ["first", "first", "transfer", "second", "return"]
                )
            )
            for stage, leg in zip(stages, legs):
                for u, v in zip(leg, leg[1:]):
                    arcs[stage, u, v] = arcs.get((stage, u, v), 0) + flow
    maxflow = max(arcs.values(), default=1)
    used = set()
    for (stage, u, v), flow in sorted(arcs.items(), key=lambda item: item[1]):
        ax.annotate(
            "",
            xy=xy[v],
            xytext=xy[u],
            arrowprops=dict(
                arrowstyle="-|>",
                color=colors[stage],
                lw=0.6 + 2.5 * flow / maxflow,
                alpha=0.65,
                shrinkA=5,
                shrinkB=5,
            ),
            zorder=2,
        )
        used.add(stage)
    opened = np.asarray(result["_solution"]["facility"]) > 0.5
    for subset, marker, name, color in [
        (np.arange(data.n_l), "s", "Base", "#353c50"),
        (np.arange(data.n_l, data.n_nodes), "o", "Locker", "#3973a5"),
    ]:
        ax.scatter(
            xy[subset, 0],
            xy[subset, 1],
            s=65,
            marker=marker,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            label=name + " candidate",
            zorder=4,
        )
        selected = subset[opened[subset]]
        ax.scatter(
            xy[selected, 0],
            xy[selected, 1],
            s=65,
            marker=marker,
            c=color,
            label=name + " open",
            zorder=5,
        )
    for i, (x, y) in enumerate(xy):
        ax.annotate(
            ("B" + str(i)) if i < data.n_l else ("H" + str(i - data.n_l)),
            (x, y),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
            zorder=6,
        )
    for stage in colors:
        if stage in used:
            ax.plot([], [], color=colors[stage], label=stage)
    ax.set(
        xlabel=f"Easting ({label})",
        ylabel=f"Northing ({label})",
        title="Selected network and physical flight stages",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    _grid(ax, "both")
    _save(fig, output, "network")
    return dict(generated=True, cached_basemap=cached_basemap, sources=sources)


def operating_figures(payload, output):
    t = payload["tables"]
    s = payload["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    costs = t["costs"]
    axes[0].barh(
        [r["component"] for r in costs],
        [r["cny_per_hour"] for r in costs],
        color="#3973a5",
    )
    axes[0].set(xlabel="Cost per hour", title="Fulfillment cost")
    _grid(axes[0])
    fleet = t["fleet"]
    axes[1].barh(
        [r["component"] for r in fleet],
        [max(0, r["drones"]) for r in fleet],
        color="#23896e",
    )
    axes[1].set(xlabel="Drones (continuous)", title="Fleet occupation")
    _grid(axes[1])
    fig.tight_layout()
    _save(fig, output, "cost_and_fleet")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lockers = t["lockers"]
    index = np.arange(len(lockers))
    axes[0].bar(
        index,
        [r["mean_occupancy"] for r in lockers],
        color="#3973a5",
        label="Mean reservation load",
    )
    axes[0].plot(
        index,
        [r["mean_threshold"] for r in lockers],
        "o-",
        color="#d28b26",
        label="Poisson mean threshold",
    )
    axes[0].set(
        xlabel="Locker", ylabel="Mean compartments", title="Locker reservation workload"
    )
    axes[0].legend()
    _grid(axes[0], "y")
    axes[1].bar(
        ["Single tasks", "Double tasks", "Drone orders", "Direct orders"],
        [s["single_tasks"], s["double_tasks"], s["served"], s["direct"]],
        color=["#3973a5", "#23896e", "#426676", "#8a939e"],
    )
    axes[1].set(ylabel="Per hour", title="Tasks and served orders")
    axes[1].tick_params(axis="x", rotation=20)
    _grid(axes[1], "y")
    fig.tight_layout()
    _save(fig, output, "workload")
    rows = [r for r in t["orders"] if r["mean_receipt_minutes"] is not None]
    if rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        deadlines = np.array([r["deadline_minutes"] for r in rows])
        receipt = np.array([r["mean_receipt_minutes"] for r in rows])
        ax.scatter(deadlines, receipt, c="#23896e", s=25)
        upper = max(max(deadlines), max(receipt))
        ax.plot([0, upper], [0, upper], "--", color="#8a939e")
        ax.set(
            xlabel="OD mean receipt limit (minutes)",
            ylabel="Mean receipt time (minutes)",
            title="Each order clock starts at its own request",
        )
        _grid(ax, "both")
        _save(fig, output, "receipt_times")


def run(path, output=None, draw_maps=True, *, cached_basemap=False):
    path = Path(path).resolve()
    out = Path(output).resolve() if output else path / "report"
    data, result = load_seed(path)
    out.mkdir(parents=True, exist_ok=True)
    payload = export_operations(data, result, out)
    _style()
    operating_figures(payload, out)
    map_info = (
        network_figure(data, result, payload, out, cached_basemap=cached_basemap)
        if draw_maps
        else dict(generated=False, reason="disabled")
    )
    s = payload["summary"]
    lines = [
        "# 连续配送网络实验报告",
        "",
        f"模型：`{MODEL_VERSION}`；算法：`{result.get('method')}`；路径范围：`{result.get('path_scope')}`。",
        f"流量约束：`{s['flow_policy']}`；prefix 为逐首段比例，aggregate 为跨首段汇总任务量。",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 总成本/小时 | {s['objective']:.6f} |",
        f"| 认证下界 | {result.get('bound')} |",
        f"| gap | {result.get('gap')} |",
        f"| 定价完整 | {result.get('pricing_complete')} |",
        f"| 停止原因 | {result.get('termination_reason')} |",
        f"| 原始订单/小时 | {s['demand']:.6f} |",
        f"| 无人机订单/小时 | {s['served']:.6f} |",
        f"| 单单任务/小时 | {s['single_tasks']:.6f} |",
        f"| 双单任务/小时 | {s['double_tasks']:.6f} |",
        f"| 连续机队规模 | {s['fleet']:.6f} |",
        "",
        "订单和任务分别计数：一条双单路径贡献一个任务和两个订单，同 OD 两次也分别计数。",
        "送达时间和完整运行时间采用均值限制；柜格风险采用潜在预约负荷的 Poisson 近似。",
        "所有运营指标从保存的物理路径独立重算。限时可行解不代表已达到最优。",
        "",
        "![成本与机队](cost_and_fleet.png)",
        "",
        "![任务和柜占用](workload.png)",
    ]
    if map_info["generated"]:
        lines += ["", "![网络](network.png)"]
    if (out / "receipt_times.png").exists():
        lines += ["", "![送达时间](receipt_times.png)"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = dict(
        model_version=MODEL_VERSION,
        flow_policy=s["flow_policy"],
        independent_validation_passed=True,
        optimization_rerun=False,
        map=map_info,
        sources={
            name: sha256(path / name)
            for name in (
                "input_metadata.json",
                "input_arrays.npz",
                "result.json",
                "solution.npz",
                "path_flows.json.gz",
            )
        },
        figures=sorted(p.name for p in out.glob("*.png")),
    )
    write_json(out / "source_manifest.json", manifest)
    return dict(
        output=str(out),
        objective=s["objective"],
        passed=True,
        figures=manifest["figures"],
    )
