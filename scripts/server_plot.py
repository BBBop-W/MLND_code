"""Render a hash-bound feasible network over the suite's offline OSM background.

This consumes the current mlnd_runtime primal certificate; it neither runs a
solver nor certifies the global lower bound. No network requests are made.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contained_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or outside-root file: {name}")
    return path


def projection_metadata(value):
    if isinstance(value, dict):
        required = {"projection_center_lon", "projection_center_lat", "crop_bounds_local_km"}
        if required.issubset(value):
            return value
        for child in value.values():
            found = projection_metadata(child)
            if found is not None:
                return found
    return None


def load_input(instance):
    """Read only frozen site geometry, checking every manifest member."""
    instance = Path(instance).resolve()
    manifest = read_json(instance / "manifest.json")
    if not {"model.json", "metadata.json", "sites.csv"}.issubset(manifest["files"]):
        raise ValueError("Input manifest must bind model, metadata and sites")
    for name, expected in manifest["files"].items():
        if sha256(contained_file(instance, name)) != expected:
            raise ValueError(f"Input SHA-256 mismatch: {name}")
    ids = read_json(instance / "model.json")["ids"]
    if ids["flight_nodes"] != ids["bases"] + ids["hubs"]:
        raise ValueError("Flight-node order must be bases followed by hubs")
    with (instance / "sites.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    sites = {row["site_id"]: row for row in rows}
    if len(sites) != len(rows) or set(sites) != set(ids["flight_nodes"]):
        raise ValueError("Inconsistent or duplicate site IDs")
    rows = [sites[name] for name in ids["flight_nodes"]]
    count_l = len(ids["bases"])
    if any(row["type"] != ("L" if i < count_l else "H") for i, row in enumerate(rows)):
        raise ValueError("Site types disagree with model IDs")
    metadata = projection_metadata(read_json(instance / "metadata.json"))
    if metadata is None:
        raise ValueError("No saved study window and projection metadata")
    center_lon = float(metadata["projection_center_lon"])
    center_lat = float(metadata["projection_center_lat"])
    xy = []
    for row in rows:
        lon = float(row["x_km"]) / (111.32 * math.cos(math.radians(center_lat))) + center_lon
        lat = float(row["y_km"]) / 111.32 + center_lat
        if not math.isfinite(lon) or not math.isfinite(lat) or not -85 < lat < 85:
            raise ValueError("Invalid projected coordinate")
        xy.append((6378137.0 * math.radians(lon),
                   6378137.0 * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))))
    return {"path": instance, "manifest": manifest, "metadata": metadata,
            "ids": ids, "xy": xy, "n_l": count_l, "n_h": len(ids["hubs"])}


def select_background(data, maps_root):
    maps_root = Path(maps_root).resolve()
    suite = maps_root.parent
    manifest_path = suite / "provenance" / "map_backgrounds.json"
    manifest = read_json(manifest_path)
    for entry in manifest["backgrounds"]:
        meta = data["metadata"]
        same_center = all(math.isclose(float(entry[k]), float(meta[k]), abs_tol=1e-8, rel_tol=0)
                          for k in ("projection_center_lon", "projection_center_lat"))
        same_window = all(math.isclose(float(entry["crop_bounds_local_km"][k]),
                                      float(meta["crop_bounds_local_km"][k]), abs_tol=1e-6, rel_tol=0)
                          for k in ("x_min", "x_max", "y_min", "y_max"))
        if not same_center or not same_window:
            continue
        path = contained_file(suite, entry["file"])
        if not path.is_relative_to(maps_root) or sha256(path) != entry["sha256"]:
            raise ValueError("Offline background SHA-256 mismatch or outside maps root")
        bounds = entry["extent_epsg3857"]
        if (len(bounds) != 4 or not all(math.isfinite(float(x)) for x in bounds)
                or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
            raise ValueError("Invalid offline background extent")
        if any(not (bounds[0] - 1e-5 <= x <= bounds[1] + 1e-5 and
                    bounds[2] - 1e-5 <= y <= bounds[3] + 1e-5) for x, y in data["xy"]):
            raise ValueError("Site outside saved background")
        return entry, path, manifest_path
    raise ValueError("No offline background matches this input's study window")


def load_verified_run(data, run):
    run = Path(run).resolve()
    certificate = read_json(run / "native_validation.json")
    if (certificate.get("passed") is not True or certificate.get("checks", {}).get("passed") is not True
            or certificate.get("validation_scope") != "current_model_primal_and_complete_input_preprocessing"
            or certificate.get("assignment_scope") != "task"
            or certificate.get("delivery_time_scope") != "courier_home"
            or certificate.get("native_manifest_matches") is not True):
        raise ValueError("Current independent primal validation did not pass")
    files = {"native_validation.json": sha256(run / "native_validation.json")}
    for name in ("result", "solution", "path_flows", "input_reference"):
        digest = sha256(run / (name + ".json"))
        if certificate.get(name + "_sha256") != digest:
            raise ValueError(f"Validation certificate no longer matches {name}.json")
        files[name + ".json"] = digest
    if certificate.get("manifest_sha256") != sha256(data["path"] / "manifest.json"):
        raise ValueError("Validation certificate belongs to a different input")
    reference = read_json(run / "input_reference.json")
    result = read_json(run / "result.json")
    if (reference.get("manifest") != data["manifest"]
            or result.get("max_orders") != certificate.get("max_orders")
            or reference.get("max_orders") != certificate.get("max_orders")):
        raise ValueError("Native result/input/certificate provenance mismatch")
    if (result.get("assignment_scope") != "task"
            or result.get("demand_formulation") != "preprocessed"
            or result.get("delivery_time_scope") != "courier_home"):
        raise ValueError("Unsupported native model semantics")
    solution = read_json(run / "solution.json")
    opened = solution["facility"]
    n = len(data["xy"])
    if len(opened) != n or not all(math.isfinite(float(v)) for v in opened):
        raise ValueError("Invalid facility vector")
    opened = [float(v) > .5 for v in opened]
    airlines = solution["airline"]
    if len(airlines) != n or any(len(row) != n for row in airlines):
        raise ValueError("Invalid airline matrix")
    active = set()
    for u, row in enumerate(airlines):
        for v, value in enumerate(row):
            if not math.isfinite(float(value)):
                raise ValueError("Nonfinite airline decision")
            if float(value) > .5:
                if u == v or not (opened[u] and opened[v]):
                    raise ValueError("Opened airline uses invalid/closed facilities")
                active.add((u, v))
    witness = read_json(run / "path_flows.json")
    arcs = defaultdict(lambda: [0.0, 0.0])
    for category, key in enumerate(("missions", "relocations")):
        for record in witness[key]:
            flow = float(record["flow"])
            if not math.isfinite(flow) or flow < -5e-5:
                raise ValueError("Invalid path flow")
            if flow <= 1e-8:
                continue
            for leg in record["path"]["legs"]:
                for u, v in zip(leg, leg[1:]):
                    if (isinstance(u, bool) or isinstance(v, bool) or not isinstance(u, int)
                            or not isinstance(v, int) or (u, v) not in active):
                        raise ValueError("Positive path flow uses an invalid or closed airline")
                    arcs[u, v][category] += flow
    return {"result": result, "opened": opened, "active": active,
            "arcs": dict(arcs), "files": files, "certificate": certificate}


def render_one(instance, run, maps_root, output=None):
    data = load_input(instance)
    state = load_verified_run(data, run)
    background, image_path, background_manifest = select_background(data, maps_root)
    identity = {"schema": "mlnd-current-offline-network-map/1", "script_sha256": sha256(__file__),
                "instance_manifest_sha256": sha256(data["path"] / "manifest.json"),
                "run_files": state["files"], "background_sha256": background["sha256"],
                "background_manifest_sha256": sha256(background_manifest)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = Path(output).resolve() if output is not None else Path(run).resolve() / "figures"
    output.mkdir(parents=True, exist_ok=True)
    target = output / fingerprint[:20]
    if target.exists():
        source = read_json(target / "network.sources.json")
        expected = source.get("figure_sha256", {})
        if (source.get("fingerprint") != fingerprint or set(expected) != {"network.png", "network.pdf"}
                or any(sha256(target / name) != digest for name, digest in expected.items())):
            raise ValueError("Existing figure directory failed verification; refusing overwrite")
        return figure_record(target, fingerprint, skipped=True)
    # Set before importing matplotlib, including when called from a process
    # with an unwritable home directory on a shared server.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mlnd-server-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from PIL import Image

    temporary = Path(tempfile.mkdtemp(prefix=".render-", dir=output))
    fig, ax = plt.subplots(figsize=(11, 10))
    try:
        fig.subplots_adjust(left=.09, right=.97, bottom=.19, top=.88)
        bounds = background["extent_epsg3857"]
        with Image.open(image_path) as image:
            ax.imshow(image, extent=bounds, origin="upper", zorder=0)
        xy, arcs, active = data["xy"], state["arcs"], state["active"]
        maximum = max((sum(values) for values in arcs.values()), default=1.0)
        for u, v in sorted(active):
            if (u, v) not in arcs:
                ax.plot([xy[u][0], xy[v][0]], [xy[u][1], xy[v][1]],
                        color="#707b83", ls=":", lw=.8, alpha=.65, zorder=1)
        for (u, v), amounts in sorted(arcs.items()):
            for category, flow in enumerate(amounts):
                if flow <= 1e-8:
                    continue
                start, end = xy[u], xy[v]
                coincident = math.dist(start, end) < 1e-6
                if coincident:
                    shift = (bounds[1] - bounds[0]) * .008
                    start = (start[0] - shift, start[1])
                    end = (end[0] + shift, end[1])
                radius = -1.4 if coincident else (.10 if (v, u) in active else 0)
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                             color="#b87722" if category else "#2361a4",
                             linestyle="--" if category else "-", linewidth=.7 + 2.2 * flow / maximum,
                             connectionstyle=f"arc3,rad={radius}", shrinkA=5, shrinkB=5, alpha=.8, zorder=3))
        for i, point in enumerate(xy):
            is_l = i < data["n_l"]
            color = "#a94132" if is_l else "#156b87"
            ax.scatter(*point, marker="s" if is_l else "o", s=90 if is_l else 40,
                       facecolors=color if state["opened"][i] else "white", edgecolors=color,
                       alpha=1 if state["opened"][i] else .6, linewidths=1.3, zorder=5)
            label = f"L{i}" if is_l else f"H{i - data['n_l']}"
            coincident = any(j != i and math.dist(point, other) < 1e-6 for j, other in enumerate(xy))
            offset = (-6, 6) if is_l and coincident else (4, 4)
            ax.annotate(label, point, xytext=offset, textcoords="offset points",
                        ha="right" if offset[0] < 0 else "left", fontsize=7 if len(xy) > 45 else 8,
                        bbox={"fc": "white", "ec": "none", "alpha": .7, "pad": .4}, zorder=6)
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{math.degrees(x / 6378137):.3f}°"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{math.degrees(2 * math.atan(math.exp(y / 6378137)) - math.pi / 2):.3f}°"))
        ax.set(xlabel="Longitude (approximate)", ylabel="Latitude (approximate)")
        result = state["result"]
        fig.suptitle(f"Validated feasible network | {data['n_l']} L / {data['n_h']} H | K{result['max_orders']}",
                     fontsize=17, y=.97)
        fig.text(.09, .925, f"{data['path'].name} | study square: {background['area_km2']:.2f} km²", fontsize=10)
        handles = [Line2D([], [], marker="s", ls="", color="#a94132", label="Base L"),
                   Line2D([], [], marker="o", ls="", color="#156b87", label="Locker H"),
                   Line2D([], [], color="#2361a4", label="Service flight"),
                   Line2D([], [], color="#b87722", ls="--", label="Empty flight"),
                   Line2D([], [], color="#707b83", ls=":", label="Open, unused link")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .105), ncol=5, frameon=False, fontsize=8)
        fig.text(.09, .09, "Filled sites: open; hollow sites: closed. Arrows: flight direction; width: traversals/hour. No OD plotted.", fontsize=8)
        fig.text(.09, .065, "Primal feasibility validated; this figure does not independently certify the global bound or optimality.", fontsize=8)
        fig.text(.09, .04, "Approximate source coordinates; study square is not an administrative boundary. "
                 + background.get("attribution", "© OpenStreetMap contributors | openstreetmap.org/copyright"), fontsize=7)
        for extension in ("png", "pdf"):
            fig.savefig(temporary / ("network." + extension), dpi=180, facecolor="white")
        # Detect a concurrent replacement instead of publishing a stale figure.
        if any(sha256(Path(run) / name) != digest for name, digest in state["files"].items()):
            raise ValueError("Native output changed while rendering")
        if sha256(image_path) != background["sha256"]:
            raise ValueError("Offline background changed while rendering")
        if (sha256(background_manifest) != identity["background_manifest_sha256"]
                or sha256(data["path"] / "manifest.json") != identity["instance_manifest_sha256"]):
            raise ValueError("Input or background manifest changed while rendering")
        for name, digest in data["manifest"]["files"].items():
            if sha256(contained_file(data["path"], name)) != digest:
                raise ValueError("Input changed while rendering")
        sources = {**identity, "fingerprint": fingerprint, "figure_state": "validated_feasible_network",
                   "independent_global_bound_certification": False, "od_plotted": False,
                   "positive_flow_display_threshold": 1e-8, "background": background,
                   "coordinate_assumption": "Saved local projection inverted; original coordinate datum unspecified.",
                   "site_ids": data["ids"]["flight_nodes"],
                   "opened_site_ids": [name for name, flag in zip(data["ids"]["flight_nodes"], state["opened"]) if flag],
                   "open_airlines": sorted(active), "operating_arcs": [{"u": u, "v": v, "service_per_hour": f[0], "empty_per_hour": f[1]} for (u, v), f in sorted(arcs.items())],
                   "figure_sha256": {f"network.{ext}": sha256(temporary / f"network.{ext}") for ext in ("png", "pdf")}}
        (temporary / "network.sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.rename(target)
    finally:
        plt.close(fig)
        if temporary.exists():
            shutil.rmtree(temporary)
    return figure_record(target, fingerprint, skipped=False)


def figure_record(target, fingerprint, *, skipped):
    return {"state": "rendered", "skipped": skipped, "fingerprint": fingerprint,
            "png": str(target / "network.png"), "pdf": str(target / "network.pdf"),
            "sources": str(target / "network.sources.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        record = render_one(args.input, args.result, args.maps_root, args.output)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
        print(json.dumps({"state": "failed", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        return 2
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
