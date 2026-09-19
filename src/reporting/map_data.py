"""Prepare physical route flows and saved projection metadata for maps."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
from .projection import local_km_to_plot_lonlat, _mercator
from .basemap import _basemap, _boundary


def _prepare(case, cache):
    provenance = case["metadata"].get("provenance", {})
    source_meta = provenance.get("instance_metadata")
    if source_meta is not None:
        source_meta_path = case["path"] / "input_metadata.json"
    else:
        directory = Path(provenance["directory"])
        source_meta_path = directory / "metadata.json"
        source_meta = json.loads(source_meta_path.read_text())
    with np.load(case["path"] / "input_arrays.npz") as a:
        local = np.concatenate((a["bases"], a["hubs"], a["remote"]))
    lonlat = np.array([local_km_to_plot_lonlat(*p, source_meta) for p in local])
    xy = _mercator(lonlat)
    with np.load(case["path"] / "solution.npz") as s:
        opened = s["facility"] > 0.5
    # 800 ground metres of surrounding map keep the coast and adjacent streets.
    margin = 800 / math.cos(math.radians(float(lonlat[:, 1].mean())))
    lo, hi = xy.min(axis=0) - margin, xy.max(axis=0) + margin
    bounds = (lo[0], hi[0], lo[1], hi[1])
    tile_image, tile_extent, tile_sources = _basemap(bounds, cache)
    boundary, boundary_source = _boundary(cache)
    L, H = case["row"]["L"], case["row"]["H"]
    labels = (
        [f"L{i+1}" for i in range(L)]
        + [f"H{i+1}" for i in range(H)]
        + [f"E{i+1}" for i in range(len(xy) - L - H)]
    )
    # A saved segment identifies its independently reconstructed physical path.
    paths = {(p["kind"], p["index"]): p for p in case["tables"]["paths"]}
    weights = defaultdict(float)
    for s in case["segments"]:
        p = paths[(s["kind"], s["path_index"])]
        key = (
            s["kind"],
            int(p["l"]),
            int(p["m"]),
            s["phase"],
            int(s["u_index"]),
            int(s["v_index"]),
        )
        weights[key] += float(s["flow_per_hour"])
    edges = [
        dict(kind=k, base=l, return_base=m, phase=p, u=u, v=v, flow_per_hour=f)
        for (k, l, m, p, u, v), f in sorted(weights.items())
        if f > 0
    ]
    return dict(
        case=case,
        source_meta=source_meta,
        source_meta_path=source_meta_path,
        local=local,
        lonlat=lonlat,
        xy=xy,
        opened=opened,
        bounds=bounds,
        image=tile_image,
        extent=tile_extent,
        tile_sources=tile_sources,
        boundary=boundary,
        boundary_source=boundary_source,
        labels=labels,
        edges=edges,
    )
