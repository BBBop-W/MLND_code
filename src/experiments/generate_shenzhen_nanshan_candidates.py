import argparse
import html
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


RAW_COLUMNS = {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "customer_pay_time",
}


def _hour_duration(start_hour: int, end_hour: int) -> int:
    duration = (int(end_hour) - int(start_hour)) % 24
    if duration <= 0:
        raise ValueError("Hour windows must have positive duration under [start, end) semantics.")
    return duration


def _parse_hour(value) -> int:
    if pd.isna(value):
        raise ValueError("Missing time value.")
    text = str(value).strip()
    hour_text = text.split(":", 1)[0]
    return int(hour_text)


def _filter_period(frame: pd.DataFrame, dt_seq_ids: list[int] | None, start_hour: int, end_hour: int) -> pd.DataFrame:
    subset = frame.copy()
    if dt_seq_ids is not None:
        subset = subset[subset["dt_seq_id"].isin(dt_seq_ids)].copy()
    if subset.empty:
        return subset

    if "order_hour" not in subset.columns:
        subset["order_hour"] = subset["customer_pay_time"].map(_parse_hour)

    if start_hour < end_hour:
        mask = (subset["order_hour"] >= start_hour) & (subset["order_hour"] < end_hour)
    else:
        mask = (subset["order_hour"] >= start_hour) | (subset["order_hour"] < end_hour)
    return subset[mask].copy()


def _project(frame: pd.DataFrame, lon_center: float, lat_center: float) -> tuple[pd.DataFrame, float, float]:
    projected = frame.copy()
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(lat_center * math.pi / 180.0)
    projected["sender_lng"] = projected["pickup_longitude"]
    projected["sender_lat"] = projected["pickup_latitude"]
    projected["recipient_lng"] = projected["dropoff_longitude"]
    projected["recipient_lat"] = projected["dropoff_latitude"]
    projected["sender_x_km"] = (projected["sender_lng"] - lon_center) * km_per_deg_lon
    projected["sender_y_km"] = (projected["sender_lat"] - lat_center) * km_per_deg_lat
    projected["recipient_x_km"] = (projected["recipient_lng"] - lon_center) * km_per_deg_lon
    projected["recipient_y_km"] = (projected["recipient_lat"] - lat_center) * km_per_deg_lat
    return projected, km_per_deg_lon, km_per_deg_lat


def _clip_area(frame: pd.DataFrame, map_half_size_km: float) -> pd.DataFrame:
    return frame[
        frame["sender_x_km"].between(-map_half_size_km, map_half_size_km)
        & frame["sender_y_km"].between(-map_half_size_km, map_half_size_km)
        & frame["recipient_x_km"].between(-map_half_size_km, map_half_size_km)
        & frame["recipient_y_km"].between(-map_half_size_km, map_half_size_km)
    ].copy()


def _weighted_kmeans(points, weights, n_clusters: int, label: str, seed: int) -> list[tuple[float, float]]:
    if n_clusters <= 0:
        return []
    if len(points) < n_clusters:
        raise ValueError(f"Cannot generate {n_clusters} {label} candidates from only {len(points)} seed points.")
    if n_clusters == 1:
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr / max(float(weights_arr.sum()), 1.0)
        center = np.average(np.asarray(points, dtype=float), axis=0, weights=weights_arr)
        return [tuple(float(v) for v in center)]

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = km.fit_predict(np.asarray(points, dtype=float), sample_weight=np.asarray(weights, dtype=float))
    ranked_labels = (
        pd.DataFrame({"label": labels, "weight": weights})
        .groupby("label")["weight"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    return [
        (float(km.cluster_centers_[cluster][0]), float(km.cluster_centers_[cluster][1]))
        for cluster in ranked_labels
    ]


def _assign_nearest(frame: pd.DataFrame, x_col: str, y_col: str, hub_ids: list[str], center_arr: np.ndarray) -> list[str]:
    pts = frame[[x_col, y_col]].to_numpy(dtype=float)
    dist_sq = ((pts[:, None, :] - center_arr[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(dist_sq, axis=1)
    return [hub_ids[idx] for idx in nearest]


def _to_lon_lat(x_km: float, y_km: float, lon_center: float, lat_center: float, km_per_deg_lon: float, km_per_deg_lat: float):
    return lon_center + x_km / km_per_deg_lon, lat_center + y_km / km_per_deg_lat


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return sorted(set(values))


def _write_map(
    output_path: Path,
    candidates: pd.DataFrame,
    activity_points: pd.DataFrame,
    center_lon: float,
    center_lat: float,
    map_half_size_km: float,
    metadata: dict,
) -> None:
    candidate_records = candidates.to_dict(orient="records")
    activity_records = activity_points.to_dict(orient="records")
    bounds = [
        [float(candidates["latitude"].min()), float(candidates["longitude"].min())],
        [float(candidates["latitude"].max()), float(candidates["longitude"].max())],
    ]
    payload = {
        "candidates": candidate_records,
        "activityPoints": activity_records,
        "center": {"lon": center_lon, "lat": center_lat},
        "bounds": bounds,
        "metadata": metadata,
    }
    json_payload = json.dumps(payload, ensure_ascii=False)
    title = html.escape("Meituan Shenzhen Nanshan candidate points")
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIINfQpG8fEYf1CMWXnD/Txl3M3PRZ2R2Jc=" crossorigin="">
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .leaflet-container {{
      overflow: hidden;
      position: relative;
      outline-offset: 1px;
      background: #ddd;
    }}
    .leaflet-pane,
    .leaflet-tile,
    .leaflet-marker-icon,
    .leaflet-marker-shadow,
    .leaflet-tile-container,
    .leaflet-pane > svg,
    .leaflet-pane > canvas,
    .leaflet-zoom-box,
    .leaflet-image-layer,
    .leaflet-layer {{
      position: absolute;
      left: 0;
      top: 0;
    }}
    .leaflet-tile,
    .leaflet-marker-icon,
    .leaflet-marker-shadow {{
      -webkit-user-select: none;
      user-select: none;
      -webkit-user-drag: none;
    }}
    .leaflet-tile {{ visibility: hidden; }}
    .leaflet-tile-loaded {{ visibility: inherit; }}
    .leaflet-zoom-animated {{ transform-origin: 0 0; }}
    .leaflet-tile-pane {{ z-index: 200; }}
    .leaflet-overlay-pane {{ z-index: 400; }}
    .leaflet-shadow-pane {{ z-index: 500; }}
    .leaflet-marker-pane {{ z-index: 600; }}
    .leaflet-tooltip-pane {{ z-index: 650; }}
    .leaflet-popup-pane {{ z-index: 700; }}
    .leaflet-map-pane canvas {{ z-index: 100; }}
    .leaflet-control {{
      position: relative;
      z-index: 800;
      pointer-events: auto;
      float: left;
      clear: both;
      background: #fff;
    }}
    .leaflet-top,
    .leaflet-bottom {{
      position: absolute;
      z-index: 1000;
      pointer-events: none;
    }}
    .leaflet-top {{ top: 0; }}
    .leaflet-bottom {{ bottom: 0; }}
    .leaflet-left {{ left: 0; }}
    .leaflet-right {{ right: 0; }}
    .leaflet-right .leaflet-control {{ float: right; }}
    .leaflet-top .leaflet-control {{ margin-top: 10px; }}
    .leaflet-bottom .leaflet-control {{ margin-bottom: 10px; }}
    .leaflet-left .leaflet-control {{ margin-left: 10px; }}
    .leaflet-right .leaflet-control {{ margin-right: 10px; }}
    .leaflet-bar a,
    .leaflet-bar a:hover {{
      display: block;
      width: 26px;
      height: 26px;
      line-height: 26px;
      text-align: center;
      text-decoration: none;
      color: #111;
      border-bottom: 1px solid #ccc;
      font: bold 18px/26px Arial, Helvetica, sans-serif;
    }}
    .leaflet-bar,
    .leaflet-control-layers {{
      border: 2px solid rgba(0, 0, 0, 0.2);
      border-radius: 4px;
      box-shadow: 0 1px 5px rgba(0, 0, 0, 0.25);
      background-clip: padding-box;
    }}
    .leaflet-control-layers {{
      padding: 6px 10px;
      line-height: 1.3;
      color: #202124;
    }}
    .leaflet-control-layers label {{
      display: block;
      margin: 3px 0;
    }}
    .leaflet-control-attribution {{
      padding: 0 5px;
      margin: 0;
      color: #333;
      background: rgba(255, 255, 255, 0.8);
      font-size: 11px;
    }}
    .leaflet-control-attribution a {{ color: #0078a8; }}
    .leaflet-popup {{
      position: absolute;
      text-align: center;
      margin-bottom: 20px;
    }}
    .leaflet-popup-content-wrapper {{
      background: white;
      border-radius: 8px;
      box-shadow: 0 3px 14px rgba(0, 0, 0, 0.4);
    }}
    .leaflet-popup-content {{ margin: 10px 12px; line-height: 1.35; }}
    .panel {{
      position: absolute;
      top: 12px;
      right: 12px;
      z-index: 500;
      width: min(360px, calc(100vw - 24px));
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(0, 0, 0, 0.15);
      border-radius: 8px;
      box-shadow: 0 6px 22px rgba(0, 0, 0, 0.16);
      padding: 12px 14px;
      color: #202124;
      font-size: 13px;
      line-height: 1.35;
    }}
    .panel h1 {{
      margin: 0 0 8px;
      font-size: 15px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .legend {{ display: grid; gap: 6px; margin-top: 10px; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 2px solid currentColor;
      box-sizing: border-box;
      flex: 0 0 auto;
    }}
    .swatch.h {{ color: #d73a31; background: #d73a31; }}
    .swatch.l {{ color: #1f6feb; background: #1f6feb; }}
    .swatch.e {{ color: #9a6700; background: #f2cc60; }}
    .swatch.raw {{ color: #5f6368; background: rgba(95, 99, 104, 0.28); }}
    @media (max-width: 640px) {{
      .panel {{
        left: 8px;
        right: 8px;
        top: 8px;
        width: auto;
        font-size: 12px;
      }}
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <section class="panel">
    <h1>Shenzhen Nanshan candidate points</h1>
    <div>H={metadata["num_hubs"]}, L={metadata["num_bases"]}, E={metadata["num_swapping"]}; candidate window {metadata["candidate_hours"][0]}:00-{metadata["candidate_hours"][1]}:00; map half-size {map_half_size_km:g} km.</div>
    <div class="legend">
      <div class="legend-row"><span class="swatch h"></span><span>Hub candidates (H)</span></div>
      <div class="legend-row"><span class="swatch l"></span><span>Base candidates (L)</span></div>
      <div class="legend-row"><span class="swatch e"></span><span>Swap candidates (E)</span></div>
      <div class="legend-row"><span class="swatch raw"></span><span>Raw pickup/dropoff activity</span></div>
    </div>
  </section>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <script>
    const data = {json_payload};
    const map = L.map("map", {{ preferCanvas: true }}).setView([data.center.lat, data.center.lon], 13);
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }}).addTo(map);

    const rawLayer = L.layerGroup().addTo(map);
    const hLayer = L.layerGroup().addTo(map);
    const lLayer = L.layerGroup().addTo(map);
    const eLayer = L.layerGroup().addTo(map);

    const maxActivity = Math.max(...data.activityPoints.map(p => Number(p.activity_count) || 1));
    for (const p of data.activityPoints) {{
      const radius = 2 + 8 * Math.sqrt((Number(p.activity_count) || 1) / maxActivity);
      L.circleMarker([p.latitude, p.longitude], {{
        radius,
        color: "#5f6368",
        weight: 1,
        fillColor: "#5f6368",
        fillOpacity: 0.18,
        opacity: 0.35
      }}).bindPopup(`<b>Raw activity</b><br>${{p.activity_count}} orders<br>${{p.latitude.toFixed(6)}}, ${{p.longitude.toFixed(6)}}`).addTo(rawLayer);
    }}

    const styles = {{
      H: {{ layer: hLayer, radius: 7, color: "#d73a31", fillColor: "#d73a31", fillOpacity: 0.82, weight: 2 }},
      L: {{ layer: lLayer, radius: 8, color: "#1f6feb", fillColor: "#1f6feb", fillOpacity: 0.82, weight: 2 }},
      E: {{ layer: eLayer, radius: 11, color: "#9a6700", fillColor: "#f2cc60", fillOpacity: 0.35, weight: 3 }}
    }};
    for (const p of data.candidates) {{
      const st = styles[p.type];
      L.circleMarker([p.latitude, p.longitude], {{
        radius: st.radius,
        color: st.color,
        weight: st.weight,
        fillColor: st.fillColor,
        fillOpacity: st.fillOpacity
      }}).bindPopup(`<b>${{p.index}} (${{p.type}})</b><br>${{p.latitude.toFixed(6)}}, ${{p.longitude.toFixed(6)}}<br>x=${{p.x.toFixed(3)}} km, y=${{p.y.toFixed(3)}} km`).addTo(st.layer);
    }}

    L.control.layers(
      null,
      {{
        "Raw activity": rawLayer,
        "H candidates": hLayer,
        "L candidates": lLayer,
        "E candidates": eLayer
      }},
      {{ collapsed: false }}
    ).addTo(map);

    L.circle([data.center.lat, data.center.lon], {{
      radius: data.metadata.map_half_size_km * 1000,
      color: "#0969da",
      opacity: 0.12,
      fillOpacity: 0,
      weight: 2
    }}).addTo(map);
    map.fitBounds(data.bounds, {{ padding: [36, 36] }});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _write_png(output_path: Path, candidates: pd.DataFrame, activity_points: pd.DataFrame) -> None:
    cache_dir = output_path.parent / ".cache"
    mpl_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(exist_ok=True)
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    max_activity = max(float(activity_points["activity_count"].max()), 1.0)
    sizes = 10 + 80 * np.sqrt(activity_points["activity_count"].to_numpy(dtype=float) / max_activity)
    ax.scatter(
        activity_points["longitude"],
        activity_points["latitude"],
        s=sizes,
        c="#777777",
        alpha=0.18,
        linewidths=0,
        label="raw pickup/dropoff activity",
    )

    style = {
        "H": ("#d73a31", "o", 42, "H candidates"),
        "L": ("#1f6feb", "^", 72, "L candidates"),
        "E": ("#9a6700", "s", 58, "E candidates"),
    }
    for kind, (color, marker, size, label) in style.items():
        part = candidates[candidates["type"] == kind]
        ax.scatter(
            part["longitude"],
            part["latitude"],
            s=size,
            marker=marker,
            c=color,
            edgecolors="white",
            linewidths=0.7,
            label=label,
            zorder=5,
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Meituan Shenzhen Nanshan candidate points")
    ax.legend(loc="best", frameon=True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.4, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "orders").mkdir(exist_ok=True)

    df = pd.read_csv(args.raw_data, encoding="utf-8-sig")
    missing = RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing)}")
    if "dt_seq_id" not in df.columns:
        df["dt_seq_id"] = 1

    candidate_dt_ids = _parse_int_list(args.candidate_dt_seq_ids)
    demand_dt_ids = _parse_int_list(args.demand_dt_seq_ids)
    if candidate_dt_ids is None:
        candidate_dt_ids = sorted(int(v) for v in df["dt_seq_id"].dropna().unique())
    if demand_dt_ids is None:
        demand_dt_ids = sorted(int(v) for v in df["dt_seq_id"].dropna().unique())

    candidate_hour_count = _hour_duration(args.candidate_hour_start, args.candidate_hour_end)
    demand_hour_count = _hour_duration(args.demand_hour_start, args.demand_hour_end)

    candidate_df = _filter_period(df, candidate_dt_ids, args.candidate_hour_start, args.candidate_hour_end)
    if candidate_df.empty:
        raise ValueError("No candidate orders after dt/hour filtering.")

    demand_df = _filter_period(df, demand_dt_ids, args.demand_hour_start, args.demand_hour_end)
    if demand_df.empty:
        raise ValueError("No demand orders after dt/hour filtering.")

    top_pickup = (
        candidate_df.groupby(["pickup_longitude", "pickup_latitude"])
        .size()
        .sort_values(ascending=False)
        .index[0]
    )
    lon_center, lat_center = float(top_pickup[0]), float(top_pickup[1])

    candidate_projected, km_per_deg_lon, km_per_deg_lat = _project(candidate_df, lon_center, lat_center)
    demand_projected, _, _ = _project(demand_df, lon_center, lat_center)
    candidate_area = _clip_area(candidate_projected, args.map_half_size_km)
    demand_area = _clip_area(demand_projected, args.map_half_size_km)
    if candidate_area.empty:
        raise ValueError(f"No candidate orders in map window half-size {args.map_half_size_km}km.")
    if demand_area.empty:
        raise ValueError(f"No demand orders in map window half-size {args.map_half_size_km}km.")

    pickup_stats = (
        candidate_area.groupby(["sender_lng", "sender_lat", "sender_x_km", "sender_y_km"])
        .size()
        .reset_index(name="activity_count")
    )
    dropoff_stats = (
        candidate_area.groupby(["recipient_lng", "recipient_lat", "recipient_x_km", "recipient_y_km"])
        .size()
        .reset_index(name="activity_count")
    )
    pickup_stats.columns = ["longitude", "latitude", "x_km", "y_km", "activity_count"]
    dropoff_stats.columns = ["longitude", "latitude", "x_km", "y_km", "activity_count"]
    node_stats = (
        pd.concat([pickup_stats, dropoff_stats])
        .groupby(["longitude", "latitude", "x_km", "y_km"])["activity_count"]
        .sum()
        .reset_index()
        .sort_values("activity_count", ascending=False)
    )

    n_h = max(1, min(args.num_hubs, len(node_stats)))
    h_xy = node_stats[["x_km", "y_km"]].to_numpy()
    h_w = node_stats["activity_count"].to_numpy()
    h_kmeans = KMeans(n_clusters=n_h, random_state=args.seed, n_init="auto")
    node_stats["cluster"] = h_kmeans.fit_predict(h_xy, sample_weight=h_w)

    h_cluster_rank = (
        node_stats.groupby("cluster")["activity_count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    h_activity = node_stats.groupby("cluster")["activity_count"].sum().to_dict()

    locations = []
    candidate_rows = []
    h_id_to_xy = {}
    h_id_to_activity = {}
    for rank, cluster_id in enumerate(h_cluster_rank, start=1):
        h_id = f"H{rank}"
        x_val = float(h_kmeans.cluster_centers_[cluster_id][0])
        y_val = float(h_kmeans.cluster_centers_[cluster_id][1])
        lon, lat = _to_lon_lat(x_val, y_val, lon_center, lat_center, km_per_deg_lon, km_per_deg_lat)
        locations.append({"index": h_id, "type": "H", "x": x_val, "y": y_val})
        candidate_rows.append(
            {
                "index": h_id,
                "type": "H",
                "x": x_val,
                "y": y_val,
                "longitude": lon,
                "latitude": lat,
                "activity_count": float(h_activity[cluster_id]),
            }
        )
        h_id_to_xy[h_id] = (x_val, y_val)
        h_id_to_activity[h_id] = float(h_activity[cluster_id])

    hub_ids = list(h_id_to_xy.keys())
    center_arr = np.asarray([h_id_to_xy[h] for h in hub_ids], dtype=float)
    hub_points = [h_id_to_xy[h] for h in hub_ids]
    hub_weights = [h_id_to_activity[h] for h in hub_ids]

    base_locations = _weighted_kmeans(hub_points, hub_weights, args.num_bases, "base", args.seed)
    for i, (x_val, y_val) in enumerate(base_locations, 1):
        lon, lat = _to_lon_lat(float(x_val), float(y_val), lon_center, lat_center, km_per_deg_lon, km_per_deg_lat)
        locations.append({"index": f"L{i}", "type": "L", "x": float(x_val), "y": float(y_val)})
        candidate_rows.append(
            {
                "index": f"L{i}",
                "type": "L",
                "x": float(x_val),
                "y": float(y_val),
                "longitude": lon,
                "latitude": lat,
                "activity_count": "",
            }
        )

    demand_area = demand_area.copy()
    demand_area["p_id"] = _assign_nearest(demand_area, "sender_x_km", "sender_y_km", hub_ids, center_arr)
    demand_area["d_id"] = _assign_nearest(demand_area, "recipient_x_km", "recipient_y_km", hub_ids, center_arr)
    demand_area = demand_area[demand_area["p_id"] != demand_area["d_id"]].copy()
    pair_orders = demand_area.groupby(["p_id", "d_id"]).size().reset_index(name="order_count")
    demand_denominator = float(len(demand_dt_ids) * demand_hour_count)
    pair_orders["demand"] = pair_orders["order_count"] / demand_denominator
    orders_out = (
        pair_orders.rename(columns={"p_id": "start_node", "d_id": "end_node"})[
            ["start_node", "end_node", "demand"]
        ]
        .sort_values(["start_node", "end_node"])
        .reset_index(drop=True)
    )

    candidate_mapped = candidate_area.copy()
    candidate_mapped["p_id"] = _assign_nearest(candidate_mapped, "sender_x_km", "sender_y_km", hub_ids, center_arr)
    candidate_mapped["d_id"] = _assign_nearest(candidate_mapped, "recipient_x_km", "recipient_y_km", hub_ids, center_arr)
    candidate_mapped = candidate_mapped[candidate_mapped["p_id"] != candidate_mapped["d_id"]]
    candidate_pairs = candidate_mapped.groupby(["p_id", "d_id"]).size().reset_index(name="count")

    swap_seed_points = list(hub_points)
    swap_seed_weights = list(hub_weights)
    for row in candidate_pairs.itertuples(index=False):
        p_xy = h_id_to_xy[row.p_id]
        d_xy = h_id_to_xy[row.d_id]
        swap_seed_points.append(((p_xy[0] + d_xy[0]) / 2.0, (p_xy[1] + d_xy[1]) / 2.0))
        swap_seed_weights.append(float(row.count))

    e_locations = []
    if args.colocate_base_swaps:
        e_locations.extend(base_locations[: args.num_bases])
    extra_swaps = args.num_swapping - len(e_locations)
    e_locations.extend(_weighted_kmeans(swap_seed_points, swap_seed_weights, extra_swaps, "swap", args.seed))

    for i, (x_val, y_val) in enumerate(e_locations, 1):
        lon, lat = _to_lon_lat(float(x_val), float(y_val), lon_center, lat_center, km_per_deg_lon, km_per_deg_lat)
        locations.append({"index": f"E{i}", "type": "E", "x": float(x_val), "y": float(y_val)})
        candidate_rows.append(
            {
                "index": f"E{i}",
                "type": "E",
                "x": float(x_val),
                "y": float(y_val),
                "longitude": lon,
                "latitude": lat,
                "activity_count": "",
            }
        )

    locations_df = pd.DataFrame(locations)
    candidates_df = pd.DataFrame(candidate_rows)
    activity_df = node_stats.rename(columns={"x_km": "x", "y_km": "y"})[
        ["longitude", "latitude", "x", "y", "activity_count"]
    ].reset_index(drop=True)

    period_id = (
        f"h{args.demand_hour_start:02d}"
        if demand_hour_count == 1
        else f"h{args.demand_hour_start:02d}_{demand_hour_count}h"
    )
    locations_df.to_csv(output_dir / "locations.csv", index=False)
    orders_out.to_csv(output_dir / "orders.csv", index=False)
    orders_out.to_csv(output_dir / "orders" / f"{period_id}.csv", index=False)
    candidates_df.to_csv(output_dir / "candidate_points.csv", index=False)
    activity_df.to_csv(output_dir / "raw_activity_points.csv", index=False)

    periods_row = {
        "period_id": period_id,
        "orders_file": str(Path("orders") / f"{period_id}.csv"),
        "start_hour": args.demand_hour_start,
        "end_hour": args.demand_hour_end,
        "hour_semantics": "[start, end)",
        "dt_seq_ids": ";".join(str(v) for v in demand_dt_ids),
        "num_dt_seq_ids": len(demand_dt_ids),
        "duration_hours": demand_hour_count,
        "raw_orders": len(demand_df),
        "in_area_orders": len(demand_area),
        "retained_od_pairs": len(orders_out),
        "total_hourly_demand": float(orders_out["demand"].sum()),
    }
    pd.DataFrame([periods_row]).to_csv(output_dir / "periods.csv", index=False)

    metadata = {
        "raw_data_path": str(Path(args.raw_data).resolve()),
        "output_dir": str(output_dir),
        "num_hubs": n_h,
        "requested_num_hubs": args.num_hubs,
        "num_bases": args.num_bases,
        "num_swapping": args.num_swapping,
        "candidate_dt_seq_ids": candidate_dt_ids,
        "candidate_hours": [args.candidate_hour_start, args.candidate_hour_end],
        "candidate_duration_hours": candidate_hour_count,
        "candidate_raw_orders": len(candidate_df),
        "candidate_in_area_orders": len(candidate_area),
        "demand_dt_seq_ids": demand_dt_ids,
        "demand_hours": [args.demand_hour_start, args.demand_hour_end],
        "demand_duration_hours": demand_hour_count,
        "demand_raw_orders": len(demand_df),
        "demand_in_area_orders": len(demand_area),
        "unique_candidate_points": len(node_stats),
        "od_pairs": len(orders_out),
        "total_hourly_demand": float(orders_out["demand"].sum()),
        "map_half_size_km": args.map_half_size_km,
        "center_longitude": lon_center,
        "center_latitude": lat_center,
        "center_rule": "most frequent pickup coordinate in candidate window",
        "colocate_base_swaps": args.colocate_base_swaps,
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_map(
        output_dir / "candidate_map.html",
        candidates_df,
        activity_df,
        lon_center,
        lat_center,
        args.map_half_size_km,
        metadata,
    )
    _write_png(output_dir / "candidate_points.png", candidates_df, activity_df)

    return metadata


def build_parser() -> argparse.ArgumentParser:
    default_raw = PROJECT_ROOT / "Meituan-Shenzhen" / "深圳市南山区运单抽样_已脱敏.csv"
    default_output = PROJECT_ROOT / "data" / "instances" / "Meituan_Shenzhen_Nanshan_H040_L10_E10_h12"
    parser = argparse.ArgumentParser(
        description="Generate Shenzhen Nanshan candidate points from the anonymized Meituan order sample."
    )
    parser.add_argument("--raw-data", default=str(default_raw))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--num-hubs", type=int, default=getattr(config, "MEITUAN_NUM_HUBS", 40))
    parser.add_argument("--num-bases", type=int, default=getattr(config, "MEITUAN_NUM_BASES", 10))
    parser.add_argument("--num-swapping", type=int, default=getattr(config, "MEITUAN_NUM_SWAPPING", 10))
    parser.add_argument("--candidate-dt-seq-ids", default=None, help="Comma-separated dt_seq_id values. Default: all.")
    parser.add_argument("--demand-dt-seq-ids", default=None, help="Comma-separated dt_seq_id values. Default: all.")
    parser.add_argument(
        "--candidate-hour-start",
        type=int,
        default=getattr(config, "MEITUAN_CANDIDATE_HOUR_START", config.MEITUAN_WORK_HOUR_START),
    )
    parser.add_argument(
        "--candidate-hour-end",
        type=int,
        default=getattr(config, "MEITUAN_CANDIDATE_HOUR_END", config.MEITUAN_WORK_HOUR_END + 1),
    )
    parser.add_argument("--demand-hour-start", type=int, default=getattr(config, "MEITUAN_DEMAND_HOUR", 12))
    parser.add_argument("--demand-hour-end", type=int, default=None)
    parser.add_argument("--map-half-size-km", type=float, default=getattr(config, "MEITUAN_MAP_HALF_SIZE_KM", 8.0))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-colocate-base-swaps", dest="colocate_base_swaps", action="store_false")
    parser.set_defaults(colocate_base_swaps=bool(getattr(config, "MEITUAN_COLOCATE_BASE_SWAPS", True)))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.demand_hour_end is None:
        args.demand_hour_end = args.demand_hour_start + 1
    if args.colocate_base_swaps and args.num_swapping < args.num_bases:
        raise ValueError("num_swapping must be >= num_bases when colocate_base_swaps is enabled.")

    metadata = generate(args)
    print(f"Wrote instance and map artifacts to {metadata['output_dir']}")
    print(
        "Summary: "
        f"H={metadata['num_hubs']} L={metadata['num_bases']} E={metadata['num_swapping']} "
        f"candidate_orders={metadata['candidate_in_area_orders']}/{metadata['candidate_raw_orders']} "
        f"unique_points={metadata['unique_candidate_points']} "
        f"OD={metadata['od_pairs']} demand={metadata['total_hourly_demand']:.4f}/h"
    )
    print(f"Map: {Path(metadata['output_dir']) / 'candidate_map.html'}")


if __name__ == "__main__":
    main()
