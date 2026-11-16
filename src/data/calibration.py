"""Observed clock parsing and checksum-verified canonical time inputs."""

from __future__ import annotations

import json

import numpy as np

import pandas as pd

from .instance_io import sha256, project_path


def clock_minutes(series):
    parts = series.astype(str).str.extract(r"^(\d{1,2}):(\d{2})$")
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    return (60 * hours + minutes).where(hours.between(0, 23) & minutes.between(0, 59))


def load_calibration(hubs, data, path):
    """Verify the processed calibration against canonical OD CSV checksums."""
    path = project_path(path)
    if path.is_dir():
        path /= "calibration.json"
    fit = json.loads(path.read_text())
    mean_path = path.parent / fit["first_leg_mean_fit"]
    if sha256(mean_path) != fit["first_leg_mean_fit_sha256"]:
        raise ValueError("Rider time calibration checksum mismatch")
    mean_fit = json.loads(mean_path.read_text())
    case = fit["cases"][str(hubs)]
    deadline_path = path.parent / case["file"]
    if sha256(deadline_path) != case["sha256"]:
        raise ValueError("Deadline table checksum mismatch")
    table = pd.read_csv(deadline_path)
    if len(table) != data["n_od"] or not np.array_equal(
        table.od, np.arange(data["n_od"])
    ):
        raise ValueError("Deadline table and instance OD ordering differ")
    for filename, key in [
        ("od_demand.csv", "od_demand_sha256"),
        ("od_nodes.csv", "od_nodes_sha256"),
    ]:
        if data["provenance"]["sha256"][filename] != case[key]:
            raise ValueError(f"Calibration refers to different {filename}")
    if (
        np.any(table.deadline_minutes <= 0)
        or not np.isfinite(table.deadline_minutes).all()
    ):
        raise ValueError("Delivery deadlines must be positive and finite")
    stats_path = path.parent / fit["pre_dispatch_statistics"]
    if sha256(stats_path) != fit["pre_dispatch_statistics_sha256"]:
        raise ValueError("Pre-dispatch statistics checksum mismatch")
    stats = json.loads(stats_path.read_text())["cases"][str(hubs)]
    return dict(
        manifest=fit,
        mean_fit=mean_fit,
        case=case,
        table=table,
        pre_dispatch=stats,
        path=path,
        mean_path=mean_path,
        deadline_path=deadline_path,
        stats_path=stats_path,
    )
