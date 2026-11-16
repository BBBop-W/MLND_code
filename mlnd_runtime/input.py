"""Read-only, hash-checked consecutive CSV/1 and CSV/2 input contract."""
from dataclasses import fields
import csv
import json
from pathlib import Path
import numpy as np
from .types import MODEL_VERSION
from .storage import sha256
from .capacity import occupancy_limits

def to_planning_data(data):
    from .types import PlanningData

    if isinstance(data, PlanningData):
        return data
    if data.get("model_version") != MODEL_VERSION:
        raise ValueError("Input model_version must be consecutive")
    names = {field.name for field in fields(PlanningData) if field.init}
    values = {name: data[name] for name in names if name in data}
    if values.get("home_receipt_mean") is None:
        values["home_receipt_mean"] = home_receipt_mean(data)
    return PlanningData(**values)


def home_receipt_mean(data):
    """Recover courier-home time only from explicit data or frozen provenance.

    A hand-built input without pickup-share data interprets receipt_mean as
    courier receipt. A mixed historical snapshot requires the original courier
    parameters; a mixture alone does not identify either mode's receipt time.
    """
    shape = (len(data["demand"]), int(data["n_h"]))
    if data.get("home_receipt_mean") is not None:
        value = data["home_receipt_mean"]
    elif not np.any(np.asarray(data.get("pickup_share", 0)) > 0):
        value = data["receipt_mean"]
    else:
        terminal = data.get("metadata", {}).get("resolved_config", {}).get("terminal", {})
        if terminal.get("courier_receipt_minutes") is not None:
            value = np.asarray(terminal["courier_receipt_minutes"], dtype=float) / 60
        elif "courier_removal_minutes" in terminal and "last_rider_mean" in data:
            value = (np.asarray(terminal["courier_removal_minutes"], dtype=float) / 60
                     + np.asarray(data["last_rider_mean"], dtype=float))
        else:
            raise ValueError("Mixed receipt data require explicit home_receipt_mean or frozen courier terminal parameters")
    result = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("home_receipt_mean must be finite and nonnegative")
    return result


SCHEMA = "consecutive-csv/2"


LEGACY_SCHEMA = "consecutive-csv/1"


TABLES = {
    "base_costs": (("base_id",), ("bases",), ("base_cost",)),
    "locker_levels": (
        ("hub_id", "level_id"),
        ("hubs", "levels"),
        ("slot_counts", "slot_cost", "slot_epsilon", "occupancy_limit"),
    ),
    "links": (
        ("from_site", "to_site"),
        ("flight_nodes", "flight_nodes"),
        ("allowed_links", "flight_times", "link_cost"),
    ),
    "od_costs": (("od_id",), ("ods",), ("ground_cost", "deadline")),
    "od_hub": (
        ("od_id", "hub_id"),
        ("ods", "hubs"),
        (
            "rider_mean",
            "first_cost",
            "terminal_cost",
            "receipt_mean",
            "removal_mean",
            "pickup_share",
            "last_rider_cost",
            "last_rider_mean",
            "home_receipt_mean",
        ),
    ),
    "kappa": (("first_od", "second_od"), ("ods", "ods"), ("kappa",)),
}


def tables_for_schema(schema):
    if schema not in (LEGACY_SCHEMA, SCHEMA):
        raise ValueError("Unsupported input schema")
    if schema == SCHEMA:
        return TABLES
    return {
        name: (keys, domains, tuple(c for c in columns if c != "home_receipt_mean"))
        for name, (keys, domains, columns) in TABLES.items()
    }


BASE_HEADERS = {
    "sites.csv": ("site_id", "type", "x_km", "y_km", "activity_rate"),
    "od_nodes.csv": ("node_id", "role", "x_km", "y_km", "raw_endpoint_count"),
    "od_demand.csv": (
        "od_id",
        "origin_node",
        "destination_node",
        "order_count",
        "exposure_hours",
        "lambda_per_hour",
        "direct_distance_km",
    ),
    "periods.csv": (
        "period_id",
        "start_hour",
        "end_hour",
        "dates",
        "exposure_hours",
        "order_count",
        "total_rate_per_hour",
    ),
}


INPUT_FILES = (
    *BASE_HEADERS,
    "metadata.json",
    "model.json",
    *(f"coefficients/{name}.csv" for name in TABLES),
)


UNITS = {
    "coordinates": "km, local projected coordinates",
    "demand": "orders/hour",
    "time_coefficients": "hour",
    "base_cost,slot_cost,link_cost": "CNY/hour per selected facility, level or link",
    "fleet_cost": "CNY/hour per drone",
    "first_cost,terminal_cost,ground_cost,last_rider_cost": "CNY/order",
    "flight_cost,waiting_cost": "CNY/drone-hour of flight or waiting",
    "slot_counts": "compartments",
    "occupancy_limit": "mean parcels present; not orders/hour",
    "slot_epsilon,pickup_share,kappa": "dimensionless",
}


CAPACITY_DEFINITION = {
    "version": "poisson-strict-overflow/1",
    "formula": "max mu >= 0 such that P(Poisson(mu) > slot_counts) <= slot_epsilon",
    "threshold_column": "coefficients/locker_levels.csv:occupancy_limit",
    "verification_atol": 1e-12,
    "verification_rtol": 1e-12,
}


def _read_csv(path, columns=None):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if columns is not None and reader.fieldnames != list(columns):
            raise ValueError(f"Wrong CSV header: {path.name}")
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f"Malformed CSV row: {path.name}")
        return rows


def _indexed(rows, column, ids):
    result = {}
    for row in rows:
        key = row[column]
        if key in result:
            raise ValueError(f"Duplicate {column}: {key}")
        result[key] = row
    if set(result) != set(ids):
        raise ValueError(f"Missing or unknown {column}")
    return [result[key] for key in ids]


def _number(value, dtype):
    if dtype == np.dtype(bool):
        if value not in ("0", "1"):
            raise ValueError("Boolean CSV values must be 0 or 1")
        return value == "1"
    if np.issubdtype(dtype, np.integer):
        return int(value)
    return float(value)


def load_standard(path, *, case=None, verify_hashes=True):
    """Read resolved inputs by ID, without defaults, aggregation or calibration."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    schema = manifest["schema"]
    tables = tables_for_schema(schema)
    if set(manifest["files"]) != set(INPUT_FILES):
        raise ValueError("Unsupported or incomplete manifest")
    if verify_hashes:
        for name, digest in manifest["files"].items():
            if sha256(path / name) != digest:
                raise ValueError(f"SHA-256 mismatch: {name}")
    model = json.loads((path / "model.json").read_text(encoding="utf-8"))
    if model["schema"] != schema or model["units"] != UNITS:
        raise ValueError("Unsupported schema or units")
    if model.get("capacity_definition") != CAPACITY_DEFINITION:
        raise ValueError("Unsupported capacity_definition")
    ids = model["ids"]
    for domain, values in ids.items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Empty or duplicate IDs: {domain}")
    if ids["flight_nodes"] != ids["bases"] + ids["hubs"]:
        raise ValueError("Flight node order must be bases followed by hubs")
    data = dict(model["values"])
    required_scalars = {
        "n_l",
        "n_h",
        "delta",
        "operating_limit",
        "fleet_cost",
        "flight_cost",
        "waiting_cost",
        "max_orders",
        "name",
        "candidate_routes",
        "route_mode",
        "flow_policy",
        "model_version",
        "n_od",
    }
    if set(data) != required_scalars:
        raise ValueError("Missing or unknown model scalar")
    for field in ("n_l", "n_h", "n_od", "max_orders"):
        if not isinstance(data[field], int) or isinstance(data[field], bool):
            raise ValueError(f"Integer model scalar required: {field}")
    if data["max_orders"] not in (1, 2):
        raise ValueError("max_orders must be 1 or 2")
    if model["cases"] != {"K1": {"max_orders": 1}, "K2": {"max_orders": 2}}:
        raise ValueError("Cases may change max_orders only")
    for field, domain in (("n_l", "bases"), ("n_h", "hubs"), ("n_od", "ods")):
        if data[field] != len(ids[domain]):
            raise ValueError(f"Dimension mismatch: {field}")
    sites = _indexed(
        _read_csv(path / "sites.csv", BASE_HEADERS["sites.csv"]),
        "site_id",
        ids["flight_nodes"],
    )
    nodes = _indexed(
        _read_csv(path / "od_nodes.csv", BASE_HEADERS["od_nodes.csv"]),
        "node_id",
        ids["origins"] + ids["destinations"],
    )
    for domain, role, rows, offset in (
        ("bases", "L", sites, 0),
        ("hubs", "H", sites, len(ids["bases"])),
        ("origins", "origin", nodes, 0),
        ("destinations", "destination", nodes, len(ids["origins"])),
    ):
        selected = rows[offset : offset + len(ids[domain])]
        role_key = "type" if rows is sites else "role"
        if any(row[role_key] != role for row in selected):
            raise ValueError(f"Wrong node role: {domain}")
        data[domain] = np.array(
            [[float(r["x_km"]), float(r["y_km"])] for r in selected]
        )
        if not np.isfinite(data[domain]).all():
            raise ValueError("Coordinates must be finite")
    ods = _indexed(
        _read_csv(path / "od_demand.csv", BASE_HEADERS["od_demand.csv"]),
        "od_id",
        ids["ods"],
    )
    data["od_pairs"] = [[r["origin_node"], r["destination_node"]] for r in ods]
    if len(set(map(tuple, data["od_pairs"]))) != len(ods):
        raise ValueError("Duplicate OD pair")
    for key, domain, field in (
        ("od_origin", "origins", "origin_node"),
        ("od_destination", "destinations", "destination_node"),
    ):
        lookup = {value: i for i, value in enumerate(ids[domain])}
        data[key] = np.array([lookup[r[field]] for r in ods], dtype=np.int64)
    data["demand"] = np.array([float(r["lambda_per_hour"]) for r in ods])
    period_rows = _read_csv(path / "periods.csv", BASE_HEADERS["periods.csv"])
    if len(period_rows) != 1:
        raise ValueError("This exchange profile requires one pooled demand period")
    period = period_rows[0]
    exposure = float(period["exposure_hours"])
    counts = [int(r["order_count"]) for r in ods]
    start, end = float(period["start_hour"]), float(period["end_hour"])
    if not 0 <= start < end <= 24:
        raise ValueError("Invalid period hour window")
    dates = period["dates"].split(";")
    # Match native std::getline: one terminal separator creates no final
    # field, while leading/interior empty fields and repeated dates are invalid.
    if dates[-1] == "":
        dates.pop()
    if not dates or any(not date for date in dates) or len(set(dates)) != len(dates):
        raise ValueError("Empty or duplicate period date")
    expected_exposure = len(dates) * (end - start)
    if (
        exposure <= 0
        or exposure != expected_exposure
        or min(counts) < 0
        or sum(counts) != int(period["order_count"])
        or not np.isclose(
            float(period["total_rate_per_hour"]),
            sum(counts) / exposure,
            rtol=1e-14,
            atol=1e-14,
        )
    ):
        raise ValueError("Inconsistent period counts/exposure")
    for row, count in zip(ods, counts):
        if (
            float(row["exposure_hours"]) != exposure
            or not np.isclose(
                float(row["lambda_per_hour"]), count / exposure, rtol=1e-14, atol=1e-14
            )
            or not np.isfinite(float(row["direct_distance_km"]))
            or float(row["direct_distance_km"]) < 0
        ):
            raise ValueError("Inconsistent OD counts/exposure/distance")
    epsilon = {}
    thresholds = None
    for name, (keys, domains, columns) in tables.items():
        shape = tuple(len(ids[d]) for d in domains)
        expected_layout = {
            "id_columns": list(keys),
            "domains": list(domains),
            "value_columns": list(columns),
        }
        if model["table_layouts"][name] != expected_layout:
            raise ValueError(f"Wrong table layout: {name}")
        rows = _read_csv(path / f"coefficients/{name}.csv", (*keys, *columns))
        lookup = [{value: i for i, value in enumerate(ids[d])} for d in domains]
        seen = set()
        for column in columns:
            if column not in ("slot_epsilon", "occupancy_limit"):
                data[column] = np.empty(shape, dtype=model["arrays"][column]["dtype"])
        if name == "locker_levels":
            thresholds = np.empty(shape)
        for row in rows:
            index = tuple(mapping[row[key]] for key, mapping in zip(keys, lookup))
            if index in seen:
                raise ValueError(f"Duplicate coefficient row: {name}")
            seen.add(index)
            for column in columns:
                if column == "slot_epsilon":
                    value = float(row[column])
                    if index[0] in epsilon and epsilon[index[0]] != value:
                        raise ValueError("Inconsistent epsilon across capacity levels")
                    epsilon[index[0]] = value
                elif column == "occupancy_limit":
                    thresholds[index] = float(row[column])
                else:
                    data[column][index] = _number(row[column], data[column].dtype)
        if len(seen) != int(np.prod(shape)):
            raise ValueError(f"Missing coefficient row: {name}")
    data["slot_epsilon"] = np.array([epsilon[i] for i in range(data["n_h"])])
    if np.any(np.diag(data["allowed_links"])):
        raise ValueError("Self links must be explicitly disabled")
    if np.any((data["pickup_share"] < 0) | (data["pickup_share"] > 1)):
        raise ValueError("pickup_share must lie in [0, 1]")
    if np.any(data["last_rider_mean"] < 0) or np.any(data["last_rider_cost"] < 0):
        raise ValueError("Last rider coefficients must be nonnegative")
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    if set(arrays) != set(model["arrays"]):
        raise ValueError("Array manifest is incomplete")
    for key, value in arrays.items():
        spec = model["arrays"][key]
        if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
            raise ValueError(f"Array shape or dtype mismatch: {key}")
        if key != "flight_times" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite array: {key}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata["exchange_schema"] != schema
        or metadata["coordinate_system"] != "local_km"
    ):
        raise ValueError("Unsupported metadata or coordinates")
    data["metadata"] = metadata["frozen_input_metadata"]
    if case is not None:
        if case not in model["cases"]:
            raise ValueError("case must be K1 or K2")
        data.update(model["cases"][case])
    planning = to_planning_data(data)
    expected = occupancy_limits(planning.slot_counts, planning.slot_epsilon[:, None])
    if not np.allclose(thresholds, expected, atol=1e-12, rtol=1e-12):
        raise ValueError("Poisson occupancy threshold mismatch")
    return data

