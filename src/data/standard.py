"""Lossless CSV/JSON exchange for resolved consecutive instances.

This adapter never recalibrates coefficients or reads a solver configuration.
The geographic five-file layout follows the historical data generator; the
coefficient tables are an explicit new extension for the C++ implementation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from src.data.instances import to_planning_data
from src.experiments.storage import load_inputs, sha256, write_json
from src.model.locker_capacity import occupancy_limits

SCHEMA = "consecutive-csv/1"
# Table: ID columns, ID domains, numeric columns. Names match PlanningData.
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
        ),
    ),
    "kappa": (("first_od", "second_od"), ("ods", "ods"), ("kappa",)),
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


def _write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    int(v)
                    if isinstance(v, (bool, np.bool_))
                    else format(float(v), ".17g")
                    if isinstance(v, (float, np.floating))
                    else v
                    for v in row
                ]
            )


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


def _assert_same(left, right):
    if set(left) != set(right):
        raise ValueError("Snapshot fields differ")
    for key, value in left.items():
        if isinstance(value, np.ndarray):
            other = right[key]
            if value.dtype != other.dtype or not np.array_equal(value, other):
                raise ValueError(f"Snapshot array differs: {key}")
        elif value != right[key]:
            raise ValueError(f"Snapshot value differs: {key}")


def export_standard(snapshot, market, output):
    """Export a frozen snapshot with the matching prepare_orders market CSVs.

    The market supplies observed counts/distances, not solver coefficients.
    Refuse replacement, unmatched markets and unsupported snapshot fields.
    """
    snapshot, market, output = map(Path, (snapshot, market, output))
    data = load_inputs(snapshot)
    planning = to_planning_data(data)
    source = data["metadata"]["source_metadata"]
    od_rows = _read_csv(market / "od_demand.csv", BASE_HEADERS["od_demand.csv"])
    nodes = _read_csv(market / "od_nodes.csv")
    origins = [r for r in nodes if r["role"] == "origin"]
    destinations = [r for r in nodes if r["role"] == "destination"]
    ids = {
        "bases": [f"L{i:04d}" for i in range(planning.n_l)],
        "hubs": [f"H{i:04d}" for i in range(planning.n_h)],
        "origins": [r["node_id"] for r in origins],
        "destinations": [r["node_id"] for r in destinations],
        "ods": [r["od_id"] for r in od_rows],
        "levels": [str(i + 1) for i in range(planning.slot_counts.shape[1])],
    }
    ids["flight_nodes"] = ids["bases"] + ids["hubs"]
    for key, rows in (("origins", origins), ("destinations", destinations)):
        coords = np.array([[float(r["x_km"]), float(r["y_km"])] for r in rows])
        if not np.array_equal(coords, data[key]):
            raise ValueError(f"Market coordinates differ: {key}")
    if not np.array_equal(
        [float(r["lambda_per_hour"]) for r in od_rows], data["demand"]
    ):
        raise ValueError("Market demand differs")
    if [[r["origin_node"], r["destination_node"]] for r in od_rows] != data["od_pairs"]:
        raise ValueError("Market OD pairs differ")
    daily = _read_csv(market / "periods.csv")
    if [r["day"] for r in daily] != source["selected_days"]:
        raise ValueError("Market days differ")
    exposure = sum(float(r["demand_exposure_hours"]) for r in daily)
    count = sum(int(r["demand_orders"]) for r in daily)
    if (
        count != sum(int(r["order_count"]) for r in od_rows)
        or count != source["demand_orders_in_area"]
        or exposure != source["demand_exposure_hours"]
    ):
        raise ValueError("Market exposure or counts differ")
    known_arrays = {
        "bases",
        "hubs",
        "origins",
        "destinations",
        "od_origin",
        "od_destination",
        "demand",
        *(c for _, _, cols in TABLES.values() for c in cols if c != "occupancy_limit"),
    }
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    if set(arrays) != known_arrays:
        raise ValueError("Unsupported or missing snapshot arrays")
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(
        output / "sites.csv",
        BASE_HEADERS["sites.csv"],
        (
            (sid, role, *xy, "")
            for domain, role in (("bases", "L"), ("hubs", "H"))
            for sid, xy in zip(ids[domain], data[domain])
        ),
    )
    _write_csv(
        output / "od_nodes.csv",
        BASE_HEADERS["od_nodes.csv"],
        (
            (nid, role, *xy, "")
            for domain, role in (("origins", "origin"), ("destinations", "destination"))
            for nid, xy in zip(ids[domain], data[domain])
        ),
    )
    _write_csv(
        output / "od_demand.csv",
        BASE_HEADERS["od_demand.csv"],
        ([row[col] for col in BASE_HEADERS["od_demand.csv"]] for row in od_rows),
    )
    start, end = source["demand_hour_window"]
    _write_csv(
        output / "periods.csv",
        BASE_HEADERS["periods.csv"],
        [
            (
                f"h{start}_{end}",
                start,
                end,
                ";".join(source["selected_days"]),
                exposure,
                count,
                count / exposure,
            )
        ],
    )
    metadata = dict(source)
    metadata.update(
        schema_version=SCHEMA,
        legacy_layout="five-file-v2",
        exchange_schema=SCHEMA,
        frozen_input_metadata=data["metadata"],
        daily_periods=daily,
        optional_missing={
            "sites.activity_rate": "Per-site cluster activity counts are not retained in this exchange table; blank is unknown, not zero.",
            "od_nodes.raw_endpoint_count": "Legacy grid-node statistic is not used by this joint-OD clustering; blank.",
        },
        direct_distance_semantics="Mean direct distance of actual orders within each joint OD cluster, not distance between representative nodes.",
        source_files={
            "snapshot": {
                name: sha256(snapshot / name)
                for name in ("input_arrays.npz", "input_metadata.json")
            },
            "market": {
                name: sha256(market / name)
                for name in ("od_nodes.csv", "od_demand.csv", "periods.csv")
            },
        },
    )
    write_json(output / "metadata.json", metadata)
    model = {
        "schema": SCHEMA,
        "values": {
            k: v
            for k, v in data.items()
            if k not in arrays and k not in ("metadata", "od_pairs")
        },
        "ids": ids,
        "arrays": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in arrays.items()
        },
        "cases": {"K1": {"max_orders": 1}, "K2": {"max_orders": 2}},
        "units": UNITS,
        "capacity_definition": CAPACITY_DEFINITION,
        "table_layouts": {
            name: {"id_columns": cols, "domains": domains, "value_columns": values}
            for name, (cols, domains, values) in TABLES.items()
        },
        "numeric_encoding": "UTF-8 RFC4180 CSV; decimal round-trip precision; bool=0/1; inf only for unavailable flight times",
    }
    write_json(output / "model.json", model)
    thresholds = occupancy_limits(planning.slot_counts, planning.slot_epsilon[:, None])
    for name, (keys, domains, columns) in TABLES.items():

        def rows():
            for index in itertools.product(*(range(len(ids[d])) for d in domains)):
                values = []
                for column in columns:
                    if column == "occupancy_limit":
                        value = thresholds[index]
                    elif column == "slot_epsilon":
                        value = data[column][index[0]]
                    else:
                        value = data[column][index]
                    values.append(value)
                yield [ids[d][i] for d, i in zip(domains, index)] + values

        _write_csv(output / f"coefficients/{name}.csv", (*keys, *columns), rows())
    write_json(
        output / "manifest.json",
        {
            "schema": SCHEMA,
            "files": {name: sha256(output / name) for name in INPUT_FILES},
        },
    )
    report = validate_standard(output, snapshot)
    write_json(output / "verification.json", report)
    return report


def load_standard(path, *, case=None, verify_hashes=True):
    """Read resolved inputs by ID, without defaults, aggregation or calibration."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or set(manifest["files"]) != set(INPUT_FILES):
        raise ValueError("Unsupported or incomplete manifest")
    if verify_hashes:
        for name, digest in manifest["files"].items():
            if sha256(path / name) != digest:
                raise ValueError(f"SHA-256 mismatch: {name}")
    model = json.loads((path / "model.json").read_text())
    if model["schema"] != SCHEMA or model["units"] != UNITS:
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
    expected_exposure = len(set(period["dates"].split(";"))) * (
        float(period["end_hour"]) - float(period["start_hour"])
    )
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
    for name, (keys, domains, columns) in TABLES.items():
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
    metadata = json.loads((path / "metadata.json").read_text())
    if (
        metadata["exchange_schema"] != SCHEMA
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


def validate_standard(path, reference=None):
    data = load_standard(path)
    if reference is not None:
        _assert_same(load_inputs(reference), data)
    for case, k in (("K1", 1), ("K2", 2)):
        variant = load_standard(path, case=case)
        expected = dict(data, max_orders=k)
        _assert_same(expected, variant)
    return {
        "schema": SCHEMA,
        "passed": True,
        "reference_exact_match": reference is not None,
        "reference_files": {
            name: sha256(Path(reference) / name)
            for name in ("input_arrays.npz", "input_metadata.json")
        }
        if reference is not None
        else None,
        "snapshot_fields": len(data),
        "arrays_compared": sum(isinstance(v, np.ndarray) for v in data.values()),
        "dimensions": {k: data[k] for k in ("n_l", "n_h", "n_od")},
        "demand_per_hour": float(data["demand"].sum()),
        "nonzero_kappa": int(np.count_nonzero(data["kappa"])),
        "cases_change_max_orders_only": True,
        "solver_run": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--snapshot", required=True)
    export.add_argument("--market", required=True)
    export.add_argument("--output", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("path")
    validate.add_argument("--reference")
    args = parser.parse_args()
    if args.command == "export":
        report = export_standard(args.snapshot, args.market, args.output)
    else:
        report = validate_standard(args.path, args.reference)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
