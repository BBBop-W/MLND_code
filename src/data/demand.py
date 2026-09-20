"""Demand identity, independent of the candidate facility set."""

import hashlib
import json
import numpy as np


# Include physical OD identities, rates and their service benchmarks, not just totals.
DEMAND_ARRAYS = (
    "origins",
    "destinations",
    "od_origin",
    "od_destination",
    "demand",
    "ground",
    "direct_rider_hours",
    "delivery_deadline",
    "observed_postfetch_hours",
    "observed_postfetch_count",
)


def demand_signature(data):
    digest = hashlib.sha256(b"mlnd-demand-v1")
    digest.update(json.dumps(data.get("od_pairs"), separators=(",", ":")).encode())
    for key in DEMAND_ARRAYS:
        digest.update(key.encode())
        if key not in data:
            digest.update(b"missing")
            continue
        array = np.asarray(data[key], dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    digest.update(str(data.get("pre_dispatch_mean_hours")).encode())
    return digest.hexdigest()


def describe_demand(data, expected_signature=None):
    signature = demand_signature(data)
    if expected_signature is not None and signature != expected_signature:
        raise ValueError(
            "Fixed demand changed: OD identities, rates, rider costs or delivery benchmarks differ"
        )
    selection = data.get("provenance", {}).get("demand_selection", {})
    return dict(
        mode=selection.get("mode", "saved"),
        source_hubs=selection.get("source_hubs"),
        n_od=int(data["n_od"]),
        hourly_demand=float(np.sum(data["demand"])),
        all_ground_cost=float(np.asarray(data["demand"]) @ data["ground"]),
        signature=signature,
    )


def audit_instances(hubs, config=None):
    """Load and compare demand inputs; never construct an optimization model."""
    from src.config import load_config
    from .instances import load_instance

    cfg = load_config(config)
    fixed = cfg["data"]["demand_hubs"] is not None
    expected = None
    rows = []
    for size in dict.fromkeys(hubs):
        data = load_instance(size, cfg)
        demand = describe_demand(data, expected if fixed else None)
        if expected is None:
            expected = demand["signature"]
        rows.append(
            dict(
                candidates={"L": data["n_l"], "H": data["n_h"], "E": data["n_k"]},
                **demand,
            )
        )
    if not rows:
        raise ValueError("At least one candidate size is required")
    return dict(
        optimization_started=False,
        mode="fixed" if fixed else "instance",
        demand_source_hubs=cfg["data"]["demand_hubs"],
        demand_scale=cfg["data"]["demand_scale"],
        same_demand=len({row["signature"] for row in rows}) == 1,
        expected_signature=expected if fixed else None,
        cases=rows,
    )
