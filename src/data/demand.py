"""Demand identity checks independent of facilities and solution choices."""

import hashlib
import json
import numpy as np


def demand_signature(data):
    digest = hashlib.sha256(b"mlnd-market-v2")
    digest.update(json.dumps(data.get("od_pairs"), separators=(",", ":")).encode())
    for key in (
        "origins",
        "destinations",
        "od_origin",
        "od_destination",
        "demand",
        "ground_cost",
        "deadline",
    ):
        digest.update(key.encode())
        value = np.asarray(data.get(key, []), dtype="<f8")
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def describe_demand(data, expected_signature=None):
    signature = demand_signature(data)
    if expected_signature is not None and signature != expected_signature:
        raise ValueError(
            "Demand identity, volume, courier baseline or deadline changed"
        )
    return dict(
        n_od=len(data["demand"]),
        hourly_demand=float(np.sum(data["demand"])),
        all_ground_cost=float(np.asarray(data["demand"]) @ data["ground_cost"]),
        signature=signature,
    )
