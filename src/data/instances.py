"""Market inputs and explicit scenarios for consecutive (all durations hours).

Saved legacy inputs are data sources only. No legacy model or solver is imported.
"""

from copy import deepcopy
from dataclasses import fields, replace
import json
import numpy as np

from src.config import load_config
from src.model.types import MODEL_VERSION
from .instance_io import project_path, read_instance, sha256
from .choice import pickup_probability, rider_tariff

GEOMETRY = ("bases", "hubs", "origins", "destinations", "od_origin", "od_destination")


def to_planning_data(data):
    from src.model.types import PlanningData

    if isinstance(data, PlanningData):
        return data
    if data.get("model_version") != MODEL_VERSION:
        raise ValueError("Use load_saved to explicitly convert legacy input data")
    names = {field.name for field in fields(PlanningData) if field.init}
    values = {name: data[name] for name in names if name in data}
    # Unlabelled historical snapshots used the shared-prefix branch equations.
    values.setdefault("flow_policy", "prefix")
    return PlanningData(**values)


def pack_planning_data(planning, metadata=None):
    planning = to_planning_data(planning)
    payload = {
        field.name: deepcopy(getattr(planning, field.name))
        for field in fields(planning)
        if field.init
    }
    payload.update(model_version=MODEL_VERSION, metadata=deepcopy(metadata or {}))
    return payload


def _read_saved(path):
    path = project_path(path)
    result = json.loads((path / "input_metadata.json").read_text())["values"]
    with np.load(path / "input_arrays.npz", allow_pickle=False) as arrays:
        result.update({key: arrays[key].copy() for key in arrays.files})
    return result, path


def load_saved(path, config=None):
    """Read a new-model snapshot or convert only raw market inputs of an old one.

    Solver-only changes replay new snapshots exactly. Explicit physical changes
    merge over the saved scenario rather than silently applying current defaults.
    """
    raw, path = _read_saved(path)
    if raw.get("model_version") == "consecutive_v2":
        # This historical name describes the exact same physical contract.
        # Normalize only the name on read; retain every saved array and input file.
        raw["model_version"] = MODEL_VERSION
        metadata = raw.setdefault("metadata", {})
        metadata["model_name_migration"] = {
            "source_name": "consecutive_v2",
            "target_name": MODEL_VERSION,
            "physical_inputs_changed": False,
        }
        if "resolved_config" in metadata:
            metadata["resolved_config"]["model"] = MODEL_VERSION
    if raw.get("model_version") == MODEL_VERSION:
        planning = to_planning_data(raw)
        if config is None:
            return raw
        changes = (
            deepcopy(config)
            if isinstance(config, dict)
            else json.loads(project_path(config).read_text())
        )
        load_config(changes)  # Validate explicit keys even for solver-only replay.
        physical = set(changes) - {
            "solver",
            "name",
            "schema_version",
            "model",
            "max_orders",
            "flow_policy",
        }
        if not physical:
            policy_changes = {
                key: changes[key]
                for key in ("max_orders", "flow_policy")
                if key in changes
            }
            if policy_changes:
                metadata = deepcopy(raw.get("metadata", {}))
                if "resolved_config" in metadata:
                    metadata["resolved_config"].update(policy_changes)
                metadata.setdefault("scenario_overrides", []).append(policy_changes)
                raw.update(
                    pack_planning_data(replace(planning, **policy_changes), metadata)
                )
            return raw
        saved = deepcopy(raw.get("metadata", {}).get("resolved_config"))
        if saved is None:
            defaults = load_config()
            sections = (
                "operations",
                "terminal",
                "compatibility",
                "choice",
                "costs",
                "depreciation",
                "data",
            )
            if any(
                section not in changes
                or set(changes[section]) != set(defaults[section])
                for section in sections
            ):
                raise ValueError(
                    "Physical changes require saved resolved_config or a complete explicit scenario"
                )
            cfg = load_config(changes)
        else:
            # Historical snapshots predate these controls.  Unrelated physical
            # changes must not silently adopt the new policy or kappa screen.
            saved.setdefault("flow_policy", raw.get("flow_policy", "prefix"))
            saved.setdefault("compatibility", {}).setdefault(
                "require_feasible_path", False
            )
            # Obsolete solver controls do not alter the saved physical scenario.
            supported_solver = load_config()["solver"]
            saved["solver"] = {
                key: value
                for key, value in saved.get("solver", {}).items()
                if key in supported_solver
            }
            cfg = load_config(saved, overrides=changes)
        source = dict(
            path=str(path),
            model=MODEL_VERSION,
            hashes={
                name: sha256(path / name)
                for name in ("input_arrays.npz", "input_metadata.json")
            },
            conversion="explicit changes merged over saved scenario",
            overrides=changes,
        )
        return _scenario(raw, cfg, source)
    cfg = load_config(config)
    if raw.get("model_version") not in (
        MODEL_VERSION,
        "path_cqn_v1",
        "continuation_v1",
    ):
        raise ValueError(
            "Unknown saved input model; supply a documented market adapter"
        )
    source = dict(
        path=str(path),
        model=raw.get("model_version"),
        hashes={
            name: sha256(path / name)
            for name in ("input_arrays.npz", "input_metadata.json")
        },
        conversion="market-only; legacy decisions, paths and solution ignored",
    )
    return _scenario(raw, cfg, source)


def load_instance(hubs=None, config=None):
    """Load the default saved market, or a requested canonical locker count."""
    cfg = load_config(config)
    if hubs is None:
        return load_saved(cfg["data"]["input_path"], cfg)
    from .calibration import load_calibration

    raw, direct, _ = read_instance(
        hubs, cfg["data"]["instance_root"], demand_hubs=cfg["data"]["demand_hubs"]
    )
    selected = raw["provenance"]["demand_selection"]["source_hubs"]
    cal = load_calibration(selected, raw, cfg["data"]["calibration"])
    raw["rider_slope_minutes_per_straight_km"] = cal["mean_fit"][
        "slope_minutes_per_straight_km"
    ]
    raw["delivery_deadline"] = cal["table"].deadline_minutes.to_numpy() / 60
    raw["ground"] = rider_tariff(direct, cfg)
    return _scenario(
        raw,
        cfg,
        dict(
            kind="canonical_csv",
            provenance=raw["provenance"],
            calibration=str(cal["path"]),
        ),
    )


def _matrix(value, shape, name):
    try:
        result = np.broadcast_to(np.asarray(value, float), shape).copy()
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _scenario(raw, cfg, source):
    from src.model.types import PlanningData

    missing = [key for key in GEOMETRY if key not in raw]
    if missing:
        raise ValueError("Market geometry missing: " + ", ".join(missing))
    coords = {key: np.array(raw[key], copy=True) for key in GEOMETRY}
    coords["od_origin"] = coords["od_origin"].astype(int)
    coords["od_destination"] = coords["od_destination"].astype(int)
    base, hub = coords["bases"], coords["hubs"]
    origin = coords["origins"][coords["od_origin"]]
    destination = coords["destinations"][coords["od_destination"]]
    previous = (
        raw.get("metadata", {}).get("resolved_config", {})
        if raw.get("model_version") == MODEL_VERSION
        else {}
    )
    prior_demand_scale = previous.get("data", {}).get("demand_scale", 1.0)
    demand = (
        np.asarray(raw["demand"], float)
        / prior_demand_scale
        * cfg["data"]["demand_scale"]
    )
    R, L, H = len(demand), len(base), len(hub)
    first_distance = np.linalg.norm(origin[:, None] - hub[None], axis=2)
    last_distance = np.linalg.norm(destination[:, None] - hub[None], axis=2)
    xy = np.concatenate((base, hub))
    distance = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    op, terminal, costs = cfg["operations"], cfg["terminal"], cfg["costs"]
    slope = op["rider_minutes_per_km"]
    if slope is None:
        slope = raw.get("rider_slope_minutes_per_straight_km")
    means_source = (
        "operations.rider_minutes_per_km"
        if op["rider_minutes_per_km"] is not None
        else "saved/calibrated empirical distance slope"
    )
    if slope is not None:
        first_mean = first_distance * float(slope) / 60
        last_mean = last_distance * float(slope) / 60
    elif "ground_time" in raw and "last_ground_time" in raw:
        first_mean = np.asarray(raw["ground_time"])[coords["od_origin"]]
        last_mean = np.asarray(raw["last_ground_time"])[coords["od_destination"]]
        means_source = "saved first/last courier travel means"
    elif "rider_mean" in raw and "last_rider_mean" in raw:
        first_mean, last_mean = raw["rider_mean"], raw["last_rider_mean"]
        means_source = "saved new-model courier means"
    else:
        raise ValueError(
            "A calibrated rider slope or both saved courier travel mean arrays are required"
        )
    sigma = pickup_probability(last_distance, **cfg["choice"])
    removal_s = _matrix(
        np.asarray(terminal["self_pickup_removal_minutes"]) / 60,
        (R, H),
        "self pickup removal",
    )
    receipt_s = _matrix(
        np.asarray(terminal["self_pickup_receipt_minutes"]) / 60,
        (R, H),
        "self pickup receipt",
    )
    removal_r = _matrix(
        np.asarray(terminal["courier_removal_minutes"]) / 60, (R, H), "courier removal"
    )
    if terminal["courier_receipt_minutes"] is None:
        receipt_r = removal_r + last_mean
        receipt_source = (
            "scenario: courier removal delay plus calibrated locker-to-customer travel"
        )
    else:
        receipt_r = _matrix(
            np.asarray(terminal["courier_receipt_minutes"]) / 60,
            (R, H),
            "courier receipt",
        )
        receipt_source = "terminal.courier_receipt_minutes"
    if np.any(receipt_s + 1e-12 < removal_s) or np.any(receipt_r + 1e-12 < removal_r):
        raise ValueError(
            "Final receipt cannot precede meal removal in either terminal mode"
        )
    deadline = op["deadline_minutes"]
    if deadline is None:
        deadline = raw.get("deadline", raw.get("delivery_deadline"))
        deadline_source = "saved/calibrated OD mean delivery deadline"
        if deadline is None:
            raise ValueError(
                "Explicit operations.deadline_minutes or calibrated OD deadline required"
            )
        deadline = np.asarray(deadline) / previous.get("operations", {}).get(
            "deadline_multiplier", 1.0
        )
    else:
        deadline = np.asarray(deadline) / 60
        deadline_source = "operations.deadline_minutes"
    deadline = _matrix(deadline, (R,), "deadline") * op["deadline_multiplier"]
    ground = np.asarray(
        raw.get(
            "ground_cost",
            raw.get(
                "ground",
                rider_tariff(np.linalg.norm(origin - destination, axis=1), cfg),
            ),
        )
    )
    if "first_cost" in raw:
        first_cost = np.asarray(raw["first_cost"])
        if raw.get("model_version") != MODEL_VERSION:
            first_cost = first_cost[coords["od_origin"]]
    else:
        first_cost = costs["feeder_rider_multiplier"] * rider_tariff(
            first_distance, cfg
        )
    if "last_rider_cost" in raw:
        last_cost = np.asarray(raw["last_rider_cost"])
        if raw.get("model_version") != MODEL_VERSION:
            last_cost = last_cost[coords["od_destination"]]
    else:
        last_cost = costs["feeder_rider_multiplier"] * rider_tariff(last_distance, cfg)
    compatibility = cfg["compatibility"]
    if compatibility["mode"] == "matrix":
        kappa = _matrix(compatibility["matrix"], (R, R), "kappa")
    elif compatibility["mode"] == "distance":
        kappa = (
            np.linalg.norm(destination[:, None] - origin[None], axis=2)
            <= compatibility["max_distance_km"]
        ).astype(float)
    else:
        kappa = np.full((R, R), float(compatibility["mode"] == "all"))
    if np.any(kappa > 1):
        raise ValueError("kappa must lie in [0, 1]")
    amortization = float(np.prod(list(cfg["depreciation"].values())))
    slots = np.asarray(op["slot_counts"])
    planning = PlanningData(
        n_l=L,
        n_h=H,
        demand=demand,
        flight_times=distance / op["drone_speed_kmh"],
        rider_mean=first_mean,
        first_cost=first_cost,
        terminal_cost=(1 - sigma) * last_cost,
        receipt_mean=sigma * receipt_s + (1 - sigma) * receipt_r,
        removal_mean=sigma * removal_s + (1 - sigma) * removal_r,
        ground_cost=ground,
        deadline=deadline,
        base_cost=costs["base_investment"] / amortization,
        slot_counts=slots,
        slot_cost=(
            costs["compartment_investment"] * slots + costs["locker_site_investment"]
        )
        / amortization,
        slot_epsilon=op["slot_epsilon"],
        link_cost=distance * costs["directed_route_investment_per_km"] / amortization,
        kappa=kappa,
        delta=op["window_minutes"] / 60,
        operating_limit=op["operating_limit_minutes"] / 60,
        fleet_cost=costs["drone_investment"] / amortization,
        flight_cost=costs["flight_per_hour"],
        waiting_cost=costs["waiting_per_hour"],
        max_orders=cfg["max_orders"],
        flow_policy=cfg["flow_policy"],
        route_mode=op["route_mode"],
        name=str(raw.get("name", cfg["name"])),
        allowed_links=raw.get("allowed_links")
        if raw.get("model_version") == MODEL_VERSION
        else None,
        candidate_routes=raw.get("candidate_routes")
        if raw.get("model_version") == MODEL_VERSION
        else None,
    )
    compatibility_screen = None
    if compatibility["require_feasible_path"]:
        from src.model.compatibility import screen_compatibility

        planning, compatibility_screen = screen_compatibility(planning)
    metadata = dict(
        source=source,
        resolved_config=cfg,
        scenario_name=cfg["name"],
        coordinate_system="local_km",
        source_metadata=raw.get("provenance", {}).get(
            "instance_metadata", raw.get("metadata", {}).get("source_metadata", {})
        ),
        assumptions={
            "compatibility": f"explicit {compatibility['mode']} scenario; not estimated from observed continuation",
            "rider_mean": means_source,
            "rider_distribution": "exponential working assumption; data identify mean only",
            "receipt": receipt_source,
            "removal": "separate terminal self_pickup_removal_minutes/courier_removal_minutes scenario inputs",
            "choice": "fixed exogenous pickup-minus-home distance/time/price differences; separate walking penalty",
            "deadline": deadline_source,
            "infrastructure_costs": "explicit scenario investment divided by operating depreciation hours",
            "market_costs": "saved OD courier baseline and feeder tariffs preserved when available",
            "capacity": "configured integer physical compartments, never legacy workload-rate capacity",
            "demand": "original OD identities and rates preserved, then explicit demand_scale applied",
            "flight_network": "L/H only; E facilities and legacy paths excluded",
        },
    )
    if compatibility_screen is not None:
        metadata["compatibility_screen"] = compatibility_screen
        metadata["assumptions"]["compatibility"] += (
            "; intersected with complete feasible two-order path support over the fixed candidate network"
        )
    result = pack_planning_data(planning, metadata)
    result.update(
        coords,
        pickup_share=sigma,
        last_rider_cost=last_cost,
        last_rider_mean=np.asarray(last_mean),
        od_pairs=deepcopy(raw.get("od_pairs")),
        n_od=R,
    )
    return result
