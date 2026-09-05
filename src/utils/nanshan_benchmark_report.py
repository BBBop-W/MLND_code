"""Publication-style comparison figures for one Nanshan MLND benchmark.

The report consumes only canonical instance files and current result-schema
artifacts.  It deliberately refuses legacy summaries so a visually polished
figure cannot accidentally mix formulations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlnd-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mlnd-xdg-cache")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from src.model_contract import FORMULATION_ID as FORMULATION_ID
from src.model_contract import (
    require_result_contract, require_same_result_schema,
    runtime_contract,
)
from src.utils.instance_plotter import StudyLayout, load_study_layout


MODES = ("MX", "HD", "LO")
REQUIRED_RUN_ORDER = ("HD", "MX", "LO")
MODE_LABELS = {
    "MX": "Mixed delivery",
    "HD": "Home delivery",
    "LO": "Locker pickup",
}
MODE_COLORS = {
    "MX": "#2F6B9A",
    "HD": "#D9901A",
    "LO": "#718355",
}
SITE_STYLES = {
    "H": ("o", "#2F6B9A", "Smart-locker candidate"),
    "L": ("^", "#D9901A", "Drone-base candidate"),
    "E": ("D", "#718355", "Battery-swap candidate"),
}

_FACILITY_TYPE_CODES = {
    "base": "L",
    "locker": "H",
    "swap": "E",
}
_SUMMARY_ARTIFACTS = ("open_facilities.csv", "corridor_flows.csv")
_MODE_REQUIRED_FILES = ("summary.json", *_SUMMARY_ARTIFACTS, "incumbent.sol")
_CORRIDOR_FIELDS = {
    "origin_locker",
    "destination_locker",
    "connection_open",
    "od_offered_per_hour",
    "source_allocated_offered_per_hour",
    "offered_reconciliation_error",
    "realized_from_chi_per_hour",
    "loaded_psi_per_hour",
    "realized_reconciliation_error",
    "offered_flow_weighted_rho",
    "direct_flow_per_hour",
    "swap_flow_per_hour",
    "swap_share",
    "flight_km_per_hour",
}
_REL_TOL = 1e-7
_ABS_TOL = 2e-6
# Gurobi writes each sparse flow separately. Re-summing hundreds of values can
# differ from the aggregate solver expression by a few parts per million; the
# absolute guard still rejects a material corridor-level discrepancy.
_FLOW_REL_TOL = 2e-6
_FLOW_ABS_TOL = 3e-5
_ENERGY_ABS_TOL = 2e-6


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_csv_table(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        return fields, list(reader)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_current_contract(payload: Mapping[str, Any], path: Path) -> None:
    require_result_contract(payload, f"result {path}")


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
    context: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} is missing mapping field {key!r}")
    return value


def _required_number(
    payload: Mapping[str, Any],
    key: str,
    context: str,
) -> float:
    if key not in payload:
        raise ValueError(f"{context} is missing numeric field {key!r}")
    number = _optional_finite(payload.get(key))
    if number is None:
        raise ValueError(f"{context} has non-finite numeric field {key!r}")
    return number


def _required_integer(
    payload: Mapping[str, Any],
    key: str,
    context: str,
    *,
    minimum: int = 0,
) -> int:
    number = _required_number(payload, key, context)
    value = int(round(number))
    if abs(number - value) > _ABS_TOL or value < minimum:
        raise ValueError(
            f"{context} has invalid integer field {key!r}: {payload.get(key)!r}"
        )
    return value


def _required_bool(
    payload: Mapping[str, Any],
    key: str,
    context: str,
) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context} is missing boolean field {key!r}")
    return value


def _required_sha256(
    payload: Mapping[str, Any],
    key: str,
    context: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{context} is missing lowercase sha256 field {key!r}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{context} has invalid sha256 field {key!r}") from exc
    return value


def _require_close(left: float, right: float, context: str) -> None:
    if not math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
        raise ValueError(f"{context} mismatch: {left!r} != {right!r}")


def _require_flow_close(left: float, right: float, context: str) -> None:
    """Compare reconstructed flows at Gurobi-scale summation tolerance."""

    if not math.isclose(
        left,
        right,
        rel_tol=_FLOW_REL_TOL,
        abs_tol=_FLOW_ABS_TOL,
    ):
        raise ValueError(f"{context} mismatch: {left!r} != {right!r}")


def _require_complete_result_files(results_root: Path) -> None:
    required = [results_root / "run_config.json", results_root / "mode_comparison.csv"]
    required.extend(
        results_root / mode / filename
        for mode in MODES
        for filename in _MODE_REQUIRED_FILES
    )
    missing = [
        str(path.relative_to(results_root)) for path in required if not path.is_file()
    ]
    if missing:
        raise ValueError(
            "incomplete three-mode result package; missing files: " + ", ".join(missing)
        )


def _load_benchmark(
    instance_dir: Path,
    results_root: Path,
) -> tuple[StudyLayout, dict[str, Any], dict[str, dict[str, Any]]]:
    _require_complete_result_files(results_root)
    run_config_path = results_root / "run_config.json"
    run_config = _read_json(run_config_path)
    _require_current_contract(run_config, run_config_path)

    configured_modes = run_config.get("modes")
    if configured_modes != list(REQUIRED_RUN_ORDER):
        raise ValueError(
            "run_config modes must be ordered HD, MX, LO so MX can receive the "
            "HD-sourced external start; "
            f"got {configured_modes!r}"
        )
    if run_config.get("cross_mode_start_policy") != "HD_to_later_MX_by_variable_name":
        raise ValueError("run_config has an invalid HD-to-MX start policy")

    recorded_hashes = _mapping(
        _mapping(run_config.get("instance")).get("canonical_file_sha256")
    )
    for filename in (
        "metadata.json",
        "sites.csv",
        "od_nodes.csv",
        "od_demand.csv",
        "periods.csv",
    ):
        expected = recorded_hashes.get(filename)
        actual = _sha256_file(instance_dir / filename)
        if expected != actual:
            raise ValueError(
                f"canonical input mismatch for {filename}: "
                f"run={expected!r}, current={actual!r}"
            )

    summaries: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        path = results_root / mode / "summary.json"
        summary = _read_json(path)
        _require_current_contract(summary, path)
        if summary.get("mode") != mode:
            raise ValueError(f"mode mismatch in {path}")
        summaries[mode] = summary

    layout = load_study_layout(instance_dir)
    _validate_benchmark_consistency(layout, run_config, summaries)
    _validate_mode_artifacts(layout, summaries, results_root)
    _validate_mode_comparison(summaries, results_root / "mode_comparison.csv")
    return layout, run_config, summaries


def _add_research_blossom(fig: plt.Figure) -> None:
    """Add the restrained research-chart blossom in the top-right corner."""

    centre_x, centre_y = 0.975, 0.965
    for angle in range(0, 360, 72):
        radians = math.radians(angle)
        fig.add_artist(
            Circle(
                (
                    centre_x + 0.008 * math.cos(radians),
                    centre_y + 0.012 * math.sin(radians),
                ),
                0.0032,
                transform=fig.transFigure,
                facecolor="#93B7D3",
                edgecolor="none",
                alpha=0.85,
                zorder=20,
            )
        )
    fig.add_artist(
        Circle(
            (centre_x, centre_y),
            0.0034,
            transform=fig.transFigure,
            facecolor="#2F6B9A",
            edgecolor="none",
            zorder=21,
        )
    )


def _activity_scale(layout: StudyLayout) -> float:
    return max(layout.endpoint_rates.values(), default=1.0)


def _draw_activity(
    ax: plt.Axes,
    layout: StudyLayout,
    maximum_rate: float,
) -> None:
    active_nodes = [
        node
        for node in layout.nodes
        if layout.endpoint_rates.get(str(node["id"]), 0.0) > 0.0
    ]
    ax.scatter(
        [float(node["x"]) for node in active_nodes],
        [float(node["y"]) for node in active_nodes],
        s=[
            5.0
            + 36.0 * math.sqrt(layout.endpoint_rates[str(node["id"])] / maximum_rate)
            for node in active_nodes
        ],
        color="#64748B",
        alpha=0.13,
        linewidths=0,
        zorder=0,
    )


def _open_facilities(mode_dir: Path) -> set[str]:
    path = mode_dir / "open_facilities.csv"
    if not path.is_file():
        raise ValueError(f"missing required result artifact: {path}")
    return {row["facility"] for row in _read_csv(path)}


def _corridor_rows(mode_dir: Path) -> list[dict[str, str]]:
    path = mode_dir / "corridor_flows.csv"
    if not path.is_file():
        raise ValueError(f"missing required result artifact: {path}")
    return _read_csv(path)


def _physical_route_rows(mode_dir: Path) -> list[dict[str, str | float]]:
    """Read positive physical route flows from the auditable Gurobi solution.

    ``incumbent.sol`` is written for every feasible mode by :mod:`main`.  Reading
    the physical ``varphi`` variables lets the comparison map show base
    dispatch, loaded H--H travel, recovery, and the actual two-leg geometry of
    one-swap routes without introducing a second, unverified flow export.
    """

    path = mode_dir / "incumbent.sol"
    if not path.is_file():
        raise ValueError(f"missing required result artifact: {path}")
    rows: list[dict[str, str | float]] = []
    seen: set[tuple[str, str, str]] = set()
    variable_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line.startswith("varphi["):
                continue
            try:
                name, value_text = line.rsplit(maxsplit=1)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} has a malformed varphi assignment"
                ) from exc
            if not name.endswith("]"):
                raise ValueError(
                    f"{path}:{line_number} has a malformed varphi variable name"
                )
            parts = name[len("varphi[") : -1].split(",")
            if len(parts) != 3 or any(not part for part in parts):
                raise ValueError(
                    f"{path}:{line_number} has a malformed varphi variable name"
                )
            key = (parts[0], parts[1], parts[2])
            if key in seen:
                raise ValueError(f"{path} contains duplicate varphi variable {key!r}")
            seen.add(key)
            variable_count += 1
            value = _optional_finite(value_text)
            if value is None:
                raise ValueError(f"{path}:{line_number} has a non-finite varphi value")
            if value < -_ABS_TOL:
                raise ValueError(f"{path}:{line_number} has negative physical flow")
            if value > _ABS_TOL:
                rows.append(
                    {
                        "origin": key[0],
                        "destination": key[1],
                        "option": key[2],
                        "flow_per_hour": value,
                    }
                )
    if variable_count == 0:
        raise ValueError(f"{path} contains no varphi variables")
    return rows


def _draw_sites(
    ax: plt.Axes,
    layout: StudyLayout,
    opened: set[str],
) -> None:
    # Draw larger L/E marks first; H remains visible at co-located candidates.
    for site_type in ("L", "E", "H"):
        marker, color, _label = SITE_STYLES[site_type]
        sites = [site for site in layout.sites if site["type"] == site_type]
        size = {"L": 68.0, "E": 50.0, "H": 34.0}[site_type]
        ax.scatter(
            [float(site["x"]) for site in sites],
            [float(site["y"]) for site in sites],
            marker=marker,
            s=size,
            facecolors="white",
            edgecolors=color,
            linewidths=1.05,
            alpha=0.86,
            zorder=3,
        )
        active = [site for site in sites if str(site["id"]) in opened]
        if active:
            ax.scatter(
                [float(site["x"]) for site in active],
                [float(site["y"]) for site in active],
                marker=marker,
                s=1.8 * size,
                facecolors=color,
                edgecolors="white",
                linewidths=0.9,
                zorder=6,
            )


def _maximum_physical_flow(results_root: Path) -> float:
    maximum = 0.0
    for mode in MODES:
        for row in _physical_route_rows(results_root / mode):
            maximum = max(maximum, float(row["flow_per_hour"]))
    return maximum


def _draw_physical_routes(
    ax: plt.Axes,
    layout: StudyLayout,
    rows: list[dict[str, str | float]],
    maximum_flow: float,
) -> None:
    if maximum_flow <= 1e-9:
        return
    coordinates = {
        str(site["id"]): (float(site["x"]), float(site["y"])) for site in layout.sites
    }
    site_types = {str(site["id"]): str(site["type"]) for site in layout.sites}
    for row in rows:
        flow = float(row["flow_per_hour"])
        if flow <= 1e-9:
            continue
        origin_id = str(row["origin"])
        destination_id = str(row["destination"])
        option = str(row["option"])
        origin = coordinates[origin_id]
        destination = coordinates[destination_id]
        points = [origin, destination]
        if option != "__direct__":
            points.insert(1, coordinates[option])
        loaded = site_types[origin_id] == site_types[destination_id] == "H"
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color="#334155" if loaded else "#94A3B8",
            linewidth=0.65 + 3.0 * math.sqrt(flow / maximum_flow),
            alpha=0.62 if loaded else 0.48,
            solid_capstyle="round",
            zorder=2 if loaded else 1,
        )


def _gap_percent(summary: Mapping[str, Any]) -> float:
    return 100.0 * _required_number(summary, "mip_gap", "result summary")


def _optional_finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _summary_baseline(summary: Mapping[str, Any]) -> float:
    context = f"{summary.get('mode', 'unknown')} summary"
    baseline = _required_number(summary, "courier_only_baseline", context)
    rider_baseline = _required_mapping(summary, "all_rider_baseline", context)
    nested = _required_number(rider_baseline, "cost_per_hour", f"{context} baseline")
    _require_close(baseline, nested, f"{context} all-courier baseline")
    if baseline <= 0.0:
        raise ValueError(f"{context} all-courier baseline must be positive")
    return baseline


def _candidate_site_counts(
    summaries: Mapping[str, Mapping[str, Any]],
    run_config: Mapping[str, Any] | None = None,
    layout: StudyLayout | None = None,
) -> dict[str, int]:
    """Resolve and cross-check candidate L/H/E counts from available artifacts."""

    evidence: dict[str, list[tuple[str, int]]] = {
        "bases": [],
        "lockers": [],
        "swap_stations": [],
    }

    def add(kind: str, source: str, value: Any) -> None:
        number = _optional_finite(value)
        if number is None:
            raise ValueError(f"missing candidate-{kind} count from {source}")
        count = int(round(number))
        minimum = 1 if kind == "lockers" else 0
        if count < minimum or abs(number - count) > _ABS_TOL:
            raise ValueError(f"invalid candidate-{kind} count from {source}: {value!r}")
        evidence[kind].append((source, count))

    instance_keys = {
        "bases": "candidate_bases",
        "lockers": "candidate_lockers",
        "swap_stations": "candidate_swap_stations",
    }
    for mode in MODES:
        if mode not in summaries:
            raise ValueError(f"missing {mode} summary")
        summary = summaries[mode]
        instance = _required_mapping(summary, "instance", f"{mode} summary")
        facilities = _required_mapping(summary, "facilities", f"{mode} summary")
        candidates = _required_mapping(
            facilities,
            "candidates",
            f"{mode} facility summary",
        )
        for kind, instance_key in instance_keys.items():
            add(kind, f"{mode} summary", instance.get(instance_key))
            add(kind, f"{mode} facility summary", candidates.get(kind))

    if run_config is not None:
        run_instance = _required_mapping(run_config, "instance", "run_config")
        metadata = _required_mapping(run_instance, "metadata", "run_config instance")
        data_config = _required_mapping(metadata, "data_config", "run metadata")
        for kind, key in (
            ("bases", "num_bases"),
            ("lockers", "candidate_count"),
            ("swap_stations", "num_swaps"),
        ):
            add(kind, "run_config", data_config.get(key))

    if layout is not None:
        for kind, site_type in (
            ("bases", "L"),
            ("lockers", "H"),
            ("swap_stations", "E"),
        ):
            add(
                kind,
                "canonical sites.csv",
                sum(site.get("type") == site_type for site in layout.sites),
            )

    counts: dict[str, int] = {}
    for kind, items in evidence.items():
        distinct = {count for _source, count in items}
        if len(distinct) != 1:
            details = ", ".join(f"{source}={count}" for source, count in items)
            label = {
                "bases": "base",
                "lockers": "locker",
                "swap_stations": "swap-station",
            }[kind]
            raise ValueError(f"inconsistent candidate-{label} counts: {details}")
        counts[kind] = items[0][1]
    return counts


def _candidate_locker_count(
    summaries: Mapping[str, Mapping[str, Any]],
    run_config: Mapping[str, Any] | None = None,
    layout: StudyLayout | None = None,
) -> int:
    """Resolve and cross-check the candidate-H count across report artifacts."""

    return _candidate_site_counts(summaries, run_config, layout)["lockers"]


def _artifact_prefix(candidate_lockers: int) -> str:
    if candidate_lockers <= 0:
        raise ValueError("candidate_lockers must be positive")
    return f"h{candidate_lockers}"


def _screening_description(
    run_config: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    model = _required_mapping(run_config, "model", "run_config")
    display = model.get("display_origin_choices")
    general = model.get("general_corridor_choices")
    configured = _required_mapping(
        run_config,
        "screening_is_exact",
        "run_config",
    )
    exact_by_mode = {
        mode: _required_bool(configured, mode, "run_config screening") for mode in MODES
    }
    if summaries is not None:
        for mode in MODES:
            exact = _required_bool(
                summaries[mode], "screening_is_exact", f"{mode} summary"
            )
            if exact != exact_by_mode[mode]:
                raise ValueError(f"{mode} screening flag disagrees with run_config")
    if all(exact_by_mode.values()):
        return "Full formulation index retained (screening_is_exact=True)."
    display_text = "all" if display is None else str(display)
    general_text = "all" if general is None else str(general)
    screened_modes = "/".join(mode for mode in MODES if not exact_by_mode[mode])
    return (
        f"Screened index ({screened_modes}: top-{display_text} display origins, "
        f"top-{general_text} general corridors); MIP gaps apply only to this "
        "screened formulation."
    )


def _energy_metrics(summary: Mapping[str, Any], context: str) -> dict[str, float]:
    energy = _required_mapping(summary, "energy_balance", context)
    expected_scope = "system_wide_base_departure_plus_swap"
    if energy.get("balance_scope") != expected_scope:
        raise ValueError(f"{context} energy balance has an invalid scope")

    speed = _required_number(energy, "drone_speed_mps", f"{context} energy")
    battery_range = _required_number(
        energy,
        "battery_range_km",
        f"{context} energy",
    )
    usable_hours = _required_number(
        energy,
        "usable_battery_time_hours",
        f"{context} energy",
    )
    usable_minutes = _required_number(
        energy,
        "usable_battery_time_minutes",
        f"{context} energy",
    )
    if speed <= 0.0 or battery_range <= 0.0:
        raise ValueError(f"{context} speed and battery range must be positive")

    def require_energy_close(left: float, right: float, label: str) -> None:
        if not math.isclose(
            left,
            right,
            rel_tol=_REL_TOL,
            abs_tol=_ENERGY_ABS_TOL,
        ):
            raise ValueError(f"{label} mismatch: {left!r} != {right!r}")

    require_energy_close(
        usable_hours,
        battery_range / (3.6 * speed),
        f"{context} usable battery time",
    )
    require_energy_close(
        usable_minutes,
        60.0 * usable_hours,
        f"{context} usable battery minutes",
    )

    energy_use = _required_number(
        energy,
        "total_energy_use_equivalent_drones",
        f"{context} energy",
    )
    energy_supply = _required_number(
        energy,
        "total_energy_supply_equivalent_drones",
        f"{context} energy",
    )
    base_departures = _required_number(
        energy,
        "base_full_battery_departures_per_hour",
        f"{context} energy",
    )
    swap_events = _required_number(
        energy,
        "swap_events_per_hour",
        f"{context} energy",
    )
    supply_events = _required_number(
        energy,
        "battery_supply_events_per_hour",
        f"{context} energy",
    )
    slack = _required_number(
        energy,
        "system_slack_equivalent_drones",
        f"{context} energy",
    )
    violation = _required_number(
        energy,
        "maximum_constraint_violation",
        f"{context} energy",
    )
    if (
        min(
            energy_use,
            energy_supply,
            base_departures,
            swap_events,
            supply_events,
            violation,
        )
        < -_ENERGY_ABS_TOL
    ):
        raise ValueError(f"{context} energy quantities must be nonnegative")
    require_energy_close(
        supply_events,
        base_departures + swap_events,
        f"{context} battery supply events",
    )
    require_energy_close(
        energy_supply,
        usable_hours * supply_events,
        f"{context} aggregate energy supply",
    )
    require_energy_close(
        slack,
        energy_supply - energy_use,
        f"{context} aggregate energy slack",
    )
    require_energy_close(
        violation,
        max(0.0, energy_use - energy_supply),
        f"{context} aggregate energy violation",
    )
    if violation > _ENERGY_ABS_TOL:
        raise ValueError(
            f"{context} maximum energy constraint violation exceeds tolerance: "
            f"{violation:g}"
        )
    return {
        "energy_use": energy_use,
        "energy_supply": energy_supply,
        "base_battery_departures": base_departures,
        "swap_events": swap_events,
        "battery_supply_events": supply_events,
        "energy_slack": slack,
        "energy_violation": violation,
    }


def _solution_metrics(summary: Mapping[str, Any]) -> dict[str, float | int | bool]:
    """Extract comparable design and service metrics from a current summary."""

    mode = str(summary.get("mode", "unknown"))
    context = f"{mode} summary"
    funnel = _required_mapping(summary, "demand_funnel_orders_per_hour", context)
    facilities = _required_mapping(summary, "facilities", context)
    opened = _required_mapping(facilities, "open", f"{context} facilities")
    fleet = _required_mapping(summary, "fleet", context)
    locker_queue = _required_mapping(summary, "locker_queue", context)
    energy_metrics = _energy_metrics(summary, context)

    total_demand = _required_number(funnel, "total_demand", f"{context} funnel")
    realized = _required_number(
        funnel,
        "total_drone_realized",
        f"{context} funnel",
    )
    fallback = _required_number(
        funnel,
        "direct_courier_fallback",
        f"{context} funnel",
    )
    service_fraction = _required_number(
        funnel,
        "drone_service_fraction_of_demand",
        f"{context} funnel",
    )
    if total_demand <= 0.0:
        raise ValueError(f"{context} total demand must be positive")
    if realized < -_ABS_TOL or fallback < -_ABS_TOL:
        raise ValueError(f"{context} demand-funnel rates must be nonnegative")
    _require_close(realized + fallback, total_demand, f"{context} demand conservation")
    _require_close(
        service_fraction,
        realized / total_demand,
        f"{context} service fraction",
    )
    _require_close(
        _required_number(summary, "drone_served_rate", context),
        service_fraction,
        f"{context} drone service rate",
    )

    baseline = _summary_baseline(summary)
    objective = _required_number(summary, "objective", context)
    saving = _required_number(summary, "cost_saving", context)
    saving_percent = _required_number(summary, "cost_saving_percent", context)
    _require_close(saving, baseline - objective, f"{context} cost saving")
    _require_close(
        saving_percent,
        100.0 * saving / baseline,
        f"{context} cost-saving percentage",
    )

    def opened_count(summary_key: str, nested_key: str) -> int:
        top = _required_integer(summary, summary_key, context)
        nested = _required_integer(opened, nested_key, f"{context} facilities")
        if top != nested:
            raise ValueError(f"{context} {summary_key} mismatch: {top!r} != {nested!r}")
        return top

    open_bases = opened_count("open_bases", "bases")
    open_lockers = opened_count("open_lockers", "lockers")
    open_swap_stations = opened_count("open_swap_stations", "swap_stations")
    active_lockers = _required_integer(summary, "active_lockers", context)
    nested_active_lockers = _required_integer(
        facilities,
        "active_lockers",
        f"{context} facilities",
    )
    idle_open_lockers = _required_integer(summary, "idle_open_lockers", context)
    nested_idle_open_lockers = _required_integer(
        facilities,
        "idle_open_lockers",
        f"{context} facilities",
    )
    if active_lockers != nested_active_lockers:
        raise ValueError(f"{context} active-locker count disagrees with facilities")
    if idle_open_lockers != nested_idle_open_lockers:
        raise ValueError(f"{context} idle-open-locker count disagrees with facilities")
    if active_lockers > open_lockers:
        raise ValueError(
            f"{context} active_lockers must satisfy 0 <= active <= open lockers"
        )
    if idle_open_lockers != open_lockers - active_lockers:
        raise ValueError(
            f"{context} idle_open_lockers must equal open_lockers - active_lockers"
        )

    fleet_size = _required_number(summary, "fleet_size", context)
    nested_fleet = _required_number(fleet, "gamma_drones", f"{context} fleet")
    _require_close(fleet_size, nested_fleet, f"{context} fleet size")
    maximum_utilization = _required_number(
        locker_queue,
        "maximum_utilization",
        f"{context} locker queue",
    )
    active_connections = _required_integer(
        facilities,
        "active_connections",
        f"{context} facilities",
    )
    network_is_nontrivial = _required_bool(summary, "network_is_nontrivial", context)
    network_opened = _required_bool(summary, "network_opened", context)
    courier_only = _required_bool(summary, "courier_only_incumbent", context)
    if network_is_nontrivial != (realized > _ABS_TOL):
        raise ValueError(
            f"{context} positive-drone-service flag disagrees with realized service"
        )
    if network_opened != any(
        count > 0 for count in (open_bases, open_lockers, open_swap_stations)
    ):
        raise ValueError(f"{context} network_opened flag disagrees with open counts")
    if courier_only == network_is_nontrivial:
        raise ValueError(
            f"{context} courier-only flag disagrees with positive drone service"
        )

    return {
        "total_demand": total_demand,
        "drone_realized": realized,
        "service_fraction": service_fraction,
        "cost_saving": saving,
        "cost_saving_percent": saving_percent,
        "open_bases": open_bases,
        "open_lockers": open_lockers,
        "active_lockers": active_lockers,
        "idle_open_lockers": idle_open_lockers,
        "open_swap_stations": open_swap_stations,
        "active_connections": active_connections,
        "fleet_size": fleet_size,
        "maximum_locker_utilization": maximum_utilization,
        "network_is_nontrivial": network_is_nontrivial,
        "network_opened": network_opened,
        **energy_metrics,
    }


def _validate_runtime_accounting(summary: Mapping[str, Any], context: str) -> None:
    version = require_result_contract(summary, context)
    solver = _required_mapping(summary, "solver", context)
    total = _required_number(summary, "runtime_seconds", context)
    phases = {
        key: _required_number(solver, key, f"{context} solver")
        for key in ("network_start_runtime_seconds", "main_solve_runtime_seconds")
    }
    if version == 6:
        # Preparation and network subphases can overlap. They are not an
        # additive partition of algorithm wall time and must not be summed.
        for key in ("algorithm_prepare_runtime_seconds", "start_preparation_runtime_seconds", "network_construction_runtime_seconds"):
            if key in solver:
                phases[key] = _required_number(solver, key, f"{context} solver")
    if total < 0 or any(value < 0 for value in phases.values()):
        raise ValueError(f"{context} has negative runtime")
    if version == 5:
        _require_close(sum(phases.values()), total, f"{context} runtime phases")
    elif any(value > total + max(0.05, total * 1e-5) for value in phases.values()):
        raise ValueError(f"{context} runtime phase exceeds algorithm wall time")


def _validate_summary_fields(
    mode: str,
    summary: Mapping[str, Any],
    run_config: Mapping[str, Any] | None = None,
) -> None:
    context = f"{mode} summary"
    require_result_contract(summary, context)
    if run_config is not None:
        require_same_result_schema(run_config, summary, context)
    status = _required_integer(summary, "status", context, minimum=1)
    status_name = summary.get("status_name")
    if not isinstance(status_name, str) or not status_name:
        raise ValueError(f"{context} is missing status_name")
    objective = _required_number(summary, "objective", context)
    bound = _required_number(summary, "objective_bound", context)
    gap = _required_number(summary, "mip_gap", context)
    requested_gap = _required_number(summary, "requested_mip_gap", context)
    _required_number(summary, "runtime_seconds", context)
    if objective < 0.0 or bound > objective + _ABS_TOL or gap < 0.0:
        raise ValueError(f"{context} has invalid minimization certificate")
    computed_gap = abs(objective - bound) / max(abs(objective), _ABS_TOL)
    _require_close(gap, computed_gap, f"{context} MIP gap")

    solver = _required_mapping(summary, "solver", context)
    if _required_integer(solver, "status", f"{context} solver", minimum=1) != status:
        raise ValueError(f"{context} top-level and solver status disagree")
    if solver.get("status_name") != status_name:
        raise ValueError(f"{context} top-level and solver status_name disagree")
    if _required_integer(solver, "solution_count", f"{context} solver", minimum=1) < 1:
        raise ValueError(f"{context} has no incumbent")
    for top_key, nested_key in (
        ("objective", "objective_cost_per_hour"),
        ("objective_bound", "objective_bound"),
        ("mip_gap", "mip_gap"),
        ("requested_mip_gap", "requested_mip_gap"),
        ("runtime_seconds", "runtime_seconds"),
    ):
        _require_close(
            _required_number(summary, top_key, context),
            _required_number(solver, nested_key, f"{context} solver"),
            f"{context} {top_key}",
        )
    _validate_runtime_accounting(summary, context)

    instance = _required_mapping(summary, "instance", context)
    summary_screening = _required_bool(summary, "screening_is_exact", context)
    if (
        _required_bool(instance, "screening_is_exact", f"{context} instance")
        != summary_screening
    ):
        raise ValueError(f"{context} screening flags disagree")
    _required_integer(instance, "markets", f"{context} instance", minimum=1)
    _required_integer(instance, "chi_keys", f"{context} instance")

    energy = _required_mapping(summary, "energy_balance", context)
    speed = _required_number(energy, "drone_speed_mps", f"{context} energy")
    battery_range = _required_number(
        energy,
        "battery_range_km",
        f"{context} energy",
    )
    _energy_metrics(summary, context)

    model_size = _required_mapping(summary, "model_size", context)
    for key in (
        "variables",
        "binary_variables",
        "linear_constraints",
        "quadratic_constraints",
    ):
        _required_integer(model_size, key, f"{context} model size")

    artifacts = _required_mapping(summary, "artifacts", context)
    for filename in _SUMMARY_ARTIFACTS:
        key = filename.removesuffix(".csv")
        if artifacts.get(key) != filename:
            raise ValueError(f"{context} artifact declaration for {key!r} is invalid")

    _solution_metrics(summary)

    if run_config is not None:
        configured_screening = _required_mapping(
            run_config,
            "screening_is_exact",
            "run_config",
        )
        if (
            _required_bool(configured_screening, mode, "run_config screening")
            != summary_screening
        ):
            raise ValueError(f"{context} screening flag disagrees with run_config")
        model = _required_mapping(run_config, "model", "run_config")
        _require_close(
            speed,
            _required_number(model, "drone_speed_mps", "run_config model"),
            f"{context} drone speed",
        )
        _require_close(
            battery_range,
            _required_number(model, "battery_range_km", "run_config model"),
            f"{context} battery range",
        )
        solver_config = _required_mapping(run_config, "solver", "run_config")
        _require_close(
            requested_gap,
            _required_number(solver_config, "mip_gap", "run_config solver"),
            f"{context} requested MIP gap",
        )


def _validate_benchmark_consistency(
    layout: StudyLayout,
    run_config: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate every cross-mode invariant exposed by current artifacts.

    The shared signature binds each summary to the declared run configuration.
    It is the strongest available boundary without a separate completion
    manifest, so this validator makes no stronger same-process claim.
    """

    if run_config.get("modes") != list(REQUIRED_RUN_ORDER):
        raise ValueError(
            "run_config modes must be ordered HD, MX, LO so MX can receive the "
            "HD-sourced external start"
        )
    if run_config.get("cross_mode_start_policy") != "HD_to_later_MX_by_variable_name":
        raise ValueError("run_config has an invalid HD-to-MX start policy")

    run_signature = _required_sha256(
        run_config,
        "run_signature_sha256",
        "run_config",
    )
    for mode in MODES:
        require_same_result_schema(run_config, summaries[mode], f"{mode} summary")
        summary_signature = _required_sha256(
            summaries[mode],
            "run_signature_sha256",
            f"{mode} summary",
        )
        if summary_signature != run_signature:
            raise ValueError(f"{mode} summary run signature disagrees with run_config")

    _candidate_site_counts(summaries, run_config, layout)
    for mode in MODES:
        _validate_summary_fields(mode, summaries[mode], run_config)

    mx_solver = _required_mapping(summaries["MX"], "solver", "MX summary")
    if not _required_bool(
        mx_solver,
        "external_start_installed",
        "MX summary solver",
    ):
        raise ValueError(
            "MX summary must record installation of the HD-sourced external start"
        )
    if mx_solver.get("external_start_source") != "HD":
        raise ValueError("MX installed external-start source must be HD")
    _required_integer(
        mx_solver,
        "external_start_matched_variables",
        "MX summary solver",
        minimum=1,
    )

    representative = summaries[MODES[0]]
    reference_metrics = _solution_metrics(representative)
    reference_baseline = _summary_baseline(representative)
    reference_instance = _required_mapping(
        representative,
        "instance",
        f"{MODES[0]} summary",
    )
    reference_markets = _required_integer(
        reference_instance,
        "markets",
        f"{MODES[0]} instance",
        minimum=1,
    )
    for mode in MODES[1:]:
        summary = summaries[mode]
        metrics = _solution_metrics(summary)
        _require_close(
            float(metrics["total_demand"]),
            float(reference_metrics["total_demand"]),
            f"{mode} versus {MODES[0]} total demand",
        )
        _require_close(
            _summary_baseline(summary),
            reference_baseline,
            f"{mode} versus {MODES[0]} all-courier baseline",
        )
        instance = _required_mapping(summary, "instance", f"{mode} summary")
        markets = _required_integer(
            instance,
            "markets",
            f"{mode} instance",
            minimum=1,
        )
        if markets != reference_markets:
            raise ValueError(
                f"{mode} versus {MODES[0]} OD-market count mismatch: "
                f"{markets} != {reference_markets}"
            )


def _validate_mode_artifacts(
    layout: StudyLayout,
    summaries: Mapping[str, Mapping[str, Any]],
    results_root: Path,
) -> None:
    canonical_types = {str(site["id"]): str(site["type"]) for site in layout.sites}
    for mode in MODES:
        metrics = _solution_metrics(summaries[mode])
        mode_dir = results_root / mode
        facility_path = mode_dir / "open_facilities.csv"
        facility_fields, facility_rows = _read_csv_table(facility_path)
        required_facility_fields = {"facility", "type"}
        if not required_facility_fields.issubset(facility_fields):
            raise ValueError(f"{facility_path} is missing required columns")

        opened: set[str] = set()
        observed_counts = {"L": 0, "H": 0, "E": 0}
        for row in facility_rows:
            facility = row.get("facility", "")
            if not facility or facility in opened:
                raise ValueError(
                    f"{facility_path} has a missing or duplicate facility id"
                )
            opened.add(facility)
            reported_type = _FACILITY_TYPE_CODES.get(row.get("type", ""))
            canonical_type = canonical_types.get(facility)
            if reported_type is None or canonical_type != reported_type:
                raise ValueError(
                    f"{facility_path} has invalid type for {facility!r}: "
                    f"reported={row.get('type')!r}, canonical={canonical_type!r}"
                )
            observed_counts[reported_type] += 1

        expected_counts = {
            "L": int(metrics["open_bases"]),
            "H": int(metrics["open_lockers"]),
            "E": int(metrics["open_swap_stations"]),
        }
        if observed_counts != expected_counts:
            raise ValueError(
                f"{mode} open-facility CSV counts disagree with summary: "
                f"csv={observed_counts}, summary={expected_counts}"
            )

        corridor_path = mode_dir / "corridor_flows.csv"
        corridor_fields, corridor_rows = _read_csv_table(corridor_path)
        if set(corridor_fields) != _CORRIDOR_FIELDS:
            missing = sorted(_CORRIDOR_FIELDS - set(corridor_fields))
            unexpected = sorted(set(corridor_fields) - _CORRIDOR_FIELDS)
            raise ValueError(
                f"{corridor_path} has incompatible columns: "
                f"missing={missing}, unexpected={unexpected}"
            )
        instance = _required_mapping(summaries[mode], "instance", f"{mode} summary")
        expected_corridors = _required_integer(
            instance,
            "endogenous_corridors",
            f"{mode} summary instance",
        )
        if len(corridor_rows) != expected_corridors:
            raise ValueError(
                f"{corridor_path} row count disagrees with summary: "
                f"csv={len(corridor_rows)}, summary={expected_corridors}"
            )
        seen_corridors: set[tuple[str, str]] = set()
        loaded_total = 0.0
        direct_total = 0.0
        swap_total = 0.0
        realized_total = 0.0
        loaded_flight_km_total = 0.0
        for row in corridor_rows:
            origin = row.get("origin_locker", "")
            destination = row.get("destination_locker", "")
            if (
                canonical_types.get(origin) != "H"
                or canonical_types.get(destination) != "H"
            ):
                raise ValueError(f"{corridor_path} contains a non-locker corridor")
            corridor = (origin, destination)
            if corridor in seen_corridors:
                raise ValueError(f"{corridor_path} contains a duplicate corridor")
            seen_corridors.add(corridor)
            connection_open = _required_number(
                row,
                "connection_open",
                str(corridor_path),
            )
            loaded_flow = _required_number(
                row,
                "loaded_psi_per_hour",
                str(corridor_path),
            )
            realized_flow = _required_number(
                row,
                "realized_from_chi_per_hour",
                str(corridor_path),
            )
            direct_flow = _required_number(
                row,
                "direct_flow_per_hour",
                str(corridor_path),
            )
            swap_flow = _required_number(
                row,
                "swap_flow_per_hour",
                str(corridor_path),
            )
            flight_km = _required_number(
                row,
                "flight_km_per_hour",
                str(corridor_path),
            )
            offered_flow = _required_number(
                row,
                "od_offered_per_hour",
                str(corridor_path),
            )
            source_offered_flow = _required_number(
                row,
                "source_allocated_offered_per_hour",
                str(corridor_path),
            )
            offered_error = _required_number(
                row,
                "offered_reconciliation_error",
                str(corridor_path),
            )
            realized_error = _required_number(
                row,
                "realized_reconciliation_error",
                str(corridor_path),
            )
            if not (-_ABS_TOL <= connection_open <= 1.0 + _ABS_TOL):
                raise ValueError(f"{corridor_path} has an invalid connection flag")
            if (
                min(loaded_flow, realized_flow, direct_flow, swap_flow, flight_km)
                < -_ABS_TOL
            ):
                raise ValueError(f"{corridor_path} contains negative physical flow")
            _require_flow_close(
                realized_flow,
                loaded_flow,
                f"{corridor_path} realized-to-loaded corridor flow",
            )
            _require_flow_close(
                offered_flow - source_offered_flow,
                offered_error,
                f"{corridor_path} reported offered reconciliation error",
            )
            _require_flow_close(
                realized_flow - loaded_flow,
                realized_error,
                f"{corridor_path} reported realized reconciliation error",
            )
            if abs(offered_error) > _FLOW_ABS_TOL:
                raise ValueError(
                    f"{corridor_path} offered reconciliation exceeds tolerance"
                )
            if abs(realized_error) > _FLOW_ABS_TOL:
                raise ValueError(
                    f"{corridor_path} realized reconciliation exceeds tolerance"
                )
            _require_close(
                direct_flow + swap_flow,
                loaded_flow,
                f"{corridor_path} physical route flow",
            )
            if loaded_flow > _ABS_TOL and connection_open < 0.5:
                raise ValueError(f"{corridor_path} has flow on a closed connection")
            if connection_open >= 0.5 and (
                origin not in opened or destination not in opened
            ):
                raise ValueError(f"{corridor_path} connects a closed locker")
            loaded_total += loaded_flow
            realized_total += realized_flow
            direct_total += direct_flow
            swap_total += swap_flow
            loaded_flight_km_total += flight_km

        funnel = _required_mapping(
            summaries[mode],
            "demand_funnel_orders_per_hour",
            f"{mode} summary",
        )
        flight_operations = _required_mapping(
            summaries[mode],
            "flight_operations",
            f"{mode} summary",
        )
        summary_realized = _required_number(
            funnel,
            "total_drone_realized",
            f"{mode} demand funnel",
        )
        summary_loaded = _required_number(
            flight_operations,
            "loaded_flow_per_hour",
            f"{mode} flight operations",
        )
        summary_loaded_km = _required_number(
            flight_operations,
            "loaded_flight_km_per_hour",
            f"{mode} flight operations",
        )
        _require_flow_close(
            loaded_total,
            summary_realized,
            f"{mode} CSV loaded vs realized",
        )
        _require_flow_close(
            realized_total,
            summary_realized,
            f"{mode} CSV chi vs realized",
        )
        _require_close(
            direct_total + swap_total,
            summary_loaded,
            f"{mode} CSV physical vs loaded flow",
        )
        _require_close(
            loaded_flight_km_total,
            summary_loaded_km,
            f"{mode} CSV loaded flight kilometres",
        )

        route_rows = _physical_route_rows(mode_dir)
        physical_totals = {
            "loaded_direct": 0.0,
            "loaded_swap": 0.0,
            "empty_direct": 0.0,
            "empty_swap": 0.0,
            "base_departure": 0.0,
            "base_recovery": 0.0,
        }
        for row in route_rows:
            origin = str(row["origin"])
            destination = str(row["destination"])
            option = str(row["option"])
            flow = float(row["flow_per_hour"])
            origin_type = canonical_types.get(origin)
            destination_type = canonical_types.get(destination)
            if (origin_type, destination_type) not in {
                ("L", "H"),
                ("H", "H"),
                ("H", "L"),
            }:
                raise ValueError(
                    f"{mode_dir / 'incumbent.sol'} has an invalid physical arc "
                    f"{origin!r}->{destination!r}"
                )
            if origin not in opened or destination not in opened:
                raise ValueError(
                    f"{mode_dir / 'incumbent.sol'} has positive flow through a "
                    "closed endpoint"
                )
            route_type = "direct"
            if option != "__direct__":
                if canonical_types.get(option) != "E" or option not in opened:
                    raise ValueError(
                        f"{mode_dir / 'incumbent.sol'} has positive flow through "
                        "a closed or invalid swap station"
                    )
                route_type = "swap"
            load_type = "loaded" if origin_type == destination_type == "H" else "empty"
            physical_totals[f"{load_type}_{route_type}"] += flow
            if origin_type == "L":
                physical_totals["base_departure"] += flow
            if destination_type == "L":
                physical_totals["base_recovery"] += flow

        loaded_direct = physical_totals["loaded_direct"]
        loaded_swap = physical_totals["loaded_swap"]
        empty_direct = physical_totals["empty_direct"]
        empty_swap = physical_totals["empty_swap"]
        checks = {
            "loaded_direct_flow_per_hour": loaded_direct,
            "loaded_swap_flow_per_hour": loaded_swap,
            "loaded_flow_per_hour": loaded_direct + loaded_swap,
            "empty_flow_per_hour": empty_direct + empty_swap,
            "all_direct_flow_per_hour": loaded_direct + empty_direct,
            "all_swap_flow_per_hour": loaded_swap + empty_swap,
            "empty_dispatch_per_hour": physical_totals["base_departure"],
            "empty_recovery_per_hour": physical_totals["base_recovery"],
        }
        for field, reconstructed in checks.items():
            _require_flow_close(
                reconstructed,
                _required_number(
                    flight_operations,
                    field,
                    f"{mode} flight operations",
                ),
                f"{mode} incumbent.sol {field}",
            )
        energy = _required_mapping(
            summaries[mode],
            "energy_balance",
            f"{mode} summary",
        )
        _require_flow_close(
            physical_totals["base_departure"],
            _required_number(
                energy,
                "base_full_battery_departures_per_hour",
                f"{mode} energy balance",
            ),
            f"{mode} incumbent.sol base battery departures",
        )
        _require_flow_close(
            loaded_swap + empty_swap,
            _required_number(
                energy,
                "swap_events_per_hour",
                f"{mode} energy balance",
            ),
            f"{mode} incumbent.sol swap events",
        )


def _csv_bool(value: str | None, context: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{context} has invalid boolean value {value!r}")


def _validate_mode_comparison(
    summaries: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    fields, rows = _read_csv_table(path)
    required = {
        "mode",
        "status_name",
        "objective",
        "runtime_seconds",
        "mip_gap",
        "requested_mip_gap",
        "screening_is_exact",
    }
    if not required.issubset(fields):
        raise ValueError(f"{path} is missing required columns")
    by_mode: dict[str, dict[str, str]] = {}
    for row in rows:
        mode = row.get("mode", "")
        if mode not in MODES or mode in by_mode:
            raise ValueError(f"{path} has an invalid or duplicate mode {mode!r}")
        by_mode[mode] = row
    if set(by_mode) != set(MODES):
        raise ValueError(f"{path} must contain MX, HD, and LO exactly once")
    for mode in MODES:
        row = by_mode[mode]
        summary = summaries[mode]
        if row.get("status_name") != summary.get("status_name"):
            raise ValueError(f"{path} status disagrees with {mode} summary")
        for key in ("objective", "runtime_seconds", "mip_gap", "requested_mip_gap"):
            _require_close(
                _required_number(row, key, str(path)),
                _required_number(summary, key, f"{mode} summary"),
                f"{path} {mode} {key}",
            )
        if _csv_bool(row.get("screening_is_exact"), str(path)) != _required_bool(
            summary,
            "screening_is_exact",
            f"{mode} summary",
        ):
            raise ValueError(f"{path} screening flag disagrees with {mode} summary")


def _certificate_x_limits(
    baselines: list[float],
    objectives: list[float],
    bounds: list[float],
) -> tuple[float, float]:
    values = [*baselines, *objectives, *bounds]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("certificate values must be finite")
    lower = min(values)
    upper = max(values)
    padding = max(20.0, 0.08 * (upper - lower))
    return lower - padding, upper + padding


def _certificate_subtitle(
    summaries: Mapping[str, Mapping[str, Any]],
) -> str:
    metrics = [_solution_metrics(summaries[mode]) for mode in MODES]
    opened_count = sum(bool(item["network_opened"]) for item in metrics)
    service_count = sum(bool(item["network_is_nontrivial"]) for item in metrics)
    if opened_count == 0:
        lead = "No reported incumbent opens drone facilities or provides drone service."
    elif opened_count == len(MODES) and service_count == len(MODES):
        lead = (
            "All three reported incumbents open facilities and provide drone service."
        )
    else:
        lead = (
            f"Facilities open in {opened_count} of {len(MODES)} incumbents; "
            f"positive drone service occurs in {service_count} of {len(MODES)}."
        )
    return (
        f"{lead} The gap panel uses a logarithmic scale; each row reports its "
        "incumbent and current valid bound."
    )


def _save_figure(fig: plt.Figure, stem: Path) -> tuple[Path, Path]:
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, svg


def _write_mode_summary(
    output_dir: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    candidate_lockers: int | None = None,
) -> Path:
    candidate_counts = _candidate_site_counts(summaries)
    candidate_count = candidate_lockers or candidate_counts["lockers"]
    if candidate_count != candidate_counts["lockers"]:
        raise ValueError("explicit candidate-locker count disagrees with summaries")
    fields = (
        "mode",
        "result_schema_version",
        "runtime_contract",
        "run_signature_sha256",
        "candidate_bases",
        "candidate_lockers",
        "candidate_swap_stations",
        "status",
        "objective_cny_per_hour",
        "best_bound_cny_per_hour",
        "remaining_improvement_cny_per_hour",
        "mip_gap_percent",
        "runtime_seconds",
        "network_start_seconds",
        "network_start_found",
        "network_start_key",
        "network_start_candidate_count",
        "network_start_capacity_rejected_count",
        "network_start_attempted_keys",
        "main_solve_seconds",
        "model_variables",
        "binary_variables",
        "chi_variables",
        "linear_constraints",
        "quadratic_constraints",
        "open_bases",
        "open_lockers",
        "active_lockers",
        "idle_open_lockers",
        "open_swap_stations",
        "active_connections",
        "fleet_size_drones",
        "maximum_locker_utilization_percent",
        "energy_use_equivalent_drones",
        "energy_supply_equivalent_drones",
        "base_full_battery_departures_per_hour",
        "swap_events_per_hour",
        "battery_supply_events_per_hour",
        "energy_slack_equivalent_drones",
        "maximum_energy_violation_equivalent_drones",
        "total_demand_orders_per_hour",
        "drone_realized_orders_per_hour",
        "drone_service_fraction_percent",
        "courier_fallback_orders_per_hour",
        "cost_saving_cny_per_hour",
        "cost_saving_percent",
        "positive_drone_service",
        "network_opened",
        "network_is_nontrivial",
        "screening_is_exact",
    )
    rows: list[dict[str, object]] = []
    for mode in MODES:
        summary = summaries[mode]
        _validate_summary_fields(mode, summary)
        solver = _mapping(summary.get("solver"))
        algorithm = _mapping(solver.get("algorithm"))
        model_size = _mapping(summary.get("model_size"))
        instance = _mapping(summary.get("instance"))
        funnel = _mapping(summary.get("demand_funnel_orders_per_hour"))
        metrics = _solution_metrics(summary)
        objective = float(summary["objective"])
        bound = float(summary["objective_bound"])
        rows.append(
            {
                "mode": mode,
                "result_schema_version": summary["schema_version"],
                "runtime_contract": runtime_contract(summary["schema_version"]),
                "run_signature_sha256": summary["run_signature_sha256"],
                "candidate_bases": candidate_counts["bases"],
                "candidate_lockers": candidate_count,
                "candidate_swap_stations": candidate_counts["swap_stations"],
                "status": summary["status_name"],
                "objective_cny_per_hour": objective,
                "best_bound_cny_per_hour": bound,
                "remaining_improvement_cny_per_hour": objective - bound,
                "mip_gap_percent": _gap_percent(summary),
                "runtime_seconds": summary["runtime_seconds"],
                "network_start_seconds": solver["network_start_runtime_seconds"],
                "network_start_found": bool(solver.get("network_start_found", False)),
                "network_start_key": "-".join(
                    str(value) for value in (solver.get("network_start_key") or ())
                ),
                "network_start_candidate_count": int(
                    algorithm.get("network_start_candidate_count", 0)
                ),
                "network_start_capacity_rejected_count": int(
                    algorithm.get("network_start_capacity_rejected_count", 0)
                ),
                "network_start_attempted_keys": ";".join(
                    "-".join(str(value) for value in key)
                    for key in algorithm.get("network_start_attempted_keys", ())
                ),
                "main_solve_seconds": solver["main_solve_runtime_seconds"],
                "model_variables": model_size["variables"],
                "binary_variables": model_size["binary_variables"],
                "chi_variables": instance["chi_keys"],
                "linear_constraints": model_size["linear_constraints"],
                "quadratic_constraints": model_size["quadratic_constraints"],
                "open_bases": metrics["open_bases"],
                "open_lockers": metrics["open_lockers"],
                "active_lockers": metrics["active_lockers"],
                "idle_open_lockers": metrics["idle_open_lockers"],
                "open_swap_stations": metrics["open_swap_stations"],
                "active_connections": metrics["active_connections"],
                "fleet_size_drones": metrics["fleet_size"],
                "maximum_locker_utilization_percent": (
                    100.0 * float(metrics["maximum_locker_utilization"])
                ),
                "energy_use_equivalent_drones": metrics["energy_use"],
                "energy_supply_equivalent_drones": metrics["energy_supply"],
                "base_full_battery_departures_per_hour": metrics[
                    "base_battery_departures"
                ],
                "swap_events_per_hour": metrics["swap_events"],
                "battery_supply_events_per_hour": metrics["battery_supply_events"],
                "energy_slack_equivalent_drones": metrics["energy_slack"],
                "maximum_energy_violation_equivalent_drones": metrics[
                    "energy_violation"
                ],
                "total_demand_orders_per_hour": metrics["total_demand"],
                "drone_realized_orders_per_hour": metrics["drone_realized"],
                "drone_service_fraction_percent": (
                    100.0 * float(metrics["service_fraction"])
                ),
                "courier_fallback_orders_per_hour": funnel["direct_courier_fallback"],
                "cost_saving_cny_per_hour": metrics["cost_saving"],
                "cost_saving_percent": metrics["cost_saving_percent"],
                "positive_drone_service": metrics["network_is_nontrivial"],
                "network_opened": metrics["network_opened"],
                "network_is_nontrivial": metrics["network_is_nontrivial"],
                "screening_is_exact": summary["screening_is_exact"],
            }
        )
    path = output_dir / f"{_artifact_prefix(candidate_count)}_mode_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot_mode_networks(
    layout: StudyLayout,
    run_config: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    results_root: Path,
    output_dir: Path,
    candidate_lockers: int | None = None,
) -> tuple[Path, Path]:
    candidate_counts = _candidate_site_counts(summaries, run_config, layout)
    candidate_count = candidate_lockers or candidate_counts["lockers"]
    if candidate_count != candidate_counts["lockers"]:
        raise ValueError("explicit candidate-locker count disagrees with benchmark")
    figure, axes = plt.subplots(1, 3, figsize=(16.6, 7.55), sharex=True, sharey=True)
    maximum_rate = _activity_scale(layout)
    maximum_flow = _maximum_physical_flow(results_root)
    half_size = layout.half_size_km

    for axis, mode in zip(axes, MODES):
        summary = summaries[mode]
        metrics = _solution_metrics(summary)
        _draw_activity(axis, layout, maximum_rate)
        _draw_physical_routes(
            axis,
            layout,
            _physical_route_rows(results_root / mode),
            maximum_flow,
        )
        opened = _open_facilities(results_root / mode)
        _draw_sites(axis, layout, opened)

        gap = _gap_percent(summary)
        status = str(summary["status_name"])
        axis.set_title(
            f"{mode} · {MODE_LABELS[mode]}\n"
            f"{status} incumbent · {float(summary['runtime_seconds']):.1f}s · "
            f"gap {gap:.3f}%\n"
            f"service {100.0 * float(metrics['service_fraction']):.1f}% · "
            f"save {float(metrics['cost_saving_percent']):.2f}%\n"
            f"open L/H/E {metrics['open_bases']}/{metrics['open_lockers']}/"
            f"{metrics['open_swap_stations']} · H with flow {metrics['active_lockers']}\n"
            f"links {metrics['active_connections']} · "
            f"fleet {float(metrics['fleet_size']):.0f} · "
            f"max util. {100.0 * float(metrics['maximum_locker_utilization']):.1f}%",
            loc="left",
            fontsize=9.8,
            fontweight="semibold",
            color=MODE_COLORS[mode],
            pad=10,
        )
        if not bool(metrics["network_opened"]):
            funnel = _mapping(summary.get("demand_funnel_orders_per_hour"))
            axis.text(
                0.5,
                0.09,
                "No drone facilities opened\n"
                f"{float(funnel['direct_courier_fallback']):,.1f} orders/h by courier · "
                f"saving {float(metrics['cost_saving']):,.1f} CNY/h",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#7C2D12",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#FFF7ED",
                    "edgecolor": "#FDBA74",
                    "linewidth": 0.8,
                    "alpha": 0.94,
                },
                zorder=10,
            )
        elif not bool(metrics["network_is_nontrivial"]):
            axis.text(
                0.5,
                0.09,
                "Facilities opened; no positive drone service\n"
                f"saving {float(metrics['cost_saving']):,.1f} CNY/h",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#7C2D12",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#FFF7ED",
                    "edgecolor": "#FDBA74",
                    "linewidth": 0.8,
                    "alpha": 0.94,
                },
                zorder=10,
            )
        else:
            axis.text(
                0.5,
                0.09,
                "Positive drone service\n"
                f"{float(metrics['drone_realized']):,.1f} orders/h · "
                f"saving {float(metrics['cost_saving']):,.1f} CNY/h",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#14532D",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#F0FDF4",
                    "edgecolor": "#86EFAC",
                    "linewidth": 0.8,
                    "alpha": 0.94,
                },
                zorder=10,
            )
        axis.set_xlim(-half_size, half_size)
        axis.set_ylim(-half_size, half_size)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Local x coordinate (km)")
        axis.grid(color="#CBD5E1", linewidth=0.55, alpha=0.46)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Local y coordinate (km)")

    representative = summaries[MODES[0]]
    instance = _required_mapping(representative, "instance", "MX summary")
    funnel = _required_mapping(
        representative,
        "demand_funnel_orders_per_hour",
        "MX summary",
    )
    markets = _required_integer(instance, "markets", "MX instance", minimum=1)
    total_demand = _required_number(funnel, "total_demand", "MX demand funnel")
    model = _required_mapping(run_config, "model", "run_config")
    speed = _required_number(model, "drone_speed_mps", "run_config model")
    battery_range = _required_number(model, "battery_range_km", "run_config model")
    speed_text = f"{speed:g} m/s"
    range_text = f"{battery_range:g} km/battery"
    side_length = 2.0 * layout.half_size_km
    figure.suptitle(
        f"Nanshan H{candidate_count} candidate set · three service modes",
        x=0.055,
        y=0.988,
        ha="left",
        fontsize=15,
        fontweight="semibold",
        color="#172033",
    )
    figure.text(
        0.055,
        0.943,
        f"Candidate sites L/H/E {candidate_counts['bases']}/{candidate_count}/"
        f"{candidate_counts['swap_stations']} · same {side_length:g}×{side_length:g} km window · "
        f"{markets:,} OD markets · "
        f"{total_demand:,.2f} orders/h · {speed_text} · {range_text} · "
        "filled/enlarged symbols are opened facilities; H with flow is reported per panel",
        ha="left",
        fontsize=9.2,
        color="#475569",
    )
    figure.text(
        0.055,
        0.916,
        _screening_description(run_config, summaries),
        ha="left",
        fontsize=8.7,
        color="#7C2D12",
    )
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#64748B",
            markeredgecolor="none",
            alpha=0.30,
            markersize=6,
            label="OD endpoint activity",
        )
    ]
    for site_type in ("H", "L", "E"):
        marker, color, label = SITE_STYLES[site_type]
        legend_handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                markersize=7,
                label=label,
            )
        )
    legend_handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#334155",
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=9,
            label="Opened facility (filled, enlarged)",
        )
    )
    if maximum_flow > 1e-9:
        legend_handles.append(
            Line2D(
                [],
                [],
                color="#334155",
                linewidth=2.0,
                label="Loaded H–H physical flow",
            )
        )
        legend_handles.append(
            Line2D(
                [],
                [],
                color="#94A3B8",
                linewidth=2.0,
                label="Base dispatch/recovery flow",
            )
        )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.048),
        fontsize=8.7,
    )
    figure.text(
        0.5,
        0.015,
        "Swap-assisted routes bend through the swap-site diamond. Demand bubbles count each order at both OD endpoints.",
        ha="center",
        fontsize=7.8,
        color="#64748B",
    )
    _add_research_blossom(figure)
    figure.tight_layout(rect=(0.025, 0.135, 0.99, 0.86), w_pad=1.1)
    return _save_figure(
        figure,
        output_dir / f"{_artifact_prefix(candidate_count)}_three_mode_networks",
    )


def _plot_solver_certificate(
    summaries: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    candidate_lockers: int | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    effective_run_config = run_config or {}
    candidate_counts = _candidate_site_counts(summaries, run_config)
    candidate_count = candidate_lockers or candidate_counts["lockers"]
    if candidate_count != candidate_counts["lockers"]:
        raise ValueError("explicit candidate-locker count disagrees with benchmark")
    figure, (bound_axis, runtime_axis) = plt.subplots(
        1,
        2,
        figsize=(14.8, 5.8),
        gridspec_kw={"width_ratios": (1.55, 1.0)},
    )
    baselines = [_summary_baseline(summaries[mode]) for mode in MODES]
    baseline = baselines[0]
    positions = list(reversed(range(len(MODES))))
    gaps = [_gap_percent(summaries[mode]) for mode in MODES]
    gap_floor = 0.01

    for position, mode, gap in zip(
        positions,
        MODES,
        gaps,
    ):
        metrics = _solution_metrics(summaries[mode])
        color = MODE_COLORS[mode]
        objective = float(summaries[mode]["objective"])
        bound = float(summaries[mode]["objective_bound"])
        plotted_gap = max(gap, gap_floor)
        bound_axis.hlines(
            position,
            gap_floor,
            plotted_gap,
            color=color,
            linewidth=4.2,
            alpha=0.30,
            zorder=1,
        )
        bound_axis.scatter(
            plotted_gap,
            position,
            s=58,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        bound_axis.text(
            plotted_gap,
            position + 0.16,
            f"gap {gap:,.3f}%",
            ha="left",
            va="bottom",
            fontsize=8,
            color=color,
            fontweight="semibold",
        )
        bound_axis.text(
            0.99,
            position - 0.10,
            f"{summaries[mode]['status_name']} incumbent · "
            f"obj / bound {objective:,.2f} / {bound:,.2f} CNY/h\n"
            f"service {100.0 * float(metrics['service_fraction']):.1f}% · "
            f"save {float(metrics['cost_saving']):,.1f} CNY/h\n"
            f"open L/H/E {metrics['open_bases']}/{metrics['open_lockers']}/"
            f"{metrics['open_swap_stations']} · H with flow {metrics['active_lockers']} · "
            f"links {metrics['active_connections']} · "
            f"fleet {float(metrics['fleet_size']):.0f} · "
            f"max util. {100.0 * float(metrics['maximum_locker_utilization']):.1f}%",
            transform=bound_axis.get_yaxis_transform(),
            ha="right",
            va="top",
            fontsize=7.6,
            color="#334155",
        )
    target_gap = (
        _required_number(
            summaries[MODES[0]],
            "requested_mip_gap",
            f"{MODES[0]} summary",
        )
        * 100.0
    )
    bound_axis.axvline(
        target_gap,
        color="#B45309",
        linestyle="--",
        linewidth=1.1,
    )
    bound_axis.text(
        target_gap * 1.06,
        2.43,
        f"requested {target_gap:g}%",
        ha="left",
        va="top",
        fontsize=8,
        color="#B45309",
    )
    bound_axis.set_yticks(
        positions, [f"{mode} · {MODE_LABELS[mode]}" for mode in MODES]
    )
    bound_axis.set_xscale("log")
    bound_axis.set_xlim(gap_floor, max(10.0, max(gaps) * 2.0))
    bound_axis.set_ylim(-0.72, 2.52)
    bound_axis.set_xlabel("Certified MIP gap (%) · logarithmic scale")
    screening = _screening_description(effective_run_config, summaries)
    screened = any(
        not _required_bool(summaries[mode], "screening_is_exact", f"{mode} summary")
        for mode in MODES
    )
    certificate_scope = "Screened-model" if screened else "Full-model"
    bound_axis.set_title(
        f"{certificate_scope} optimality certificate",
        loc="left",
        fontsize=11,
    )
    bound_axis.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    bound_axis.set_axisbelow(True)
    bound_axis.spines[["top", "right", "left"]].set_visible(False)

    source_versions = {require_result_contract(summaries[mode], f"{mode} summary") for mode in MODES}
    if len(source_versions) != 1:
        raise ValueError("runtime comparison requires one source schema")
    source_version = next(iter(source_versions))
    main_times = [
        float(_mapping(summaries[mode]["solver"])["main_solve_runtime_seconds"])
        for mode in MODES
    ]
    starts = [
        float(_mapping(summaries[mode]["solver"])["network_start_runtime_seconds"])
        if source_version == 5
        else max(0.0, float(summaries[mode]["runtime_seconds"]) - main_time)
        for mode, main_time in zip(MODES, main_times)
    ]
    x_positions = list(range(len(MODES)))
    runtime_axis.bar(
        x_positions,
        starts,
        width=0.62,
        color="#BFDBFE",
        edgecolor="#1D4ED8",
        linewidth=0.8,
        label="Network-start phase" if source_version == 5 else "Preparation and other wall time",
    )
    runtime_axis.bar(
        x_positions,
        main_times,
        width=0.62,
        bottom=starts,
        color="#2563EB",
        edgecolor="#1D4ED8",
        linewidth=0.8,
        label="Main solve",
    )
    for index, (start, main_time) in enumerate(zip(starts, main_times)):
        runtime_axis.text(
            index,
            start + main_time + 1.0,
            f"{start + main_time:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#334155",
        )
    runtime_axis.set_xticks(x_positions, MODES)
    runtime_axis.set_ylabel("Network + main solver time (seconds)" if source_version == 5 else "Algorithm wall-clock time (seconds)")
    runtime_axis.set_title("Runtime composition", loc="left", fontsize=11)
    runtime_axis.set_ylim(0.0, 1.20 * max(s + m for s, m in zip(starts, main_times)))
    runtime_axis.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    runtime_axis.set_axisbelow(True)
    runtime_axis.spines[["top", "right"]].set_visible(False)
    runtime_axis.legend(loc="upper right", frameon=False, fontsize=8.3)

    figure.suptitle(
        f"Nanshan H{candidate_count} candidate set · solver status and runtime",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="semibold",
        color="#172033",
    )
    figure.text(
        0.055,
        0.925,
        f"Candidate sites L/H/E {candidate_counts['bases']}/{candidate_count}/"
        f"{candidate_counts['swap_stations']} · all-courier benchmark "
        f"{baseline:,.2f} CNY/h. {_certificate_subtitle(summaries)}",
        ha="left",
        fontsize=9,
        color="#475569",
    )
    figure.text(
        0.5,
        0.018,
        f"{screening} Runtime includes network-start "
        "and main-solve phases, but excludes data preparation, model construction, "
        "and plotting.",
        ha="center",
        fontsize=8,
        color="#64748B",
    )
    _add_research_blossom(figure)
    figure.tight_layout(rect=(0.035, 0.07, 0.99, 0.88), w_pad=3.0)
    return _save_figure(
        figure,
        output_dir / f"{_artifact_prefix(candidate_count)}_solver_certificate",
    )


def write_nanshan_benchmark_report(
    instance_dir: str | Path,
    results_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    instance = Path(instance_dir).expanduser().resolve()
    results = Path(results_root).expanduser().resolve()
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else results / "figures"
    )
    layout, run_config, summaries = _load_benchmark(instance, results)
    output.mkdir(parents=True, exist_ok=True)
    candidate_lockers = _candidate_locker_count(summaries, run_config, layout)
    networks_png, networks_svg = _plot_mode_networks(
        layout,
        run_config,
        summaries,
        results,
        output,
        candidate_lockers,
    )
    certificate_png, certificate_svg = _plot_solver_certificate(
        summaries,
        output,
        candidate_lockers,
        run_config,
    )
    summary_csv = _write_mode_summary(output, summaries, candidate_lockers)
    return {
        "networks_png": networks_png,
        "networks_svg": networks_svg,
        "certificate_png": certificate_png,
        "certificate_svg": certificate_svg,
        "summary_csv": summary_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create current-contract Nanshan benchmark comparison figures."
    )
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    artifacts = write_nanshan_benchmark_report(
        args.instance_dir,
        args.results_root,
        args.output_dir,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
