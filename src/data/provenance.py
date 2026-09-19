"""Auditable parameter and input descriptions used by result exports."""

from __future__ import annotations
from src.config import DEFAULT_CONFIG
from .instance_io import sha256


def attach_provenance(data, config, calibration):
    p = data["provenance"]
    costs, operations = config["costs"], config["operations"]
    fit, mean, case = (calibration[k] for k in ("manifest", "mean_fit", "case"))
    dep = config["depreciation"]
    p.update(
        config=str(DEFAULT_CONFIG),
        resolved_config=config,
        config_sha256=sha256(DEFAULT_CONFIG),
        model_version="path_cqn_v1",
        demand_rescaled=config["data"]["demand_scale"] != 1,
        demand_scale=config["data"]["demand_scale"],
        rider_speed_kmh_scenario=data["rider_speed"],
        pickup_timing=calibration["pre_dispatch"],
        dispatch_clock="fetch",
        dispatch_rule="Rider fetch triggers a drone request. An available drone leaves its base immediately.",
        pickup_expectation="Analytic E[max(a,T)] for first-mile rider travel; no scenario variables",
        mission_energy="Flight, handling and quantile pickup hover consume battery; at most one swap per stage; no transit bases",
        terminal_base_replenishment="Instantaneous boundary operation with zero extra mission time",
        customer_pre_dispatch="Observed payment-to-fetch mean, excluded from drone workload and post-fetch delivery time",
        fixed_corridor_cost="Charged once per opened directed link, independently of flown volume",
        locker_capacity="Installed cabinet count times handling-event throughput per cabinet",
        pickup_discount="Self-pickup has zero terminal cost; utility discount has no separate subsidy charge",
    )
    p["rider_travel_time"] = dict(
        family=data["rider_time_family"],
        mean="Distance slope times straight-line kilometers",
        slope_minutes_per_straight_km=data["rider_slope_minutes_per_straight_km"],
        independent_per_mission=True,
        calibration=str(calibration["mean_path"]),
        calibration_sha256=sha256(calibration["mean_path"]),
        mean_source=mean["mean_source"],
        transfer_assumption=mean["transfer_assumption"],
        family_status=mean["family_status"],
        shape_parameter_estimated=False,
        first_leg_observed=False,
    )
    p["airlines"] = dict(
        scope="Directed physical flight links and zero-fee co-located connectors",
        hh_cost_per_km_hour=data["airline_hh_cost_per_km_hour"],
        connection_cost_per_km_hour=data["airline_connection_cost_per_km_hour"],
        multiplier=data["airline_cost_multiplier"],
        source="Resolved production configuration",
        cost_period="hour",
        upper_bounds="Demand and acyclic empty-flow activation bounds",
    )
    p["cabinet_capacity"] = dict(
        max_units_per_site=data["max_cabinets"],
        handling_events_per_unit_hour=data["unit_capacity"],
        cost_per_unit_hour=data["hub_cost"],
        cost_source="Resolved production configuration; setup fee charged once per location",
        visit_accounting="Each served order consumes one pickup and one delivery event",
        delay="Mission durations include handling; capacity adds no separate locker queue",
        choice="One binary level for the installed cabinet count",
    )
    p["scenario_settings"] = dict(
        name=config["name"],
        source="Resolved production configuration",
        **{
            k: data[k]
            for k in (
                "base_cost",
                "hub_cost",
                "swap_cost",
                "fleet_cost",
                "base_site_cost",
                "hub_site_cost",
                "swap_site_cost",
                "airline_hh_cost_per_km_hour",
                "airline_connection_cost_per_km_hour",
            )
        },
        choice=config["choice"],
        cost_period="operating hour",
        amortization_years=dep["years"],
        operating_days_per_year=dep["days_per_year"],
        operating_hours_per_day=dep["hours_per_day"],
        rider_initial_charge_cny=costs["rider_initial_charge"],
        rider_included_distance_km=costs["rider_included_distance_km"],
        rider_incremental_cost_per_km=costs["rider_additional_per_km"],
        road_distance_factor=operations["road_distance_factor"],
        feeder_rider_cost_factor=costs["feeder_rider_multiplier"],
        first_leg_cost="Feeder multiplier times rider tariff",
        last_leg_cost="Home-delivery share times feeder multiplier times rider tariff",
    )
    p["calibration"] = dict(
        source=str(calibration["path"]),
        sha256=sha256(calibration["path"]),
        deadline_table=str(calibration["deadline_path"]),
        deadline_table_sha256=sha256(calibration["deadline_path"]),
        pre_dispatch_statistics=str(calibration["stats_path"]),
        pre_dispatch_sha256=sha256(calibration["stats_path"]),
        dispatch_mean_minutes=data["terminal_dispatch_time"] * 60,
        dispatch_valid_rows=fit["dispatch"]["valid_rows"],
        terminal_time_definition=fit["terminal_time"],
        terminal_time_choice=fit["terminal_time_choice"],
        deadline_prior_weight=case["prior_weight"],
        preparation_time_included=False,
        pickup_handling_minutes=operations["pickup_minutes"],
        delivery_handling_minutes=operations["unloading_minutes"],
        handling_source="Configured action constants",
        hover_cost_source="Resolved configuration or flight cost",
        epsilon=data["epsilon"],
    )
