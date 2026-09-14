"""Operational metrics recomputed from solved global-column flow arrays.

``theta[p]`` refers to ``inst.service[p]`` and ``phi[r]`` to
``inst.empty[r]``, regardless of the restricted master column pool.
An empty flow list is the explicitly supported all-ground incumbent.
"""

from __future__ import annotations

import math
from typing import Any

from .data import Instance


def _dense_flows(values, expected_length: int, name: str, tolerance: float):
    if values is None or len(values) == 0:
        return [0.0] * expected_length
    if len(values) != expected_length:
        raise ValueError(f"{name} must be indexed by all {expected_length} global columns; "
                         f"received {len(values)} values. Restricted-pool positions are not column ids.")
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} contains non-finite flow values")
    if min(values, default=0.0) < -tolerance:
        raise ValueError(f"{name} contains flow below {-tolerance:g}")
    return [max(0.0, value) for value in values]


def summarize_solution(inst: Instance, result: dict[str, Any],
                       tolerance: float = 1e-6) -> dict[str, Any]:
    """Return facility, service-mode, swapping, fleet and balance statistics.

    All flow and swap rates are per hour.  ``intermediate_swap_rate`` excludes
    the one compulsory terminal swap on every nonzero service/empty mission;
    ``remote_swap_rate`` includes all actual visits to remote facilities.
    Home-delivery and pickup rates both belong to served drone orders.  Pure
    ground fallback is reported separately.
    """
    if result.get('objective') is None:
        return {'has_incumbent': False, 'status': result.get('status')}
    theta = _dense_flows(result.get('theta'), len(inst.service), 'theta', tolerance)
    phi = _dense_flows(result.get('phi'), len(inst.empty), 'phi', tolerance)
    design = result.get('design') or inst.zero_design()
    rho = _dense_flows(result.get('rho'), inst.n_l, 'rho', tolerance)
    open_bases = [l for l in range(inst.n_l) if design.get(f'X{l}', 0.0) > .5]
    open_hubs = [i for i in range(inst.n_h) if design.get(f'Y{i}', 0.0) > .5]
    open_remote = [k for k in range(inst.n_l, inst.n_l + inst.n_k)
                   if design.get(f'S{k}', 0.0) > .5]
    service_out = [0.0] * inst.n_l
    service_in = [0.0] * inst.n_l
    empty_out = [0.0] * inst.n_l
    empty_in = [0.0] * inst.n_l
    served_od = [0.0] * inst.n_od
    swap_rates = [0.0] * (inst.n_l + inst.n_k)
    pickup = home = cross_base = service_time = empty_time = 0.0
    service_cost = empty_cost = 0.0
    using_remote = using_intermediate = 0.0
    active_service = active_empty = 0
    for p, rate in zip(inst.service, theta):
        if rate > tolerance:
            active_service += 1
        served_od[p['od']] += rate
        service_out[p['l']] += rate
        service_in[p['m']] += rate
        service_time += rate * p['time']
        service_cost += rate * p['cost']
        pickup += rate * p['pickup_share']
        home += rate * (1.0 - p['pickup_share'])
        cross_base += rate * (p['l'] != p['m'])
        using_remote += rate * any(p['swaps'][inst.n_l:])
        using_intermediate += rate * (sum(p['swaps']) > 1)
        for k, visits in enumerate(p['swaps']):
            swap_rates[k] += rate * visits
    for r, rate in zip(inst.empty, phi):
        if rate > tolerance:
            active_empty += 1
        empty_out[r['l']] += rate
        empty_in[r['m']] += rate
        empty_time += rate * r['time']
        empty_cost += rate * r['cost']
        for k, visits in enumerate(r['swaps']):
            swap_rates[k] += rate * visits
    served = sum(theta)
    empty_rate = sum(phi)
    demand = sum(inst.demand)
    fleet_reported = result.get('fleet')
    if result.get('idle') is not None:
        idle_fleet = sum(_dense_flows(result['idle'], inst.n_l, 'idle', tolerance))
        idle_source = 'reported idle variables'
    elif fleet_reported is not None:
        idle_fleet = float(fleet_reported) - service_time - empty_time
        idle_source = 'reported total fleet minus route workload'
    else:
        idle_fleet = sum(value / (1.0 - value) for value in rho)
        idle_source = 'rho/(1-rho) reconstructed at minimum fleet cost'
    fleet = idle_fleet + service_time + empty_time
    ground_cost = sum((h - z) * c for h, z, c in zip(inst.demand, served_od, inst.ground_cost))
    fixed_cost = inst.fixed_cost(design)
    recomputed_objective = fixed_cost + ground_cost + service_cost + empty_cost + inst.fleet_cost * fleet
    balance = [service_out[l] + empty_out[l] - service_in[l] - empty_in[l]
               for l in range(inst.n_l)]
    return {
        'has_incumbent': True, 'status': result.get('status'),
        'flow_indexing': 'theta[p] and phi[r] use full global-column ids; [] means zero flows',
        'open_base_count': len(open_bases), 'open_hub_count': len(open_hubs),
        'open_remote_swap_count': len(open_remote),
        'open_base_ids': open_bases, 'open_hub_ids': open_hubs,
        'open_remote_swap_global_ids': open_remote,
        'assigned_od_count': sum(any(design.get(f'B{od}:{l}:{m}', 0.0) > .5
                                     for l in range(inst.n_l) for m in range(inst.n_l))
                                 for od in range(inst.n_od)),
        'active_service_columns': active_service, 'active_empty_columns': active_empty,
        'potential_order_rate': demand, 'served_order_rate': served,
        'served_fraction': served / demand if demand else 0.0,
        'pure_ground_order_rate': demand - served,
        'pickup_order_rate': pickup, 'home_delivery_order_rate': home,
        'pickup_fraction_of_served': pickup / served if served else 0.0,
        'rho_by_base': rho, 'fleet': fleet, 'idle_fleet': idle_fleet,
        'service_fleet_workload': service_time, 'empty_fleet_workload': empty_time,
        'idle_fleet_source': idle_source,
        'mean_served_task_time_hours': service_time / served if served else 0.0,
        'total_swap_rate': sum(swap_rates),
        'remote_swap_rate': sum(swap_rates[inst.n_l:]),
        'intermediate_swap_rate': sum(swap_rates) - served - empty_rate,
        'terminal_swap_rate': served + empty_rate,
        'swap_rate_by_global_station': swap_rates,
        'service_order_rate_using_remote_swap': using_remote,
        'service_order_rate_using_intermediate_swap': using_intermediate,
        'cross_base_service_rate': cross_base,
        'cross_base_fraction_of_served': cross_base / served if served else 0.0,
        'empty_reposition_rate': empty_rate,
        'service_out_by_base': service_out, 'service_in_by_base': service_in,
        'empty_out_by_base': empty_out, 'empty_in_by_base': empty_in,
        'served_rate_by_od': served_od,
        'maximum_base_flow_imbalance': max(map(abs, balance), default=0.0),
        'mode_split_identity_error': abs(pickup + home - served),
        'reported_fleet_error': abs(fleet - float(fleet_reported)) if fleet_reported is not None else None,
        'fixed_facility_cost': fixed_cost, 'ground_fallback_cost': ground_cost,
        'service_operating_cost': service_cost, 'empty_operating_cost': empty_cost,
        'fleet_cost': inst.fleet_cost * fleet,
        'recomputed_objective': recomputed_objective,
        'objective_recomputation_error': abs(recomputed_objective - result['objective']),
    }


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    from .instances import make_instance
    parser = argparse.ArgumentParser()
    parser.add_argument('result_json', type=Path)
    args = parser.parse_args()
    saved = json.loads(args.result_json.read_text())
    instance = make_instance(saved['n_l'], saved['n_h'], saved['n_od'], saved['n_k'],
                             seed=saved['seed'], name=saved['instance'])
    print(json.dumps(summarize_solution(instance, saved), indent=2))
