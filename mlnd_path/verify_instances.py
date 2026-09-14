"""Independent exhaustive checks of the small-instance route generator."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

from .instances import Geometry, _label_route, make_instance


def _stage_sequences(stations):
    for length in range(len(stations) + 1):
        yield from itertools.permutations(stations, length)


def _exhaustive_best(geo, l, m, i, j, allowed, mean, ground):
    """Enumerate every stage-simple route and carry energy across both H.

    A revisit of the initial full base before pickup is a removable cycle;
    arrival at the terminal base in the final stage ends the mission.
    """
    starts = (("S", l), ("H", i), ("H", j))
    ends = (("H", i), ("H", j), ("S", m))
    options = (tuple(_stage_sequences(tuple(s for s in allowed if s != l))),
               tuple(_stage_sequences(allowed)),
               tuple(_stage_sequences(tuple(s for s in allowed if s != m))))
    best = math.inf
    evaluated = 0
    feasible = 0
    for stage_routes in itertools.product(*options):
        energy = geo.battery
        elapsed = []
        cost = 0.0
        valid = True
        for stage, swaps in enumerate(stage_routes):
            now = starts[stage]
            stage_time = 0.0
            for next_node in [*(('S', s) for s in swaps), ends[stage]]:
                distance = math.dist(geo.location(now), geo.location(next_node))
                energy -= distance * (geo.loaded_energy if stage == 1 else 1.0)
                if energy < geo.reserve - 1e-9:
                    valid = False
                    break
                stage_time += distance / geo.speed
                cost += distance * geo.flight_cost * (1.06 if stage == 1 else 1.0)
                if next_node[0] == 'S':
                    energy = geo.battery
                    stage_time += geo.swap_time
                    cost += geo.swap_fee
                now = next_node
            elapsed.append(stage_time)
            if not valid:
                break
        evaluated += 1
        if not valid:
            continue
        feasible += 1
        first = ground + mean if elapsed[0] <= ground else elapsed[0] + mean * math.exp(-(elapsed[0] - ground) / mean)
        duration = first + elapsed[1] + elapsed[2] + geo.handling_pickup + geo.handling_delivery
        best = min(best, cost + geo.fleet_cost * duration)
    return best, evaluated, feasible


def verify(seed=17):
    inst = make_instance(2, 3, 2, 1, seed)
    points = inst.metadata['coordinates']
    geo = Geometry(tuple(points['bases']), tuple(points['hubs']), tuple(points['remote_swaps']))
    errors = []
    checked = evaluated = feasible = 0
    # Exhaustive comparison over every candidate pair and every legal facility
    # design, including no-remote designs and same-base round trips.
    for od in range(inst.n_od):
        merchant = inst.metadata['od_merchant'][od]
        mean = inst.metadata['merchant_ready_means'][merchant]
        for mask in range(1 << (inst.n_l + inst.n_k)):
            allowed = tuple(s for s in range(inst.n_l + inst.n_k) if mask & (1 << s))
            for l in range(inst.n_l):
                for m in range(inst.n_l):
                    if l not in allowed or m not in allowed:
                        continue
                    for i in range(inst.n_h):
                        ground = 1.2 * math.dist(points['merchants'][merchant], points['hubs'][i]) / 18.0
                        for j in range(inst.n_h):
                            if i == j:
                                continue
                            exact, count, n_feasible = _exhaustive_best(geo, l, m, i, j, allowed, mean, ground)
                            candidate, _ = _label_route(geo, l, m, allowed, i, j, mean, ground)
                            value = math.inf if candidate is None else candidate['search_objective']
                            union = min((p['search_objective'] for p in inst.service
                                         if (p['od'], p['l'], p['m'], p['i'], p['j']) == (od, l, m, i, j)
                                         and all(k in allowed or count == 0 for k, count in enumerate(p['swaps']))),
                                        default=math.inf)
                            if math.isfinite(exact):
                                errors.append(abs(exact - value))
                                errors.append(abs(exact - union))
                            else:
                                assert math.isinf(value) and math.isinf(union)
                            checked += 1
                            evaluated += count
                            feasible += n_feasible
    for column in inst.service + inst.empty:
        assert column['route'][-1] == ('T', column['m'])
        assert column['swaps'][column['m']] >= 1
        assert len(column['energy_trace']) == len(column['route']) - 1
        for action, (arrival_energy, next_energy) in zip(column['route'][1:], column['energy_trace']):
            assert arrival_energy >= geo.reserve - 1e-9
            if action[0] in ('P', 'D'):
                assert abs(arrival_energy - next_energy) < 1e-9
            else:
                assert abs(next_energy - geo.battery) < 1e-9
        assert all(count <= 4 for count in column['swaps'])
    result = dict(seed=seed, fixed_od_base_hub_and_station_designs_checked=checked,
                  fully_enumerated_complete_routes=evaluated, feasible_complete_routes=feasible,
                  label_and_subset_union_max_objective_error=max(errors, default=0.0),
                  stored_columns_checked=len(inst.service) + len(inst.empty),
                  exact_per_integer_facility_design=True,
                  no_automatic_H_energy_reset=True, terminal_swap_included=True,
                  arbitrary_hop_cap=False, preserves_full_fractional_relaxation=False)
    assert result['label_and_subset_union_max_objective_error'] < 1e-8
    return result


if __name__ == '__main__':
    result = {'checks': [verify(17), verify(31)]}
    path = Path(__file__).resolve().parent / 'instance_verification.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
