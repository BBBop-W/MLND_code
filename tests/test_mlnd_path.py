"""Independent model checks for the new path formulation.

The one-base reference is solved analytically, rather than by a second copy of
the matrix builder.  This checks the idle cone, closed-facility gates, all-column
pricing certificates, and the integer objective against the same known answer.
"""
from __future__ import annotations

import itertools
import math
import copy

import pytest

pytest.importorskip("gurobipy")

from mlnd_path.data import Instance
from mlnd_path.solver import solve_cg, solve_full, solve_recourse


@pytest.fixture(scope="module")
def analytic_instance():
    inst = Instance(
        name="analytic_one_base", n_l=1, n_h=2, n_k=1,
        demand=[4.0], ground_cost=[10.0], base_cost=[1.0],
        hub_cost=[0.2, 0.2], swap_cost=[0.6], rho_max=0.8,
        kappa=8.0, fleet_cost=2.0,
    )
    inst.service = [
        dict(od=0, l=0, m=0, i=0, j=1, time=0.25, cost=2.0,
             swaps=(1, 0)),
        dict(od=0, l=0, m=0, i=0, j=1, time=0.35, cost=0.5,
             swaps=(1, 1)),
    ]
    return inst


def _designs(inst):
    """All 24 structurally feasible binary designs of the reference case."""
    result = []
    for x, y0, y1, s, b in itertools.product((0, 1), repeat=5):
        if b > x:
            continue
        result.append(dict(X0=x, Y0=y0, Y1=y1, S1=s, **{"B0:0:0": b}))
    return result


def _reference(inst, design):
    if not all(design[k] for k in ("X0", "Y0", "Y1", "B0:0:0")):
        return inst.c0, 0.0
    col = 1 if design["S1"] else 0
    slope = inst.demand[0] * inst.service_cost(col)
    rho = min(inst.rho_max, max(0.0, 1.0-math.sqrt(inst.fleet_cost/(-slope))))
    return inst.c0+slope*rho+inst.fleet_cost*rho/(1-rho), rho


def test_single_base_conic_recourse_matches_analytic_optimum(analytic_instance):
    inst = analytic_instance
    for design in _designs(inst):
        expected, rho = _reference(inst, design)
        result = solve_recourse(inst, design, time_limit=10.0)
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(expected, abs=5e-6)
        assert result["rho"][0] == pytest.approx(rho, abs=3e-4)
        assert result["primal_residual"] < 2e-6


def test_empty_pools_are_feasible_with_all_facilities_closed(analytic_instance):
    inst = analytic_instance
    result = solve_recourse(inst, inst.zero_design(), service_pool=[], empty_pool=[],
                            time_limit=10.0)
    assert result["status"] == "optimal"
    assert result["objective"] == pytest.approx(inst.c0, abs=5e-6)
    assert result["served"] == pytest.approx(0.0, abs=1e-8)


def test_full_candidate_cg_and_cross_design_cuts(analytic_instance):
    inst = analytic_instance
    designs = _designs(inst)
    # In particular, cuts obtained with a closed remote swap station must also
    # hold after that station opens and makes a cheaper column available.
    origins = [inst.zero_design(),
               dict(X0=1, Y0=1, Y1=1, S1=0, **{"B0:0:0": 1}),
               dict(X0=1, Y0=1, Y1=1, S1=1, **{"B0:0:0": 1}),
               dict(X0=1, Y0=0, Y1=1, S1=1, **{"B0:0:0": 1})]
    for design in origins:
        result = solve_cg(inst, design, time_limit=20.0)
        expected, _ = _reference(inst, design)
        assert result["cut_valid"], result
        assert result["cg_complete"], result
        assert result["objective"] == pytest.approx(expected, abs=5e-6)
        assert len(result["rc_service"]) == len(inst.service)
        assert result["minimum_rc"] >= -1e-6
        for target in designs:
            lower = result["cut_intercept"]+sum(
                result["cut_slope"][key]*value for key, value in target.items())
            assert lower <= _reference(inst, target)[0]+1e-7


def test_full_misocp_matches_exhaustive_analytic_designs(analytic_instance):
    inst = analytic_instance
    reference = min(inst.fixed_cost(d)+_reference(inst, d)[0] for d in _designs(inst))
    result = solve_full(inst, time_limit=20.0, mip_gap=1e-6)
    assert result["status"] == "optimal"
    assert result["objective"] == pytest.approx(reference, abs=1e-5)
    assert result["bound"] <= reference+1e-6
    assert result["gap"] < 1e-6


def test_physical_route_energy_and_terminal_swaps():
    from mlnd_path.instances import make_instance

    inst = make_instance(2, 3, 2, n_k=1, seed=17)
    geo = inst.metadata
    battery, reserve = geo["battery_capacity"], geo["battery_reserve"]
    stations = geo["coordinates"]["bases"]+geo["coordinates"]["remote_swaps"]
    hubs = geo["coordinates"]["hubs"]

    def location(node):
        return hubs[node[1]] if node[0] in ("P", "D") else stations[node[1]]

    assert inst.service, "The physical fixture should contain service routes"
    for col in inst.service+inst.empty:
        assert col["route"][-1] == ("T", col["m"])
        assert col["swaps"][col["m"]] >= 1
        assert len(col["energy_trace"]) == len(col["route"])-1
        counts = [0]*(inst.n_l+inst.n_k)
        previous, energy, loaded = col["route"][0], battery, False
        for node, (before, after) in zip(col["route"][1:], col["energy_trace"]):
            assert before >= reserve-1e-8
            factor = geo["loaded_energy_factor"] if loaded else 1.0
            assert before == pytest.approx(energy-factor*math.dist(location(previous), location(node)))
            if node[0] in ("S", "T"):
                counts[node[1]] += 1
                assert after == pytest.approx(battery)
            else:
                # Visiting either locker never restores the battery.
                assert after == pytest.approx(before)
            if node[0] == "P":
                loaded = True
            elif node[0] == "D":
                loaded = False
            previous, energy = node, after
        assert tuple(counts) == tuple(col["swaps"])
    assert all(col["i"] != col["j"] for col in inst.service)


def test_initial_idle_allocated_bound_is_globally_valid(analytic_instance):
    from mlnd_path.benders import initial_bound_coefficients

    inst = analytic_instance
    coefficients = initial_bound_coefficients(inst)
    for design in _designs(inst):
        lower = inst.c0+sum(value*design[key] for key, value in coefficients.items())
        assert lower <= _reference(inst, design)[0]+1e-9


@pytest.mark.parametrize("mode", ["loop", "lazy"])
def test_benders_bounds_and_objective_match_analytic_designs(analytic_instance, mode):
    from mlnd_path.benders import solve_benders

    inst = analytic_instance
    reference = min(inst.fixed_cost(d)+_reference(inst, d)[0] for d in _designs(inst))
    result = solve_benders(inst, time_limit=20.0, mip_gap=1e-5, mode=mode)
    assert result["status"] == "OPTIMAL_TOL", result
    assert result["objective"] == pytest.approx(reference, abs=1e-5)
    assert result["bound"] <= reference+1e-6
    assert result["gap"] <= 1e-5
    assert not result["errors"]


def test_physical_instance_full_cg_and_benders_agree():
    from mlnd_path.benders import initial_bound_coefficients, solve_benders
    from mlnd_path.instances import make_instance

    inst = make_instance(2, 3, 2, n_k=1, seed=17)
    full = solve_full(inst, time_limit=30.0)
    assert full["status"] == "optimal", full
    fixed = {key: float(round(value)) for key, value in full["design"].items()}
    cg = solve_cg(inst, fixed, time_limit=20.0)
    assert cg.get("cg_complete"), cg
    assert inst.fixed_cost(fixed)+cg["objective"] == pytest.approx(full["objective"], abs=1e-4)
    coefficients = initial_bound_coefficients(inst)
    initial = inst.c0+sum(value*fixed[key] for key, value in coefficients.items())
    assert initial <= cg["objective"]+1e-7
    for mode in ("loop", "lazy"):
        result = solve_benders(inst, time_limit=30.0, mip_gap=1e-4, mode=mode)
        assert result["bound"] <= full["objective"]+1e-5
        assert result["objective"] >= full["bound"]-1e-5
        assert result["objective"] == pytest.approx(full["objective"], abs=1e-4)
        assert result["gap"] <= 1e-4, result


def test_deadline_preserves_incumbent_without_false_completion(analytic_instance):
    from mlnd_path.benders import solve_benders

    inst = analytic_instance
    reference = min(inst.fixed_cost(d)+_reference(inst, d)[0] for d in _designs(inst))
    result = solve_benders(inst, time_limit=0.001, mode="loop")
    assert result["objective"] >= reference-1e-7
    assert result["bound"] <= reference+1e-7
    assert result["objective"] <= inst.c0+1e-7
    assert result["status"] != "OPTIMAL_TOL" or result["gap"] <= 1e-4


def test_pareto_cut_improves_core_and_remains_valid_across_all_designs(analytic_instance):
    from mlnd_path.pareto import strengthen_cut

    inst = analytic_instance
    origin = inst.zero_design()
    core = dict.fromkeys(inst.design_keys(), 0.5)
    core["B0:0:0"] = 0.25  # Strictly below the fractional base opening.
    baseline = solve_cg(inst, origin, time_limit=10.0)
    result = strengthen_cut(inst, origin, baseline, core,
                            pools={"service": [], "empty": []}, time_limit=10.0)
    assert result["use"], result
    assert result["core_improvement"] > 1e-5
    assert result["cg_complete"] and result["cut_valid"]
    assert abs(result["tight_gap"]) < 1e-5
    for target in _designs(inst):
        value = result["cut_intercept"]+sum(c*target[k] for k, c in result["cut_slope"].items())
        assert value <= _reference(inst, target)[0]+1e-7


def test_disaggregated_gates_preserve_integer_values_and_rebuild_safely():
    from mlnd_path.instances import make_instance
    from mlnd_path.solver import _matrix
    from mlnd_path.strengthening import add_disaggregated_gates

    original = make_instance(2, 3, 2, n_k=1, seed=17)
    strengthened = copy.deepcopy(original)
    original_rows = len(_matrix(original).names)
    add_disaggregated_gates(strengthened)
    total_rows = len(_matrix(strengthened).names)
    assert total_rows > original_rows
    add_disaggregated_gates(strengthened)
    assert len(_matrix(strengthened).names) == total_rows
    # Benchmark fairness resets the matrix cache without deleting metadata.
    del strengthened._solver_matrix
    add_disaggregated_gates(strengthened)
    assert len(_matrix(strengthened).names) == total_rows
    designs = [original.zero_design()]
    for remote, cross in itertools.product((0, 1), repeat=2):
        point = original.zero_design()
        for key in point:
            if key.startswith(("X", "Y")):
                point[key] = 1.0
        point["S2"] = float(remote)
        for od in range(original.n_od):
            origin = od % 2
            end = 1-origin if cross else origin
            point[f"B{od}:{origin}:{end}"] = 1.0
        designs.append(point)
    for point in designs:
        baseline = solve_recourse(original, point, time_limit=10.0)
        strong = solve_recourse(strengthened, point, time_limit=10.0)
        assert strong["objective"] == pytest.approx(baseline["objective"], abs=2e-5)
    baseline = solve_full(original, time_limit=20.0, mip_gap=1e-6)
    strong = solve_full(strengthened, time_limit=20.0, mip_gap=1e-6)
    assert baseline["status"] == strong["status"] == "optimal"
    assert strong["objective"] == pytest.approx(baseline["objective"], abs=2e-5)


def test_cross_base_order_requires_positive_empty_return_flow():
    from scipy.optimize import minimize_scalar
    from mlnd_path.strengthening import add_disaggregated_gates

    inst = Instance("cross_base_return", 2, 2, 0, [4.0], [10.0],
                    [1.0, 1.0], [0.2, 0.2], [], rho_max=0.8, kappa=8.0, fleet_cost=2.0)
    inst.service = [dict(od=0, l=0, m=1, i=0, j=1, time=0.25, cost=2.0, swaps=(0, 1))]
    inst.empty = [dict(l=1, m=0, time=0.1, cost=0.1, swaps=(1, 0))]
    design = inst.zero_design()
    design.update(X0=1.0, X1=1.0, Y0=1.0, Y1=1.0)
    design["B0:0:1"] = 1.0
    # Balance requires phi=theta=4*rho0. The base-1 clock then gives
    # rho1>=phi/8=rho0/2; positive idle cost makes this inequality tight.
    reference = minimize_scalar(
        lambda r: 40.0-28.8*r+2.0*r/(1.0-r)+r/(1.0-r/2.0),
        bounds=(0.0, 0.8), method="bounded", options={"xatol": 1e-12})
    strengthened = add_disaggregated_gates(copy.deepcopy(inst))
    for candidate in (inst, strengthened):
        result = solve_cg(candidate, design, time_limit=20.0)
        assert result["cg_complete"], result
        assert result["objective"] == pytest.approx(reference.fun, abs=2e-5)
        assert result["phi"][0] > 1.0
        assert result["phi"][0] == pytest.approx(result["theta"][0], abs=1e-6)
        assert result["rho"][1] == pytest.approx(result["rho"][0]/2.0, abs=2e-5)
