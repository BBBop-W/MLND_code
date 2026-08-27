from __future__ import annotations

from types import SimpleNamespace

import gurobipy as gp
import pytest
from gurobipy import GRB

from src.config import ChoiceConfig, ModelConfig, SolverConfig
from src.data_processor.instance import Instance
from src.model.gurobi_model import (
    MLNDModel,
    _LOStartCandidate,
    _NetworkStartAttempt,
)
from src.model.primal_heuristic import PrimalSeed
from src.model.valid_inequalities import ChoiceCorridorCoverData
from src.utils.output_processor import ResultProcessor


class _ExplodingChiIndex:
    def __iter__(self) -> object:
        raise AssertionError("LO candidate ranking must not traverse chi_keys")


class _RankingInstance:
    def __init__(self, candidates: tuple[ChoiceCorridorCoverData, ...]):
        self.config = ModelConfig(choice=ChoiceConfig(max_walk_km=None))
        self.display_origins = {
            (market[0], market[1], data.locker): (f"I_{data.locker}",)
            for data in candidates
            for market, _amount in data.market_contributions
        }

    @staticmethod
    def direct_rider_cost(_origin: str, _destination: str) -> float:
        return 10.0

    @staticmethod
    def feeder_rider_cost(_origin: str, _locker: str) -> float:
        return 1.0

    @staticmethod
    def locker_pickup_cost(_locker: str) -> float:
        return 1.0

    @staticmethod
    def distance(_origin: str, _locker: str) -> float:
        return 0.0


def _cover(destination: str, locker: str, offered: float) -> ChoiceCorridorCoverData:
    market = (f"O_{destination}", destination)
    return ChoiceCorridorCoverData(
        destination=destination,
        locker=locker,
        offered_demand=offered,
        market_contributions=((market, offered),),
        incoming_corridors=((f"I_{locker}", locker),),
    )


def _ranking_wrapper(
    candidates: tuple[ChoiceCorridorCoverData, ...],
) -> MLNDModel:
    wrapper = MLNDModel.__new__(MLNDModel)
    wrapper.model = gp.Model("lo_candidate_ranking")
    wrapper.model.Params.OutputFlag = 0
    wrapper.inst = _RankingInstance(candidates)
    wrapper.alpha_keys = tuple(data.alpha_key for data in candidates)
    wrapper._choice_cover_data = candidates
    wrapper.chi_keys = _ExplodingChiIndex()
    return wrapper


def test_lo_candidate_capacity_screen_is_safe_stable_and_chi_free() -> None:
    candidates = (
        _cover("D2", "H2", 10.0),
        _cover("D0", "H0", 80.0),
        _cover("D3", "H3", 76.00004),
        _cover("D1", "H1", 10.0),
    )
    solver = SolverConfig(output_flag=0)

    forward, rejected = _ranking_wrapper(candidates)._rank_lo_network_start_candidates(
        solver
    )
    reverse, reverse_rejected = _ranking_wrapper(
        tuple(reversed(candidates))
    )._rank_lo_network_start_candidates(solver)

    assert rejected == reverse_rejected == 1
    assert [candidate.alpha_key for candidate in forward] == [
        ("D3", "H3"),
        ("D1", "H1"),
        ("D2", "H2"),
    ]
    assert [candidate.alpha_key for candidate in reverse] == [
        candidate.alpha_key for candidate in forward
    ]
    # rho_min=0.5 and the best locker capacity is 50-1/(5/60)=38.
    # The near-boundary candidate remains admissible within solver tolerance.
    assert forward[0].effective_min_arrival == pytest.approx(38.00002)


def _start_values(model: gp.Model, index: int) -> tuple[float, ...]:
    model.Params.StartNumber = index
    model.update()
    return tuple(float(variable.Start) for variable in model.getVars())


def test_network_start_attempt_retries_from_clean_state_and_restores_everything() -> (
    None
):
    model = gp.Model("lo_attempt_transaction")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    x = model.addVars(("L",), vtype=GRB.BINARY, name="x")
    z = model.addVars(("H",), vtype=GRB.BINARY, name="z")
    delta = model.addVars(("S",), vtype=GRB.BINARY, name="delta")
    alpha = model.addVars(
        (("D", "bad"), ("D", "good")),
        vtype=GRB.BINARY,
        name="alpha",
    )
    chi = model.addVars((), lb=0.0, name="chi")
    model.addConstr(alpha["D", "bad"] == 0.0, name="bad_is_infeasible")
    model.setObjective(x["L"] + z["H"] + delta["S"], GRB.MINIMIZE)
    model.update()

    wrapper = MLNDModel.__new__(MLNDModel)
    wrapper.model = model
    wrapper.vars = {
        "x": x,
        "z": z,
        "delta": delta,
        "alpha": alpha,
        "chi": chi,
    }
    original_bounds = {
        variable.VarName: (float(variable.LB), float(variable.UB))
        for variable in (*x.values(), *alpha.values())
    }
    original_constraint_count = model.NumConstrs
    original_parameters = {
        name: getattr(model.Params, name)
        for name in ("TimeLimit", "MIPGap", "MIPFocus", "Heuristics", "NoRelHeurTime")
    }
    model.NumStart = 1
    model.update()
    model.Params.StartNumber = 0
    for index, variable in enumerate(model.getVars()):
        variable.Start = float(index % 2)
    model.update()
    original_start = _start_values(model, 0)

    failed = wrapper._solve_network_start_attempt(
        time_limit=0.5,
        alpha_target=("D", "bad"),
    )
    assert failed is None
    feasible = wrapper._solve_network_start_attempt(
        time_limit=0.5,
        alpha_target=("D", "good"),
    )

    assert feasible is not None
    assert feasible.seed.structure["alpha"] == {
        ("D", "bad"): 0,
        ("D", "good"): 1,
    }
    assert {
        variable.VarName: (float(variable.LB), float(variable.UB))
        for variable in (*x.values(), *alpha.values())
    } == original_bounds
    assert model.NumConstrs == original_constraint_count
    assert model.NumStart == 1
    assert _start_values(model, 0) == original_start
    assert {
        name: getattr(model.Params, name) for name in original_parameters
    } == original_parameters


def test_lo_retry_stops_at_first_feasible_seed_and_records_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = gp.Model("lo_retry_control")
    model.Params.OutputFlag = 0
    x = model.addVars(("L",), vtype=GRB.BINARY, name="x")
    alpha = model.addVars(("A",), vtype=GRB.BINARY, name="alpha")
    model.update()

    wrapper = MLNDModel.__new__(MLNDModel)
    wrapper.model = model
    wrapper.mode = "LO"
    wrapper.inst = SimpleNamespace(L=["L"], H=[], E=[])
    wrapper.vars = {"x": x, "alpha": alpha}
    wrapper._choice_cover_data = tuple(range(7))
    wrapper.network_start_found = False
    wrapper.network_start_used = False
    wrapper.network_start_key = None
    wrapper.network_start_objective = None
    wrapper.network_start_candidate_count = 0
    wrapper.network_start_capacity_rejected_count = 0
    wrapper.network_start_attempted_keys = ()
    wrapper.network_start_candidate_selection_seconds = 0.0
    wrapper.network_start_seed_open_counts = {}
    wrapper.local_branching_runtime = 0.0
    wrapper.local_branching_improved = False
    wrapper.local_branching_objective = None
    wrapper.local_branching_status = None
    wrapper.local_branching_solution_count = 0
    wrapper.local_branching_hamming_distance = None

    candidates = tuple(
        _LOStartCandidate((f"D{index}", f"H{index}"), 1.0, 0.5, 10.0 - index)
        for index in range(7)
    )
    wrapper._rank_lo_network_start_candidates = lambda _solver: (candidates, 2)
    calls: list[tuple[str, str]] = []
    model_values = {variable: 0.0 for variable in model.getVars()}
    seed = PrimalSeed(
        objective=1.0,
        structure={"x": {"L": 1}},
        values=model_values,
    )

    def fake_attempt(**kwargs: object) -> _NetworkStartAttempt | None:
        key = kwargs["alpha_target"]
        assert isinstance(key, tuple)
        calls.append(key)
        if len(calls) == 1:
            return None
        return _NetworkStartAttempt(seed=seed, objective=1.0, values=model_values)

    wrapper._solve_network_start_attempt = fake_attempt
    fallback_calls: list[str] = []
    wrapper._append_network_fallback_starts = lambda: fallback_calls.append("network")
    wrapper._set_courier_only_start = lambda: fallback_calls.append("courier")
    local_branching_calls: list[PrimalSeed] = []

    def fake_local_branching(
        _model: gp.Model,
        _variables: object,
        candidate_seed: PrimalSeed,
        _config: object,
    ) -> SimpleNamespace:
        local_branching_calls.append(candidate_seed)
        return SimpleNamespace(
            runtime=0.0,
            improved=False,
            installed_objective=candidate_seed.objective,
            status=GRB.OPTIMAL,
            solution_count=1,
            hamming_distance=0,
        )

    monkeypatch.setattr(
        "src.model.gurobi_model.improve_mip_start",
        fake_local_branching,
    )

    wrapper._construct_network_start(
        1.0,
        SolverConfig(output_flag=0, use_local_branching=True),
    )

    assert calls == [("D0", "H0"), ("D1", "H1")]
    assert wrapper.network_start_attempted_keys == tuple(calls)
    assert wrapper.network_start_key == ("D1", "H1")
    assert wrapper.network_start_candidate_count == 7
    assert wrapper.network_start_capacity_rejected_count == 2
    assert wrapper.network_start_candidate_selection_seconds >= 0.0
    assert wrapper.network_start_found and wrapper.network_start_used
    assert local_branching_calls == [seed]
    assert fallback_calls == ["network"]


def _tiny_lo_instance() -> Instance:
    config = ModelConfig(
        choice=ChoiceConfig(max_walk_km=None),
        battery_range_km=10.0,
        base_cost_per_hour=0.01,
        locker_fixed_cost_per_hour=0.01,
        locker_capacity_costs=(0.01, 0.02, 0.03),
        corridor_cost_per_km_hour=0.0001,
        connection_cost_per_km_hour=0.0001,
        flight_cost_per_km=0.001,
        drone_cost_per_hour=0.01,
        feeder_rider_cost_factor=0.01,
        locker_pickup_cost_per_order=0.0,
        display_origin_choices=None,
        general_corridor_choices=None,
    )
    return Instance(
        L=["L1"],
        H=["H1", "H2", "H3"],
        E=[],
        N=["O1", "D1"],
        coords={
            "L1": (0.0, 0.0),
            "H1": (3.0, 0.0),
            "H2": (2.0, 0.0),
            "H3": (1.0, 0.0),
            "O1": (0.0, 0.0),
            "D1": (3.0, 0.0),
        },
        activity_rate={"L1": 1.0, "H1": 1.0, "H2": 1.0, "H3": 1.0},
        h_od={("O1", "D1"): 160.0},
        order_count={("O1", "D1"): 160},
        metadata={"source": "test", "demand_exposure_hours": 1.0},
        config=config,
    ).build()


def test_tiny_real_lo_solve_skips_overcapacity_choice_and_builds_network() -> None:
    wrapper = MLNDModel(_tiny_lo_instance(), "LO").build()
    original_bounds = {
        variable.VarName: (float(variable.LB), float(variable.UB))
        for variable in (
            *wrapper.vars["x"].values(),
            *wrapper.vars["alpha"].values(),
        )
    }
    solved = wrapper.solve(
        SolverConfig(
            time_limit=10.0,
            mip_gap=0.0,
            threads=1,
            output_flag=0,
            network_start_seconds=4.0,
            use_valid_inequalities=False,
            use_user_cuts=False,
            use_local_branching=False,
        )
    )

    assert solved and wrapper.model.Status == GRB.OPTIMAL
    assert wrapper.network_start_found and wrapper.network_start_used
    assert wrapper.network_start_candidate_count == 3
    assert wrapper.network_start_capacity_rejected_count == 1
    assert wrapper.network_start_attempted_keys == (("D1", "H2"),)
    assert wrapper.network_start_key == ("D1", "H2")
    assert sum(variable.X for variable in wrapper.vars["chi"].values()) > 1e-8
    assert sum(variable.X for variable in wrapper.vars["alpha"].values()) > 0.5
    assert {
        variable.VarName: (float(variable.LB), float(variable.UB))
        for variable in (
            *wrapper.vars["x"].values(),
            *wrapper.vars["alpha"].values(),
        )
    } == original_bounds

    processor = ResultProcessor(
        wrapper,
        wrapper.inst,
        run_signature_sha256="a" * 64,
    )
    algorithm = processor.summary["solver"]["algorithm"]
    assert algorithm["network_start_candidate_count"] == 3
    assert algorithm["network_start_capacity_rejected_count"] == 1
    assert algorithm["network_start_attempted_keys"] == [["D1", "H2"]]
    assert algorithm["network_start_candidate_selection_seconds"] >= 0.0
