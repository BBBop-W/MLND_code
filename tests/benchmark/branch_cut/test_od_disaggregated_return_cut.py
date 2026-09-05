from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

import src.model.gurobi_model as gurobi_model
import src.model.valid_inequalities as inequalities
from src.config import SolverConfig
from src.model.gurobi_model import MLNDModel
from src.model.valid_inequalities import (
    separate_directed_return_cuts,
    separate_od_disaggregated_directed_return_cuts,
)


def _base_network() -> tuple[
    tuple[tuple[str, str], ...],
    dict[tuple[str, str], float],
]:
    arcs = (
        ("H1", "H2"),
        ("H2", "H3"),
        ("H2", "H4"),
        ("H3", "H1"),
        ("H4", "H1"),
    )
    y_values = {
        ("H1", "H2"): 1.0,
        ("H2", "H3"): 0.10,
        ("H2", "H4"): 0.20,
        ("H3", "H1"): 0.80,
        ("H4", "H1"): 0.70,
    }
    return arcs, y_values


def _has_directed_path(
    arcs: tuple[tuple[str, str], ...],
    values: dict[tuple[str, str], float],
    source: str,
    sink: str,
) -> bool:
    reachable = {source}
    while True:
        expanded = reachable | {
            head
            for tail, head in arcs
            if values[tail, head] > 0.5 and tail in reachable
        }
        if expanded == reachable:
            return sink in reachable
        reachable = expanded


def test_od_disaggregated_cut_has_expected_min_cut_and_coefficients() -> None:
    arcs, y_values = _base_network()
    key = ("O1", "D1", "H1", "H2")

    cuts = separate_od_disaggregated_directed_return_cuts(
        arcs,
        y_values,
        offered_values={key: 5.0},
        upper_bounds={key: 10.0},
    )

    assert len(cuts) == 1
    cut = cuts[0]
    assert cut.od_corridor == key
    assert cut.market == ("O1", "D1")
    assert cut.corridor == ("H1", "H2")
    assert cut.source_side == ("H2",)
    assert cut.crossing_arcs == (("H2", "H3"), ("H2", "H4"))
    assert cut.cut_capacity == pytest.approx(0.30)
    assert cut.upper_bound == pytest.approx(10.0)
    assert cut.violation == pytest.approx(2.0)


def test_od_disaggregated_cut_is_valid_for_all_integral_return_networks() -> None:
    arcs, y_values = _base_network()
    key = ("O1", "D1", "H1", "H2")
    (cut,) = separate_od_disaggregated_directed_return_cuts(
        arcs,
        y_values,
        offered_values={key: 5.0},
        upper_bounds={key: 10.0},
    )

    for assignment in itertools.product((0.0, 1.0), repeat=len(arcs)):
        integral_y = dict(zip(arcs, assignment))
        return_exists = _has_directed_path(
            arcs,
            integral_y,
            source="H2",
            sink="H1",
        )
        # A balanced integral design can carry this OD/corridor flow only when
        # a return path exists.  This is the exact disjunction behind the cut.
        offered = cut.upper_bound if return_exists else 0.0
        rhs = cut.upper_bound * sum(integral_y[arc] for arc in cut.crossing_arcs)
        assert offered <= rhs


def test_od_disaggregation_strictly_strengthens_aggregate_return_cut() -> None:
    arcs = (("H1", "H2"), ("H2", "H1"))
    y_values = {("H1", "H2"): 1.0, ("H2", "H1"): 0.10}
    key = ("O-small", "D-small", "H1", "H2")

    # The aggregate row is satisfied because its corridor M is much larger.
    assert (
        separate_directed_return_cuts(
            arcs,
            y_values,
            offered_values={("H1", "H2"): 5.0},
            big_m={("H1", "H2"): 100.0},
        )
        == ()
    )

    (cut,) = separate_od_disaggregated_directed_return_cuts(
        arcs,
        y_values,
        offered_values={key: 5.0},
        upper_bounds={key: 10.0},
    )
    assert cut.violation == pytest.approx(4.0)


def test_min_cut_is_cached_once_per_corridor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arcs, y_values = _base_network()
    keys = (
        ("O1", "D1", "H1", "H2"),
        ("O2", "D2", "H1", "H2"),
        ("O3", "D3", "H1", "H2"),
    )
    calls: list[tuple[str, str]] = []
    original = inequalities._minimum_directed_cut

    def counting_min_cut(*args: object, **kwargs: object):
        calls.append((str(args[2]), str(args[3])))
        return original(*args, **kwargs)

    monkeypatch.setattr(inequalities, "_minimum_directed_cut", counting_min_cut)
    separate_od_disaggregated_directed_return_cuts(
        arcs,
        y_values,
        offered_values={key: 5.0 for key in keys},
        upper_bounds={key: 10.0 for key in keys},
    )

    assert calls == [("H2", "H1")]


def test_virtual_state_contraction_preserves_original_crossing_arcs() -> None:
    arcs = (
        ("H1", "H2"),
        ("H2", "S_H2"),
        ("S_H2", "H3"),
        ("H3", "H1"),
    )
    key = ("O1", "D1", "H1", "H2")
    (cut,) = separate_od_disaggregated_directed_return_cuts(
        arcs,
        y_values={arc: value for arc, value in zip(arcs, (1.0, 0.0, 0.10, 0.20))},
        offered_values={key: 3.0},
        upper_bounds={key: 10.0},
        contraction={"S_H2": "H2"},
    )

    assert ("H2", "S_H2") not in cut.crossing_arcs
    assert cut.crossing_arcs == (("S_H2", "H3"),)
    assert cut.violation == pytest.approx(2.0)


def test_duplicate_signature_is_suppressed() -> None:
    arcs = (("H1", "H2"), ("H2", "H1"))
    y_values = {("H1", "H2"): 1.0, ("H2", "H1"): 0.0}
    key = ("O1", "D1", "H1", "H2")
    first = separate_od_disaggregated_directed_return_cuts(
        arcs,
        y_values,
        offered_values={key: 1.0},
        upper_bounds={key: 4.0},
    )

    assert len(first) == 1
    assert (
        separate_od_disaggregated_directed_return_cuts(
            arcs,
            y_values,
            offered_values={key: 1.0},
            upper_bounds={key: 4.0},
            seen_signatures={first[0].signature()},
        )
        == ()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("y", float("nan")),
        ("offered", float("inf")),
        ("upper", float("nan")),
        ("violation_tolerance", float("nan")),
        ("feasibility_tolerance", float("inf")),
    ),
)
def test_nonfinite_inputs_fail_closed(field: str, value: float) -> None:
    arcs = (("H1", "H2"), ("H2", "H1"))
    key = ("O1", "D1", "H1", "H2")
    kwargs: dict[str, object] = {
        "physical_arcs": arcs,
        "y_values": {("H1", "H2"): 1.0, ("H2", "H1"): 0.0},
        "offered_values": {key: 1.0},
        "upper_bounds": {key: 4.0},
    }
    if field == "y":
        kwargs["y_values"] = {("H1", "H2"): 1.0, ("H2", "H1"): value}
    elif field == "offered":
        kwargs["offered_values"] = {key: value}
    elif field == "upper":
        kwargs["upper_bounds"] = {key: value}
    else:
        kwargs[field] = value

    with pytest.raises(ValueError, match="finite"):
        separate_od_disaggregated_directed_return_cuts(**kwargs)  # type: ignore[arg-type]


def test_missing_bound_and_malformed_key_fail_closed() -> None:
    arcs = (("H1", "H2"), ("H2", "H1"))
    y_values = {("H1", "H2"): 1.0, ("H2", "H1"): 0.0}
    key = ("O1", "D1", "H1", "H2")

    with pytest.raises(ValueError, match="missing upper bound"):
        separate_od_disaggregated_directed_return_cuts(
            arcs,
            y_values,
            offered_values={key: 1.0},
            upper_bounds={},
        )
    with pytest.raises(ValueError, match="origin,destination,i,j"):
        separate_od_disaggregated_directed_return_cuts(
            arcs,
            y_values,
            offered_values={("O1", "D1", "H1"): 1.0},  # type: ignore[dict-item]
            upper_bounds={},
        )


def test_lo_upper_bound_uses_choice_share_and_corridor_cap() -> None:
    key = ("O1", "D1", "H1", "H2")
    wrapper = object.__new__(MLNDModel)
    wrapper.mode = "LO"
    wrapper.inst = SimpleNamespace(
        h_od={("O1", "D1"): 20.0},
        choice_share={key[:2] + (key[3],): 0.25},
    )
    wrapper.chi_by_od_corridor = {key: []}

    wrapper.M_offer = {("H1", "H2"): 30.0}
    assert wrapper._od_disaggregated_return_upper_bound(*key) == pytest.approx(5.0)

    wrapper.M_offer = {("H1", "H2"): 3.0}
    assert wrapper._od_disaggregated_return_upper_bound(*key) == pytest.approx(3.0)

    wrapper.mode = "MX"
    assert wrapper._od_disaggregated_return_upper_bound(*key) == pytest.approx(3.0)


def test_callback_family_has_independent_counts_time_and_deduplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gurobi_model.gp, "quicksum", sum)
    corridor = ("H1", "H2")
    return_arc = ("H2", "H1")
    od_corridor = ("O1", "D1", *corridor)
    chi_keys = (
        ("O1", "D1", *corridor, 0),
        ("O1", "D1", *corridor, 1),
    )
    wrapper = object.__new__(MLNDModel)
    wrapper.mode = "MX"
    wrapper.inst = SimpleNamespace(
        config=SimpleNamespace(),
        A_F=(corridor, return_arc),
        s_to_h={},
        h_od={("O1", "D1"): 10.0},
        choice_share={},
    )
    wrapper.vars = {
        "chi": {chi_keys[0]: 2.0, chi_keys[1]: 3.0},
        "alpha": {},
        "psi": {},
        "x": {},
        "y": {corridor: 1.0, return_arc: 0.1},
        "z": {},
        "xi": {},
        "varphi": {},
    }
    wrapper.chi_by_od_corridor = {od_corridor: list(chi_keys)}
    wrapper.chi_by_corridor = {corridor: list(chi_keys)}
    wrapper.corridors = (corridor,)
    wrapper.M_offer = {corridor: 100.0}
    wrapper._choice_cover_data = ()
    wrapper._route_bundle_indices_prepared = False
    wrapper._user_cut_signatures = set()
    wrapper.user_cut_counts = {}
    wrapper.user_cut_candidate_counts = {}
    wrapper.user_cut_duplicate_counts = {}
    wrapper.user_cut_rejection_counts = {}
    wrapper.user_cut_separation_seconds = {}

    class CallbackProbe:
        def __init__(self) -> None:
            self.cuts: list[object] = []

        @staticmethod
        def cbGetNodeRel(variables: list[float]) -> list[float]:
            return variables

        def cbCut(self, cut: object) -> None:
            self.cuts.append(cut)

    solver = SolverConfig(
        cut_chi_vub=False,
        cut_chi_facility_vub=False,
        cut_route_swap_station_vub=False,
        cut_route_swap_leg_vub=False,
        cut_queue_perspective=False,
        cut_lo_choice_corridor=False,
        cut_lo_choice_capacity=False,
        cut_directed_return=False,
        cut_od_disaggregated_return=True,
        max_cuts_per_round=10,
    )
    callback = CallbackProbe()

    assert wrapper._add_root_user_cuts(callback, solver) == 1
    assert wrapper.user_cut_counts == {"od_disaggregated_directed_return": 1}
    assert wrapper.user_cut_candidate_counts == {"od_disaggregated_directed_return": 1}
    assert (
        wrapper.user_cut_separation_seconds["od_disaggregated_directed_return"] >= 0.0
    )
    assert len(callback.cuts) == 1

    assert wrapper._add_root_user_cuts(callback, solver) == 0
    assert wrapper.user_cut_duplicate_counts == {"od_disaggregated_directed_return": 1}
    assert len(callback.cuts) == 1
