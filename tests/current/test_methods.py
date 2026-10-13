"""Method profiles and experiment planning; no optimization is needed."""

from dataclasses import asdict
import json

import pytest

from src.cli import main
from src.experiments.plan import build_plan
from src.solvers.methods import METHODS, get_method


def test_plain_disables_every_custom_acceleration():
    plain = get_method("benders_cg_plain")
    assert plain.pricing == "label"
    assert not any(asdict(plain.accelerations).values())
    assert plain.describe()["benders_cuts"]
    assert plain.describe()["extra_valid_inequalities"] == []


def test_full_and_solver_pricing_differ_only_in_backend():
    label = get_method("benders_cg")
    solver = get_method("benders_cg_solver")
    assert label.accelerations == solver.accelerations
    assert label.formulation == solver.formulation == "benders"
    assert label.pricing == "label" and solver.pricing == "solver"
    assert label.accelerations.pareto_cuts
    assert label.accelerations.benders_options()["pareto_cuts"]


def test_pareto_can_be_disabled_without_changing_other_accelerations():
    profile = get_method("benders_cg", {"pareto_cuts": False})
    assert not profile.accelerations.pareto_cuts
    assert profile.accelerations.initial_bound
    assert profile.accelerations.primal_search
    assert get_method("benders_cg").accelerations.pareto_cuts


def test_overrides_do_not_mutate_profiles():
    custom = get_method("benders_cg", {"initial_bound": False})
    assert not custom.accelerations.initial_bound
    assert get_method("benders_cg").accelerations.initial_bound
    with pytest.raises(ValueError):
        get_method("arc", {"initial_bound": False})
    with pytest.raises(ValueError):
        get_method("benders_cg", {"unknown": True})


def test_plan_creates_no_solver_or_output(tmp_path, monkeypatch, capsys):
    import gurobipy as gp
    import src.data.instances as instances

    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not load instances or create an optimizer")

    monkeypatch.setattr(gp, "Model", forbidden)
    monkeypatch.setattr(gp, "Env", forbidden)
    monkeypatch.setattr(instances, "load_instance", forbidden)
    target = tmp_path / "future"
    assert (
        main(["plan", "--hubs", "11", "120", "--methods", "all", "--out", str(target)])
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "planned_only"
    assert not plan["optimization_started"]
    assert len(plan["cases"]) == 10
    assert len({case["output"] for case in plan["cases"]}) == 10
    assert not target.exists()


def test_plan_uses_common_solver_settings():
    plan = build_plan([11], METHODS, options={"time_limit": 120, "threads": 2})
    assert len(plan["cases"]) == 5
    assert plan["solver"]["time_limit"] == 120
    assert plan["solver"]["threads"] == 2
