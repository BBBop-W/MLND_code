from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import pytest

import main as cli_main
from main import build_parser
from src.config import SolverConfig
from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION


@pytest.mark.parametrize(
    "modes",
    [
        ("HD",),
        ("MX",),
        ("LO",),
        ("HD", "MX"),
        ("HD", "LO"),
        ("MX", "LO"),
        ("HD", "MX", "LO"),
    ],
)
def test_mode_validation_accepts_only_legal_subsequences(
    modes: tuple[str, ...],
) -> None:
    assert cli_main._validated_modes(modes) == modes


@pytest.mark.parametrize(
    "modes",
    [
        ("MX", "HD"),
        ("LO", "MX"),
        ("HD", "LO", "MX"),
        ("HD", "HD"),
        ("MX", "MX", "LO"),
    ],
)
def test_mode_validation_rejects_reverse_order_and_duplicates(
    modes: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="order|duplicates"):
        cli_main._validated_modes(modes)


def test_cross_mode_start_policy_is_declared_only_for_hd_then_mx() -> None:
    assert (
        cli_main._cross_mode_start_policy(("HD", "MX", "LO"))
        == "HD_to_later_MX_by_variable_name"
    )
    assert cli_main._cross_mode_start_policy(("HD", "LO")) is None
    assert cli_main._cross_mode_start_policy(("MX", "LO")) is None


def test_results_root_must_be_empty(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    (results_root / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be empty"):
        cli_main._prepare_empty_results_root(results_root)


def test_gurobi_solution_loader_returns_exact_values_and_provenance(
    tmp_path: Path,
) -> None:
    solution = tmp_path / "incumbent.sol"
    payload = (
        "# Solution for model MLND_HD\n"
        "# Objective value = 123.5\n"
        "gamma 4\n"
        "y[H001] 1.0\n"
        "q[L001,H001,t0] -2.5e-3\n"
    )
    solution.write_text(payload, encoding="utf-8")

    path, sha256, values = cli_main._load_gurobi_solution(solution)

    assert path == solution.resolve()
    assert sha256 == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert values == {
        "gamma": 4.0,
        "y[H001]": 1.0,
        "q[L001,H001,t0]": -2.5e-3,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("", "no variable assignments"),
        ("# comments only\n", "no variable assignments"),
        ("x 1\nx 2\n", "duplicate warm-start variable"),
        ("x not-a-number\n", "invalid warm-start value"),
        ("x nan\n", "nonfinite warm-start value"),
        ("x inf\n", "nonfinite warm-start value"),
        ("x 1 extra-field\n", "expected '<variable-name> <finite-value>'"),
    ],
)
def test_gurobi_solution_loader_rejects_bad_content(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    solution = tmp_path / "bad.sol"
    solution.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        cli_main._load_gurobi_solution(solution)


def test_gurobi_solution_loader_requires_sol_file(tmp_path: Path) -> None:
    solution = tmp_path / "incumbent.txt"
    solution.write_text("x 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.sol extension"):
        cli_main._load_gurobi_solution(solution)


def test_cli_applies_file_start_once_then_prefers_hd_and_writes_solutions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.data_processor.loader as loader_module
    import src.model.gurobi_model as gurobi_model_module
    import src.utils.output_processor as output_processor_module

    instance_dir = tmp_path / "instance"
    instance_dir.mkdir()
    (instance_dir / "metadata.json").write_text("{}", encoding="utf-8")
    results_dir = tmp_path / "results"
    input_solution = tmp_path / "previous_hd.sol"
    input_payload = "# prior HD incumbent\ngamma 7\ny[H001] 1\n"
    input_solution.write_text(input_payload, encoding="utf-8")

    class FakeInstance:
        def screening_is_exact(self, _mode: str) -> bool:
            return True

    class FakeNativeModel:
        def __init__(self, mode: str):
            self.mode = mode
            self.Status = 2
            self.Params = type("FakeParams", (), {"LogFile": ""})()

        def write(self, raw_path: str) -> None:
            Path(raw_path).write_text(
                f"# generated {self.mode} incumbent\ngamma 1\n",
                encoding="utf-8",
            )

        def dispose(self) -> None:
            return None

    solve_calls: list[tuple[str, dict[str, float] | None, str | None]] = []

    class FakeModel:
        def __init__(self, _instance: object, mode: str):
            self.mode = mode
            self.model = FakeNativeModel(mode)
            self.external_start_installed = False
            self.external_start_source: str | None = None
            self.external_start_matched_variables = 0

        def build(self) -> FakeModel:
            return self

        def solve(
            self,
            _solver: SolverConfig,
            *,
            external_start: Mapping[str, float] | None = None,
            external_start_source: str | None = None,
        ) -> bool:
            copied_start = dict(external_start) if external_start is not None else None
            solve_calls.append((self.mode, copied_start, external_start_source))
            self.external_start_installed = external_start is not None
            self.external_start_source = external_start_source
            self.external_start_matched_variables = len(copied_start or {})
            return True

        def export_incumbent_start(self) -> dict[str, float]:
            return {"current_hd": 11.0}

    class FakeProcessor:
        def __init__(
            self,
            wrapper: FakeModel,
            _instance: object,
            *,
            run_signature_sha256: str,
        ):
            self.wrapper = wrapper
            self.run_signature_sha256 = run_signature_sha256

        def print_summary(self) -> None:
            return None

        def write_all(self, _output_dir: str) -> dict[str, object]:
            return {
                "schema_version": RESULT_SCHEMA_VERSION,
                "formulation_id": FORMULATION_ID,
                "run_signature_sha256": self.run_signature_sha256,
                "mode": self.wrapper.mode,
                "solver": {
                    "external_start_installed": self.wrapper.external_start_installed,
                    "external_start_source": self.wrapper.external_start_source,
                    "external_start_matched_variables": (
                        self.wrapper.external_start_matched_variables
                    ),
                },
            }

    monkeypatch.setattr(
        loader_module.DataLoader,
        "load_instance_dir",
        lambda _self, *_args, **_kwargs: FakeInstance(),
    )
    monkeypatch.setattr(gurobi_model_module, "MLNDModel", FakeModel)
    monkeypatch.setattr(output_processor_module, "ResultProcessor", FakeProcessor)
    monkeypatch.setattr(cli_main, "_write_mode_comparison", lambda *_args: None)

    args = build_parser().parse_args(
        [
            "solve",
            "--instance-dir",
            str(instance_dir),
            "--results-dir",
            str(results_dir),
            "--modes",
            "HD,MX,LO",
            "--warm-start-from",
            str(input_solution),
            "--quiet",
        ]
    )
    summaries = cli_main._solve(args, str(instance_dir))

    absolute_source = f"file:{input_solution.resolve()}"
    assert solve_calls == [
        ("HD", {"gamma": 7.0, "y[H001]": 1.0}, absolute_source),
        ("MX", {"current_hd": 11.0}, "HD"),
        ("LO", None, None),
    ]
    assert [summary["solver"]["external_start_source"] for summary in summaries] == [
        absolute_source,
        "HD",
        None,
    ]
    for mode in ("HD", "MX", "LO"):
        written_solution = results_dir / mode / "incumbent.sol"
        assert written_solution.is_file()
        assert f"generated {mode} incumbent" in written_solution.read_text(
            encoding="utf-8"
        )

    run_config = json.loads((results_dir / "run_config.json").read_text())
    assert run_config["warm_start_from"] == {
        "absolute_path": str(input_solution.resolve()),
        "sha256": hashlib.sha256(input_payload.encode("utf-8")).hexdigest(),
        "variable_count": 2,
    }
    assert run_config["file_warm_start_policy"] == "first_mode_only"
    assert run_config["cross_mode_start_policy"] == "HD_to_later_MX_by_variable_name"
    assert len(run_config["run_signature_sha256"]) == 64
    assert {summary["run_signature_sha256"] for summary in summaries} == {
        run_config["run_signature_sha256"]
    }

    with pytest.raises(FileExistsError, match="must be empty"):
        cli_main._solve(args, str(instance_dir))
    assert len(solve_calls) == 3
