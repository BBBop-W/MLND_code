from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.benchmark.branch_cut.hashing import canonical_json
from src.benchmark.branch_cut.telemetry import (
    GurobiParameterAuditError,
    collect_gurobi_parameter_audit,
    enumerate_public_gurobi_parameter_names,
)


class _Backend:
    def __init__(
        self,
        records: dict[str, tuple[type[object], object, object]],
        *,
        failing_name: str | None = None,
    ) -> None:
        self.records = records
        self.failing_name = failing_name

    def getParamInfo(self, name: str) -> tuple[object, ...]:
        if name == self.failing_name:
            raise RuntimeError("SENSITIVE_ERROR_TEXT_MUST_NOT_BE_PERSISTED")
        value_type, current, default = self.records[name]
        return name, value_type, current, None, None, default


def _wrapper(
    records: dict[str, tuple[type[object], object, object]],
    *,
    failing_name: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(model=_Backend(records, failing_name=failing_name))


def test_public_parameter_catalog_comes_from_grb_param_without_environment() -> None:
    names = enumerate_public_gurobi_parameter_names()

    assert len(names) == len(set(names))
    assert names == tuple(sorted(names))
    assert {
        "CloudSecretKey",
        "ComputeServer",
        "MIPFocus",
        "Threads",
    } <= set(names)


def test_full_nondefault_audit_redacts_credentials_and_identifies_remote_mode() -> None:
    records: dict[str, tuple[type[object], object, object]] = {
        "BranchDir": (int, 1, 0),
        "CloudAccessID": (str, "RAW_CLOUD_ACCESS_ID", ""),
        "CloudHost": (str, "RAW_REMOTE_HOST", ""),
        "CloudSecretKey": (str, "RAW_CLOUD_SECRET", ""),
        "Cuts": (int, -1, -1),
        "LicenseID": (int, 8_675_309, 0),
        "LogFile": (str, "/private/raw/solver.log", ""),
        "Threads": (int, 1, 0),
        "UserName": (str, "RAW_USER_NAME", ""),
    }
    names = tuple(records)

    first = collect_gurobi_parameter_audit(
        _wrapper(records),
        strict=True,
        parameter_name_provider=lambda: names,
    )
    second = collect_gurobi_parameter_audit(
        _wrapper(records),
        strict=True,
        parameter_name_provider=lambda: tuple(reversed(names)),
    )

    assert first == second
    assert first["enumeration_complete"] is True
    assert first["enumerated_parameter_count"] == len(records)
    assert first["unallowed_nondefault_parameters"] == ["BranchDir"]
    effective = first["effective_nondefault_parameters"]
    assert "Cuts" not in effective
    assert effective["Threads"] == {
        "classification": "core_solver_allowed",
        "current": 1,
        "default": 0,
        "value_type": "int",
    }
    assert effective["BranchDir"]["classification"] == "unallowed"
    assert effective["LogFile"]["classification"] == "benchmark_operational"
    assert effective["CloudSecretKey"]["current"] == {
        "is_set": True,
        "redacted": True,
    }
    assert "sha256" not in effective["CloudSecretKey"]["current"]
    assert effective["CloudHost"]["current"]["redacted"] is True
    assert first["execution_context"]["execution_mode"] == "remote"
    assert first["execution_context"]["remote_mechanisms"] == ["instant_cloud"]
    assert len(first["execution_context"]["remote_identity_sha256"]) == 64

    serialized = canonical_json(first)
    for raw_value in (
        "RAW_CLOUD_ACCESS_ID",
        "RAW_REMOTE_HOST",
        "RAW_CLOUD_SECRET",
        "RAW_USER_NAME",
        "/private/raw/solver.log",
        "8675309",
    ):
        assert raw_value not in serialized


def test_local_license_identity_is_hashed_without_becoming_remote() -> None:
    records: dict[str, tuple[type[object], object, object]] = {
        "LicenseID": (int, 1234, 0),
        "Threads": (int, 1, 0),
        "UserName": (str, "LOCAL_RAW_USER", ""),
    }
    audit = collect_gurobi_parameter_audit(
        _wrapper(records),
        strict=True,
        parameter_name_provider=lambda: records,
    )

    assert audit["execution_context"] == {
        "execution_mode": "local",
        "observation_basis": "gurobi_effective_parameters",
        "remote_identity_sha256": None,
        "remote_mechanisms": [],
    }
    assert audit["unallowed_nondefault_parameters"] == []
    assert "LOCAL_RAW_USER" not in canonical_json(audit)


def test_remote_secret_without_observable_identity_fails_closed() -> None:
    records: dict[str, tuple[type[object], object, object]] = {
        "CloudSecretKey": (str, "RAW_ORPHAN_SECRET", ""),
    }

    with pytest.raises(GurobiParameterAuditError, match="identity is unobservable"):
        collect_gurobi_parameter_audit(
            _wrapper(records),
            strict=True,
            parameter_name_provider=lambda: records,
        )

    fallback = collect_gurobi_parameter_audit(
        _wrapper(records),
        strict=False,
        parameter_name_provider=lambda: records,
    )
    assert fallback["enumeration_complete"] is False
    assert fallback["execution_context"]["execution_mode"] == "unobserved"
    assert "RAW_ORPHAN_SECRET" not in canonical_json(fallback)


def test_enumeration_error_discards_partial_state_and_error_text() -> None:
    records: dict[str, tuple[type[object], object, object]] = {
        "BranchDir": (int, 1, 0),
        "Threads": (int, 1, 0),
    }
    fallback = collect_gurobi_parameter_audit(
        _wrapper(records, failing_name="Threads"),
        strict=False,
        parameter_name_provider=lambda: records,
    )

    assert fallback["enumeration_complete"] is False
    assert fallback["effective_nondefault_parameters"] == {}
    assert "SENSITIVE_ERROR_TEXT" not in canonical_json(fallback)

    with pytest.raises(GurobiParameterAuditError, match="parameter_enumeration_failed"):
        collect_gurobi_parameter_audit(
            _wrapper(records, failing_name="Threads"),
            strict=True,
            parameter_name_provider=lambda: records,
        )


def test_missing_public_observation_api_fails_closed_only_when_strict() -> None:
    wrapper = SimpleNamespace(model=SimpleNamespace())
    ordinary = collect_gurobi_parameter_audit(wrapper, strict=False)
    assert ordinary["enumeration_complete"] is False
    assert ordinary["reason_code"] == "get_param_info_unavailable"

    with pytest.raises(GurobiParameterAuditError, match="get_param_info_unavailable"):
        collect_gurobi_parameter_audit(wrapper, strict=True)
