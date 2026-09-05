"""Versioned solver output and explicitly supported historical readers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


FORMULATION_ID = "mlnd-unified-chi-epsilon-energy-v3"
# Writers emit only this version. Reading an older result never upgrades it.
RESULT_SCHEMA_VERSION = 6
READABLE_RESULT_SCHEMA_VERSIONS = (5, 6)
RUNTIME_CONTRACTS = {
    5: "network_start_plus_main_solver",
    6: "algorithm_wall_excluding_model_build",
}


def is_readable_result_schema(value: object) -> bool:
    return type(value) is int and value in READABLE_RESULT_SCHEMA_VERSIONS


def require_result_contract(payload: Mapping[str, Any], context: object) -> int:
    """Read the common mathematical contract without coercing its version."""
    version = payload.get("schema_version")
    if (
        not is_readable_result_schema(version)
        or payload.get("formulation_id") != FORMULATION_ID
    ):
        raise ValueError(
            f"legacy/incompatible {context}: expected schema_version in "
            f"{READABLE_RESULT_SCHEMA_VERSIONS} and formulation_id={FORMULATION_ID!r}; "
            f"got schema_version={version!r}, formulation_id={payload.get('formulation_id')!r}"
        )
    return version


def require_same_result_schema(
    run_config: Mapping[str, Any], summary: Mapping[str, Any], context: object
) -> None:
    """Forbid cross-version assembly even when a run signature was copied."""
    version = require_result_contract(run_config, "run_config")
    if require_result_contract(summary, context) != version:
        raise ValueError(f"{context}: result schema disagrees with run_config")


def runtime_contract(version: object) -> str:
    if not is_readable_result_schema(version):
        raise ValueError(f"unsupported result schema_version: {version!r}")
    return RUNTIME_CONTRACTS[version]


def result_contract_metadata(versions: Iterable[int]) -> dict[str, Any]:
    """Describe source versions rather than the version of the running writer."""
    actual = sorted(set(versions))
    if not actual or any(not is_readable_result_schema(value) for value in actual):
        raise ValueError("result contract requires supported source schema versions")
    return {
        "schema_version": actual[0] if len(actual) == 1 else None,
        "schema_versions": actual,
        "formulation_id": FORMULATION_ID,
        "runtime_contracts": {str(value): runtime_contract(value) for value in actual},
    }


def require_validation_binding(
    validation: Mapping[str, Any], run_config: Mapping[str, Any], context: object
) -> None:
    version = require_result_contract(run_config, "run_config")
    if (
        type(validation.get("result_schema_version")) is not int
        or validation.get("result_schema_version") != version
        or validation.get("formulation_id") != run_config.get("formulation_id")
        or validation.get("run_signature_sha256")
        != run_config.get("run_signature_sha256")
    ):
        raise ValueError(
            f"{context}: validation result contract disagrees with run_config"
        )
