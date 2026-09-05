"""Artifact-safe telemetry extraction for branch-and-cut executions.

The helpers in this module use duck typing and do not import :mod:`gurobipy` at
module-import time.  The complete parameter audit lazily reads the installed
``GRB.Param`` constant namespace, which does not create a Gurobi environment;
tests can inject that namespace boundary without a license.
"""

from __future__ import annotations

import math
import os
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .artifacts import AttemptClaim, write_canonical_json_exclusive
from .hashing import canonical_json, canonical_sha256
from .schema import TelemetryConfig


_STATUS_NAMES = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT",
    17: "MEM_LIMIT",
}

APPLIED_GUROBI_PARAMETER_ATTRIBUTES = (
    ("time_limit", "TimeLimit"),
    ("mip_gap", "MIPGap"),
    ("threads", "Threads"),
    ("output_flag", "OutputFlag"),
    ("seed", "Seed"),
    ("method", "Method"),
    ("miqcp_method", "MIQCPMethod"),
    ("mip_focus", "MIPFocus"),
    ("heuristics", "Heuristics"),
    ("no_rel_heur_time", "NoRelHeurTime"),
    ("presolve", "Presolve"),
    ("cuts", "Cuts"),
    ("root_cut_passes", "CutPasses"),
    ("numeric_focus", "NumericFocus"),
    ("feasibility_tol", "FeasibilityTol"),
    ("optimality_tol", "OptimalityTol"),
    ("integer_feasibility_tol", "IntFeasTol"),
    ("pre_crush", "PreCrush"),
)

GUROBI_PARAMETER_AUDIT_SCHEMA_VERSION = 1
GUROBI_PARAMETER_CATALOG_HASH_DOMAIN = "mlnd-gurobi-parameter-catalog-v1"
GUROBI_PARAMETER_IDENTITY_HASH_DOMAIN = "mlnd-gurobi-parameter-identity-v1"
GUROBI_REMOTE_IDENTITY_HASH_DOMAIN = "mlnd-gurobi-remote-identity-v1"

# Exact public Gurobi parameter names whose effective values are intentionally
# controlled by CoreSolverConfig or by the benchmark artifact layer.  Any
# other nondefault algorithmic/operational parameter remains visible in the
# audit's unallowed list rather than silently inheriting process state.
CORE_SOLVER_ALLOWED_GUROBI_PARAMETERS = frozenset(
    {
        "Cuts",
        "CutPasses",
        "FeasibilityTol",
        "Heuristics",
        "IntFeasTol",
        "Method",
        "MIPFocus",
        "MIPGap",
        "MIQCPMethod",
        "NoRelHeurTime",
        "NumericFocus",
        "OptimalityTol",
        "OutputFlag",
        "PreCrush",
        "Presolve",
        "Seed",
        "Threads",
        "TimeLimit",
    }
)
BENCHMARK_OPERATIONAL_GUROBI_PARAMETERS = frozenset({"LogFile"})

# Credential material is never hashed because low-entropy passwords or tokens
# must not become offline-verifiable artifacts.  Only presence is retained.
_GUROBI_SECRET_PARAMETERS = frozenset(
    {
        "CSAPISecret",
        "CSAuthToken",
        "CloudSecretKey",
        "ServerPassword",
        "WLSSecret",
        "WLSToken",
        "WorkerPassword",
    }
)

# License, user, endpoint, and path identities are domain-separated hashes.
# This retains reproducible identity comparisons without writing raw values.
_GUROBI_IDENTITY_PARAMETERS = frozenset(
    {
        "CSAPIAccessID",
        "CSAppName",
        "CSGroup",
        "CSManager",
        "CSRouter",
        "CloudAccessID",
        "CloudHost",
        "CloudPool",
        "ComputeServer",
        "JobID",
        "LicenseID",
        "LogFile",
        "NodefileDir",
        "ResultFile",
        "SolFiles",
        "TokenServer",
        "UserName",
        "WLSAccessID",
        "WLSConfig",
        "WLSProxy",
        "WorkerPool",
    }
)

# These parameters describe licensing or a remote execution connection rather
# than an optimization algorithm.  They remain in the effective-state mapping
# (redacted as above) and execution-context record, but are not mislabeled as
# uncontrolled algorithm knobs.
GUROBI_RUNTIME_CONTEXT_PARAMETERS = frozenset(
    {
        "CSAPIAccessID",
        "CSAPISecret",
        "CSAppName",
        "CSAuthToken",
        "CSBatchMode",
        "CSClientLog",
        "CSGroup",
        "CSIdleTimeout",
        "CSManager",
        "CSPriority",
        "CSQueueTimeout",
        "CSRouter",
        "CSTLSInsecure",
        "CloudAccessID",
        "CloudHost",
        "CloudPool",
        "CloudSecretKey",
        "ComputeServer",
        "JobID",
        "LicenseID",
        "ServerPassword",
        "ServerTimeout",
        "TokenServer",
        "TSPort",
        "UserName",
        "WLSAccessID",
        "WLSConfig",
        "WLSProxy",
        "WLSSecret",
        "WLSToken",
        "WLSTokenDuration",
        "WLSTokenRefresh",
        "WorkerPassword",
        "WorkerPool",
    }
)

GurobiParameterNameProvider = Callable[[], Iterable[str]]


class GurobiParameterAuditError(RuntimeError):
    """The effective Gurobi parameter state cannot be observed safely."""


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """All files owned by one claimed execution attempt."""

    directory: Path
    solver_log: Path
    run_spec: Path
    run_record: Path
    progress: Path
    root_rounds: Path
    cut_stats: Path
    failure: Path

    @classmethod
    def from_claim(cls, claim: AttemptClaim) -> ArtifactPaths:
        directory = claim.directory
        return cls(
            directory=directory,
            solver_log=directory / "solver.log",
            run_spec=directory / "run_spec.json",
            run_record=directory / "run_record.json",
            progress=directory / "progress.jsonl",
            root_rounds=directory / "root_rounds.jsonl",
            cut_stats=directory / "cut_stats.json",
            failure=directory / "failure.json",
        )

    def generated_paths(self) -> tuple[Path, ...]:
        return (
            self.solver_log,
            self.run_spec,
            self.run_record,
            self.progress,
            self.root_rounds,
            self.cut_stats,
            self.failure,
        )


def assert_fresh_artifact_paths(paths: ArtifactPaths) -> None:
    """Refuse to execute in an attempt containing any generated artifact."""

    existing = tuple(path for path in paths.generated_paths() if path.exists())
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite attempt artifacts: {names}")


def _write_text_exclusive(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def write_jsonl_exclusive(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write canonical JSON Lines, including a valid empty artifact."""

    target = Path(path)
    lines = tuple(canonical_json(_json_safe(dict(row))) for row in rows)
    text = "" if not lines else "\n".join(lines) + "\n"
    return _write_text_exclusive(target, text)


def ensure_solver_log(path: str | Path) -> Path:
    """Create an empty log only when the backend did not create one."""

    target = Path(path)
    if target.exists():
        return target
    return _write_text_exclusive(target, "")


def _safe_attribute(value: object, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return default


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    # Gurobi's infinity sentinel is 1e100.
    if not math.isfinite(number) or abs(number) >= 0.99e100:
        return None
    return number


def _integer(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) and abs(value) < 0.99e100 else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return str(value)


def enumerate_public_gurobi_parameter_names() -> tuple[str, ...]:
    """Enumerate the installed public parameter catalog through ``GRB.Param``.

    Importing the constant namespace does not create or start a Gurobi
    environment.  Effective values are read later from the explicitly owned
    model through its public ``getParamInfo`` API.
    """

    from gurobipy import GRB

    names = {
        value
        for attribute in dir(GRB.Param)
        if not attribute.startswith("_")
        and isinstance((value := getattr(GRB.Param, attribute)), str)
        and value
    }
    if not names:
        raise GurobiParameterAuditError("public Gurobi parameter catalog is empty")
    folded = {name.casefold() for name in names}
    if len(folded) != len(names):
        raise GurobiParameterAuditError(
            "public Gurobi parameter catalog contains case-insensitive duplicates"
        )
    return tuple(sorted(names))


def _parameter_type_name(parameter_type: object) -> str:
    if parameter_type is int:
        return "int"
    if parameter_type is float:
        return "float"
    if parameter_type is str:
        return "str"
    raise GurobiParameterAuditError("Gurobi returned an unsupported parameter type")


def _plain_parameter_scalar(value: object, value_type: str) -> Any:
    """Return a canonical scalar without collapsing Gurobi infinity sentinels."""

    if value_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise GurobiParameterAuditError("Gurobi returned an invalid integer value")
        return value
    if value_type == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GurobiParameterAuditError("Gurobi returned an invalid float value")
        number = float(value)
        if math.isnan(number):
            raise GurobiParameterAuditError("Gurobi returned a NaN parameter value")
        if number >= 0.99e100:
            return {"sentinel": "GRB_INFINITY"}
        if number <= -0.99e100:
            return {"sentinel": "GRB_NEGATIVE_INFINITY"}
        if not math.isfinite(number):
            raise GurobiParameterAuditError(
                "Gurobi returned a noncanonical infinite parameter value"
            )
        return number
    if value_type == "str":
        if not isinstance(value, str):
            raise GurobiParameterAuditError("Gurobi returned an invalid string value")
        return value
    raise GurobiParameterAuditError("unknown canonical parameter value type")


def _parameter_is_set(value: object, value_type: str) -> bool:
    if value_type == "str":
        return isinstance(value, str) and bool(value.strip())
    if value_type == "int":
        return not isinstance(value, bool) and isinstance(value, int) and value != 0
    if value_type == "float":
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) != 0.0
        )
    return value is not None


def _parameter_sensitivity(parameter_name: str) -> str:
    """Classify known and future credential-like names before serialization."""

    if parameter_name in _GUROBI_SECRET_PARAMETERS:
        return "secret_presence"
    if parameter_name in _GUROBI_IDENTITY_PARAMETERS:
        return "identity_hash"
    folded = parameter_name.casefold()
    if "secret" in folded or "password" in folded:
        return "secret_presence"
    if "token" in folded and parameter_name not in {
        "WLSTokenDuration",
        "WLSTokenRefresh",
    }:
        return "secret_presence"
    if any(marker in folded for marker in ("accessid", "username", "licenseid")):
        return "identity_hash"
    return "plain"


def _parameter_identity_hash(parameter_name: str, value: object) -> str:
    return canonical_sha256(
        {
            "domain": GUROBI_PARAMETER_IDENTITY_HASH_DOMAIN,
            "parameter_name": parameter_name,
            "value": value,
        }
    )


def _redacted_parameter_value(
    parameter_name: str,
    value: object,
    value_type: str,
) -> dict[str, Any] | Any:
    sensitivity = _parameter_sensitivity(parameter_name)
    if sensitivity == "secret_presence":
        return {
            "is_set": _parameter_is_set(value, value_type),
            "redacted": True,
        }
    if sensitivity == "identity_hash":
        return {
            "redacted": True,
            "sha256": _parameter_identity_hash(parameter_name, value),
        }
    return _plain_parameter_scalar(value, value_type)


def _effective_parameter_classification(parameter_name: str) -> str:
    if parameter_name in CORE_SOLVER_ALLOWED_GUROBI_PARAMETERS:
        return "core_solver_allowed"
    if parameter_name in BENCHMARK_OPERATIONAL_GUROBI_PARAMETERS:
        return "benchmark_operational"
    if parameter_name in GUROBI_RUNTIME_CONTEXT_PARAMETERS:
        return "runtime_context"
    return "unallowed"


def _incomplete_gurobi_parameter_audit(reason_code: str) -> dict[str, Any]:
    return {
        "schema_version": GUROBI_PARAMETER_AUDIT_SCHEMA_VERSION,
        "enumeration_complete": False,
        "reason_code": reason_code,
        "enumerated_parameter_count": 0,
        "parameter_catalog_sha256": None,
        "effective_nondefault_parameters": {},
        "unallowed_nondefault_parameters": [],
        "execution_context": {
            "execution_mode": "unobserved",
            "observation_basis": "gurobi_effective_parameters",
            "remote_identity_sha256": None,
            "remote_mechanisms": [],
        },
    }


def _remote_execution_context(
    parameter_records: Mapping[str, tuple[object, object, str]],
) -> dict[str, Any]:
    """Infer local/remote execution solely from effective public parameters."""

    def active(parameter_name: str) -> bool:
        record = parameter_records.get(parameter_name)
        if record is None:
            return False
        current, default, value_type = record
        if current == default:
            return False
        if value_type == "str":
            return _parameter_is_set(current, value_type)
        return True

    mechanism_parameters = {
        "compute_server": {
            "ComputeServer",
            "ServerPassword",
            "ServerTimeout",
        },
        "cluster_manager": {
            "CSAPIAccessID",
            "CSAPISecret",
            "CSAppName",
            "CSAuthToken",
            "CSBatchMode",
            "CSClientLog",
            "CSGroup",
            "CSIdleTimeout",
            "CSManager",
            "CSPriority",
            "CSQueueTimeout",
            "CSRouter",
            "CSTLSInsecure",
        },
        "instant_cloud": {
            "CloudAccessID",
            "CloudHost",
            "CloudPool",
            "CloudSecretKey",
        },
        "distributed_worker_pool": {
            "WorkerPassword",
            "WorkerPool",
        },
        "remote_job": {"JobID"},
    }
    mechanisms = tuple(
        sorted(
            mechanism
            for mechanism, names in mechanism_parameters.items()
            if any(active(name) for name in names)
        )
    )
    if not mechanisms:
        return {
            "execution_mode": "local",
            "observation_basis": "gurobi_effective_parameters",
            "remote_identity_sha256": None,
            "remote_mechanisms": [],
        }

    remote_identity_parameters = {
        "CSAPIAccessID",
        "CSGroup",
        "CSManager",
        "CSRouter",
        "CloudAccessID",
        "CloudHost",
        "CloudPool",
        "ComputeServer",
        "WorkerPool",
    }
    identity_components = {
        name: _parameter_identity_hash(name, parameter_records[name][0])
        for name in sorted(remote_identity_parameters)
        if active(name)
    }
    if not identity_components:
        raise GurobiParameterAuditError(
            "remote Gurobi execution is active but its identity is unobservable"
        )
    return {
        "execution_mode": "remote",
        "observation_basis": "gurobi_effective_parameters",
        "remote_identity_sha256": canonical_sha256(
            {
                "domain": GUROBI_REMOTE_IDENTITY_HASH_DOMAIN,
                "identity_components": identity_components,
                "remote_mechanisms": list(mechanisms),
            }
        ),
        "remote_mechanisms": list(mechanisms),
    }


def collect_gurobi_parameter_audit(
    model_wrapper: object | None,
    *,
    strict: bool = False,
    parameter_name_provider: GurobiParameterNameProvider | None = None,
) -> dict[str, Any]:
    """Audit every installed Gurobi parameter without leaking credentials.

    In strict mode, used by formal execution, an unavailable/incomplete public
    catalog or an unidentifiable remote connection raises and therefore fails
    the attempt closed.  Ordinary runs receive an explicit incomplete payload
    so legacy injected model factories remain compatible.
    """

    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")

    def fail(reason_code: str, error: BaseException | None = None) -> dict[str, Any]:
        if strict:
            if isinstance(error, GurobiParameterAuditError):
                raise error
            raise GurobiParameterAuditError(
                f"Gurobi parameter audit failed: {reason_code}"
            ) from error
        return _incomplete_gurobi_parameter_audit(reason_code)

    if model_wrapper is None:
        return fail("model_unavailable")
    backend = _safe_attribute(model_wrapper, "model")
    get_parameter_info = _safe_attribute(backend, "getParamInfo")
    if not callable(get_parameter_info):
        return fail("get_param_info_unavailable")

    try:
        provider = parameter_name_provider or enumerate_public_gurobi_parameter_names
        provided_names = tuple(provider())
        if not provided_names:
            raise GurobiParameterAuditError("Gurobi parameter catalog is empty")
        if any(not isinstance(name, str) or not name for name in provided_names):
            raise GurobiParameterAuditError(
                "Gurobi parameter catalog contains an invalid name"
            )
        names = tuple(sorted(set(provided_names)))
        if len(names) != len(provided_names):
            raise GurobiParameterAuditError(
                "Gurobi parameter catalog contains duplicate names"
            )
        if len({name.casefold() for name in names}) != len(names):
            raise GurobiParameterAuditError(
                "Gurobi parameter catalog contains case-insensitive duplicates"
            )

        parameter_records: dict[str, tuple[object, object, str]] = {}
        catalog_entries: list[dict[str, Any]] = []
        effective: dict[str, dict[str, Any]] = {}
        unallowed: list[str] = []
        for requested_name in names:
            info = get_parameter_info(requested_name)
            if not isinstance(info, tuple) or len(info) != 6:
                raise GurobiParameterAuditError(
                    "Gurobi returned malformed parameter information"
                )
            canonical_name, parameter_type, current, _minimum, _maximum, default = info
            if canonical_name != requested_name:
                raise GurobiParameterAuditError(
                    "Gurobi returned a mismatched canonical parameter name"
                )
            value_type = _parameter_type_name(parameter_type)
            # Validate even default-valued parameters before incorporating them
            # into the catalog digest.
            current_safe = _redacted_parameter_value(
                canonical_name,
                current,
                value_type,
            )
            default_safe = _redacted_parameter_value(
                canonical_name,
                default,
                value_type,
            )
            parameter_records[canonical_name] = (current, default, value_type)
            catalog_entries.append(
                {
                    "default": default_safe,
                    "name": canonical_name,
                    "value_type": value_type,
                }
            )
            if current == default:
                continue
            classification = _effective_parameter_classification(canonical_name)
            effective[canonical_name] = {
                "classification": classification,
                "current": current_safe,
                "default": default_safe,
                "value_type": value_type,
            }
            if classification == "unallowed":
                unallowed.append(canonical_name)

        execution_context = _remote_execution_context(parameter_records)
        return {
            "schema_version": GUROBI_PARAMETER_AUDIT_SCHEMA_VERSION,
            "enumeration_complete": True,
            "reason_code": None,
            "enumerated_parameter_count": len(names),
            "parameter_catalog_sha256": canonical_sha256(
                {
                    "domain": GUROBI_PARAMETER_CATALOG_HASH_DOMAIN,
                    "parameters": catalog_entries,
                }
            ),
            "effective_nondefault_parameters": effective,
            "unallowed_nondefault_parameters": sorted(unallowed),
            "execution_context": execution_context,
        }
    except MemoryError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        return fail("parameter_enumeration_failed", error)


def relative_gap(incumbent: float | None, bound: float | None) -> float | None:
    if incumbent is None or bound is None:
        return None
    return abs(incumbent - bound) / max(1e-10, abs(incumbent))


def collect_solver_metrics(model_wrapper: object | None) -> dict[str, Any]:
    """Collect an outcome without assuming that an incumbent exists."""

    if model_wrapper is None:
        return {
            "status": None,
            "status_name": "NOT_CREATED",
            "has_incumbent": False,
            "solution_count": 0,
            "objective": None,
            "best_bound": None,
            "relative_gap": None,
            "node_count": None,
            "main_solve_runtime_seconds": None,
            "total_algorithm_runtime_seconds": None,
        }

    backend = _safe_attribute(model_wrapper, "model")
    status_value = _safe_attribute(backend, "Status")
    status = _integer(status_value, default=-1) if status_value is not None else None
    solution_count = _integer(_safe_attribute(backend, "SolCount"), default=0)
    has_incumbent = solution_count > 0
    objective = (
        _finite_number(_safe_attribute(backend, "ObjVal")) if has_incumbent else None
    )
    bound = _finite_number(_safe_attribute(backend, "ObjBound"))
    reported_gap = (
        _finite_number(_safe_attribute(backend, "MIPGap")) if has_incumbent else None
    )
    gap = reported_gap if reported_gap is not None else relative_gap(objective, bound)
    main_runtime = _finite_number(
        _safe_attribute(
            model_wrapper,
            "main_solve_runtime",
            _safe_attribute(backend, "Runtime"),
        )
    )
    total_runtime = _finite_number(
        _safe_attribute(model_wrapper, "total_runtime", main_runtime)
    )
    metrics = {
        "status": status,
        "status_name": _STATUS_NAMES.get(status, "UNKNOWN") if status else "UNKNOWN",
        "has_incumbent": has_incumbent,
        "solution_count": solution_count,
        "objective": objective,
        "best_bound": bound,
        "relative_gap": gap,
        "node_count": _finite_number(_safe_attribute(backend, "NodeCount")),
        "main_solve_runtime_seconds": main_runtime,
        "total_algorithm_runtime_seconds": total_runtime,
        "algorithm_prepare_runtime_seconds": _finite_number(
            _safe_attribute(model_wrapper, "algorithm_prepare_runtime")
        ),
        "start_preparation_runtime_seconds": _finite_number(
            _safe_attribute(model_wrapper, "start_preparation_runtime")
        ),
        "objective_partial_start": _safe_attribute(
            model_wrapper,
            "objective_partial_start_telemetry",
            {},
        ),
        "installed_mip_start_count": _integer(
            _safe_attribute(backend, "NumStart"), default=0
        ),
        "external_start_installed": bool(
            _safe_attribute(model_wrapper, "external_start_installed", False)
        ),
        "external_start_source": _safe_attribute(
            model_wrapper, "external_start_source"
        ),
        "external_start_matched_variables": _integer(
            _safe_attribute(model_wrapper, "external_start_matched_variables"),
            default=0,
        ),
        "fixed_integer_repair_runtime_seconds": _finite_number(
            _safe_attribute(model_wrapper, "fixed_integer_repair_runtime")
        ),
        "fixed_integer_repair_found": bool(
            _safe_attribute(model_wrapper, "fixed_integer_repair_found", False)
        ),
        "fixed_integer_repair_objective": _finite_number(
            _safe_attribute(model_wrapper, "fixed_integer_repair_objective")
        ),
        "fixed_integer_repair_status": _safe_attribute(
            model_wrapper, "fixed_integer_repair_status"
        ),
        "fixed_integer_repair_solution_count": int(
            _safe_attribute(model_wrapper, "fixed_integer_repair_solution_count", 0)
        ),
        "fixed_integer_repair_fixed_count": int(
            _safe_attribute(model_wrapper, "fixed_integer_repair_fixed_count", 0)
        ),
        "network_start_runtime_seconds": _finite_number(
            _safe_attribute(model_wrapper, "network_start_runtime")
        ),
        "network_construction_runtime_seconds": _finite_number(
            _safe_attribute(model_wrapper, "network_construction_runtime")
        ),
        "local_branching_runtime_seconds": _finite_number(
            _safe_attribute(model_wrapper, "local_branching_runtime")
        ),
        "callback_runtime_seconds": _finite_number(
            _safe_attribute(model_wrapper, "callback_runtime")
        ),
        "first_incumbent_time_seconds": _finite_number(
            _safe_attribute(model_wrapper, "first_incumbent_time")
        ),
        "first_incumbent_objective": _finite_number(
            _safe_attribute(model_wrapper, "first_incumbent_objective")
        ),
        "root_bound_before_cuts": _finite_number(
            _safe_attribute(model_wrapper, "root_bound_before_cuts")
        ),
        "root_bound_after_cuts": _finite_number(
            _safe_attribute(model_wrapper, "root_bound_after_cuts")
        ),
        "network_start_found": bool(
            _safe_attribute(model_wrapper, "network_start_found", False)
        ),
        "network_start_used": bool(
            _safe_attribute(model_wrapper, "network_start_used", False)
        ),
        "network_start_open_h_count": int(
            _safe_attribute(model_wrapper, "network_start_open_h_count", 0)
        ),
        "network_start_open_facility_count": int(
            _safe_attribute(model_wrapper, "network_start_open_facility_count", 0)
        ),
        "network_start_service_per_hour": _finite_number(
            _safe_attribute(model_wrapper, "network_start_service_per_hour")
        ),
        "network_start_relevant_chi_count": int(
            _safe_attribute(model_wrapper, "network_start_relevant_chi_count", 0)
        ),
        "local_branching_improved": bool(
            _safe_attribute(model_wrapper, "local_branching_improved", False)
        ),
    }
    return cast(dict[str, Any], _json_safe(metrics))


def collect_progress_rows(
    model_wrapper: object | None,
    config: TelemetryConfig,
) -> tuple[dict[str, Any], ...]:
    events = (
        _safe_attribute(model_wrapper, "progress_events", ()) if model_wrapper else ()
    )
    rows: list[dict[str, Any]] = []
    for event in events or ():
        row = _json_safe(dict(event))
        if not config.capture_incumbent_events and row.get("event") == "incumbent":
            continue
        rows.append(row)
    return tuple(rows)


def collect_root_round_rows(
    model_wrapper: object | None,
    config: TelemetryConfig,
) -> tuple[dict[str, Any], ...]:
    if not config.capture_root_rounds or model_wrapper is None:
        return ()
    events = _safe_attribute(model_wrapper, "root_round_events", ()) or ()
    return tuple(_json_safe(dict(event)) for event in events)


def collect_cut_stats(
    model_wrapper: object | None,
    config: TelemetryConfig,
) -> dict[str, Any]:
    if not config.capture_cut_stats or model_wrapper is None:
        return {"captured": False}

    def mapping(name: str) -> dict[str, Any]:
        value = _safe_attribute(model_wrapper, name, {}) or {}
        return cast(dict[str, Any], _json_safe(dict(sorted(dict(value).items()))))

    static_added = mapping("static_cut_counts")
    user_added = mapping("user_cut_counts")
    candidates = mapping("user_cut_candidate_counts")
    duplicates = mapping("user_cut_duplicate_counts")
    rejections = mapping("user_cut_rejection_counts")
    separation = mapping("user_cut_separation_seconds")
    route_bundle_index_counts = mapping("route_bundle_index_counts")
    return {
        "captured": True,
        "static_rows_added_by_family": static_added,
        "user_cuts_added_by_family": user_added,
        "user_cut_candidates_by_family": candidates,
        "user_cut_duplicates_by_family": duplicates,
        "user_cut_rejections_by_family": rejections,
        "user_cut_separation_seconds_by_family": separation,
        "route_bundle_index_counts": route_bundle_index_counts,
        "static_rows_added_total": sum(
            _integer(value) for value in static_added.values()
        ),
        "user_cuts_added_total": sum(_integer(value) for value in user_added.values()),
        "root_cut_rounds": _integer(
            _safe_attribute(model_wrapper, "root_cut_rounds"), default=0
        ),
        "callback_runtime_seconds": _finite_number(
            _safe_attribute(model_wrapper, "callback_runtime")
        ),
    }


def collect_applied_gurobi_parameters(model_wrapper: object | None) -> dict[str, Any]:
    """Read the final backend parameter values when a model was created."""

    if model_wrapper is None:
        return {}
    backend = _safe_attribute(model_wrapper, "model")
    parameters = _safe_attribute(backend, "Params")
    if parameters is None:
        return {}
    applied: dict[str, Any] = {}
    for output_name, attribute in APPLIED_GUROBI_PARAMETER_ATTRIBUTES:
        value = _safe_attribute(parameters, attribute)
        if value is not None:
            applied[output_name] = _json_safe(value)
    return applied


def exception_payload(error: BaseException) -> dict[str, Any]:
    """Return a structured error while retaining a debugging traceback."""

    error_type = type(error)
    return {
        "exception_type": f"{error_type.__module__}.{error_type.__qualname__}",
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(error_type, error, error.__traceback__)
        ),
    }


def write_telemetry_artifacts(
    paths: ArtifactPaths,
    model_wrapper: object | None,
    config: TelemetryConfig,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    """Extract and exclusively write progress, root-round, and cut artifacts."""

    progress = collect_progress_rows(model_wrapper, config)
    root_rounds = collect_root_round_rows(model_wrapper, config)
    cut_stats = collect_cut_stats(model_wrapper, config)
    write_jsonl_exclusive(paths.progress, progress)
    write_jsonl_exclusive(paths.root_rounds, root_rounds)
    write_canonical_json_exclusive(paths.cut_stats, cut_stats)
    return progress, root_rounds, cut_stats


__all__ = [
    "APPLIED_GUROBI_PARAMETER_ATTRIBUTES",
    "ArtifactPaths",
    "BENCHMARK_OPERATIONAL_GUROBI_PARAMETERS",
    "CORE_SOLVER_ALLOWED_GUROBI_PARAMETERS",
    "GUROBI_PARAMETER_AUDIT_SCHEMA_VERSION",
    "GUROBI_RUNTIME_CONTEXT_PARAMETERS",
    "GurobiParameterAuditError",
    "GurobiParameterNameProvider",
    "assert_fresh_artifact_paths",
    "collect_applied_gurobi_parameters",
    "collect_gurobi_parameter_audit",
    "collect_cut_stats",
    "collect_progress_rows",
    "collect_root_round_rows",
    "collect_solver_metrics",
    "ensure_solver_log",
    "enumerate_public_gurobi_parameter_names",
    "exception_payload",
    "relative_gap",
    "write_jsonl_exclusive",
    "write_telemetry_artifacts",
]
