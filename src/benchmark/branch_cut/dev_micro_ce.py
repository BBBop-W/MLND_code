"""Fail-closed development runner for the five-second micro C/E pilot.

This module is deliberately separate from the formal benchmark scheduler.  It
exists only to decide whether the lightweight objective-aware partial start is
worth promoting into the immutable A--E protocol.  It never changes the model,
instances, or formal plan, and its artifacts are explicitly development-only.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import concurrent.futures
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from main import FORMULATION_IMPLEMENTATION_FILES, REPO_ROOT as MAIN_REPO_ROOT
from src.model_contract import FORMULATION_ID, RESULT_SCHEMA_VERSION
from src.model.primal_heuristic import (
    OBJECTIVE_START_PROXY_ENERGY_REL_TOLERANCE,
    OBJECTIVE_START_PROXY_QOS_ABS_TOLERANCE,
    OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE,
)

from .artifacts import write_canonical_json_exclusive
from .case_builder import canonical_instance_hash
from .qualification import QualificationPolicy, objective_margin, service_tolerance
from .resource_lock import (
    DEFAULT_BENCHMARK_RESOURCE_LOCK,
    BenchmarkResourceLock,
    BenchmarkResourceLockError,
)


EXPECTED_FORMULATION_SHA256 = (
    "6575f7ad5af2e62ba2c73f141ccd3f0b39d7c90a5e08868ac439d5b7d3531720"
)
REQUIRED_BASELINE_COMMIT = "f8849384058f86f4da9e39337b91d0697462f42f"
FORMULATION_RELATIVE_PATH = Path("MLND/main.tex")
DEFAULT_FORMULATION_SOURCE = MAIN_REPO_ROOT / FORMULATION_RELATIVE_PATH
IMPLEMENTATION_RELATIVE_PATHS = tuple(
    path.relative_to(MAIN_REPO_ROOT) for path in FORMULATION_IMPLEMENTATION_FILES
)
if len(IMPLEMENTATION_RELATIVE_PATHS) != 9:  # pragma: no cover - import-time guard
    raise RuntimeError("development C/E protocol expects exactly nine executable files")

SEEDS = (11, 29, 47)
PROFILES = ("C", "E")
TIME_LIMIT_SECONDS = 5.0
START_BUDGET_SECONDS = 0.25
MAX_CONCURRENCY = 2
ROOT_CUT_PASSES = 2
FEEDER_RIDER_COST_FACTOR = 0.30
OBJECTIVE_RELATIVE_TOLERANCE = 1e-8
OBJECTIVE_ABSOLUTE_TOLERANCE = 1e-7
GAP_CONSISTENCY_TOLERANCE = 1e-7
TIME_BUDGET_TOLERANCE_SECONDS = 0.10
PREPARATION_TELEMETRY_TOLERANCE_SECONDS = 0.001
OBJECTIVE_PARTIAL_POLICY_VERSION = "objective_partial_v3_system_energy"
ACTIVE_POLICY = QualificationPolicy()


class DevMicroCEError(RuntimeError):
    """The development pilot cannot proceed without weakening its contract."""


@dataclass(frozen=True, slots=True)
class InstanceSpec:
    stage: str
    relative_path: Path
    expected_sha256: str
    candidate_h: int


INSTANCE_SPECS = {
    "H4": InstanceSpec(
        stage="H4",
        relative_path=Path(
            "results/branch_cut_paper/dev/micro_crop_matrix_v2/"
            "nanshan_d04_05_noon_12_13/H004_L02_E02_R02p00"
        ),
        expected_sha256=(
            "7a6f524f9a77d395e8e6bd455e752a1d81494137b4dca1d0af105af0ae8d5f11"
        ),
        candidate_h=4,
    ),
    "H6": InstanceSpec(
        stage="H6",
        relative_path=Path(
            "results/branch_cut_paper/dev/micro_crop_matrix_v2/"
            "nanshan_d04_05_noon_12_13/H006_L02_E02_R02p20"
        ),
        expected_sha256=(
            "d6d70495f240c12b35ad3b1fc6d817c7f7b3948a8c62e5662f21fb77790e7617"
        ),
        candidate_h=6,
    ),
    "H9": InstanceSpec(
        stage="H9",
        relative_path=Path(
            "results/branch_cut_paper/dev/micro_crop_matrix_v1/"
            "nanshan_d04_05_noon_12_13/H009_L02_E02_R02p50"
        ),
        expected_sha256=(
            "2a471ca655e0c3720dc5c165896364f47b63117ed6adaa6cac2aac0b7af36798"
        ),
        candidate_h=9,
    ),
    "H10": InstanceSpec(
        stage="H10",
        relative_path=Path(
            "results/branch_cut_paper/dev/micro_crop_matrix_v1/"
            "nanshan_d04_05_noon_12_13/H010_L02_E02_R02p60"
        ),
        expected_sha256=(
            "f0976a770550c14dbafb0b515d962b7ab936b19e2e123938afd8d4cb6c6428f3"
        ),
        candidate_h=10,
    ),
}


@dataclass(frozen=True, slots=True)
class BatchConfig:
    repo_root: Path
    output_root: Path
    formulation_source: Path | None = None
    expected_formulation_sha256: str = EXPECTED_FORMULATION_SHA256
    requested_stages: tuple[str, ...] | None = None
    include_h10: bool = False
    confirm_h10: bool = False
    jobs: int = MAX_CONCURRENCY
    resource_lock: Path = DEFAULT_BENCHMARK_RESOURCE_LOCK

    @property
    def stages(self) -> tuple[str, ...]:
        if self.requested_stages is not None:
            return tuple(self.requested_stages)
        return ("H9", "H10") if self.include_h10 else ("H9",)


@dataclass(frozen=True, slots=True)
class FrozenInputs:
    commit: str
    branch: str
    implementation_sha256: str
    implementation_files: Mapping[str, str]
    formulation_source: str
    formulation_sha256: str
    instances: Mapping[str, Mapping[str, Any]]
    process_table_sha256: str


@dataclass(frozen=True, slots=True)
class JobSpec:
    stage: str
    profile: str
    seed: int
    instance_dir: Path
    results_dir: Path
    driver_log: Path
    cache_dir: Path

    @property
    def job_id(self) -> str:
        return f"{self.stage}-{self.profile}-seed{self.seed:03d}"


ProcessRunner = Callable[..., Any]
_NO_INCUMBENT_PATTERN = re.compile(
    r"MX finished without a feasible incumbent \(status=(?P<status>-?\d+)\)"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_provenance(repo_root: Path) -> tuple[str, dict[str, str]]:
    files: dict[str, str] = {}
    for relative in IMPLEMENTATION_RELATIVE_PATHS:
        path = _require_real_path(repo_root / relative, kind="executable file")
        if not path.is_file():
            raise DevMicroCEError(f"executable path is not a file: {path}")
        files[relative.as_posix()] = _sha256_file(path)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), files


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise DevMicroCEError(f"git {' '.join(arguments)} failed: {message}")
    return completed.stdout.strip()


def _require_baseline_ancestor(repo_root: Path, commit: str) -> None:
    """Require the handed-off f884 baseline to be in the executed history."""

    try:
        _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            REQUIRED_BASELINE_COMMIT,
            commit,
        )
    except DevMicroCEError as error:
        raise DevMicroCEError(
            "development pilot HEAD must descend from the frozen f884 baseline "
            f"{REQUIRED_BASELINE_COMMIT}"
        ) from error


def _process_table() -> str:
    completed = subprocess.run(
        ("ps", "-axo", "pid=,ppid=,command="),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise DevMicroCEError(
            "cannot audit concurrent solver processes: "
            + (completed.stderr or completed.stdout).strip()
        )
    return completed.stdout


def _conflicting_processes(process_table: str) -> tuple[str, ...]:
    markers = (
        "main.py solve",
        "main.py run",
        "branch_cut qualify",
        "branch_cut run ",
        "branch_cut run-formal",
        "mlnd_qualification",
        "gurobi_cl",
        "pytest",
    )
    own_pid = os.getpid()
    conflicts: list[str] = []
    for raw_line in process_table.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=2)
        try:
            pid = int(fields[0])
        except (IndexError, ValueError):
            continue
        if pid == own_pid:
            continue
        if any(marker in line for marker in markers):
            conflicts.append(line)
    return tuple(conflicts)


def _require_real_path(path: Path, *, kind: str) -> Path:
    absolute = path.expanduser().absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise DevMicroCEError(
            f"{kind} does not resolve: {absolute}: {error}"
        ) from error
    if absolute != resolved or absolute.is_symlink():
        raise DevMicroCEError(f"{kind} must be a real non-symlink path: {absolute}")
    return resolved


def _require_regular_file(path: Path, *, kind: str) -> Path:
    resolved = _require_real_path(path, kind=kind)
    if not resolved.is_file():
        raise DevMicroCEError(f"{kind} must be a regular non-symlink file: {resolved}")
    return resolved


def _require_real_output_path(path: Path, *, kind: str) -> Path:
    """Validate a prospective artifact path without following a final symlink."""

    absolute = path.expanduser().absolute()
    if absolute.is_symlink():
        raise DevMicroCEError(f"{kind} must not be a symlink: {absolute}")
    parent = _require_real_path(absolute.parent, kind=f"{kind} parent")
    if parent != absolute.parent:
        raise DevMicroCEError(f"{kind} parent must not traverse a symlink: {absolute}")
    if absolute.exists():
        resolved = _require_real_path(absolute, kind=kind)
        if resolved != absolute:
            raise DevMicroCEError(f"{kind} must not traverse a symlink: {absolute}")
    return absolute


def _canonical_lock_path(path: Path) -> Path:
    """Canonicalize the platform temp parent without following the lock entry."""

    absolute = path.expanduser().absolute()
    if absolute.is_symlink():
        return absolute
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as error:
        raise DevMicroCEError(
            f"benchmark resource-lock parent does not resolve: {absolute.parent}: {error}"
        ) from error
    return parent / absolute.name


def _require_shared_resource_lock(path: Path) -> Path:
    """Validate the exact shared lock before any lock-file mutation occurs."""

    configured_lock = _require_real_output_path(path, kind="benchmark resource lock")
    shared_lock = _canonical_lock_path(Path(DEFAULT_BENCHMARK_RESOURCE_LOCK))
    if configured_lock != shared_lock:
        raise DevMicroCEError(
            "development pilot must use the single shared benchmark resource lock: "
            f"{shared_lock}"
        )
    if configured_lock.exists() and not configured_lock.is_file():
        raise DevMicroCEError(
            f"benchmark resource lock must be a regular file: {configured_lock}"
        )
    return configured_lock


def _read_json_object(path: Path) -> dict[str, Any]:
    path = _require_regular_file(path, kind="JSON artifact")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DevMicroCEError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DevMicroCEError(f"expected JSON object: {path}")
    return payload


def _driver_no_incumbent_status(path: Path) -> int | None:
    """Recognize only main.py's exact terminal no-incumbent exception."""

    artifact = _require_regular_file(path, kind="driver-log artifact")
    try:
        with artifact.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 131_072))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError as error:
        raise DevMicroCEError(
            f"cannot read driver-log artifact {artifact}: {error}"
        ) from error
    matches = tuple(_NO_INCUMBENT_PATTERN.finditer(tail))
    return int(matches[-1].group("status")) if matches else None


def _clean_repository(repo_root: Path) -> tuple[str, str]:
    if _git(repo_root, "rev-parse", "--is-inside-work-tree") != "true":
        raise DevMicroCEError("repository root is not a Git worktree")
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise DevMicroCEError("development pilot requires a clean committed worktree")
    commit = _git(repo_root, "rev-parse", "--verify", "HEAD")
    if len(commit) != 40:
        raise DevMicroCEError("HEAD is not a full commit identity")
    branch = _git(repo_root, "branch", "--show-current")
    if not branch:
        raise DevMicroCEError("detached HEAD is not allowed for the development pilot")
    _require_baseline_ancestor(repo_root, commit)
    return commit, branch


def _freeze_inputs(config: BatchConfig) -> FrozenInputs:
    if config.jobs < 1 or config.jobs > MAX_CONCURRENCY:
        raise DevMicroCEError("jobs must be one or two")
    if config.requested_stages is not None and config.include_h10:
        raise DevMicroCEError("--stages and --include-h10 cannot be combined")
    stages = config.stages
    if not stages:
        raise DevMicroCEError("at least one micro stage must be selected")
    unknown_stages = tuple(stage for stage in stages if stage not in INSTANCE_SPECS)
    if unknown_stages:
        raise DevMicroCEError(
            "unknown micro stage selection: " + ", ".join(unknown_stages)
        )
    duplicate_stages = tuple(
        stage for index, stage in enumerate(stages) if stage in stages[:index]
    )
    if duplicate_stages:
        raise DevMicroCEError(
            "micro stages must be unique; repeated selection: "
            + ", ".join(dict.fromkeys(duplicate_stages))
        )
    if ("H10" in stages) != config.confirm_h10:
        raise DevMicroCEError(
            "H10 requires both an explicit H10 selection and --confirm-h10"
        )

    repo_root = _require_real_path(config.repo_root, kind="repository root")
    if not repo_root.is_dir():
        raise DevMicroCEError(f"repository root is not a directory: {repo_root}")
    output_root = config.output_root.expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise DevMicroCEError(f"output root must not exist: {output_root}")
    output_parent = _require_real_path(output_root.parent, kind="output parent")
    if output_root.parent.absolute() != output_parent:
        raise DevMicroCEError("output parent must not traverse a symlink")

    _require_shared_resource_lock(Path(config.resource_lock))

    commit, branch = _clean_repository(repo_root)
    implementation_sha256, implementation_files = _implementation_provenance(repo_root)

    if config.formulation_source is None:  # pragma: no cover - normalized by run_batch
        raise DevMicroCEError("formulation source was not normalized")
    source = _require_regular_file(config.formulation_source, kind="formulation source")
    expected_source = _require_regular_file(
        repo_root / FORMULATION_RELATIVE_PATH,
        kind="worktree formulation source",
    )
    if source != expected_source:
        raise DevMicroCEError(
            "formulation source must be this worktree's MLND/main.tex: "
            f"{expected_source}"
        )
    source_sha256 = _sha256_file(source)
    if source_sha256 != config.expected_formulation_sha256:
        raise DevMicroCEError(
            "formulation source hash mismatch: "
            f"expected {config.expected_formulation_sha256}, got {source_sha256}"
        )

    instances: dict[str, Mapping[str, Any]] = {}
    for stage in config.stages:
        spec = INSTANCE_SPECS[stage]
        instance_dir = _require_real_path(
            repo_root / spec.relative_path,
            kind=f"{stage} instance",
        )
        aggregate, files = canonical_instance_hash(instance_dir)
        if aggregate != spec.expected_sha256:
            raise DevMicroCEError(
                f"{stage} canonical hash mismatch: expected {spec.expected_sha256}, "
                f"got {aggregate}"
            )
        for filename in files:
            _require_regular_file(instance_dir / filename, kind=f"{stage} {filename}")
        instances[stage] = {
            "path": str(instance_dir),
            "sha256": aggregate,
            "canonical_file_sha256": files,
            "candidate_h": spec.candidate_h,
        }

    process_table = _process_table()
    conflicts = _conflicting_processes(process_table)
    if conflicts:
        raise DevMicroCEError(
            "conflicting solver/test processes are active: " + " | ".join(conflicts)
        )
    return FrozenInputs(
        commit=commit,
        branch=branch,
        implementation_sha256=implementation_sha256,
        implementation_files=implementation_files,
        formulation_source=str(source),
        formulation_sha256=source_sha256,
        instances=instances,
        process_table_sha256=hashlib.sha256(process_table.encode("utf-8")).hexdigest(),
    )


def _job_specs(config: BatchConfig, frozen: FrozenInputs) -> tuple[JobSpec, ...]:
    jobs: list[JobSpec] = []
    for stage in config.stages:
        instance_dir = Path(str(frozen.instances[stage]["path"]))
        for seed in SEEDS:
            for profile in PROFILES:
                stem = f"{stage}-{profile}-seed{seed:03d}"
                jobs.append(
                    JobSpec(
                        stage=stage,
                        profile=profile,
                        seed=seed,
                        instance_dir=instance_dir,
                        results_dir=config.output_root
                        / "runs"
                        / stage
                        / profile
                        / f"seed{seed:03d}",
                        driver_log=config.output_root / "driver_logs" / f"{stem}.log",
                        cache_dir=config.output_root / "cache" / stem,
                    )
                )
    return tuple(jobs)


def _job_command(config: BatchConfig, job: JobSpec) -> tuple[str, ...]:
    profile_flag = (
        "--dev-objective-partial-start"
        if job.profile == "E"
        else "--no-dev-objective-partial-start"
    )
    return (
        "/opt/anaconda3/bin/python",
        str(config.repo_root / "main.py"),
        "solve",
        "--instance-dir",
        str(job.instance_dir),
        "--results-dir",
        str(job.results_dir),
        "--modes",
        "MX",
        "--demand-scale",
        "1",
        "--display-origin-choices",
        "all",
        "--general-corridor-choices",
        "all",
        "--feeder-rider-cost-factor",
        "0.30",
        "--time-limit",
        "5",
        "--mip-gap",
        "0",
        "--threads",
        "1",
        "--start-mode",
        "none",
        "--network-start-seconds",
        "0",
        "--fixed-integer-repair-seconds",
        "0",
        "--no-local-branching",
        "--local-branching-seconds",
        "0",
        "--valid-inequalities",
        "--max-root-cut-rounds",
        "3",
        "--root-cut-passes",
        str(ROOT_CUT_PASSES),
        "--max-cuts-per-round",
        "250",
        "--cut-violation-tolerance",
        "1e-6",
        "--cut-efficacy-tolerance",
        "1e-9",
        "--static-od-vub-per-corridor",
        "0",
        "--no-nearest-display-suffix-rows",
        "--no-od-facility-vub-cuts",
        "--no-route-swap-station-vub-cuts",
        "--no-route-swap-leg-vub-cuts",
        "--root-method",
        "-1",
        "--miqcp-method",
        "-1",
        "--mip-focus",
        "1",
        "--heuristics",
        "0.2",
        "--no-rel-heur-seconds",
        "0",
        "--no-rel-heur-time",
        "0",
        "--presolve",
        "-1",
        "--native-cuts",
        "-1",
        "--numeric-focus",
        "0",
        "--collect-progress",
        "--progress-interval-seconds",
        "0.25",
        "--objective-partial-start-seconds",
        "0.25",
        "--objective-partial-start-max-candidates",
        "3",
        "--solver-seed",
        str(job.seed),
        profile_flag,
    )


def _controlled_environment(config: BatchConfig, job: JobSpec) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "MLND_FORMULATION_SOURCE": str(config.formulation_source),
            "MPLCONFIGDIR": str(job.cache_dir / "mpl"),
            "XDG_CACHE_HOME": str(job.cache_dir / "xdg"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def _open_binary_exclusive_nofollow(path: Path) -> Any:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    return os.fdopen(descriptor, "wb")


def _run_one_job(
    config: BatchConfig,
    job: JobSpec,
    process_runner: ProcessRunner,
) -> dict[str, Any]:
    if job.results_dir.exists() or job.results_dir.is_symlink():
        return {
            "job_id": job.job_id,
            "stage": job.stage,
            "profile": job.profile,
            "seed": job.seed,
            "returncode": None,
            "error": "results_directory_already_exists",
        }
    command = _job_command(config, job)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        _require_real_path(config.output_root, kind="development output root")
        job.results_dir.parent.mkdir(parents=True, exist_ok=True)
        job.driver_log.parent.mkdir(parents=True, exist_ok=True)
        _require_real_path(job.results_dir.parent, kind="job results parent")
        _require_real_path(job.driver_log.parent, kind="driver-log parent")
        job.cache_dir.mkdir(parents=True, exist_ok=False)
        _require_real_path(job.cache_dir, kind="job cache directory")
        with _open_binary_exclusive_nofollow(job.driver_log) as log_handle:
            completed = process_runner(
                command,
                cwd=config.repo_root,
                env=_controlled_environment(config, job),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        returncode = int(completed.returncode)
        error = None if returncode == 0 else f"subprocess_returncode_{returncode}"
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        returncode = None
        error = f"{type(exc).__name__}: {exc}"
    return {
        "job_id": job.job_id,
        "stage": job.stage,
        "profile": job.profile,
        "seed": job.seed,
        "returncode": returncode,
        "error": error,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "results_dir": str(job.results_dir),
        "driver_log": str(job.driver_log),
    }


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DevMicroCEError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DevMicroCEError(f"{label} must be finite")
    return result


def _optional_number(value: Any, label: str) -> float | None:
    return None if value is None else _number(value, label)


def _integer_field(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DevMicroCEError(f"{label} must be an integer")
    return value


def _scale_tolerance(*values: float) -> float:
    scale = max((abs(value) for value in values), default=0.0)
    return max(OBJECTIVE_ABSOLUTE_TOLERANCE, OBJECTIVE_RELATIVE_TOLERANCE * scale)


def _numbers_close(left: float, right: float) -> bool:
    return abs(left - right) <= _scale_tolerance(left, right)


def _optional_numbers_close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return _numbers_close(left, right)


def _reported_relative_gap(objective: float, bound: float) -> float:
    return abs(objective - bound) / max(1e-10, abs(objective))


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise DevMicroCEError(f"missing artifact field: {'.'.join(keys)}")
        value = value[key]
    return value


def _run_signature_payload(run_config: Mapping[str, Any]) -> dict[str, Any]:
    instance = _nested(run_config, "instance")
    return {
        "schema_version": run_config.get("schema_version"),
        "formulation_id": run_config.get("formulation_id"),
        "formulation_source": _nested(run_config, "formulation_source"),
        "implementation": _nested(run_config, "implementation"),
        "instance_dir": _nested(instance, "directory"),
        "canonical_file_sha256": _nested(instance, "canonical_file_sha256"),
        "demand_scale": run_config.get("demand_scale"),
        "modes": run_config.get("modes"),
        "cross_mode_start_policy": run_config.get("cross_mode_start_policy"),
        "warm_start_from": run_config.get("warm_start_from"),
        "file_warm_start_policy": run_config.get("file_warm_start_policy"),
        "screening_is_exact": run_config.get("screening_is_exact"),
        "model": _nested(run_config, "model"),
        "model_derived": _nested(run_config, "model_derived"),
        "solver": _nested(run_config, "solver"),
    }


def _computed_run_signature_sha256(run_config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _run_signature_payload(run_config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_run_config(
    job: JobSpec,
    run_config: Mapping[str, Any],
    frozen: FrozenInputs,
) -> tuple[str, ...]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(run_config.get("schema_version") == RESULT_SCHEMA_VERSION, "schema_version")
    require(run_config.get("formulation_id") == FORMULATION_ID, "formulation_id")
    signature = run_config.get("run_signature_sha256")
    require(
        isinstance(signature, str)
        and len(signature) == 64
        and signature == signature.lower(),
        "run_signature_sha256_format",
    )
    try:
        computed_signature = _computed_run_signature_sha256(run_config)
    except (TypeError, ValueError, DevMicroCEError):
        computed_signature = None
    require(signature == computed_signature, "run_signature_sha256_recomputation")
    require(
        _nested(run_config, "formulation_source", "sha256")
        == frozen.formulation_sha256,
        "formulation_sha256",
    )
    require(
        _nested(run_config, "formulation_source", "path") == frozen.formulation_source,
        "formulation_path",
    )
    require(
        _nested(run_config, "implementation", "sha256") == frozen.implementation_sha256,
        "implementation_sha256",
    )
    require(
        _nested(run_config, "implementation", "files")
        == dict(frozen.implementation_files),
        "implementation_files",
    )
    require(
        _nested(run_config, "instance", "directory") == str(job.instance_dir),
        "instance_directory",
    )
    require(
        _nested(run_config, "instance", "canonical_file_sha256")
        == frozen.instances[job.stage]["canonical_file_sha256"],
        "instance_files",
    )
    require(run_config.get("demand_scale") == 1.0, "demand_scale")
    require(run_config.get("modes") == ["MX"], "modes")
    require(
        _nested(run_config, "screening_is_exact", "MX") is True, "screening_is_exact"
    )
    require(
        run_config.get("cross_mode_start_policy") is None, "cross_mode_start_policy"
    )
    require(run_config.get("warm_start_from") is None, "warm_start_from")
    require(
        run_config.get("file_warm_start_policy") == "first_mode_only",
        "file_warm_start_policy",
    )
    model = _nested(run_config, "model")
    require(_nested(model, "display_origin_choices") is None, "display_origin_choices")
    require(
        _nested(model, "general_corridor_choices") is None, "general_corridor_choices"
    )
    require(
        _nested(model, "feeder_rider_cost_factor") == FEEDER_RIDER_COST_FACTOR,
        "feeder_rider_cost_factor",
    )

    solver = _nested(run_config, "solver")
    expected = {
        "time_limit": 5.0,
        "mip_gap": 0.0,
        "threads": 1,
        "output_flag": 1,
        "seed": job.seed,
        "network_start_seconds": 0.0,
        "network_start_method": 2,
        "use_valid_inequalities": True,
        "use_user_cuts": True,
        "max_root_cut_rounds": 3,
        "root_cut_passes": ROOT_CUT_PASSES,
        "max_cuts_per_round": 250,
        "cut_violation_tolerance": 1e-6,
        "cut_efficacy_tolerance": 1e-9,
        "fixed_integer_repair_seconds": 0.0,
        "use_local_branching": False,
        "local_branching_seconds": 0.0,
        "root_relaxation_method": -1,
        "miqcp_method": -1,
        "start_mode": "none",
        "use_objective_partial_start": job.profile == "E",
        "objective_partial_start_seconds": 0.25,
        "objective_partial_start_max_candidates": 3,
        "mip_focus": 1,
        "heuristics": 0.2,
        "no_rel_heur_seconds": 0.0,
        "no_rel_heur_time": 0.0,
        "presolve": -1,
        "cuts": -1,
        "numeric_focus": 0,
        "static_nearest_display_suffix": False,
        "static_chi_vub_top_k_per_corridor": 0,
        "cut_chi_facility_vub": False,
        "cut_route_swap_station_vub": False,
        "cut_route_swap_leg_vub": False,
        "collect_progress": True,
        "progress_interval_seconds": 0.25,
    }
    for key, expected_value in expected.items():
        require(solver.get(key) == expected_value, f"solver.{key}")
    return tuple(errors)


def _normalized_run_config_sha256(run_config: Mapping[str, Any]) -> str:
    """Hash a run config after removing the sole permitted C/E difference."""

    normalized = json.loads(json.dumps(run_config, sort_keys=True, allow_nan=False))
    normalized.pop("run_signature_sha256", None)
    solver = normalized.get("solver")
    if not isinstance(solver, dict):
        raise DevMicroCEError("run config solver must be an object")
    solver.pop("use_objective_partial_start", None)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _qualification_audit(job: JobSpec, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one schema-current incumbent without treating no-incumbent as I/O."""

    errors: list[str] = []
    if summary.get("schema_version") != RESULT_SCHEMA_VERSION:
        errors.append("summary_schema_version")
    if summary.get("formulation_id") != FORMULATION_ID:
        errors.append("summary_formulation_id")
    if summary.get("mode") != "MX":
        errors.append("summary_mode")

    solver = _nested(summary, "solver")
    solution_count = _integer_field(
        _nested(solver, "solution_count"), "solver.solution_count"
    )
    if solution_count <= 0:
        return {
            "valid": False,
            "outcome": "no_incumbent",
            "errors": [*errors, "no_incumbent"],
            "status_name": summary.get("status_name"),
            "objective": None,
            "objective_bound": _optional_number(
                summary.get("objective_bound"), "objective_bound"
            ),
            "mip_gap": None,
            "total_algorithm_runtime_seconds": _optional_number(
                summary.get("runtime_seconds"), "runtime_seconds"
            ),
            "main_solve_runtime_seconds": _optional_number(
                solver.get("main_solve_runtime_seconds"),
                "main_solve_runtime_seconds",
            ),
            "algorithm_prepare_runtime_seconds": _optional_number(
                solver.get("algorithm_prepare_runtime_seconds"),
                "algorithm_prepare_runtime_seconds",
            ),
            "start_preparation_runtime_seconds": _optional_number(
                solver.get("start_preparation_runtime_seconds"),
                "start_preparation_runtime_seconds",
            ),
            "first_incumbent_time_seconds": None,
            "node_count": _optional_number(solver.get("node_count"), "node_count"),
            "candidate_h": INSTANCE_SPECS[job.stage].candidate_h,
            "total_demand_per_hour": None,
        }

    objective = _number(summary.get("objective"), "objective")
    # Gurobi can return a feasible incumbent before it has established a
    # finite dual bound.  ResultProcessor persists that honest state as JSON
    # null for both ObjBound and MIPGap.  It is an algorithmic observation,
    # not a corrupt artifact or an infrastructure failure.
    bound = _optional_number(summary.get("objective_bound"), "objective_bound")
    gap = _optional_number(summary.get("mip_gap"), "mip_gap")
    runtime = _number(summary.get("runtime_seconds"), "runtime_seconds")
    baseline = _number(
        _nested(summary, "all_rider_baseline", "cost_per_hour"),
        "all_rider_baseline.cost_per_hour",
    )
    top_level_baseline = _number(
        summary.get("courier_only_baseline"), "courier_only_baseline"
    )
    demand = _number(
        _nested(summary, "demand_funnel_orders_per_hour", "total_demand"),
        "total_demand",
    )
    service = _number(
        _nested(summary, "demand_funnel_orders_per_hour", "total_drone_realized"),
        "total_drone_realized",
    )
    candidate_h = _integer_field(
        _nested(summary, "facilities", "candidates", "lockers"), "candidate_h"
    )
    open_h = _integer_field(_nested(summary, "facilities", "open", "lockers"), "open_h")
    reported_active_h = _integer_field(
        _nested(summary, "facilities", "active_lockers"), "active_lockers"
    )
    reported_idle_h = _integer_field(
        _nested(summary, "facilities", "idle_open_lockers"), "idle_open_lockers"
    )

    facility_path = _require_regular_file(
        job.results_dir / "MX" / "open_facilities.csv",
        kind="open-facilities artifact",
    )
    active_threshold = service_tolerance(demand)
    active_h = 0
    positive_cost_open = 0
    open_rows = 0
    with facility_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "type",
            "fixed_cost_per_hour",
            "incident_physical_flow_per_hour",
        }
        if reader.fieldnames is None or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            raise DevMicroCEError("open-facilities artifact has an invalid schema")
        for row in reader:
            open_rows += 1
            if _number(float(row["fixed_cost_per_hour"]), "facility fixed cost") > 0.0:
                positive_cost_open += 1
            if (
                row["type"] == "locker"
                and _number(
                    float(row["incident_physical_flow_per_hour"]),
                    "facility incident flow",
                )
                > active_threshold
            ):
                active_h += 1

    solver_objective = _number(
        _nested(solver, "objective_cost_per_hour"), "solver.objective"
    )
    solver_bound = _optional_number(
        _nested(solver, "objective_bound"), "solver.bound"
    )
    solver_gap = _optional_number(_nested(solver, "mip_gap"), "solver.gap")
    solver_runtime = _number(_nested(solver, "runtime_seconds"), "solver.runtime")
    main_runtime = _number(
        _nested(solver, "main_solve_runtime_seconds"), "main_solve_runtime_seconds"
    )
    prepare_runtime = _number(
        _nested(solver, "algorithm_prepare_runtime_seconds"),
        "algorithm_prepare_runtime_seconds",
    )
    start_runtime = _number(
        _nested(solver, "start_preparation_runtime_seconds"),
        "start_preparation_runtime_seconds",
    )
    first_incumbent = _optional_number(
        solver.get("first_incumbent_time_seconds"), "first_incumbent_time_seconds"
    )
    node_count = _number(_nested(solver, "node_count"), "node_count")

    if not _numbers_close(objective, solver_objective):
        errors.append("objective_summary_solver_mismatch")
    if not _optional_numbers_close(bound, solver_bound):
        errors.append("bound_summary_solver_mismatch")
    if not _optional_numbers_close(gap, solver_gap):
        errors.append("gap_summary_solver_mismatch")
    if abs(runtime - solver_runtime) > TIME_BUDGET_TOLERANCE_SECONDS:
        errors.append("runtime_summary_solver_mismatch")
    if not _numbers_close(baseline, top_level_baseline):
        errors.append("courier_baseline_summary_mismatch")
    if summary.get("status") != solver.get("status"):
        errors.append("status_summary_solver_mismatch")
    if summary.get("status_name") != solver.get("status_name"):
        errors.append("status_name_summary_solver_mismatch")
    recomputed_gap: float | None = None
    if bound is None:
        if gap is not None:
            errors.append("gap_present_without_finite_bound")
    else:
        objective_tolerance = _scale_tolerance(objective, bound)
        if bound > objective + objective_tolerance:
            errors.append("minimization_bound_exceeds_incumbent")
        recomputed_gap = _reported_relative_gap(objective, bound)
        if gap is None:
            errors.append("gap_missing_with_finite_bound")
        elif abs(gap - recomputed_gap) > GAP_CONSISTENCY_TOLERANCE:
            errors.append("reported_gap_inconsistent_with_objective_and_bound")

    time_fields = {
        "runtime": runtime,
        "main_runtime": main_runtime,
        "algorithm_prepare_runtime": prepare_runtime,
        "start_preparation_runtime": start_runtime,
        "node_count": node_count,
    }
    if any(value < 0.0 for value in time_fields.values()):
        errors.append("negative_runtime_or_node_field")
    component_runtime = main_runtime + prepare_runtime + start_runtime
    if component_runtime > runtime + TIME_BUDGET_TOLERANCE_SECONDS:
        errors.append("runtime_components_exceed_total")
    if runtime > TIME_LIMIT_SECONDS + TIME_BUDGET_TOLERANCE_SECONDS:
        errors.append("total_runtime_exceeds_fixed_budget")
    if first_incumbent is not None and (
        first_incumbent < 0.0
        or first_incumbent > main_runtime + TIME_BUDGET_TOLERANCE_SECONDS
    ):
        errors.append("invalid_first_incumbent_time")

    improvement = baseline - objective
    margin = objective_margin(baseline, ACTIVE_POLICY.optimality_tolerance)
    savings_fraction = improvement / baseline if baseline > 0.0 else -math.inf
    service_fraction = service / demand if demand > 0.0 else -math.inf
    open_fraction = open_h / candidate_h if candidate_h > 0 else math.inf
    if baseline <= 0.0:
        errors.append("nonpositive_courier_baseline")
    if improvement < margin:
        errors.append("objective_improvement_below_margin")
    if savings_fraction < ACTIVE_POLICY.minimum_savings_fraction:
        errors.append("savings_fraction_below_minimum")
    if service < active_threshold:
        errors.append("service_below_tolerance")
    if service_fraction < ACTIVE_POLICY.minimum_service_fraction:
        errors.append("service_fraction_below_minimum")
    if open_rows <= 0 or positive_cost_open <= 0:
        errors.append("no_positive_cost_open_facility")
    if not (
        ACTIVE_POLICY.minimum_h_open_fraction
        <= open_fraction
        <= ACTIVE_POLICY.maximum_h_open_fraction
    ):
        errors.append("h_open_fraction_outside_interior")
    if active_h <= 0:
        errors.append("no_active_h")
    if active_h != reported_active_h or summary.get("active_lockers") != active_h:
        errors.append("active_h_report_mismatch")
    if (
        reported_idle_h != open_h - active_h
        or summary.get("idle_open_lockers") != reported_idle_h
    ):
        errors.append("idle_open_h_report_mismatch")
    if summary.get("open_lockers") != open_h:
        errors.append("open_h_report_mismatch")
    if candidate_h != INSTANCE_SPECS[job.stage].candidate_h:
        errors.append("candidate_h_mismatch")
    if summary.get("network_is_nontrivial") is not True:
        errors.append("network_not_nontrivial")
    if summary.get("network_opened") is not True:
        errors.append("network_not_opened")
    if summary.get("courier_only_incumbent") is not False:
        errors.append("courier_only_incumbent")
    if (
        summary.get("screening_is_exact") is not True
        or _nested(summary, "instance", "screening_is_exact") is not True
    ):
        errors.append("summary_screening_not_exact")

    energy = _nested(summary, "energy_balance")
    if energy.get("balance_scope") != "system_wide_base_departure_plus_swap":
        errors.append("energy_balance_scope")
    energy_use = _number(energy.get("total_energy_use_equivalent_drones"), "energy_use")
    energy_supply = _number(
        energy.get("total_energy_supply_equivalent_drones"), "energy_supply"
    )
    energy_violation = _number(
        energy.get("maximum_constraint_violation"), "energy_violation"
    )
    if energy_use > energy_supply + _scale_tolerance(energy_use, energy_supply):
        errors.append("system_energy_use_exceeds_supply")
    if energy_violation > _scale_tolerance(energy_use, energy_supply):
        errors.append("system_energy_violation")

    return {
        "valid": not errors,
        "outcome": "incumbent",
        "errors": errors,
        "status_name": summary.get("status_name"),
        "objective": objective,
        "objective_bound": bound,
        "mip_gap": gap,
        "recomputed_mip_gap": recomputed_gap,
        "total_algorithm_runtime_seconds": runtime,
        "main_solve_runtime_seconds": main_runtime,
        "algorithm_prepare_runtime_seconds": prepare_runtime,
        "start_preparation_runtime_seconds": start_runtime,
        "first_incumbent_time_seconds": first_incumbent,
        "node_count": node_count,
        "courier_only_baseline": baseline,
        "objective_improvement": improvement,
        "objective_margin": margin,
        "savings_fraction": savings_fraction,
        "total_demand_per_hour": demand,
        "service_per_hour": service,
        "service_tolerance": active_threshold,
        "service_fraction": service_fraction,
        "candidate_h": candidate_h,
        "open_h": open_h,
        "open_h_fraction": open_fraction,
        "active_h": active_h,
        "idle_open_h": reported_idle_h,
        "positive_cost_open_facilities": positive_cost_open,
        "energy_balance_scope": energy.get("balance_scope"),
        "energy_use_equivalent_drones": energy_use,
        "energy_supply_equivalent_drones": energy_supply,
    }


def _telemetry_audit(
    job: JobSpec,
    summary: Mapping[str, Any],
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    algorithm = _nested(summary, "solver", "algorithm")
    telemetry = _nested(algorithm, "objective_partial_start")
    errors: list[str] = []
    solver = _nested(summary, "solver")
    if telemetry.get("policy_version") != OBJECTIVE_PARTIAL_POLICY_VERSION:
        errors.append("objective_partial_policy_version")
    if telemetry.get("proxy_only") is not True:
        errors.append("objective_partial_proxy_only_flag")
    expected_proxy_tolerances = {
        "qos_arrival_absolute": OBJECTIVE_START_PROXY_QOS_ABS_TOLERANCE,
        "system_energy_relative": OBJECTIVE_START_PROXY_ENERGY_REL_TOLERANCE,
        "construction_search_relative": (
            OBJECTIVE_START_PROXY_SEARCH_REL_TOLERANCE
        ),
        "solver_feasibility_tolerance_linked": False,
    }
    if telemetry.get("proxy_feasibility_tolerances") != expected_proxy_tolerances:
        errors.append("objective_partial_proxy_feasibility_tolerances")
    if algorithm.get("native_root_cut_passes") != ROOT_CUT_PASSES:
        errors.append("native_root_cut_passes")
    if algorithm.get("applied_native_root_cut_passes") != ROOT_CUT_PASSES:
        errors.append("applied_native_root_cut_passes")
    if algorithm.get("no_rel_heur_seconds") != 0.0:
        errors.append("no_rel_heur_seconds")
    if algorithm.get("no_rel_heur_time") != 0.0:
        errors.append("no_rel_heur_time")
    if algorithm.get("applied_no_rel_heur_time") != 0.0:
        errors.append("applied_no_rel_heur_time")
    if algorithm.get("external_start_installed") is not False:
        errors.append("external_start_installed")
    if algorithm.get("external_start_source") is not None:
        errors.append("external_start_source")
    if int(algorithm.get("external_start_matched_variables", -1)) != 0:
        errors.append("external_start_matched_variables")
    if solver.get("external_start_installed") is not False:
        errors.append("solver_external_start_installed")
    if solver.get("external_start_source") is not None:
        errors.append("solver_external_start_source")
    if int(solver.get("external_start_matched_variables", -1)) != 0:
        errors.append("solver_external_start_matched_variables")
    if job.profile == "C":
        if telemetry.get("enabled") is not False:
            errors.append("c_partial_start_enabled")
        if int(algorithm.get("installed_mip_start_count", -1)) != 0:
            errors.append("c_installed_mip_start")
        if algorithm.get("all_courier_start_installed") is not False:
            errors.append("c_all_courier_start_installed")
        c_preparation = qualification.get("start_preparation_runtime_seconds")
        if (
            c_preparation is not None
            and _number(c_preparation, "start_preparation_runtime_seconds") != 0.0
        ):
            errors.append("c_start_preparation_nonzero")
        return {"valid": not errors, "errors": errors, "telemetry": telemetry}

    network_count = int(telemetry.get("network_candidate_count", 0))
    installed_count = int(telemetry.get("installed_start_count", 0))
    if telemetry.get("enabled") is not True:
        errors.append("e_partial_start_not_enabled")
    if telemetry.get("used") is not True:
        errors.append("e_partial_start_not_used")
    if telemetry.get("timed_out") is not False:
        errors.append("e_partial_start_timed_out")
    if telemetry.get("fallback_reason") is not None:
        errors.append("e_partial_start_fallback")
    if network_count < 1:
        errors.append("e_no_network_candidate")
    if int(telemetry.get("safety_start_count", 0)) != 1:
        errors.append("e_safety_start_count")
    if installed_count != network_count + 1:
        errors.append("e_installed_start_count")
    if int(algorithm.get("installed_mip_start_count", -1)) != installed_count:
        errors.append("e_backend_start_count_mismatch")
    if algorithm.get("all_courier_start_installed") is not True:
        errors.append("e_safety_start_not_reported")
    preparation = _number(
        telemetry.get("preparation_seconds"), "objective_start.preparation_seconds"
    )
    solver_preparation = _number(
        qualification["start_preparation_runtime_seconds"],
        "start_preparation_runtime_seconds",
    )
    budget = _number(telemetry.get("budget_seconds"), "objective_start.budget_seconds")
    if preparation < 0.0 or solver_preparation < 0.0 or budget < 0.0:
        errors.append("e_negative_preparation_or_budget")
    if budget > START_BUDGET_SECONDS + 1e-12:
        errors.append("e_declared_budget_exceeds_hard_budget")
    if preparation > START_BUDGET_SECONDS + 1e-6:
        errors.append("e_preparation_exceeds_hard_budget")
    if preparation > 0.05 * TIME_LIMIT_SECONDS + 1e-6:
        errors.append("e_preparation_exceeds_five_percent_of_fixed_time_limit")
    # The outer solver timer brackets the call while the proxy timer starts
    # inside it, so a few hundred microseconds of Python call/return overhead
    # is expected.  Keep the reconciliation strict but above timer jitter.
    if (
        abs(preparation - solver_preparation)
        > PREPARATION_TELEMETRY_TOLERANCE_SECONDS
    ):
        errors.append("e_preparation_telemetry_mismatch")
    candidates = telemetry.get("candidate_supports")
    if not isinstance(candidates, list) or len(candidates) != network_count:
        errors.append("e_candidate_support_count")
        candidates = []
    candidate_h = int(qualification["candidate_h"])
    demand_value = qualification.get("total_demand_per_hour")
    demand = (
        _number(demand_value, "total_demand_per_hour")
        if demand_value is not None
        else 0.0
    )
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            errors.append(f"e_candidate_{index}_not_mapping")
            continue
        open_h = candidate.get("open_h")
        if (
            not isinstance(open_h, list)
            or len(open_h) != len(set(open_h))
            or not 2 <= len(open_h) < candidate_h
        ):
            errors.append(f"e_candidate_{index}_open_h_gate")
        if _number(candidate.get("estimated_service"), "estimated_service") < max(
            1e-6, 0.01 * demand
        ):
            errors.append(f"e_candidate_{index}_service_gate")
        estimated_service = _number(
            candidate.get("estimated_service"), "estimated_service"
        )
        estimated_fraction = _number(
            candidate.get("estimated_service_fraction"),
            "estimated_service_fraction",
        )
        fraction_basis = _number(
            candidate.get("fraction_basis_service"), "fraction_basis_service"
        )
        if candidate.get("proxy_only") is not True:
            errors.append(f"e_candidate_{index}_proxy_only_flag")
        if not 0.0 < estimated_fraction <= 1.0 or fraction_basis <= 0.0:
            errors.append(f"e_candidate_{index}_fraction_gate")
        elif not _numbers_close(
            estimated_service / fraction_basis,
            estimated_fraction,
        ):
            errors.append(f"e_candidate_{index}_fraction_identity")
        if _number(candidate.get("score"), "score") <= 0.0:
            errors.append(f"e_candidate_{index}_score_gate")
        energy_proxy = candidate.get("energy_proxy")
        if not isinstance(energy_proxy, Mapping):
            errors.append(f"e_candidate_{index}_energy_proxy")
            continue
        energy_use = _number(
            energy_proxy.get("system_energy_use"), "candidate system energy use"
        )
        energy_supply = _number(
            energy_proxy.get("system_energy_supply"),
            "candidate system energy supply",
        )
        base_departures = _number(
            energy_proxy.get("base_departure_population"),
            "candidate base departure population",
        )
        swap_events = _number(
            energy_proxy.get("swap_event_population"),
            "candidate swap event population",
        )
        if min(energy_use, energy_supply, base_departures, swap_events) < 0.0:
            errors.append(f"e_candidate_{index}_negative_energy_proxy")
        if energy_use > energy_supply + _scale_tolerance(energy_use, energy_supply):
            errors.append(f"e_candidate_{index}_system_energy_gate")
    return {"valid": not errors, "errors": errors, "telemetry": telemetry}


def _inspect_job(
    job: JobSpec,
    process_record: Mapping[str, Any],
    frozen: FrozenInputs,
) -> dict[str, Any]:
    record = dict(process_record)
    infrastructure_errors: list[str] = []
    try:
        results_dir = _require_real_path(job.results_dir, kind="job results directory")
        if not results_dir.is_dir():
            raise DevMicroCEError("job results path is not a directory")
        driver_log = _require_regular_file(job.driver_log, kind="driver-log artifact")
        if driver_log.stat().st_size <= 0:
            infrastructure_errors.append("missing_or_empty_driver_log")
        solver_log = job.results_dir / "MX" / "solver.log"
        solver_log = _require_regular_file(solver_log, kind="solver-log artifact")
        if solver_log.stat().st_size <= 0:
            infrastructure_errors.append("missing_or_empty_solver_log")
        run_config = _read_json_object(job.results_dir / "run_config.json")
        fairness_errors = _validate_run_config(job, run_config, frozen)
        infrastructure_errors.extend(f"run_config:{error}" for error in fairness_errors)
        signature = run_config.get("run_signature_sha256")
        summary_path = job.results_dir / "MX" / "summary.json"
        no_incumbent_status = _driver_no_incumbent_status(job.driver_log)
        if summary_path.exists() or summary_path.is_symlink():
            summary = _read_json_object(summary_path)
            if summary.get("run_signature_sha256") != signature:
                infrastructure_errors.append("summary_run_signature_mismatch")
            qualification = _qualification_audit(job, summary)
            telemetry = _telemetry_audit(job, summary, qualification)
            outcome = str(qualification["outcome"])
            if process_record.get("returncode") != 0 and outcome != "no_incumbent":
                infrastructure_errors.append(
                    str(process_record.get("error") or "job_failed")
                )
            structured_by_runner = False
        elif process_record.get("returncode") != 0 and no_incumbent_status is not None:
            # f884 main.py raises before ResultProcessor can persist a summary.
            # Preserve this as an explicit algorithmic outcome instead of
            # misclassifying the absence of an incumbent as an I/O failure.
            qualification = {
                "valid": False,
                "outcome": "no_incumbent",
                "errors": ["no_incumbent", "summary_unavailable_from_f884_main"],
                "status_name": f"STATUS_{no_incumbent_status}",
                "objective": None,
                "objective_bound": None,
                "mip_gap": None,
                "total_algorithm_runtime_seconds": None,
                "main_solve_runtime_seconds": None,
                "algorithm_prepare_runtime_seconds": None,
                "start_preparation_runtime_seconds": None,
                "first_incumbent_time_seconds": None,
                "node_count": None,
                "candidate_h": INSTANCE_SPECS[job.stage].candidate_h,
                "total_demand_per_hour": None,
            }
            telemetry = {
                "valid": False,
                "errors": ["telemetry_unavailable_from_f884_main"],
                "telemetry": {},
            }
            outcome = "no_incumbent"
            structured_by_runner = True
        else:
            infrastructure_errors.append("missing_summary_artifact")
            qualification = {
                "valid": False,
                "outcome": "unavailable",
                "errors": ["summary_unavailable"],
            }
            telemetry = {
                "valid": False,
                "errors": ["telemetry_unavailable"],
                "telemetry": {},
            }
            outcome = "infrastructure_failure"
            structured_by_runner = False
            if process_record.get("returncode") != 0:
                infrastructure_errors.append(
                    str(process_record.get("error") or "job_failed")
                )
        record.update(
            {
                "run_config_valid": not fairness_errors,
                "run_config_errors": list(fairness_errors),
                "run_signature_sha256": signature,
                "normalized_run_config_sha256": _normalized_run_config_sha256(
                    run_config
                ),
                "outcome": outcome,
                "no_incumbent_structured_by_runner": structured_by_runner,
                "qualification": qualification,
                "telemetry_audit": telemetry,
            }
        )
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        infrastructure_errors.append(
            f"artifact_inspection:{type(error).__name__}: {error}"
        )
    record["infrastructure_valid"] = not infrastructure_errors
    record["infrastructure_errors"] = infrastructure_errors
    qualification_valid = bool(record.get("qualification", {}).get("valid", False))
    telemetry_valid = bool(record.get("telemetry_audit", {}).get("valid", False))
    record["qualification_valid"] = qualification_valid
    record["construction_use_gate_passed"] = telemetry_valid
    record["treatment_reliable"] = telemetry_valid
    record["effective"] = (
        not infrastructure_errors and qualification_valid and telemetry_valid
    )
    return record


def _pair_audit(
    c_record: Mapping[str, Any], e_record: Mapping[str, Any]
) -> dict[str, Any]:
    infrastructure_errors: list[str] = []
    if not c_record.get("infrastructure_valid") or not e_record.get(
        "infrastructure_valid"
    ):
        infrastructure_errors.append("invalid_job_or_artifact_infrastructure")
    if not infrastructure_errors and c_record.get(
        "normalized_run_config_sha256"
    ) != e_record.get("normalized_run_config_sha256"):
        infrastructure_errors.append("c_e_run_configs_differ_beyond_start_flag")
    c_gap: float | None = None
    e_gap: float | None = None
    c_bound: float | None = None
    e_bound: float | None = None
    c_objective: float | None = None
    e_objective: float | None = None
    try:
        c_gap = _optional_number(_nested(c_record, "qualification", "mip_gap"), "C gap")
        e_gap = _optional_number(_nested(e_record, "qualification", "mip_gap"), "E gap")
        c_bound = _optional_number(
            _nested(c_record, "qualification", "objective_bound"), "C bound"
        )
        e_bound = _optional_number(
            _nested(e_record, "qualification", "objective_bound"), "E bound"
        )
        c_objective = _optional_number(
            _nested(c_record, "qualification", "objective"), "C objective"
        )
        e_objective = _optional_number(
            _nested(e_record, "qualification", "objective"), "E objective"
        )
    except DevMicroCEError as error:
        infrastructure_errors.append(str(error))
    qualification_valid = bool(
        c_record.get("qualification_valid") and e_record.get("qualification_valid")
    )
    gap_worse = (
        e_gap > c_gap + GAP_CONSISTENCY_TOLERANCE
        if c_gap is not None and e_gap is not None
        else None
    )
    bound_improved = (
        e_bound > c_bound + _scale_tolerance(e_bound, c_bound)
        if c_bound is not None and e_bound is not None
        else None
    )
    return {
        "infrastructure_valid": not infrastructure_errors,
        "infrastructure_errors": infrastructure_errors,
        "qualification_valid": qualification_valid,
        "e_construction_use_gate_passed": bool(
            e_record.get("construction_use_gate_passed")
        ),
        "e_start_reliable": bool(e_record.get("treatment_reliable")),
        "effective_pair": bool(c_record.get("effective") and e_record.get("effective")),
        "stage": c_record.get("stage"),
        "seed": c_record.get("seed"),
        "c_job_id": c_record.get("job_id"),
        "e_job_id": e_record.get("job_id"),
        "c_gap": c_gap,
        "e_gap": e_gap,
        "c_bound": c_bound,
        "e_bound": e_bound,
        "e_gap_worse": gap_worse,
        "e_bound_improved": bound_improved,
        "e_adverse_gap": gap_worse if qualification_valid else None,
        "incumbent_delta_e_minus_c": (
            e_objective - c_objective
            if e_objective is not None and c_objective is not None
            else None
        ),
        "bound_delta_e_minus_c": (
            e_bound - c_bound if e_bound is not None and c_bound is not None else None
        ),
        "c_outcome": c_record.get("outcome"),
        "e_outcome": e_record.get("outcome"),
    }


def _postflight(config: BatchConfig, frozen: FrozenInputs) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        output_root = _require_real_path(
            config.output_root, kind="development output root"
        )
        if not output_root.is_dir():
            errors.append("output_root_not_directory")
        _require_regular_file(
            output_root / "manifest.json", kind="development manifest"
        )
        _require_real_output_path(
            Path(config.resource_lock), kind="benchmark resource lock"
        )
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        errors.append(f"artifact_container:{type(error).__name__}:{error}")
    try:
        commit, branch = _clean_repository(config.repo_root)
        if commit != frozen.commit:
            errors.append("git_commit_changed")
        if branch != frozen.branch:
            errors.append("git_branch_changed")
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        errors.append(f"repository:{error}")
    try:
        aggregate, files = _implementation_provenance(config.repo_root)
        if aggregate != frozen.implementation_sha256 or files != dict(
            frozen.implementation_files
        ):
            errors.append("implementation_changed")
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        errors.append(f"implementation:{error}")
    try:
        source = _require_regular_file(
            Path(frozen.formulation_source), kind="formulation source"
        )
        if _sha256_file(source) != frozen.formulation_sha256:
            errors.append("formulation_source_changed")
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        errors.append(f"formulation_source:{error}")
    for stage, expected in frozen.instances.items():
        try:
            instance_path = _require_real_path(
                Path(str(expected["path"])), kind=f"{stage} instance"
            )
            aggregate, files = canonical_instance_hash(instance_path)
            for filename in files:
                _require_regular_file(
                    instance_path / filename, kind=f"{stage} {filename}"
                )
            if (
                aggregate != expected["sha256"]
                or files != expected["canonical_file_sha256"]
            ):
                errors.append(f"{stage}_instance_changed")
        except (MemoryError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            errors.append(f"{stage}_instance:{error}")
    return tuple(errors)


def _manifest_payload(
    config: BatchConfig,
    frozen: FrozenInputs,
    jobs: Sequence[JobSpec],
) -> dict[str, Any]:
    return {
        "artifact_type": "mlnd_dev_micro_ce_manifest",
        "artifact_schema_version": 2,
        "development_only": True,
        "formal_evidence_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(config.repo_root),
            "commit": frozen.commit,
            "branch": frozen.branch,
            "required_baseline_ancestor": REQUIRED_BASELINE_COMMIT,
            "implementation_sha256": frozen.implementation_sha256,
            "implementation_files": dict(frozen.implementation_files),
        },
        "formulation_source": {
            "path": frozen.formulation_source,
            "sha256": frozen.formulation_sha256,
        },
        "instances": dict(frozen.instances),
        "protocol": {
            "formulation_id": FORMULATION_ID,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "stages": list(config.stages),
            "seeds": list(SEEDS),
            "profiles": list(PROFILES),
            "time_limit_seconds": TIME_LIMIT_SECONDS,
            "preparation_telemetry_reconciliation_tolerance_seconds": (
                PREPARATION_TELEMETRY_TOLERANCE_SECONDS
            ),
            "maximum_concurrency": config.jobs,
            "pairing": "only C and E of one exact stage/seed may overlap",
            "c_e_only_difference": "solver.use_objective_partial_start",
            "h10_requires_explicit_confirmation": True,
            "stop_rules": {
                "any_infrastructure_failure": "stop_immediately",
                "e_reliable_start_count_below_two_of_three": "stop_before_next_stage",
                "effective_pair_count_below_two_of_three": "stop_before_next_stage",
                "e_gap_worse_on_at_least_two_of_three": ("stop_before_next_stage"),
            },
            "qualification_policy": asdict(ACTIVE_POLICY),
            "evidence_limitations": [
                "development-only terminal pilot",
                "main.py does not persist structured progress_events",
                (
                    "f884 main.py raises before summary persistence when no incumbent; "
                    "the runner recognizes only its exact terminal error and records an "
                    "explicit algorithmic no-incumbent outcome"
                ),
                "not sufficient for primal-dual integral or time-to-common-target claims",
            ],
        },
        "controlled_environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "resource_lock": str(config.resource_lock.expanduser().resolve()),
        "process_table_sha256": frozen.process_table_sha256,
        "planned_jobs": [
            {
                "job_id": job.job_id,
                "stage": job.stage,
                "profile": job.profile,
                "seed": job.seed,
                "instance_dir": str(job.instance_dir),
                "results_dir": str(job.results_dir),
                "driver_log": str(job.driver_log),
                "command": list(_job_command(config, job)),
            }
            for job in jobs
        ],
    }


def run_batch(
    config: BatchConfig,
    *,
    process_runner: ProcessRunner = subprocess.run,
) -> dict[str, Any]:
    """Run and audit the fixed development pilot under the shared resource lock."""

    repo_root = config.repo_root.expanduser().absolute()
    formulation_source = (
        config.formulation_source.expanduser().absolute()
        if config.formulation_source is not None
        else repo_root / FORMULATION_RELATIVE_PATH
    )
    config = BatchConfig(
        repo_root=repo_root,
        output_root=config.output_root.expanduser().absolute(),
        formulation_source=formulation_source,
        expected_formulation_sha256=config.expected_formulation_sha256,
        requested_stages=config.requested_stages,
        include_h10=config.include_h10,
        confirm_h10=config.confirm_h10,
        jobs=config.jobs,
        resource_lock=_canonical_lock_path(Path(config.resource_lock)),
    )
    _require_shared_resource_lock(config.resource_lock)
    try:
        with BenchmarkResourceLock(config.resource_lock, command="dev-micro-ce"):
            frozen = _freeze_inputs(config)
            jobs = _job_specs(config, frozen)
            config.output_root.mkdir(parents=False, exist_ok=False)
            manifest = _manifest_payload(config, frozen, jobs)
            write_canonical_json_exclusive(
                config.output_root / "manifest.json", manifest
            )

            records: list[dict[str, Any]] = []
            pairs: list[dict[str, Any]] = []
            stages: list[dict[str, Any]] = []
            pair_process_audits: list[dict[str, Any]] = []
            stop_reason: str | None = None
            for stage in config.stages:
                stage_pairs: list[dict[str, Any]] = []
                for seed in SEEDS:
                    try:
                        pair_process_table = _process_table()
                        pair_conflicts = _conflicting_processes(pair_process_table)
                        pair_process_audit = {
                            "stage": stage,
                            "seed": seed,
                            "process_table_sha256": hashlib.sha256(
                                pair_process_table.encode("utf-8")
                            ).hexdigest(),
                            "conflicting_processes": list(pair_conflicts),
                            "valid": not pair_conflicts,
                        }
                    except (MemoryError, KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as error:
                        pair_process_audit = {
                            "stage": stage,
                            "seed": seed,
                            "process_table_sha256": None,
                            "conflicting_processes": [],
                            "valid": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    pair_process_audits.append(pair_process_audit)
                    if not pair_process_audit["valid"]:
                        pair = {
                            "infrastructure_valid": False,
                            "infrastructure_errors": ["pair_process_audit_failed"],
                            "qualification_valid": False,
                            "e_start_reliable": False,
                            "effective_pair": False,
                            "stage": stage,
                            "seed": seed,
                            "e_adverse_gap": None,
                            "process_table_sha256": pair_process_audit[
                                "process_table_sha256"
                            ],
                        }
                        pairs.append(pair)
                        stage_pairs.append(pair)
                        stop_reason = f"{stage}_seed{seed}_process_conflict"
                        break
                    pair_jobs = [
                        job for job in jobs if job.stage == stage and job.seed == seed
                    ]
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(config.jobs, len(pair_jobs))
                    ) as executor:
                        futures = {
                            executor.submit(
                                _run_one_job,
                                config,
                                job,
                                process_runner,
                            ): job
                            for job in pair_jobs
                        }
                        process_records = {
                            futures[future].profile: future.result()
                            for future in concurrent.futures.as_completed(futures)
                        }
                    inspected = {
                        job.profile: _inspect_job(
                            job,
                            process_records[job.profile],
                            frozen,
                        )
                        for job in pair_jobs
                    }
                    records.extend(inspected[profile] for profile in PROFILES)
                    pair = _pair_audit(inspected["C"], inspected["E"])
                    pair["process_table_sha256"] = pair_process_audit[
                        "process_table_sha256"
                    ]
                    pairs.append(pair)
                    stage_pairs.append(pair)
                    if not pair["infrastructure_valid"]:
                        stop_reason = f"{stage}_seed{seed}_infrastructure_failure"
                        break

                adverse_gap_count = sum(
                    pair.get("e_adverse_gap") is True for pair in stage_pairs
                )
                reliable_start_count = sum(
                    pair.get("e_start_reliable") is True for pair in stage_pairs
                )
                effective_pair_count = sum(
                    pair.get("effective_pair") is True for pair in stage_pairs
                )
                complete_seed_set = len(stage_pairs) == len(SEEDS)
                stage_passed = (
                    complete_seed_set
                    and all(pair["infrastructure_valid"] for pair in stage_pairs)
                    and reliable_start_count >= 2
                    and effective_pair_count >= 2
                    and adverse_gap_count < 2
                )
                if complete_seed_set and reliable_start_count < 2:
                    stop_reason = (
                        f"{stage}_e_reliable_start_on_{reliable_start_count}_of_3_seeds"
                    )
                elif complete_seed_set and effective_pair_count < 2:
                    stop_reason = (
                        f"{stage}_effective_pair_on_{effective_pair_count}_of_3_seeds"
                    )
                elif complete_seed_set and adverse_gap_count >= 2:
                    stop_reason = (
                        f"{stage}_e_adverse_gap_on_{adverse_gap_count}_of_3_seeds"
                    )
                stages.append(
                    {
                        "stage": stage,
                        "completed_seed_count": len(stage_pairs),
                        "complete_seed_set": complete_seed_set,
                        "e_reliable_start_count": reliable_start_count,
                        "effective_pair_count": effective_pair_count,
                        "e_adverse_gap_count": adverse_gap_count,
                        "passed": stage_passed,
                    }
                )
                if not stage_passed:
                    stop_reason = stop_reason or f"{stage}_stage_failed"
                    break

            postflight_errors = list(_postflight(config, frozen))
            if postflight_errors:
                stop_reason = stop_reason or "postflight_drift"
            executed_ids = {record["job_id"] for record in records}
            skipped = [job.job_id for job in jobs if job.job_id not in executed_ids]
            summary = {
                "artifact_type": "mlnd_dev_micro_ce_summary",
                "artifact_schema_version": 2,
                "development_only": True,
                "formal_evidence_eligible": False,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "passed" if stop_reason is None else "failed_closed",
                "stop_reason": stop_reason,
                "postflight_errors": postflight_errors,
                "job_records": records,
                "pair_audits": pairs,
                "pair_process_audits": pair_process_audits,
                "stage_audits": stages,
                "skipped_planned_job_ids": skipped,
                "manifest": "manifest.json",
            }
            write_canonical_json_exclusive(config.output_root / "summary.json", summary)
            return summary
    except BenchmarkResourceLockError as error:
        raise DevMicroCEError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the development-only matched micro C/E partial-start pilot."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default=str(MAIN_REPO_ROOT))
    parser.add_argument(
        "--formulation-source",
        default=None,
        help="must be this worktree's MLND/main.tex (default: inferred from --repo-root)",
    )
    parser.add_argument(
        "--expected-formulation-sha256",
        default=EXPECTED_FORMULATION_SHA256,
    )
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--stages",
        action="append",
        nargs="+",
        choices=tuple(INSTANCE_SPECS),
        default=None,
        help=(
            "ordered micro stages; may be repeated as an option, but each stage "
            "may appear only once (default: H9)"
        ),
    )
    stage_group.add_argument(
        "--include-h10",
        action="store_true",
        help="legacy compatibility shorthand for --stages H9 H10",
    )
    parser.add_argument(
        "--confirm-h10",
        action="store_true",
        help="second explicit acknowledgement required before scheduling H10",
    )
    parser.add_argument(
        "--resource-lock",
        default=str(DEFAULT_BENCHMARK_RESOURCE_LOCK),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requested_stages = (
        tuple(stage for group in args.stages for stage in group)
        if args.stages is not None
        else None
    )
    config = BatchConfig(
        repo_root=Path(args.repo_root),
        output_root=Path(args.output_root),
        formulation_source=(
            Path(args.formulation_source) if args.formulation_source else None
        ),
        expected_formulation_sha256=args.expected_formulation_sha256,
        requested_stages=requested_stages,
        include_h10=args.include_h10,
        confirm_h10=args.confirm_h10,
        jobs=args.jobs,
        resource_lock=Path(args.resource_lock),
    )
    try:
        summary = run_batch(config)
    except DevMicroCEError as error:
        print(f"development C/E pilot refused to run: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
