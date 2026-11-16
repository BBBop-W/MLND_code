"""Run the frozen capacity matrix using the current MLND_sever runner.

Only input files and the reviewed job matrix are read from the legacy bundle.
Its source, binaries, launch scripts and result files are never executed/reused.
Execution requires Linux; validation and dry-run planning are portable.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = Path(__file__).resolve().parent / "server_support"
sys.path.insert(0, str(SUPPORT))
# The frozen checkout runs directly without installing the helper modules.
from job_control import (  # noqa: E402
    ControlledProcess,
    GIB,
    atomic_json,
    ensure_no_live_attempts,
    pressure_victim,
    reservation_fits,
)
from linux_process import require_linux  # noqa: E402

STAGES = ("core", "stress")
SCHEMA = "mlnd-server-capacity-suite/1"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def number(value, name, *, integer=False, minimum=0, strict=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < minimum or strict and value == minimum:
        raise ValueError(f"Invalid {name}: {value}")
    return value


def fingerprints(instance):
    manifest = read(instance / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.json" not in files:
        raise ValueError(f"Invalid instance manifest: {instance}")
    actual = {"manifest.json": digest(instance / "manifest.json")}
    for name, expected in files.items():
        path = (instance / name).resolve()
        if not path.is_relative_to(instance) or path == instance:
            raise ValueError(f"Instance manifest escapes its directory: {name}")
        actual[name] = digest(path)
        if actual[name] != expected:
            raise ValueError(f"Instance SHA-256 mismatch: {path}")
    return actual


class OutputLock:
    def __init__(self, output):
        self.path = output / ".dispatcher.lock"
        self.guard_path = output / ".dispatcher.guard"
        self.guard = None
        self.token = {
            "pid": os.getpid(),
            "create_time": psutil.Process().create_time(),
            "started_utc": utc(),
        }

    def __enter__(self):
        # The persistent guard inode is never renamed or removed. A PID marker
        # alone has an ABA race when two callers reclaim the same stale marker.
        self.guard = self.guard_path.open("a+b")
        try:
            if self.guard_path.stat().st_size == 0:
                self.guard.write(b"\0")
                self.guard.flush()
            self.guard.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.guard.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.guard.close()
            self.guard = None
            raise RuntimeError(
                "Cannot acquire output lock; another dispatcher may own this output"
            ) from error
        try:
            if self.path.exists():
                self.path.rename(
                    self.path.with_name(f".dispatcher.stale.{time.time_ns()}.json")
                )
            atomic_json(self.path, self.token)
        except BaseException:
            self.release()
            raise
        return self

    def release(self):
        if self.guard:
            try:
                self.guard.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self.guard.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.guard.fileno(), fcntl.LOCK_UN)
            finally:
                self.guard.close()
                self.guard = None

    def __exit__(self, *exc):
        try:
            if self.path.exists() and read(self.path) == self.token:
                self.path.unlink()
        finally:
            self.release()


def host_environment():
    memory = psutil.virtual_memory()
    result = {
        "recorded_utc": utc(),
        "platform": platform.platform(),
        "python": sys.version,
        "psutil_version": psutil.__version__,
        "memory_total_bytes": memory.total,
        "memory_available_bytes": memory.available,
        "unavailable_counters": {},
    }
    for name, getter in (
        ("physical_cpu_count", lambda: psutil.cpu_count(logical=False)),
        ("logical_cpu_count", lambda: psutil.cpu_count(logical=True)),
        ("swap_total_bytes", lambda: psutil.swap_memory().total),
    ):
        try:
            result[name] = getter()
        except (OSError, psutil.Error) as error:
            result[name] = None
            result["unavailable_counters"][name] = f"{type(error).__name__}: {error}"
    return result


def load_suite(suite_root, args):
    root = Path(suite_root).resolve()
    if not (root / "suite.json").is_file() and (root / "suite/suite.json").is_file():
        root = root / "suite"
    path = root / "suite.json"
    document = read(path)
    if document.get("schema") != SCHEMA:
        raise ValueError(f"Expected suite schema {SCHEMA}")
    defaults = document.get("defaults", {})
    if any(
        defaults.get(key) != value
        for key, value in {
            "seconds": 7200,
            "gap": 0.01,
            "seed": 43,
            "threads": 1,
        }.items()
    ):
        raise ValueError(
            "Capacity matrix requires seconds=7200, gap=.01, seed=43, threads=1"
        )
    defaults = dict(defaults)
    number(defaults["threads"], "threads", integer=True, minimum=1)
    number(defaults["seed"], "seed", integer=True, minimum=0)
    for key, fallback in [
        ("solver_wall_seconds", 7500),
        ("validation_seconds", 1800),
        ("job_wall_seconds", 9400),
    ]:
        defaults.setdefault(key, fallback)
        number(defaults[key], key, strict=True)
    if (
        defaults["solver_wall_seconds"] < 7200
        or defaults["job_wall_seconds"]
        < defaults["solver_wall_seconds"] + defaults["validation_seconds"] + 30
    ):
        raise ValueError(
            "Wall limits must cover solve, output and independent validation"
        )
    defaults["map_seconds"] = 90
    defaults["job_wall_seconds"] = max(
        defaults["job_wall_seconds"],
        defaults["solver_wall_seconds"]
        + defaults["validation_seconds"]
        + defaults["map_seconds"]
        + 60,
    )
    resources = dict(
        max_parallel=8,
        max_threads=8,
        reservation_budget_gib=24,
        system_available_floor_gib=8,
        sample_seconds=1,
        admission_timeout_seconds=900,
    )
    # Concurrency defaults to the requested eight, subject to memory admission.
    for key in resources:
        if key != "max_parallel" and key in document.get("resources", {}):
            resources[key] = document["resources"][key]
    for option, key in [
        ("max_parallel", "max_parallel"),
        ("memory_budget_gib", "reservation_budget_gib"),
        ("available_floor_gib", "system_available_floor_gib"),
    ]:
        if getattr(args, option, None) is not None:
            resources[key] = getattr(args, option)
    for key in ["max_parallel", "max_threads"]:
        number(resources[key], key, integer=True, minimum=1)
        if resources[key] > 8:
            raise ValueError("At most 8 concurrent one-thread tasks are allowed")
    for key in [
        "reservation_budget_gib",
        "system_available_floor_gib",
        "sample_seconds",
        "admission_timeout_seconds",
    ]:
        number(resources[key], key, strict=True)
    if (
        resources["reservation_budget_gib"] > 24
        or resources["system_available_floor_gib"] < 8
    ):
        raise ValueError(
            "Use a reservation budget <=24 GiB and available-memory floor >=8 GiB"
        )
    resources.update(
        reservation_budget_bytes=int(resources["reservation_budget_gib"] * GIB),
        available_floor_bytes=int(resources["system_available_floor_gib"] * GIB),
    )
    raw_jobs = document.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("Suite must have a nonempty jobs list")
    jobs, ids, inputs = [], set(), {}
    for entry in raw_jobs:
        if not isinstance(entry, dict):
            raise ValueError("Each job must be an object")
        key = entry.get("id", "")
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", key)
            or key in ids
        ):
            raise ValueError(f"Unsafe or repeated job ID: {key}")
        ids.add(key)
        if type(entry.get("enabled", True)) is not bool:
            raise ValueError("enabled must be boolean")
        if (
            entry.get("stage") not in (*STAGES, "preflight")
            or type(entry.get("k")) is not int
            or entry["k"] not in (1, 2)
        ):
            raise ValueError(f"Invalid stage/K: {key}")
        for name in ["seconds", "gap", "seed", "threads"]:
            number(
                entry.get(name, defaults[name]),
                name,
                integer=name in ("seed", "threads"),
            )
            if entry.get(name, defaults[name]) != defaults[name]:
                raise ValueError(
                    f"Per-job protocol override is not allowed: {key}/{name}"
                )
        if entry.get("preset", "performance-20261105") != "performance-20261105":
            raise ValueError(
                "Legacy matrix must identify the reviewed algorithm preset"
            )
        memory = number(entry.get("memory_gib"), "memory_gib", strict=True)
        if memory not in (3, 6, 10, 16, 24):
            raise ValueError("Memory class must be 3/6/10/16/24 GiB")
        if memory > resources["reservation_budget_gib"]:
            raise ValueError(f"Job exceeds reservation budget: {key}")
        relative = entry.get("instance")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Instance must be a suite-relative path")
        instance = (root / relative).resolve()
        if not instance.is_relative_to(root) or instance == root:
            raise ValueError(f"Instance escapes suite directory: {relative}")
        if relative not in inputs:
            inputs[relative] = fingerprints(instance)
        values = read(instance / "model.json")["values"]
        jobs.append(
            {
                **entry,
                **defaults,
                # Preserve every legacy matrix row without a special gate.
                "source_stage": entry["stage"],
                "stage": "core" if entry["stage"] == "preflight" else entry["stage"],
                "instance": str(instance),
                "instance_relative": relative,
                "input_sha256": inputs[relative],
                "memory_bytes": int(memory * GIB),
                "exclusive": memory == 24,
                "method": "algorithm",
                "dimensions": {name: values[name] for name in ("n_l", "n_h", "n_od")},
            }
        )
    return path, jobs, resources


def select_jobs(jobs, args):
    selected = [
        j
        for j in jobs
        if j.get("enabled", True)
        and (args.stage == "all" or j["stage"] == args.stage)
        and (not args.job or j["id"] in args.job)
    ]
    if not selected or (args.job and set(args.job) - {j["id"] for j in selected}):
        raise ValueError(
            "Requested tasks are missing, disabled or outside the selected stage"
        )
    return selected


def source_fingerprints(root):
    version = read(root / "SOURCE_VERSION.json")
    result = {"SOURCE_VERSION.json": digest(root / "SOURCE_VERSION.json")}
    for name, entry in version["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != entry["sha256"]:
            raise ValueError(f"Source no longer matches SOURCE_VERSION.json: {name}")
        result[name] = entry["sha256"]
    for path in sorted((root / "scripts").rglob("*.py")):
        result[path.relative_to(root).as_posix()] = digest(path)
    return result


def run_identity(suite, binary, build_manifest, jobs, resources):
    manifest = read(build_manifest)
    sources = source_fingerprints(ROOT)
    version = read(ROOT / "SOURCE_VERSION.json")
    expected = {name: entry["sha256"] for name, entry in version["files"].items()}
    if (
        manifest.get("schema_version") != 1
        or manifest.get("binary_sha256") != digest(binary)
        or manifest.get("source_sha256") != expected
        or manifest.get("source_version_sha256") != sources["SOURCE_VERSION.json"]
    ):
        raise ValueError(
            "Build manifest does not bind this binary to the current source"
        )
    return {
        "schema": "mlnd-server-suite-run/2",
        "suite_sha256": digest(suite),
        "binary_sha256": manifest["binary_sha256"],
        "build_manifest_sha256": digest(build_manifest),
        "source_sha256": sources,
        "jobs_sha256": content_hash(jobs),
        "resources": resources,
        "method": "algorithm",
        "legacy_bundle_use": "input files and matrix only",
    }


def job_signature(job, identity):
    return content_hash({"job": job, "identity": identity})


def native_record(attempt):
    path = attempt / "native/task_status.json"
    try:
        result = read(path)
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def completed_attempt(attempt, signature, *, qualified=False):
    try:
        own = read(attempt / "attempt.json")
        status = native_record(attempt)
        if (
            own.get("signature") != signature
            or own.get("status") != "completed"
            or own.get("group_cleanup_complete") is not True
            or own.get("stop_reason")
            or own.get("returncode") != 0
            or status.get("status") != "completed"
            or status.get("validation_passed") is not True
            or status.get("solver_returncode") != 0
            or status.get("validation_returncode") != 0
        ):
            return False
        if (
            status.get("input_sha256") != own["job"]["input_sha256"]
            or status.get("binary_sha256") != own.get("binary_sha256")
            or status.get("build_manifest_sha256") != own.get("build_manifest_sha256")
        ):
            return False
        certificate = read(attempt / "native/native_validation.json")
        if (
            certificate.get("passed") is not True
            or certificate.get("max_orders") != own["job"]["k"]
            or certificate.get("native_manifest_matches") is not True
            or certificate.get("assignment_scope") != "task"
            or certificate.get("manifest_sha256")
            != own["job"]["input_sha256"]["manifest.json"]
        ):
            return False
        for filename, field in [
            ("result.json", "result_sha256"),
            ("solution.json", "solution_sha256"),
            ("path_flows.json", "path_flows_sha256"),
            ("input_reference.json", "input_reference_sha256"),
        ]:
            if digest(attempt / "native" / filename) != certificate.get(field):
                return False
        hashes = own.get("artifact_sha256")
        if not hashes or "native/task_status.json" not in hashes:
            return False
        for name, expected in hashes.items():
            path = (attempt / name).resolve()
            if not path.is_relative_to(attempt.resolve()) or digest(path) != expected:
                return False
        return not qualified or (
            status.get("qualified_baseline") is True
            and status.get("state") == "qualified_baseline"
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def attempt_hashes(attempt):
    # Bind the complete small native audit and output artifacts. Large append-only
    # log files are retained but do not enter success/recovery acceptance.
    return {
        p.relative_to(attempt).as_posix(): digest(p)
        for p in sorted((attempt / "native").rglob("*.json"))
        if p.is_file()
    }


def latest_attempt(output, job):
    paths = sorted((output / "jobs" / job["id"] / "attempts").glob("[0-9]*"))
    return paths[-1] if paths else None


def make_command(job, binary, build_manifest, attempt, python, maps_root=None):
    command = [
        str(python),
        str(ROOT / "scripts/server_run.py"),
        "--binary",
        str(binary),
        "--build-manifest",
        str(build_manifest),
        "--input",
        job["instance"],
        "--output",
        str(attempt / "native"),
        "--k",
        str(job["k"]),
        "--method",
        "algorithm",
        "--seconds",
        str(job["seconds"]),
        "--gap",
        str(job["gap"]),
        "--seed",
        str(job["seed"]),
        "--threads",
        "1",
        "--wall-limit",
        str(job["solver_wall_seconds"]),
        "--validation-seconds",
        str(job["validation_seconds"]),
    ]
    if maps_root is not None:
        command.extend(
            ["--maps-root", str(maps_root), "--map-seconds", str(job["map_seconds"])]
        )
    return command


def write_summary(output):
    rows = []
    for path in sorted((output / "jobs").glob("*/attempts/*/attempt.json")):
        own, status = read(path), native_record(path.parent)
        row = {
            "job_id": own["job"]["id"],
            "stage": own["job"]["stage"],
            "source_stage": own["job"].get("source_stage", own["job"]["stage"]),
            "k": own["job"]["k"],
            **own["job"]["dimensions"],
            "memory_limit_gib": own["job"]["memory_gib"],
            "attempt": path.parent.name,
            "status": own["status"],
            "stop_reason": own.get("stop_reason"),
            "state": status.get("state", "unvalidated_failed"),
            "qualified_baseline": False,
            "validation_passed": status.get("validation_passed", False),
            "objective": status.get("objective"),
            "bound": status.get("bound"),
            "gap": status.get("gap"),
            "bound_scope": status.get("bound_scope"),
            "termination_reason": status.get("termination_reason"),
            "native_global_lower_bound": status.get("native_global_lower_bound"),
            "bound_provenance": status.get("bound_provenance"),
            "global_lower_bound_independently_certified": False,
            "solver_returncode": status.get("solver_returncode"),
            "validation_returncode": status.get("validation_returncode"),
            "map_state": status.get("map_state"),
            "map_png": status.get("map_png"),
            "map_pdf": status.get("map_pdf"),
            "map_error": status.get("map_error"),
            "sampled_rss_peak_bytes": own.get("sampled_rss_peak_bytes"),
            "sampled_max_process_virtual_peak_bytes": own.get(
                "sampled_max_process_virtual_peak_bytes"
            ),
            "group_cleanup_complete": own.get("group_cleanup_complete"),
            "path": str(path.parent),
        }
        if completed_attempt(path.parent, own["signature"]):
            row["qualified_baseline"] = status.get("qualified_baseline", False)
        elif row["state"] in ("qualified_baseline", "validated_not_qualified"):
            row["state"] = (
                "validated_abnormal_termination"
                if row["validation_passed"]
                else "unvalidated_failed"
            )
        rows.append(row)
    atomic_json(
        output / "summary.json",
        {
            "updated_utc": utc(),
            "attempts": rows,
            "bound_note": "Native reported global bounds retain their provenance; primal validation is not independent bound certification.",
            "memory_note": "RLIMIT_AS hard per process; group RSS sampled only; swap not disabled.",
        },
    )
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(output / "summary.csv")


def execute(args, suite, jobs, resources):
    require_linux()
    if not args.binary or not args.build_manifest or not args.output:
        raise ValueError("Execution requires --binary, --build-manifest and --output")
    binary, manifest, output = (
        args.binary.resolve(),
        args.build_manifest.resolve(),
        args.output.resolve(),
    )
    identity = run_identity(suite, binary, manifest, jobs, resources)
    selected = select_jobs(jobs, args)
    if output.exists() and not args.resume:
        raise FileExistsError(
            "Output already exists; use --resume for the same frozen run"
        )
    output.mkdir(parents=True, exist_ok=True)
    active, finished, skipped, stop_reason = [], [], [], None
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    with OutputLock(output):
        identity_path = output / "run_identity.json"
        if identity_path.exists():
            if not args.resume or read(identity_path) != identity:
                raise ValueError(
                    "Source/binary/inputs/matrix/resource policy changed; use a new output"
                )
        elif args.resume:
            raise ValueError("No previous run identity exists in this output")
        else:
            atomic_json(identity_path, identity)
        ensure_no_live_attempts(output)
        pending = []
        for job in selected:
            previous = latest_attempt(output, job)
            if previous and completed_attempt(
                previous,
                job_signature(job, identity),
            ):
                skipped.append(
                    {
                        "job": job["id"],
                        "reason": "verified_completed",
                        "attempt": str(previous),
                    }
                )
            elif previous and not args.retry_failed:
                skipped.append(
                    {
                        "job": job["id"],
                        "reason": "prior_unsuccessful_attempt_requires_retry_failed",
                        "attempt": str(previous),
                    }
                )
            else:
                pending.append(job)
        atomic_json(
            output / f"invocation_{time.time_ns()}.json",
            {
                "started_utc": utc(),
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "skipped": skipped,
            },
        )
        if not (output / "host_environment.json").exists():
            atomic_json(output / "host_environment.json", host_environment())
        last_progress, last_check, low_count, cooldown = time.monotonic(), 0.0, 0, 0.0

        def changed():
            try:
                return (
                    run_identity(suite, binary, manifest, jobs, resources) != identity
                )
            except (OSError, ValueError, KeyError):
                return True

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"controller received signal {signum}")

        previous_handlers = {
            s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            while pending or active:
                now = time.monotonic()
                if now - last_check >= 30:
                    if changed():
                        stop_reason = "source_binary_suite_or_tools_changed"
                    last_check = now
                for item in active[:]:
                    process = item["process"]
                    done = process.tick()
                    item["sample"] = process.last_sample
                    item["stopping"] = process.stop_reason is not None
                    if not done:
                        continue
                    own = {
                        **item["own"],
                        **process.record(),
                        "finished_utc": utc(),
                        "status": "failed",
                    }
                    if own.get("group_cleanup_complete") is not True:
                        stop_reason = "owned_group_cleanup_unconfirmed"
                    if (
                        process.returncode == 0
                        and not process.stop_reason
                        and own.get("group_cleanup_complete") is True
                    ):
                        own["status"] = "completed"
                    own["artifact_sha256"] = attempt_hashes(item["attempt"])
                    atomic_json(item["attempt"] / "attempt.json", own)
                    if own["status"] == "completed" and not completed_attempt(
                        item["attempt"], own["signature"]
                    ):
                        own.update(
                            status="failed",
                            stop_reason="artifact_or_validation_check_failed",
                        )
                        atomic_json(item["attempt"] / "attempt.json", own)
                    process.close()
                    active.remove(item)
                    finished.append({"job": item["job"]["id"], "status": own["status"]})
                    print(json.dumps(finished[-1]), flush=True)
                    last_progress = now
                    write_summary(output)
                memory = psutil.virtual_memory()
                victim, low_count = pressure_victim(
                    active, memory.available, resources, low_count, cooldown, now
                )
                if victim:
                    next(i for i in active if i["job"]["id"] == victim)["process"].stop(
                        "system_available_memory_floor"
                    )
                    cooldown = now + 15
                if pending and not stop_reason:
                    stage = min(STAGES.index(j["stage"]) for j in pending)
                    if active:
                        stage = min(
                            stage, min(STAGES.index(i["job"]["stage"]) for i in active)
                        )
                    for job in pending[:]:
                        if stop_reason:
                            break
                        if STAGES.index(job["stage"]) != stage or not reservation_fits(
                            job, active, memory.available, resources
                        ):
                            continue
                        if changed():
                            stop_reason = "source_binary_suite_or_tools_changed"
                            break
                        try:
                            unchanged = (
                                fingerprints(Path(job["instance"]))
                                == job["input_sha256"]
                            )
                        except (OSError, ValueError):
                            unchanged = False
                        if not unchanged:
                            stop_reason = "input_changed"
                            break
                        attempts = output / "jobs" / job["id"] / "attempts"
                        attempts.mkdir(parents=True, exist_ok=True)
                        serial = 1 + max(
                            [
                                int(p.name)
                                for p in attempts.iterdir()
                                if p.is_dir() and p.name.isdigit()
                            ]
                            or [0]
                        )
                        attempt = attempts / f"{serial:03d}"
                        attempt.mkdir()
                        own = {
                            "schema": "mlnd-server-attempt/2",
                            "job": job,
                            "signature": job_signature(job, identity),
                            "binary_sha256": identity["binary_sha256"],
                            "build_manifest_sha256": identity["build_manifest_sha256"],
                            "started_utc": utc(),
                            "status": "running",
                        }
                        atomic_json(attempt / "attempt.json", own)
                        try:
                            process = ControlledProcess(
                                make_command(
                                    job,
                                    binary,
                                    manifest,
                                    attempt,
                                    args.python or Path(sys.executable),
                                    suite.parent / "maps",
                                ),
                                ROOT,
                                attempt,
                                job["memory_bytes"],
                                job["job_wall_seconds"],
                                env,
                                sample_seconds=resources["sample_seconds"],
                            )
                        except Exception as error:
                            own.update(
                                status="failed",
                                stop_reason="containment_or_launch_failed",
                                error=str(error),
                                finished_utc=utc(),
                            )
                            atomic_json(attempt / "attempt.json", own)
                            stop_reason = "containment_or_launch_failed"
                            pending.remove(job)
                            break
                        active.append(
                            {
                                "job": job,
                                "process": process,
                                "attempt": attempt,
                                "own": own,
                                "sample": process.last_sample,
                                "stopping": False,
                            }
                        )
                        pending.remove(job)
                        last_progress = now
                        print(
                            json.dumps(
                                {"started": job["id"], "memory_gib": job["memory_gib"]}
                            ),
                            flush=True,
                        )
                        memory = psutil.virtual_memory()
                if (
                    pending
                    and not active
                    and not stop_reason
                    and now - last_progress >= resources["admission_timeout_seconds"]
                ):
                    stop_reason = "insufficient_available_memory_admission_timeout"
                atomic_json(
                    output / "suite_status.json",
                    {
                        "updated_utc": utc(),
                        "status": "draining" if stop_reason else "running",
                        "dispatch_stop_reason": stop_reason,
                        "active": [i["job"]["id"] for i in active],
                        "pending": [j["id"] for j in pending],
                        "finished_this_invocation": finished,
                        "skipped": skipped,
                        "reserved_bytes": sum(i["job"]["memory_bytes"] for i in active),
                        "system_available_bytes": memory.available,
                    },
                )
                if stop_reason and not active:
                    break
                if pending or active:
                    time.sleep(resources["sample_seconds"])
        except BaseException as error:
            cleanup_errors = []
            for item in active:
                try:
                    item["process"].stop("dispatcher_interrupted")
                    item["process"].close()
                    item["process"].tick()
                    atomic_json(
                        item["attempt"] / "attempt.json",
                        {
                            **item["own"],
                            **item["process"].record(),
                            "status": "interrupted",
                            "finished_utc": utc(),
                        },
                    )
                except Exception as cleanup_error:
                    cleanup_errors.append(
                        {"job": item["job"]["id"], "error": str(cleanup_error)}
                    )
            active.clear()
            atomic_json(
                output / "suite_status.json",
                {
                    "updated_utc": utc(),
                    "status": "interrupted",
                    "error": str(error),
                    "cleanup_errors": cleanup_errors,
                    "pending": [j["id"] for j in pending],
                },
            )
            raise
        finally:
            for item in active:
                item["process"].close()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            write_summary(output)
        failed = any(i["status"] != "completed" for i in finished) or any(
            i["reason"] != "verified_completed" for i in skipped
        )
        atomic_json(
            output / "suite_status.json",
            {
                "updated_utc": utc(),
                "status": "stopped"
                if stop_reason
                else "completed_with_failures"
                if failed
                else "completed",
                "dispatch_stop_reason": stop_reason,
                "active": [],
                "pending": [j["id"] for j in pending],
                "finished_this_invocation": finished,
                "skipped": skipped,
            },
        )
    return 1 if stop_reason or failed else 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite-root", type=Path, required=True)
    result.add_argument("--binary", type=Path)
    result.add_argument("--build-manifest", type=Path)
    result.add_argument("--output", "--out", type=Path)
    result.add_argument("--python", type=Path)
    result.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    result.add_argument("--job", action="append")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--dry-run", "--verify", action="store_true")
    result.add_argument("--max-parallel", type=int, default=8)
    result.add_argument("--memory-budget-gib", type=float)
    result.add_argument("--available-floor-gib", type=float)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.retry_failed and not args.resume:
            raise ValueError("--retry-failed requires --resume")
        suite, jobs, resources = load_suite(args.suite_root, args)
        selected = select_jobs(jobs, args)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "verified": True,
                        "actual_solver_started": False,
                        "jobs": len(jobs),
                        "selected_jobs": len(selected),
                        "stages": {
                            s: sum(j["stage"] == s for j in jobs) for s in STAGES
                        },
                        "resources": resources,
                        "method": "algorithm",
                        "suite": str(suite),
                    },
                    indent=2,
                )
            )
            return 0
        return execute(args, suite, jobs, resources)
    except KeyboardInterrupt:
        print(
            "Suite interrupted; owned-task cleanup requested and results preserved.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Suite stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
