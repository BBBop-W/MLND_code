"""Run one version-checked MLND task and independently validate its incumbent.

The solver and validator inherit this process's process group so a Linux suite
guardian can enforce its resource policy across the complete task. No existing
output directory is reused, and a validated incumbent from an abnormal exit is
retained without being reported as a normally completed experiment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_FILES = ("result.json", "solution.json", "path_flows.json", "input_reference.json")
THREAD_ENVIRONMENT = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
RESULT_FIELDS = (
    "objective", "bound", "gap", "termination_reason", "bound_scope", "wall",
    "optimization_completed", "has_feasible_solution", "target_gap_met", "solved",
    "pricing_complete", "pricing_complete_scope", "pricing_domain", "domain_complete",
    "support_complete", "catalogue_complete", "path_storage", "failure_stage",
    "assignment_scope", "delivery_time_scope", "demand_formulation", "max_orders",
    "requested_method", "method", "budget", "threads", "seed", "requested_gap",
    "resolved_configuration", "auto_disabled_accelerations", "solver_version",
    "solver_status", "solver_status_code", "has_solver_solution",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=invalid)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def input_fingerprints(instance):
    instance = Path(instance).resolve()
    manifest = read_json(instance / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.json" not in files:
        raise ValueError("Input manifest must list model.json and its SHA-256")
    hashes = {"manifest.json": digest(instance / "manifest.json")}
    for name, expected in files.items():
        path = (instance / name).resolve()
        if path == instance or not path.is_relative_to(instance):
            raise ValueError(f"Input manifest path leaves its instance: {name}")
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Input SHA-256 mismatch: {name}")
        hashes[name] = actual
    return hashes


def source_fingerprints(root=ROOT):
    root = Path(root).resolve()
    version = read_json(root / "SOURCE_VERSION.json")
    files = version.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("SOURCE_VERSION.json has no source file manifest")
    expected_paths = set(files)
    actual_paths = set()
    for folder, suffixes in (("cpp/src", {".cpp"}), ("cpp/include", {".hpp", ".h"}),
                             ("mlnd_runtime", {".py"})):
        actual_paths.update(p.relative_to(root).as_posix()
                            for p in (root / folder).rglob("*")
                            if p.is_file() and p.suffix in suffixes)
    declared_paths = {name for name in expected_paths
                      if name.startswith(("cpp/src/", "cpp/include/", "mlnd_runtime/"))}
    if actual_paths != declared_paths:
        raise ValueError("Source files differ from the SOURCE_VERSION file set")
    hashes = {}
    for name, entry in files.items():
        path = (root / name).resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError(f"Source manifest path leaves repository: {name}")
        hashes[name] = digest(path)
        if hashes[name] != entry.get("sha256"):
            raise ValueError(f"Source differs from SOURCE_VERSION.json: {name}")
    return hashes


def verify_build(binary, manifest_path, root=ROOT):
    """Match actual binary and sources; paths in a moved build are informational."""
    binary, manifest_path, root = Path(binary), Path(manifest_path), Path(root)
    manifest = read_json(manifest_path)
    sources = source_fingerprints(root)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported build manifest schema_version")
    if manifest.get("source_sha256") != sources:
        raise ValueError("Build manifest does not match the current source files")
    if manifest.get("source_version_sha256") != digest(root / "SOURCE_VERSION.json"):
        raise ValueError("Build manifest does not match SOURCE_VERSION.json")
    if manifest.get("binary_sha256") != digest(binary):
        raise ValueError("Binary SHA-256 differs from its build manifest; rebuild it")
    if not os.access(binary, os.X_OK):
        raise ValueError(f"Native binary is not executable: {binary}")
    return manifest, sources


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Expected a finite nonnegative number")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path)
    p.add_argument("--build-manifest", type=Path)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", "--out", dest="output", type=Path, required=True)
    p.add_argument("--k", type=int, choices=(1, 2), required=True)
    p.add_argument("--method", choices=("algorithm", "benders_cg", "full_path"), default="algorithm")
    p.add_argument("--seconds", type=positive, default=7200.)
    p.add_argument("--gap", type=nonnegative, default=.01)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--solver-wall-seconds", "--wall-limit", dest="wall_limit", type=positive)
    p.add_argument("--validation-seconds", type=positive, default=1800.)
    p.add_argument("--maps-root", type=Path, help="Optional existing suite/maps directory for offline maps")
    p.add_argument("--map-seconds", type=positive, default=90.)
    p.add_argument("--disable", action="append", default=[])
    p.add_argument("--export-catalogue", action="store_true")
    p.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    return p


class RunInterrupted(Exception):
    def __init__(self, signum):
        super().__init__(f"Received signal {signum}")
        self.signum = signum
        self.returncode = None


@contextmanager
def interruption_handlers():
    def interrupted(signum, frame):
        raise RunInterrupted(signum)
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def stop_owned_child(process):
    """Only signal this live Popen child; never use PIDs recovered from disk."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def linux_rss(pid):
    """Sample direct-child RSS; suite guardian separately measures its group."""
    try:
        rows = Path(f"/proc/{pid}/status").read_text().splitlines()
        return max((int(line.split()[1]) * 1024 for line in rows
                    if line.startswith(("VmRSS:", "VmHWM:"))), default=0)
    except (OSError, ValueError):
        return None


def invoke(command, output, prefix, timeout):
    started = time.monotonic()
    peak = None
    timed_out = False
    with (output / f"{prefix}.stdout.log").open("wb") as stdout, \
         (output / f"{prefix}.stderr.log").open("wb") as stderr:
        # Inherit the guardian's process group and resource limits.
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   env={**os.environ, **THREAD_ENVIRONMENT})
        try:
            while True:
                sample = linux_rss(process.pid)
                if sample is not None:
                    peak = max(peak or 0, sample)
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    stop_owned_child(process)
                    break
                try:
                    process.wait(timeout=min(.25, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
        except RunInterrupted as error:
            stop_owned_child(process)
            error.returncode = process.returncode
            raise
        finally:
            stop_owned_child(process)
    return {"returncode": process.returncode, "timed_out": timed_out,
            "wall_seconds": time.monotonic() - started,
            "sampled_peak_rss_bytes": peak, "rss_scope": "direct_child_linux_VmHWM_sampling"}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def native_termination_error(record):
    """A retained feasible point does not turn an internal solver abort into success."""
    reason, status = record.get("termination_reason"), record.get("solver_status")
    method = record.get("native_method", record.get("requested_method", record.get("method")))
    if method == "full_path":
        # CPLEX's optimal and configured time-limit stops are expected. Other
        # stops (including memory/numerical/user aborts) retain their evidence
        # but must not be accepted as a normal completed comparison attempt.
        expected = {"Optimal", "OptimalTol", "AbortTimeLim", "AbortDetTimeLim", "time_limit"}
        expected_reasons = expected | {"gap_limit", "optimal_with_repaired_incumbent",
                                      "solver_optimal_validated_gap_above_target"}
        if status not in expected or reason not in expected_reasons:
            return f"Unexpected full-path termination: solver_status={status!r}, reason={reason!r}"
    else:
        # Benders reports its own stopping reasons, sometimes with a master_
        # prefix. Keep ordinary budget/gap/certificate stops unchanged.
        for value in (reason, status):
            if isinstance(value, str) and any(token in value.lower() for token in (
                    "memlim", "memory", "numbest", "optimalinfeas", "failfeas", "failinfeas",
                    "abortuser", "numerical", "validation_failed", "exception", "error",
                    "infeasible", "unbounded", "inforunbd")):
                return f"Abnormal native termination: {value}"
    return None


def classify(record):
    """Keep process completion, primal validity and target gap as distinct facts."""
    validated = record.get("validation_passed") is True
    provenance = all(record.get(key) is True for key in
                     ("source_unchanged", "input_unchanged", "binary_unchanged", "runner_unchanged",
                      "build_manifest_unchanged"))
    record["native_termination_error"] = native_termination_error(record)
    normal = (record.get("solver_returncode") == 0
              and record.get("validation_returncode") == 0
              and not record.get("solver_timed_out")
              and not record.get("validation_timed_out")
              and record.get("optimization_completed") is True
              and record.get("has_feasible_solution") is True
              and not record.get("error") and not record["native_termination_error"]
              and provenance and validated)
    objective, bound, gap = (record.get(key) for key in ("objective", "bound", "gap"))
    bound_consistent = (all(finite_number(value) for value in (objective, bound, gap))
                        and gap >= 0 and bound <= objective + 1e-7 * (1 + abs(objective)))
    if bound_consistent:
        recomputed = max(0., objective - bound) / max(1., abs(objective))
        bound_consistent = abs(recomputed - gap) <= 1e-7 * (1 + abs(gap))
    global_bound = bound_consistent and record.get("bound_scope") == "global"
    meets_gap = bool(global_bound and gap <= record["target_gap"])
    qualified = bool(normal and meets_gap)
    state = ("qualified_baseline" if qualified else "validated_not_qualified" if normal
             else "validated_abnormal_termination" if validated else "unvalidated_failed")
    record.update(state=state, qualified_baseline=qualified, target_gap_met_by_wrapper=meets_gap,
                  native_global_lower_bound=bound if global_bound else None,
                  global_lower_bound_independently_certified=False,
                  bound_provenance="native_result_global_bound" if global_bound else "not_usable_as_global_bound",
                  bound_note="Independent validation checks the primal and bound/gap consistency; it does not independently certify the global lower bound.")
    if record.get("status") != "interrupted":
        record["status"] = "completed" if normal else "failed"
    return record


def capture_result(record, output, args):
    result_path = output / "result.json"
    if not result_path.is_file():
        return False
    result = read_json(result_path)
    if not isinstance(result, dict):
        raise ValueError("Native result.json must be an object")
    record.update({key: result.get(key) for key in RESULT_FIELDS})
    record["result_sha256"] = digest(result_path)
    expected = {"assignment_scope": "task", "delivery_time_scope": "courier_home",
                "demand_formulation": "preprocessed", "max_orders": args.k,
                "requested_method": record["native_method"]}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Native output has unexpected {key}: {result.get(key)!r}")
    for key, expected_hash in (("binary_sha256", record["binary_sha256"]),
                               ("input_manifest_sha256", record["input_sha256"]["manifest.json"])):
        if key in result and result[key] != expected_hash:
            raise ValueError(f"Native output provenance mismatch: {key}")
    # Failed native finalization can omit numeric option metadata. Normal output cannot.
    if record.get("solver_returncode") == 0:
        for key, value in (("threads", args.threads), ("seed", args.seed),
                           ("requested_gap", args.gap), ("budget", args.seconds)):
            if result.get(key) != value:
                raise ValueError(f"Native output has unexpected {key}: {result.get(key)!r}")
    return True


def run(args):
    instance, output = resolve(args.input), resolve(args.output)
    if args.binary is None:
        raise ValueError("--binary is required for a solve")
    binary = resolve(args.binary)
    manifest_path = resolve(args.build_manifest) if args.build_manifest else binary.parent / "build_manifest.json"
    if output.is_relative_to(instance):
        raise ValueError("Output directory must not be inside the input directory")
    if args.threads < 1 or not 0 <= args.seed <= 2147483647:
        raise ValueError("threads must be positive and seed must be an unsigned 31-bit integer")
    wall_limit = args.wall_limit if args.wall_limit is not None else args.seconds + 300.
    if wall_limit < args.seconds:
        raise ValueError("External solver wall limit must cover the native budget")
    build, source = verify_build(binary, manifest_path)
    inputs = input_fingerprints(instance)
    native_method = "benders_cg" if args.method in ("algorithm", "benders_cg") else "full_path"
    command = [str(binary), "--method", native_method, "--input", str(instance), "--out", str(output),
               "--k", str(args.k), "--seconds", str(args.seconds), "--gap", str(args.gap),
               "--threads", str(args.threads), "--seed", str(args.seed)]
    for name in sorted(set(args.disable)):
        command.extend(["--disable", name])
    if args.export_catalogue:
        command.append("--export-catalogue")
    runner = Path(__file__).resolve()
    record = dict(schema_version=1, status="running", phase="solver", started_utc=utc(),
                  output=str(output), input=str(instance), binary=str(binary), native_method=native_method,
                  max_orders=args.k, command=command, binary_sha256=digest(binary), input_sha256=inputs,
                  source_sha256=source, source_version_sha256=digest(ROOT / "SOURCE_VERSION.json"),
                  runner_sha256=digest(runner), build_manifest=str(manifest_path),
                  build_manifest_sha256=digest(manifest_path), git_commit=build.get("git_commit"),
                  python=sys.version, platform=platform.platform(), target_gap=args.gap,
                  thread_environment=THREAD_ENVIRONMENT,
                  native_budget_seconds=args.seconds, solver_wall_limit_seconds=wall_limit,
                  validation_budget_seconds=args.validation_seconds,
                  map_state="pending" if args.maps_root else "not_requested",
                  map_budget_seconds=args.map_seconds,
                  validation_passed=False, solver_returncode=None, validation_returncode=None,
                  solver_timed_out=False, validation_timed_out=False,
                  disabled_accelerations=sorted(set(args.disable)))
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "task_status.json", record)
    started = time.monotonic()
    phase = "solver"
    try:
        with interruption_handlers():
            solver = invoke(command, output, "solver", wall_limit)
            record.update(solver_returncode=solver["returncode"], solver_timed_out=solver["timed_out"],
                          solver_process_seconds=solver["wall_seconds"], solver_resources=solver)
            if not capture_result(record, output, args):
                raise RuntimeError("Native solver produced no result.json")
            # Retain and validate saved incumbents even after a nonzero exit or timeout.
            if not all((output / name).is_file() for name in OUTPUT_FILES):
                raise RuntimeError("Saved incumbent is incomplete; physical validation cannot run")
            phase = record["phase"] = "validation"
            validation_command = [sys.executable, "-X", "utf8", str(runner), "--validate-only",
                                  "--input", str(instance), "--output", str(output), "--k", str(args.k)]
            record["validation_command"] = validation_command
            atomic_json(output / "task_status.json", record)
            validation = invoke(validation_command, output, "validation", args.validation_seconds)
            record.update(validation_returncode=validation["returncode"],
                          validation_timed_out=validation["timed_out"],
                          validation_seconds=validation["wall_seconds"], validation_resources=validation)
            report_path = output / "native_validation.json"
            if report_path.is_file():
                report = read_json(report_path)
                record["validation_passed"] = (report.get("passed") is True
                                                and validation["returncode"] == 0
                                                and not validation["timed_out"])
                record["validation_sha256"] = digest(report_path)
                if report.get("result_sha256") != digest(output / "result.json"):
                    raise ValueError("Physical validation does not refer to the final result.json")
            if not record["validation_passed"]:
                raise RuntimeError("Physical validation failed or exceeded its time limit")
            if args.maps_root:
                phase = record["phase"] = "maps"
                atomic_json(output / "task_status.json", record)
                map_command = [sys.executable, "-X", "utf8", str(ROOT / "scripts/server_plot.py"),
                               "--input", str(instance), "--result", str(output),
                               "--maps-root", str(resolve(args.maps_root))]
                record["map_command"] = map_command
                try:
                    rendering = invoke(map_command, output, "map", args.map_seconds)
                    record["map_resources"] = rendering
                    if rendering["timed_out"] or rendering["returncode"] != 0:
                        record["map_state"] = "timed_out" if rendering["timed_out"] else "failed"
                        with (output / "map.stderr.log").open("rb") as stream:
                            stream.seek(max(0, os.fstat(stream.fileno()).st_size - 4096))
                            record["map_error"] = stream.read().decode("utf-8", errors="replace")
                    else:
                        rendered = read_json(output / "map.stdout.log")
                        record.update(map_state=rendered["state"], map_png=rendered.get("png"),
                                      map_pdf=rendered.get("pdf"), map_sources=rendered.get("sources"))
                except (OSError, ValueError, KeyError) as error:
                    record.update(map_state="failed", map_error=f"{type(error).__name__}: {error}")
    except RunInterrupted as error:
        record.update(status="interrupted", error=str(error), interrupted_signal=error.signum,
                      failed_phase=phase)
        record[phase + "_returncode"] = error.returncode
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}", failed_phase=phase)
    finally:
        for key, check in (
            ("source_unchanged", lambda: source_fingerprints() == source),
            ("input_unchanged", lambda: input_fingerprints(instance) == inputs),
            ("binary_unchanged", lambda: digest(binary) == record["binary_sha256"]),
            ("runner_unchanged", lambda: digest(runner) == record["runner_sha256"]),
            ("build_manifest_unchanged", lambda: digest(manifest_path) == record["build_manifest_sha256"]),
        ):
            try:
                record[key] = check()
            except (OSError, ValueError, KeyError):
                record[key] = False
        changed = [key for key in ("source_unchanged", "input_unchanged", "binary_unchanged",
                                    "runner_unchanged", "build_manifest_unchanged") if not record[key]]
        if changed:
            record["provenance_error"] = "Run files changed or could not be verified: " + ", ".join(changed)
        record.update(finished_utc=utc(), wall_seconds=time.monotonic() - started, phase="finished")
        classify(record)
        atomic_json(output / "task_status.json", record)
        if args.maps_root:
            atomic_json(output / "map_status.json", {key: value for key, value in record.items()
                                                      if key.startswith("map_")})
        # A stable compact summary is consumed by the suite; detailed provenance
        # and resolved solver options remain in task_status.json and result.json.
        excluded = {"source_sha256", "input_sha256", "resolved_configuration", "validation_command", "command"}
        atomic_json(output / "summary.json", {key: value for key, value in record.items() if key not in excluded})
    return record


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.validate_only:
            from mlnd_runtime.native import validate_native
            report = validate_native(resolve(args.output), resolve(args.input), max_orders=args.k)
            print(json.dumps({"passed": report["passed"]}))
            return 0 if report["passed"] else 1
        record = run(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Experiment not started: {error}", file=sys.stderr)
        return 2
    print(json.dumps({key: record[key] for key in
                      ("status", "state", "validation_passed", "qualified_baseline", "output")}))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
