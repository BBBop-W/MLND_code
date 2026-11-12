"""Run one frozen native experiment and its independent audit on Windows/POSIX.

No shell, scheduler, POSIX process groups, or Gurobi license is required.
The native executable owns its internal time budget; the wrapper additionally
limits process wall time, including final output. Runs are serial by invocation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ALNS_FLAGS = ("primal_alns", "primal_support_completion",
              "indexed_support_completion", "cg_primal_feedback")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def input_fingerprints(instance):
    manifest = json.loads((instance / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    if not isinstance(files, dict) or "model.json" not in files:
        raise ValueError("Input manifest must contain model.json and its SHA-256")
    checked = {"manifest.json": digest(instance / "manifest.json")}
    for name, expected in files.items():
        path = (instance / name).resolve()
        if not path.is_relative_to(instance) or path == instance:
            raise ValueError(f"Input manifest path leaves instance: {name}")
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Input manifest SHA-256 mismatch: {name}")
        checked[name] = actual
    return checked


def source_fingerprints():
    paths = []
    for folder in ("cpp/src", "cpp/include", "src", "scripts", "configs"):
        paths.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix in {".cpp", ".hpp", ".py", ".json"})
    paths.extend(ROOT / name for name in
                 ("pyproject.toml", "cpp/CMakeLists.txt", "cpp/Makefile", "SERVER_MANIFEST.json")
                 if (ROOT / name).is_file())
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Expected a finite positive number")
    return result


def nonnegative(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("Expected a finite nonnegative number")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--instance", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="New parent directory; never overwritten")
    p.add_argument("--preset", choices=("performance-20261105", "full-cplex-20261029"), required=True)
    p.add_argument("--k", type=int, choices=(1, 2), required=True)
    p.add_argument("--seconds", type=positive, default=600.)
    p.add_argument("--gap", type=nonnegative, default=.01)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--disable", action="append", default=[], help="Native acceleration name; repeatable")
    p.add_argument("--export-catalogue", action="store_true")
    p.add_argument("--wall-limit", type=positive, help="External process limit; default seconds + 120")
    p.add_argument("--validation-seconds", type=positive, default=300.)
    return p


def invoke(command, parent, prefix, timeout):
    with (parent / f"{prefix}.stdout.log").open("wb") as stdout, \
         (parent / f"{prefix}.stderr.log").open("wb") as stderr:
        return subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                              timeout=timeout, check=False).returncode


def run(args):
    binary, instance, output = args.binary.resolve(), args.instance.resolve(), args.out.resolve()
    if not binary.is_file():
        raise ValueError(f"Executable not found: {binary}")
    if args.threads < 1 or args.seed < 0:
        raise ValueError("threads must be positive and seed must be nonnegative")
    if args.wall_limit is not None and args.wall_limit < args.seconds:
        raise ValueError("External wall limit must cover the native budget")
    fingerprints = input_fingerprints(instance)
    values = json.loads((instance / "model.json").read_text(encoding="utf-8"))["values"]
    disabled = set(args.disable)
    if args.preset == "performance-20261105" and (args.k != 2 or values["n_l"] != 2):
        disabled.update(ALNS_FLAGS)
    command = [str(binary), "--preset", args.preset, "--input", str(instance),
               "--out", str(output / "run"), "--k", str(args.k),
               "--seconds", str(args.seconds), "--gap", str(args.gap),
               "--threads", str(args.threads), "--seed", str(args.seed)]
    for flag in sorted(disabled):
        command.extend(["--disable", flag])
    if args.export_catalogue:
        command.append("--export-catalogue")
    source = source_fingerprints()
    binary_hash = digest(binary)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                  command=command, binary_sha256=binary_hash, input_sha256=fingerprints,
                  source_sha256=source, python=sys.version, platform=platform.platform(),
                  disabled_accelerations=sorted(disabled), native_budget_seconds=args.seconds,
                  wall_limit_seconds=args.wall_limit or args.seconds + 120.,
                  validation_budget_seconds=args.validation_seconds,
                  validation_passed=False, source_unchanged=None, input_unchanged=None,
                  binary_unchanged=None, solver_returncode=None, validation_returncode=None)
    manifest_path = output / "experiment.json"
    write_json(manifest_path, record)
    started = time.perf_counter()
    phase = "solver"
    try:
        solver_started = time.perf_counter()
        record["solver_returncode"] = invoke(command, output, "solver", record["wall_limit_seconds"])
        record["solver_process_seconds"] = time.perf_counter() - solver_started
        result_path = output / "run" / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            for key in ("objective", "bound", "gap", "termination_reason", "wall", "pricing_complete",
                        "assignment_scope", "delivery_time_scope", "max_orders", "resolved_configuration"):
                record[key] = result.get(key)
            record["result_sha256"] = digest(result_path)
            if result.get("assignment_scope") != "task" or result.get("delivery_time_scope") != "courier_home":
                raise ValueError("Executable result is not the current task/courier-home model")
            if result.get("max_orders") != args.k:
                raise ValueError("Executable result has an unexpected K")
        if record["solver_returncode"] != 0:
            raise RuntimeError(f"Native solver exited with {record['solver_returncode']}; see solver.stderr.log")
        if not result_path.is_file():
            raise RuntimeError("Native solver produced no result.json")
        phase = "validation"
        validation_command = [sys.executable, "-X", "utf8", "-m", "src.experiments.native", "validate",
                              str(output / "run"), "--instance", str(instance), "--k", str(args.k)]
        record["validation_command"] = validation_command
        validation_started = time.perf_counter()
        record["validation_returncode"] = invoke(validation_command, output, "validation", args.validation_seconds)
        record["validation_seconds"] = time.perf_counter() - validation_started
        validation = json.loads((output / "run" / "native_validation.json").read_text(encoding="utf-8"))
        record["validation_passed"] = validation.get("passed") is True
        if record["validation_returncode"] != 0 or not record["validation_passed"]:
            raise RuntimeError("Independent physical validation failed; see validation.stderr.log")
        record["validation_sha256"] = digest(output / "run" / "native_validation.json")
        record["status"] = "completed"
    except subprocess.TimeoutExpired:
        record.update(status="failed", error=f"{phase}_external_wall_limit", failed_phase=phase)
    except KeyboardInterrupt:
        record.update(status="interrupted", failed_phase=phase, error="KeyboardInterrupt")
    except Exception as error:
        record.update(status="failed", failed_phase=phase, error=f"{type(error).__name__}: {error}")
    finally:
        try:
            record["source_unchanged"] = source_fingerprints() == source
            record["input_unchanged"] = input_fingerprints(instance) == fingerprints
            record["binary_unchanged"] = digest(binary) == binary_hash
            if not all(record[key] for key in ("source_unchanged", "input_unchanged", "binary_unchanged")):
                record.update(status="failed", provenance_error="Source, input or executable changed during run")
        except Exception as error:
            record.update(status="failed", provenance_error=f"{type(error).__name__}: {error}")
        record.update(wall_seconds=time.perf_counter() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(manifest_path, record)
    return record


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        record = run(args)
    except (OSError, ValueError, KeyError) as error:
        print(f"Experiment not started: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: record[k] for k in ("status", "validation_passed", "wall_seconds")}))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
