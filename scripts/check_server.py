"""Serial correctness acceptance for a freshly built Windows/POSIX executable.

Build the probe targets first. Outputs go only to a new directory. This checks
small constructed cases, not production performance or bitwise portability.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROBES = (
    "progress", "solver_presets", "task_assignment", "preprocessed_demand",
    "catalogue_pricing", "point_assessment", "primal_feedback", "adaptive_pricing",
    "performance_state", "performance_integration", "stabilized_master",
    "queue_polymatroid", "queue_joint", "queue_joint_fractional", "design_face",
    "branch_pricing", "native_dual", "certified_cut_pool", "task_airline",
    "task_reduction", "pricing_state",
)


def executable(directory, name):
    candidate = directory / (name + (".exe" if os.name == "nt" else ""))
    if not candidate.is_file():
        raise FileNotFoundError(f"Build the {name} target first: {candidate}")
    return candidate


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--probes-only", action="store_true", help="Skip CSV and physical end-to-end checks")
    args = p.parse_args(argv)
    binaries = args.binary_dir.resolve()
    commands = []
    for name in PROBES:
        command = [str(executable(binaries, f"mlnd_{name}_probe"))]
        if name in {"performance_state", "pricing_state"}:
            command.append("--solve")
        if name == "progress":
            command.append(str(args.out.resolve() / "progress_fixture"))
        commands.append((name, command, 120))
    if not args.probes_only:
        for name in ("mlnd_cpp", "mlnd_input_probe", "mlnd_path_catalogue"):
            executable(binaries, name)
        commands.append(("task_migration_end_to_end", [
            sys.executable, "-X", "utf8", str(ROOT / "cpp/tests/task_migration_reference.py"),
            "--binary-dir", str(binaries), "--output", str(args.out.resolve() / "end_to_end"),
            "--seconds", "15"], 240))
    args.out.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                  serial=True, production_benchmark=False, passed=False, checks=[])
    for name, command, timeout in commands:
        started = time.perf_counter()
        record = dict(name=name, command=command, timeout_seconds=timeout)
        if name != "task_migration_end_to_end":
            record["binary_sha256"] = sha(Path(command[0]))
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                       timeout=timeout, check=False)
            record["exit_code"] = completed.returncode
            (args.out / f"{name}.stdout.log").write_bytes(completed.stdout)
            (args.out / f"{name}.stderr.log").write_bytes(completed.stderr)
            record["passed"] = completed.returncode == 0
            if record["passed"] and name != "task_migration_end_to_end":
                record["result"] = json.loads(completed.stdout)
                record["passed"] = record["result"].get("passed", True) is True
                record["binary_unchanged"] = sha(Path(command[0])) == record["binary_sha256"]
                record["passed"] &= record["binary_unchanged"]
        except subprocess.TimeoutExpired as error:
            record.update(passed=False, exit_code=124, error="external_wall_limit")
            (args.out / f"{name}.stdout.log").write_bytes(error.stdout or b"")
            (args.out / f"{name}.stderr.log").write_bytes(error.stderr or b"")
        except (OSError, ValueError) as error:
            record.update(passed=False, error=f"{type(error).__name__}: {error}")
        record["seconds"] = time.perf_counter() - started
        report["checks"].append(record)
        print(json.dumps({k: record[k] for k in ("name", "passed", "seconds")}), flush=True)
        report["passed"] = len(report["checks"]) == len(commands) and all(r["passed"] for r in report["checks"])
        (args.out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not record["passed"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
