#!/usr/bin/env python3
"""Prepare an isolated current-source server batch, then run or resume it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from server_build import ROOT, checked_path, sha256, snapshot, write_json


def freeze_suite(source: Path, output: Path) -> dict:
    source = source.resolve()
    if not (source / "suite.json").is_file() and (source / "suite/suite.json").is_file():
        source = source / "suite"
    suite = json.loads((source / "suite.json").read_text())
    if suite.get("schema") != "mlnd-server-capacity-suite/1":
        raise ValueError("Unsupported suite.json; use the existing Linux capacity input suite")
    names = {"suite.json", "provenance/map_backgrounds.json"}
    for optional in ("TEST_MATRIX.csv", "instance_index.json"):
        if (source / optional).is_file():
            names.add(optional)
    for job in suite["jobs"]:
        instance_name = job["instance"]
        manifest_name = f"{instance_name}/manifest.json"
        manifest = json.loads(checked_path(source, manifest_name).read_text())
        names.add(manifest_name)
        for name, expected in manifest["files"].items():
            full_name = f"{instance_name}/{name}"
            if sha256(checked_path(source, full_name)) != expected:
                raise ValueError(f"Input manifest mismatch: {full_name}")
            names.add(full_name)
    names.update(str(p.relative_to(source)) for p in (source / "maps").rglob("*") if p.is_file())
    files = {name: sha256(checked_path(source, name)) for name in sorted(names)}
    output.mkdir(parents=True, exist_ok=False)
    for name in files:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
        if sha256(destination) != files[name]:
            raise ValueError(f"Input changed while copying: {name}")
    record = {"schema_version": 1, "source": str(source), "files": files,
              "jobs": len(suite["jobs"]), "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "SERVER_INPUT_SNAPSHOT.json", record)
    return record


def verify_files(root: Path, manifest_name: str) -> None:
    manifest = json.loads((root / manifest_name).read_text())
    for name, expected in manifest["files"].items():
        if sha256(checked_path(root, name)) != expected:
            raise ValueError(f"Frozen run file changed: {root / name}")


def current_run(runs: Path) -> Path | None:
    marker = runs / "CURRENT"
    if not marker.is_file():
        return None
    name = marker.read_text().strip()
    if not name.startswith("run-") or Path(name).name != name:
        raise ValueError("Invalid CURRENT run marker")
    run = runs / name
    if not run.is_dir():
        raise ValueError(f"Latest run directory is missing: {run}")
    return run


def show_status(run: Path | None) -> None:
    if run is None:
        print("No server batch has been prepared by this checkout.")
        return
    print(f"Run: {run}")
    print(f"Setup log: {run / 'setup.log'}")
    print(f"Batch log: {run / 'batch.log'}")
    print(f"Summary: {run / 'results/summary.csv'}")
    status = run / "results/suite_status.json"
    if status.is_file():
        value = json.loads(status.read_text())
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif (run / "prepared.json").is_file():
        print("Build completed; suite has not written status yet.")
    else:
        print("Preparation is incomplete. Read setup.log; a new start preserves this directory.")


def launch(run: Path, lock, args, config: dict) -> int:
    verify_files(run / "source", "SNAPSHOT.json")
    verify_files(run / "suite", "SERVER_INPUT_SNAPSHOT.json")
    manifest = json.loads((run / "build/build_manifest.json").read_text())
    if sha256(run / "build/mlnd") != manifest["binary_sha256"]:
        raise ValueError("Frozen binary changed")
    command = [sys.executable, str(run / "source/scripts/server_suite.py"),
               "--suite-root", str(run / "suite"), "--binary", str(run / "build/mlnd"),
               "--build-manifest", str(run / "build/build_manifest.json"),
               "--output", str(run / "results"), "--stage", config["stage"],
               "--max-parallel", str(config["max_parallel"])]
    if (run / "results/run_identity.json").exists():
        command.append("--resume")
    if args.retry_failed:
        command.append("--retry-failed")
    log = (run / "batch.log").open("ab")
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock.fileno(),))
    finally:
        log.close()
    write_json(run / "launcher.json", {"pid": process.pid, "command": command,
                                      "started_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Started PID {process.pid}; single solver thread, max parallel {config['max_parallel']}.")
    print("The background process retains the global batch lock; closing this terminal is safe.")
    time.sleep(1)
    if process.poll() is not None and process.returncode != 0:
        raise ValueError(f"Batch exited immediately with {process.returncode}; see {run / 'batch.log'}")
    show_status(run)
    return process.wait() if args.foreground else 0


def configure_cplex(cplex_root: str) -> None:
    directory = Path(cplex_root).resolve() / "cplex/bin/x86-64_linux"
    for name in ("PATH", "LD_LIBRARY_PATH"):
        os.environ[name] = str(directory) + os.pathsep + os.environ.get(name, "")


def runtime_environment() -> dict:
    return {"python": sys.version, "executable": sys.executable,
            "packages": {name: version(name) for name in ("numpy", "scipy", "psutil", "matplotlib", "Pillow")}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--suite-root", type=Path, default=os.environ.get("MLND_SUITE_ROOT"))
    parser.add_argument("--cplex-root", type=Path, default=os.environ.get("CPLEX_ROOT", "/home/autumn/workspace/wangsongtao/cplex"))
    parser.add_argument("--json-include", type=Path, default=Path(os.environ.get("CONDA_PREFIX", sys.prefix)) / "include")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--stage", choices=("core", "stress", "all"), default=None)
    parser.add_argument("--build-jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs/server_runs")
    args = parser.parse_args()
    try:
        runs = args.runs_root.resolve()
        if args.status:
            show_status(current_run(runs))
            return 0
        if not sys.platform.startswith("linux"):
            raise ValueError("START_SERVER.sh is the Linux entrypoint; use server_build.py/server_run.py for local checks")
        if args.max_parallel is not None and not 1 <= args.max_parallel <= 8:
            raise ValueError("--max-parallel must be between 1 and 8")
        if args.retry_failed and not args.resume:
            raise ValueError("--retry-failed requires --resume")
        runs.mkdir(parents=True, exist_ok=True)
        lock_path = Path(os.environ.get("MLND_LOCK_FILE", str(ROOT.parent / ".mlnd-capacity-run.lock")))
        # The child inherits this open file description. Never LOCK_UN here:
        # closing our descriptor leaves its exclusive lock held by the batch.
        inherited_fd = os.environ.pop("MLND_LAUNCH_LOCK_FD", None)
        lock_stream = os.fdopen(int(inherited_fd), "a+") if inherited_fd else lock_path.open("a+")
        with lock_stream as lock:
            if os.fstat(lock.fileno()).st_ino != lock_path.stat().st_ino:
                raise ValueError("Inherited batch lock does not match the configured lock file")
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"A server batch or setup already holds {lock_path}")
                show_status(current_run(runs))
                return 75
            if args.resume:
                run = current_run(runs)
                if run is None or not (run / "prepared.json").is_file():
                    raise ValueError("No prepared run to resume; start without --resume")
                config = json.loads((run / "prepared.json").read_text())
                if config.get("runtime_environment") != runtime_environment():
                    raise ValueError("Python environment differs from this frozen run; restore its recorded environment before resuming")
                for name in ("stage", "max_parallel"):
                    if getattr(args, name) is not None and getattr(args, name) != config[name]:
                        raise ValueError(f"Resume retains {name}={config[name]}; use a new batch for changed settings")
                configure_cplex(config["cplex_root"])
            else:
                suite = args.suite_root
                if suite is None:
                    candidates = [ROOT / "MLND_LINUX_ALL_20261115/suite",
                                  Path("/home/autumn/workspace/wangsongtao/MLND_code/MLND_LINUX_ALL_20261115/suite")]
                    suite = next((p for p in candidates if (p / "suite.json").is_file()), None)
                if suite is None:
                    raise ValueError("Existing experiment inputs not found; use --suite-root /path/to/MLND_LINUX_ALL_20261115/suite")
                name = datetime.now().strftime("run-%Y%m%d-%H%M%S-") + str(os.getpid())
                run = runs / name
                run.mkdir(exist_ok=False)
                (runs / "CURRENT.tmp").write_text(name + "\n")
                (runs / "CURRENT.tmp").replace(runs / "CURRENT")
                print(f"Preparing {run}", flush=True)
                snapshot(ROOT, run / "source")
                freeze_suite(suite, run / "suite")
                config = {"stage": args.stage or "all", "max_parallel": args.max_parallel or 8,
                          "python": sys.executable, "cplex_root": str(args.cplex_root.resolve()),
                          "runtime_environment": runtime_environment(),
                          "git_commit": json.loads((run / "source/SNAPSHOT.json").read_text())["git_commit"]}
                configure_cplex(config["cplex_root"])
                with (run / "setup.log").open("w") as log:
                    command = [sys.executable, str(run / "source/scripts/server_build.py"), "build",
                               "--output", str(run / "build"), "--cplex-root", str(args.cplex_root.resolve()),
                               "--json-include", str(args.json_include.resolve()), "--jobs", str(args.build_jobs)]
                    print(f"Building current source; progress: {run / 'setup.log'}", flush=True)
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                write_json(run / "prepared.json", config)
            if args.prepare_only:
                show_status(run)
                return 0
            return launch(run, lock, args, config)
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        print(f"Server start failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
