"""Reproducible, bounded comparison of full-domain BDD root diagnostics.

The common continuous-bound/anchor initialization is charged to each BDD
budget. The full CPLEX control starts without those precomputed certificates.
This is a root-strengthening experiment, not production Benders integration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run_job(job: dict) -> dict:
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / "request.json", job)
    start = time.monotonic()
    record = {"K": job["K"], "method": job["method"], "status": "running",
              "started_unix": time.time(), "common_initialization_seconds": job["common_seconds"]}
    resources = []
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(job["command"], stdout=stdout, stderr=stderr)
        record["pid"] = process.pid
        save(directory / "execution.json", record)
        try:
            try:
                import psutil
                monitor = psutil.Process(process.pid)
            except (ImportError, OSError):
                monitor = None
            while process.poll() is None:
                elapsed = time.monotonic() - start
                if monitor is not None:
                    try:
                        cpu = monitor.cpu_times()
                        resources.append({"elapsed_seconds": elapsed, "rss_bytes": monitor.memory_info().rss,
                                          "cpu_user_seconds": cpu.user, "cpu_system_seconds": cpu.system})
                    except (OSError, psutil.Error):
                        monitor = None
                if elapsed > job["run_budget_seconds"] + 90:
                    record["watchdog_interrupted"] = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(1)
            process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    record.update(returncode=process.returncode, process_wall_seconds=time.monotonic() - start,
                  finished_unix=time.time(), status="completed" if process.returncode == 0 else "failed")
    record["charged_wall_seconds"] = record["process_wall_seconds"] + job["common_seconds"]
    result_path = Path(job["result_path"])
    if result_path.exists():
        result = read(result_path)
        record["result_sha256"] = digest(result_path)
        if job["method"] == "full_cplex":
            record.update(lower_bound=result.get("bound"), upper_bound=result.get("objective"),
                          gap=result.get("gap"), termination=result.get("termination_reason"),
                          feasible=result.get("has_feasible_solution"))
        else:
            search = result.get("search", {})
            lower = search.get("best_lower_bound")
            upper = search.get("validated_original_integer_upper_bound")
            record.update(lower_bound=lower, upper_bound=upper,
                          gap=(upper-lower)/max(1, abs(upper)) if lower is not None and upper is not None else None,
                          termination=result.get("status"),
                          root_gain_lower_bound=search.get("root_gain_lower_bound"),
                          cuts=len(search.get("cuts", [])), evaluations=search.get("evaluations_count"))
    save(directory / "resources.json", resources)
    save(directory / "execution.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("examples/tsl_2l20h50od"))
    parser.add_argument("--search-binary", type=Path, required=True)
    parser.add_argument("--full-binary", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()
    if args.seconds <= 0 or not 1 <= args.parallel <= 3:
        parser.error("Require positive seconds and 1 to 3 parallel jobs")
    root = args.root.resolve()
    experiment = root / "comparison"
    experiment.mkdir(parents=True, exist_ok=False)
    frozen = experiment / "frozen"
    frozen.mkdir()
    binaries = {}
    for name, source in (("search", args.search_binary), ("full", args.full_binary)):
        target = frozen / source.name
        shutil.copy2(source.resolve(), target)
        binaries[name] = target
    instance = experiment / "instance"
    shutil.copytree(args.input.resolve(), instance)
    prep = read(args.preparation.resolve())
    sources = {str(p): digest(p) for p in sorted(Path("cpp").rglob("*"))
               if p.is_file() and p.suffix in {".cpp", ".hpp"}}
    cases = []
    for item in prep["cases"]:
        k = int(item["K"])
        common = float(item["common_initialization_seconds"])
        remaining = args.seconds - common
        if remaining <= 30:
            raise ValueError(f"K={k} common initialization leaves only {remaining}s")
        anchor = frozen / f"anchors_K{k}.json"
        floor = frozen / f"continuous_K{k}.json"
        shutil.copy2(item["anchors_file"], anchor)
        shutil.copy2(item["continuous_file"], floor)
        for method in ("fixed", "search", "full_cplex"):
            directory = experiment / "runs" / f"K{k}_{method}"
            if method == "full_cplex":
                result = directory / "native" / "result.json"
                command = [str(binaries["full"]), "--input", str(instance), "--k", str(k),
                           "--out", str(directory / "native"), "--method", "full_path",
                           "--seconds", str(args.seconds), "--threads", "1", "--seed", "43",
                           "--gap", "0", "--tight-airline-bounds", "--path-column-limit", "0"]
            else:
                result = directory / "result.json"
                command = [str(binaries["search"]), str(instance), str(k), str(anchor), str(result),
                           "--continuous-bound-file", str(floor), "--method", method,
                           "--total-seconds", str(remaining), "--inner-seconds", "30",
                           "--master-seconds", "3", "--box-radius", "50", "--multipliers-per-root", "5",
                           "--max-evaluations", "2000", "--tight-airline-bounds", "true"]
            cases.append({"K": k, "method": method, "directory": str(directory),
                          "result_path": str(result), "command": command,
                          "common_seconds": 0 if method == "full_cplex" else common,
                          "run_budget_seconds": args.seconds if method == "full_cplex" else remaining})
    manifest = {"schema": "bdd-root-comparison/1", "created_unix": time.time(), "budget_seconds": args.seconds,
                "threads_per_solver": 1, "max_parallel": args.parallel, "seed": 43,
                "preparation_file": str(args.preparation.resolve()), "preparation_sha256": digest(args.preparation),
                "common_initialization_charged": True,
                "scope": "independent_root_diagnostic_vs_fixed_multipliers_and_full_MISOCP",
                "binary_hashes": {k: digest(v) for k, v in binaries.items()}, "source_hashes": sources,
                "frozen_files": {str(p.relative_to(experiment)): digest(p) for p in experiment.rglob("*") if p.is_file()},
                "jobs": cases}
    save(experiment / "manifest.json", manifest)
    completed = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_job, job) for job in cases]
        for future in as_completed(futures):
            completed.append(future.result())
            save(experiment / "summary.json", {"completed": len(completed), "total": len(cases), "runs": completed})
            print(json.dumps(completed[-1], ensure_ascii=False), flush=True)
    save(experiment / "summary.json", {"completed": len(completed), "total": len(cases), "runs": completed,
                                       "all_processes_returned_zero": all(r["returncode"] == 0 for r in completed)})


if __name__ == "__main__":
    main()
