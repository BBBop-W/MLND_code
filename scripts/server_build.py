#!/usr/bin/env python3
"""Freeze a Git checkout and build the current, version-checked native solver."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid relative path: {name}")
    path = root / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Expected a regular file inside source: {name}")
    return path


def source_hashes(root: Path) -> dict[str, str]:
    manifest = json.loads((root / "SOURCE_VERSION.json").read_text())
    expected = {name: row["sha256"] for name, row in manifest["files"].items()}
    actual = {name: sha256(checked_path(root, name)) for name in expected}
    if actual != expected:
        changed = [name for name in actual if actual[name] != expected[name]]
        raise ValueError(f"SOURCE_VERSION.json does not match source: {changed}")
    return actual


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def snapshot(root: Path, output: Path) -> dict:
    if git(root, "branch", "--show-current").decode().strip() != "MLND_sever":
        raise ValueError("Switch to the MLND_sever branch before starting a server batch")
    if git(root, "diff", "--name-only") or git(root, "diff", "--cached", "--name-only"):
        raise ValueError("Tracked files have local changes; commit or resolve them before starting")
    source_hashes(root)
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    output.mkdir(parents=True, exist_ok=False)
    # Read the fixed commit rather than racing later worktree edits/git add.
    # Avoid tar.extractall: only regular tracked files may enter the snapshot.
    before = {}
    with tarfile.open(fileobj=io.BytesIO(git(root, "archive", "--format=tar", commit))) as archive:
        for member in archive:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid archived path: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Snapshot requires regular files: {member.name}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = archive.extractfile(member).read()
            destination.write_bytes(data)
            destination.chmod(member.mode & 0o777)
            before[member.name] = hashlib.sha256(data).hexdigest()
    source_hashes(output)
    if (commit != git(root, "rev-parse", "HEAD").decode().strip()
            or git(root, "diff", "--name-only") or git(root, "diff", "--cached", "--name-only")):
        raise ValueError("Git checkout changed while copying the snapshot")
    record = {"schema_version": 1, "git_commit": commit, "branch": "MLND_sever",
              "created_utc": datetime.now(timezone.utc).isoformat(), "files": before}
    write_json(output / "SNAPSHOT.json", record)
    return record


def build(root: Path, directory: Path, cplex: Path, json_include: Path | None,
          cmake: str, jobs: int) -> dict:
    if not 1 <= jobs <= 2:
        raise ValueError("Build jobs must be 1 or 2")
    before = source_hashes(root)
    for name in ("cplex/include/ilcplex/ilocplex.h", "concert/include/ilconcert/iloenv.h"):
        if not (cplex / name).is_file():
            raise ValueError(f"Incomplete CPLEX/Concert installation: {cplex / name}")
    directory.mkdir(parents=True, exist_ok=False)
    command = [cmake, "-S", str(root), "-B", str(directory), "-DCMAKE_BUILD_TYPE=Release",
               f"-DCPLEX_ROOT={cplex}"]
    if json_include is not None:
        if not (json_include / "nlohmann/json.hpp").is_file():
            raise ValueError(f"nlohmann/json.hpp is missing under {json_include}")
        command.append(f"-DNLOHMANN_JSON_INCLUDE_DIR={json_include}")
    if os.environ.get("CONDA_PREFIX"):
        command.append(f"-DCMAKE_PREFIX_PATH={os.environ['CONDA_PREFIX']}")
    commands = [command, [cmake, "--build", str(directory), "--target", "mlnd", "--parallel", str(jobs)]]
    for command in commands:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    binary = directory / ("mlnd.exe" if os.name == "nt" else "mlnd")
    if not binary.is_file() or before != source_hashes(root):
        raise ValueError("Build failed to produce mlnd, or source changed during the build")
    snapshot_file = root / "SNAPSHOT.json"
    commit = (json.loads(snapshot_file.read_text())["git_commit"] if snapshot_file.exists()
              else git(root, "rev-parse", "HEAD").decode().strip())
    record = {"schema_version": 1, "git_commit": commit, "source_root": str(root),
              "source_sha256": before, "source_version_sha256": sha256(root / "SOURCE_VERSION.json"),
              "binary_path": str(binary), "binary_sha256": sha256(binary),
              "platform": platform.platform(), "machine": platform.machine(),
              "created_utc": datetime.now(timezone.utc).isoformat(), "commands": commands,
              "cmake_version": subprocess.check_output([cmake, "--version"], text=True).splitlines()[0]}
    if snapshot_file.exists():
        record["snapshot_sha256"] = sha256(snapshot_file)
    write_json(directory / "build_manifest.json", record)
    print(json.dumps({"binary": str(binary), "binary_sha256": record["binary_sha256"]}), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "build"))
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cplex-root", type=Path, default=os.environ.get("CPLEX_ROOT"))
    parser.add_argument("--json-include", type=Path)
    parser.add_argument("--cmake", default=shutil.which("cmake"))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    try:
        root, output = args.source_root.resolve(), args.output.resolve()
        if args.action == "snapshot":
            snapshot(root, output)
        else:
            if args.cplex_root is None or args.cmake is None:
                raise ValueError("Build requires --cplex-root (or CPLEX_ROOT) and cmake on PATH")
            build(root, output, args.cplex_root.resolve(), args.json_include, args.cmake, args.jobs)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Build preparation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
