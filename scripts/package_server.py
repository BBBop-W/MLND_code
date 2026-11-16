"""Build and verify the small, allowlisted MLND source delivery (stdlib only).

This copies current working files, not a Git revision. No solver is invoked,
and no excluded data/output directory is traversed. A manifest detects accidental
changes; it is not a signature or a claim that the source passed Windows tests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

MANIFEST = "SERVER_MANIFEST.json"
SCHEMA = "mlnd-server-source/1"
ROOT = Path(__file__).resolve().parents[1]

# Directory AND suffix allowlists: neither repository-wide recursion nor Git
# tracking status can accidentally pull in output, raw data, binaries or licenses.
TREES = {
    "cpp/src": {".cpp"}, "cpp/include": {".hpp"}, "cpp/tests": {".cpp", ".hpp", ".py"},
    "src": {".py"}, "tests/unit": {".py"}, "tests/integration": {".py"},
    "scripts": {".py"}, "configs": {".json", ".md"},
}
FILES = (
    "README.md", "pyproject.toml", "main.py", "cpp/README.md", "cpp/Makefile",
    "cpp/CMakeLists.txt", "tests/conftest.py",
    "tests/fixtures/baseline_release_20261029/catalogue_export_probe.json",
    "tests/fixtures/baseline_release_20261029/full_catalogue.json",
    "tests/fixtures/baseline_release_20261029/inherited_audit_ast.json",
    "docs/server_runbook_20261112.md", "docs/current_baseline.md", "docs/model.md",
    "docs/algorithm.md", "docs/architecture.md", "docs/task_assignment_migration_20261105.md",
    "docs/acceleration_implementation_20261105.md", "docs/performance_implementation_20261105.md",
    "docs/general_l_cuts_20261109.md",
)
OPTIONAL_FILES = (
    "docs/project_assessment_20261112.md", "docs/project_assessment_20261112.validation.json",
)
BENCHMARK_FILES = (
    "README.md", "benchmark_manifest.json", "verify.py", "input_arrays.npz",
    "input_metadata.json", "metadata.json", "resolved_config.json", "candidate_sites.csv",
    "source_instance_protocol.json", "source_instance_audit.json",
)
EXAMPLE_FILES = (
    "README.md", "manifest.json", "verification.json", "metadata.json", "model.json",
    "sites.csv", "od_nodes.csv", "od_demand.csv", "periods.csv",
    "coefficients/base_costs.csv", "coefficients/locker_levels.csv", "coefficients/links.csv",
    "coefficients/od_costs.csv", "coefficients/od_hub.csv", "coefficients/kappa.csv",
)
EXCLUDED_PARTS = {"outputs", "results", "build", "dist", "__pycache__", "node_modules"}


def safe_name(name: str) -> str:
    """Require portable relative names, including Windows case/path constraints."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or path.as_posix() != name:
        raise ValueError(f"Noncanonical package path: {name!r}")
    for part in path.parts:
        if part in {".", ".."} or part.endswith((" ", ".")) or re.search(r'[\\:<>"|?*\x00-\x1f]', part):
            raise ValueError(f"Unsafe package path: {name!r}")
        if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])", part.split(".")[0]):
            raise ValueError(f"Reserved Windows package path: {name!r}")
    return name


def excluded(path: Path) -> bool:
    return any(part.startswith(".") or part.lower() in EXCLUDED_PARTS or
               part.lower().startswith(("license", "licence")) for part in path.parts)


def regular_file(root: Path, name: str) -> Path:
    safe_name(name)
    path = root
    for part in PurePosixPath(name).parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"Symlinks are not packaged or verified: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Required regular file missing: {name}")
    return path


def selected_files(root: Path) -> list[str]:
    root = root.resolve()
    names = set(FILES)
    names.update(name for name in OPTIONAL_FILES if (root / name).exists())
    names.update(f"data/benchmarks/tsl_2l20h50od/{name}" for name in BENCHMARK_FILES)
    names.update(f"examples/tsl_2l20h50od/{name}" for name in EXAMPLE_FILES)
    for folder, suffixes in TREES.items():
        base = root / folder
        ancestors = [root.joinpath(*Path(folder).parts[:i]) for i in range(1, len(Path(folder).parts) + 1)]
        if not base.is_dir() or any(path.is_symlink() for path in ancestors):
            raise ValueError(f"Required source directory missing or symlinked: {folder}")
        for directory, dirs, filenames in os.walk(base, followlinks=False):
            dirs[:] = [name for name in dirs if not excluded(Path(directory, name).relative_to(root))]
            for name in [*dirs, *filenames]:
                path = Path(directory, name)
                relative = path.relative_to(root)
                if excluded(relative):
                    continue
                if path.is_symlink():
                    raise ValueError(f"Symlink in allowlisted source tree: {relative}")
                if path.is_file() and path.suffix in suffixes:
                    names.add(relative.as_posix())
    folded = set()
    for name in sorted(names):
        regular_file(root, name)
        if name.casefold() in folded:
            raise ValueError(f"Case-insensitive path collision: {name}")
        folded.add(name.casefold())
    return sorted(names)


def fingerprint(data: bytes) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def checked_manifest(data: bytes) -> dict:
    manifest = json.loads(data)
    if manifest.get("schema") != SCHEMA or manifest.get("hash_algorithm") != "sha256":
        raise ValueError("Unsupported server manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or MANIFEST in files:
        raise ValueError("Invalid manifest file table")
    folded = set()
    for name, entry in files.items():
        safe_name(name)
        if name.casefold() in folded:
            raise ValueError(f"Case-insensitive manifest path collision: {name}")
        folded.add(name.casefold())
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size"} or \
                not isinstance(entry["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", entry["sha256"]) or \
                type(entry["size"]) is not int or entry["size"] < 0:
            raise ValueError(f"Invalid fingerprint: {name}")
    return manifest


def verify_package(target: Path, *, allow_extra_files: bool = False) -> dict:
    target = target.resolve()
    if target.is_dir():
        manifest_bytes = regular_file(target, MANIFEST).read_bytes()
        manifest = checked_manifest(manifest_bytes)
        if not allow_extra_files:
            actual = set()
            for path in target.rglob("*"):
                if path.is_symlink():
                    raise ValueError(f"Symlink in package: {path.relative_to(target)}")
                if path.is_file():
                    actual.add(path.relative_to(target).as_posix())
            expected = set(manifest["files"]) | {MANIFEST}
            if actual != expected:
                raise ValueError(f"Package file set differs: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        for name, expected in manifest["files"].items():
            if fingerprint(regular_file(target, name).read_bytes()) != expected:
                raise ValueError(f"SHA-256/size mismatch: {name}")
    else:
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP entries")
            for entry in archive.infolist():
                safe_name(entry.filename)
                if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f"Unexpected non-file ZIP entry: {entry.filename}")
            manifest_bytes = archive.read(MANIFEST)
            manifest = checked_manifest(manifest_bytes)
            expected_names = set(manifest["files"]) | {MANIFEST}
            if not expected_names.issubset(names) or (not allow_extra_files and set(names) != expected_names):
                raise ValueError("ZIP file set differs from manifest")
            for name, expected in manifest["files"].items():
                if fingerprint(archive.read(name)) != expected:
                    raise ValueError(f"SHA-256/size mismatch: {name}")
    return {"passed": True, "files": len(manifest["files"]),
            "bytes": sum(entry["size"] for entry in manifest["files"].values()),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "extra_files_checked": not allow_extra_files}


def create_package(source: Path, output: Path, *, make_zip: bool = False) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the source directory")
    archive_path = output.with_name(output.name + ".zip")
    if output.exists() or (make_zip and archive_path.exists()):
        raise FileExistsError("Use a new output directory and ZIP filename; existing files are never overwritten")
    names = selected_files(source)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in names:
        path = regular_file(source, name)
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed while packaging: {name}; incomplete package retained at {output}")
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
        files[name] = fingerprint(data)
    manifest = {"schema": SCHEMA, "hash_algorithm": "sha256",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": "current working files; independent of Git commit/dirty status",
                "scope": "allowlisted source, tests, documentation and frozen derived benchmarks",
                "excluded": ["outputs/results", "executables/build products", "Git history",
                             "solver installations/licenses", "raw user datasets", "old temporary trees"],
                "files": files}
    with (output / MANIFEST).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    report = verify_package(output)
    report["directory"] = str(output)
    if make_zip:
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted([*names, MANIFEST]):
                archive.write(output / name, name)
        verify_package(archive_path)
        report.update(archive=str(archive_path), archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Copy only allowed files to a new directory")
    create.add_argument("--source", type=Path, default=ROOT)
    create.add_argument("--out", type=Path, required=True)
    create.add_argument("--zip", action="store_true", help="Also create OUT.zip, without overwriting")
    verify = commands.add_parser("verify", help="Verify an extracted source directory or ZIP")
    verify.add_argument("target", type=Path)
    verify.add_argument("--allow-extra-files", action="store_true", help="Verify listed files after installing/building; do not scan extra files")
    args = parser.parse_args(argv)
    try:
        result = create_package(args.source, args.out, make_zip=args.zip) if args.command == "create" else \
            verify_package(args.target, allow_extra_files=args.allow_extra_files)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Package error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
