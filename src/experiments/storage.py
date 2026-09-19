"""Portable inputs, solution files, and source snapshots for an experiment."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_payload(value):
    missing = []

    def convert(item, key="$"):
        if isinstance(item, (np.ndarray, np.generic)):
            return convert(item.tolist(), key)
        if isinstance(item, dict):
            return {str(k): convert(v, f"{key}.{k}") for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v, f"{key}[{i}]") for i, v in enumerate(item)]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, float) and not math.isfinite(item):
            missing.append(key)
            return None
        return item

    result = convert(value)
    if missing and isinstance(result, dict):
        result["nonfinite_fields_serialized_as_null"] = missing
    return result


def write_json(path, value):
    """Atomically replace one JSON file; readers never see a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_payload(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_arrays(path, arrays):
    for name, array in arrays.items():
        if np.asarray(array).dtype.hasobject:
            raise ValueError(f"Object arrays cannot be saved: {name}")
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_inputs(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    values = {k: v for k, v in data.items() if k not in arrays}
    _save_arrays(output / "input_arrays.npz", arrays)
    write_json(
        output / "input_metadata.json",
        {
            "schema": 1,
            "values": values,
            "arrays": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in arrays.items()
            },
        },
    )
    return {
        name: sha256(output / name)
        for name in ("input_arrays.npz", "input_metadata.json")
    }


def load_inputs(path):
    path = Path(path)
    metadata = json.loads((path / "input_metadata.json").read_text())
    data = metadata["values"]
    with np.load(path / "input_arrays.npz", allow_pickle=False) as arrays:
        data.update({key: arrays[key].copy() for key in arrays.files})
    return data


def save_solution(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if result.get("_solution") is not None:
        _save_arrays(output / "solution.npz", result["_solution"])
        paths = {
            "missions": result.get("_missions", []),
            "relocations": result.get("_relocations", []),
        }
        temporary = output / "path_flows.json.gz.tmp"
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(json_payload(paths), stream, allow_nan=False)
        temporary.replace(output / "path_flows.json.gz")
    write_json(
        output / "result.json",
        {k: v for k, v in result.items() if not k.startswith("_")},
    )


def load_seed(path):
    """Load a finished run, or the latest complete checkpoint of a running run."""
    path = Path(path)
    data = load_inputs(path)
    solution_path = path
    if not (path / "solution.npz").exists() and (path / "checkpoint.json").exists():
        relative = json.loads((path / "checkpoint.json").read_text())["directory"]
        solution_path = (path / relative).resolve()
        if path.resolve() not in solution_path.parents:
            raise ValueError("Checkpoint must belong to the experiment directory")
    result = json.loads((solution_path / "result.json").read_text())
    with np.load(solution_path / "solution.npz", allow_pickle=False) as arrays:
        result["_solution"] = {key: arrays[key].copy() for key in arrays.files}
    with gzip.open(
        solution_path / "path_flows.json.gz", "rt", encoding="utf-8"
    ) as stream:
        paths = json.load(stream)
    result["_missions"] = paths["missions"]
    result["_relocations"] = paths["relocations"]
    return data, result


def source_hashes():
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "main.py",
        ROOT / "pyproject.toml",
    ]
    paths += sorted((ROOT / "configs").glob("*.json"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths if p.is_file()}


def snapshot_sources(output):
    output = Path(output)
    manifest = {}
    for name, digest in source_hashes().items():
        source = ROOT / name
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[name] = {
            "source": str(source),
            "snapshot": str(target.relative_to(output)),
            "sha256": digest,
        }
    return manifest


def verify_sources(manifest):
    changed = [
        name
        for name, spec in manifest.items()
        if not Path(spec["source"]).is_file()
        or sha256(spec["source"]) != spec["sha256"]
    ]
    if changed:
        raise RuntimeError(f"Source files changed during the experiment: {changed}")


def snapshot_calibration(output, data=None):
    """The saved input arrays remain sufficient to replay without calibration files."""
    root = ROOT / "data" / "model_inputs"
    records = {}
    if data is None:
        candidates = list(root.rglob("*"))
    else:
        provenance = data.get("provenance", {})
        calibration = provenance.get("calibration", {})
        candidates = [
            Path(calibration[k])
            for k in ("source", "deadline_table", "pre_dispatch_statistics")
            if calibration.get(k)
        ]
        rider = provenance.get("rider_travel_time", {})
        if rider.get("calibration"):
            candidates.append(Path(rider["calibration"]))
    for source in sorted(set(candidates)):
        if source.is_file():
            relative = (
                source.relative_to(root)
                if source.is_relative_to(root)
                else Path(sha256(source)[:12]) / source.name
            )
            target = Path(output) / "calibration_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            records[str(relative)] = {"source": str(source), "sha256": sha256(target)}
    return records


# Kept for downstream file readers while all new call sites use the public names.
_write_json = write_json
_save_solution = save_solution
