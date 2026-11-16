"""Portable arrays, atomic JSON and source fingerprints for reproducible runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_payload(value):
    def convert(item):
        if isinstance(item, (np.ndarray, np.generic)):
            return convert(item.tolist())
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    return convert(value)


def write_json(path, value):
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
    if metadata.get("schema") != 1:
        raise ValueError("Unsupported saved input schema")
    data = dict(metadata["values"])
    with np.load(path / "input_arrays.npz", allow_pickle=False) as arrays:
        for key in arrays.files:
            spec = metadata["arrays"].get(key)
            value = arrays[key]
            if (
                spec is None
                or list(value.shape) != spec["shape"]
                or str(value.dtype) != spec["dtype"]
            ):
                raise ValueError(f"Saved array metadata mismatch: {key}")
            data[key] = value.copy()
        if set(arrays.files) != set(metadata["arrays"]):
            raise ValueError("Saved array names disagree with metadata")
    return data


def save_solution(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if result.get("_solution") is not None:
        _save_arrays(output / "solution.npz", result["_solution"])
        paths = {
            name: result.get("_" + name, [])
            for name in ("missions", "relocations", "segments")
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
    for name in ("missions", "relocations", "segments"):
        result["_" + name] = paths.get(name, [])
    return data, result


def source_hashes():
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "main.py",
        ROOT / "pyproject.toml",
    ]
    paths += sorted((ROOT / "configs").rglob("*.json"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths if p.is_file()}
