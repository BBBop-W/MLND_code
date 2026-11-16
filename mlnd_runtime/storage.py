"""Streaming file fingerprints and atomic JSON output."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

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

