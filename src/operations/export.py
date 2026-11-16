"""Persist operating metrics, path-level checks and tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from .validation import build_operations
from .text import render_markdown


def export_operations(data, result, output_dir):
    payload = build_operations(data, result)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in payload["tables"].items():
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out / (name + ".csv")).open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (out / "operations.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "operations_summary.json").write_text(
        json.dumps(payload["summary"], ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (out / "OPERATIONS.md").write_text(render_markdown(payload), encoding="utf-8")
    if not payload["summary"]["checks"]["passed"]:
        raise ValueError(
            f"Current-model operating validation failed: {payload['summary']['checks']}"
        )
    return payload
