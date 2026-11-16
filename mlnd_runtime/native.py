"""Validate portable native output without an optimizer or reporting framework."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .input import load_standard
from .storage import sha256, write_json
from .validation import validate_solution


def load_native_result(directory):
    """Read current task decisions and their compact physical witnesses."""
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    solution = json.loads((directory / "solution.json").read_text(encoding="utf-8"))
    scope = solution.pop("assignment_scope", None)
    if scope is not None:
        if result.get("assignment_scope", scope) != scope:
            raise ValueError("Native result and solution assignment_scope disagree")
        result["assignment_scope"] = scope
    for key in ("task_indexing", "task_table", "delivery_time_scope"):
        if key in solution:
            value = solution.pop(key)
            if key in result and result[key] != value:
                raise ValueError(f"Native result and solution {key} disagree")
            result[key] = value
    records = json.loads((directory / "path_flows.json").read_text(encoding="utf-8"))
    if set(records) != {"segments", "missions", "relocations"}:
        raise ValueError("Native path output must contain segments, missions, relocations")
    result["_solution"] = {key: np.asarray(value, dtype=float) for key, value in solution.items()}
    for name, values in result["_solution"].items():
        if not np.isfinite(values).all():
            raise ValueError(f"Native decision array must be finite: {name}")
    for key, value in records.items():
        result["_" + key] = value
    return result


def validate_native(directory, instance, *, max_orders=None):
    """Check the original current-model incumbent and write its audit record.

    Complete physical-domain reconstruction is mandatory. The caller may impose
    an external validation deadline; an interrupted or rejected validation does
    not create a successful audit. A feasible primal does not certify the native
    optimizer's global lower bound or optimality claim.
    """
    started = time.perf_counter()
    directory, instance = Path(directory).resolve(), Path(instance).resolve()
    output_names = ("result.json", "solution.json", "path_flows.json", "input_reference.json")
    output_hashes = {name: sha256(directory / name) for name in output_names}
    result = load_native_result(directory)
    reference = json.loads((directory / "input_reference.json").read_text(encoding="utf-8"))
    if not isinstance(reference, dict) or "manifest" not in reference or "max_orders" not in reference:
        raise ValueError("Native input reference must contain manifest and max_orders")
    k = max_orders if max_orders is not None else result.get("max_orders", reference["max_orders"])
    if isinstance(k, bool) or k not in (1, 2):
        raise ValueError("Cannot determine K=1/2 from result/provenance")
    if result.get("max_orders", k) != k or reference["max_orders"] != k:
        raise ValueError("Requested K differs from native run provenance")
    if (result.get("assignment_scope") != "task"
            or result.get("demand_formulation") != "preprocessed"
            or result.get("delivery_time_scope") != "courier_home"):
        raise ValueError("Only the current preprocessed task/courier-home model is supported")
    if reference.get("assignment_scope", "task") != "task":
        raise ValueError("Native input reference has an obsolete assignment scope")
    manifest = json.loads((instance / "manifest.json").read_text(encoding="utf-8"))
    if reference["manifest"] != manifest:
        raise ValueError("Native run manifest differs from the requested instance")
    manifest_hash = sha256(instance / "manifest.json")
    if result.get("input_manifest_sha256", manifest_hash) != manifest_hash:
        raise ValueError("Native result input manifest SHA-256 differs from the requested instance")
    inputs = load_standard(instance, case=f"K{k}")
    read_seconds = time.perf_counter() - started
    validation_started = time.perf_counter()
    summary = validate_solution(inputs, result)
    validation_seconds = time.perf_counter() - validation_started
    if any(sha256(directory / name) != expected for name, expected in output_hashes.items()):
        raise ValueError("Native output changed during independent validation")
    if (sha256(instance / "manifest.json") != manifest_hash
            or any(sha256(instance / name) != expected for name, expected in manifest["files"].items())):
        raise ValueError("Input changed during independent validation")
    report = {
        "passed": summary["checks"]["passed"],
        "validation_scope": "current_model_primal_and_complete_input_preprocessing",
        "global_lower_bound_independently_certified": False,
        "max_orders": k,
        "assignment_scope": "task",
        "delivery_time_scope": "courier_home",
        "instance": str(instance),
        "manifest_sha256": manifest_hash,
        "native_manifest_matches": True,
        "result_sha256": output_hashes["result.json"],
        "solution_sha256": output_hashes["solution.json"],
        "path_flows_sha256": output_hashes["path_flows.json"],
        "input_reference_sha256": output_hashes["input_reference.json"],
        "read_seconds": read_seconds,
        "validation_seconds": validation_seconds,
        "checks": summary["checks"],
        "summary": summary,
    }
    write_json(directory / "native_validation.json", report)
    return report
