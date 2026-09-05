"""Canonical serialization and domain-separated SHA256 experiment IDs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .schema import ModelIdentity, PairSpec, ProfileConfig, SolverConfig


IDENTITY_SCHEMA = "mlnd-branch-cut-identity-v1"
_IDENTITY_RE = re.compile(r"^(model|solver|profile|pair|run)-[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^run-[0-9a-f]{64}$")


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not permit NaN or infinity")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        return {
            key: _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("sets are not accepted because their order is not canonical")
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return UTF-8-safe, whitespace-free JSON with deterministic key order."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    """Hash a canonical JSON value with SHA256."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _identity(kind: str, payload: Any) -> str:
    digest = canonical_sha256(
        {
            "identity_schema": IDENTITY_SCHEMA,
            "kind": kind,
            "payload": payload,
        }
    )
    return f"{kind}-{digest}"


def build_model_id(model: ModelIdentity) -> str:
    return _identity("model", model)


def build_solver_id(solver: SolverConfig) -> str:
    return _identity("solver", solver)


def build_profile_id(profile: ProfileConfig) -> str:
    return _identity("profile", profile)


def build_pair_id(pair: PairSpec) -> str:
    """Identify one manifest/case/mode/seed/budget/common-solver comparison."""

    return _identity("pair", pair)


def build_logical_run_id(pair: PairSpec, profile: ProfileConfig) -> str:
    """Identify a profile run within a matched pair without attempt state."""

    return _identity(
        "run",
        {
            "pair_id": build_pair_id(pair),
            "profile_id": build_profile_id(profile),
        },
    )


def build_attempt_id(logical_run_id: str, attempt_number: int) -> str:
    """Append a bounded one-based attempt suffix (for example ``-a0001``)."""

    if _RUN_ID_RE.fullmatch(logical_run_id) is None:
        raise ValueError("logical_run_id must be a canonical run identity")
    if (
        isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or not 1 <= attempt_number <= 9999
    ):
        raise ValueError("attempt_number must be an integer in [1, 9999]")
    return f"{logical_run_id}-a{attempt_number:04d}"


def is_canonical_identity(value: str) -> bool:
    """Return whether *value* is a model/solver/profile/pair/run identity."""

    return isinstance(value, str) and _IDENTITY_RE.fullmatch(value) is not None
