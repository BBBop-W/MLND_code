"""Non-overwriting filesystem claims for benchmark attempts."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import build_attempt_id, canonical_json


_RUN_ID_RE = re.compile(r"^run-[0-9a-f]{64}$")
ATOMIC_WRITE_TEMP_PREFIX = ".__mlnd_atomic_write__."


@dataclass(frozen=True, slots=True)
class AttemptClaim:
    """The exclusive directory reserved for one execution attempt."""

    logical_run_id: str
    attempt_id: str
    attempt_number: int
    directory: Path

    @property
    def claim_file(self) -> Path:
        return self.directory / "claim.json"


def write_canonical_json_exclusive(path: str | Path, payload: Any) -> Path:
    """Atomically publish one complete canonical artifact without overwriting.

    The payload is fully written and synced to a same-directory temporary file
    before an atomic hard-link publication.  A crash can therefore leave either
    no target or one complete target, never a partially written final artifact.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_json(payload) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{ATOMIC_WRITE_TEMP_PREFIX}{target.name}.",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    published = False
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        # link(2) is atomic and fails with FileExistsError if another writer has
        # already published this immutable target.  Unlike replace/rename, it
        # never overwrites the winner.
        os.link(temporary, target)
        published = True
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
        finally:
            if published:
                # Persist removal of the temporary directory entry as well.
                directory_descriptor = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
    return target


def attempt_directory(
    root: str | Path,
    logical_run_id: str,
    attempt_number: int,
) -> Path:
    """Return the canonical nested path for an attempt without creating it."""

    if _RUN_ID_RE.fullmatch(logical_run_id) is None:
        raise ValueError("logical_run_id must be a canonical run identity")
    attempt_id = build_attempt_id(logical_run_id, attempt_number)
    return Path(root) / logical_run_id / attempt_id


def claim_attempt_directory(
    root: str | Path,
    logical_run_id: str,
    attempt_number: int | None = None,
) -> AttemptClaim:
    """Atomically reserve an attempt directory using ``mkdir(exist_ok=False)``.

    With no explicit number, concurrent workers race safely over attempts
    1--9999: only one process can create a given directory, and losers advance
    to the next number.  Existing attempts are never reused or overwritten.
    """

    if _RUN_ID_RE.fullmatch(logical_run_id) is None:
        raise ValueError("logical_run_id must be a canonical run identity")
    candidates: Iterable[int]
    if attempt_number is not None:
        # Validate before creating any directory for an invalid request.
        build_attempt_id(logical_run_id, attempt_number)
        candidates = (attempt_number,)
    else:
        candidates = range(1, 10000)

    logical_root = Path(root) / logical_run_id
    logical_root.mkdir(parents=True, exist_ok=True)
    for number in candidates:
        directory = attempt_directory(root, logical_run_id, number)
        try:
            directory.mkdir(exist_ok=False)
        except FileExistsError:
            if attempt_number is not None:
                raise
            continue
        attempt_id = build_attempt_id(logical_run_id, number)
        claim = AttemptClaim(
            logical_run_id=logical_run_id,
            attempt_id=attempt_id,
            attempt_number=number,
            directory=directory,
        )
        write_canonical_json_exclusive(
            claim.claim_file,
            {
                "attempt_id": attempt_id,
                "attempt_number": number,
                "logical_run_id": logical_run_id,
            },
        )
        return claim
    raise RuntimeError(f"all 9999 attempts are already claimed for {logical_run_id}")
