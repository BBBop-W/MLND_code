"""Cross-process exclusion for memory-intensive benchmark solver batches.

The lock is deliberately advisory: every current benchmark entry point uses
the same stable path, while older or unrelated solver commands still require
the documented manual process check.  The lock file is persistent and must
not be deleted; ownership is represented by the operating-system lock on its
open file descriptor, not by file existence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import IO


DEFAULT_BENCHMARK_RESOURCE_LOCK = (
    Path(tempfile.gettempdir())
    / f"mlnd-branch-cut-benchmark-{os.getuid()}.lock"
)


class BenchmarkResourceLockError(RuntimeError):
    """The shared benchmark resource lock could not be acquired or managed."""


def _attach_cleanup_note(
    primary_error: BaseException,
    *,
    stage: str,
    cleanup_error: BaseException,
) -> None:
    primary_error.add_note(
        f"secondary resource-lock cleanup failure during {stage}: "
        f"{type(cleanup_error).__name__}: {cleanup_error}"
    )


def _close_preserving_primary(handle: IO[str], primary_error: BaseException) -> None:
    try:
        handle.close()
    except BaseException as cleanup_error:
        _attach_cleanup_note(
            primary_error,
            stage="close",
            cleanup_error=cleanup_error,
        )


@dataclass(slots=True)
class BenchmarkResourceLock:
    """Non-blocking, process-wide advisory lock held for one solver batch."""

    path: str | Path = DEFAULT_BENCHMARK_RESOURCE_LOCK
    command: str = "benchmark"
    _handle: IO[str] | None = field(default=None, init=False, repr=False)

    @property
    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()

    def acquire(self) -> BenchmarkResourceLock:
        if self._handle is not None:
            raise BenchmarkResourceLockError("benchmark resource lock is already held")
        target = self.resolved_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(target, flags, 0o600)
            handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise BenchmarkResourceLockError(
                f"cannot open benchmark resource lock {target}: {error}"
            ) from error

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            _close_preserving_primary(handle, error)
            raise BenchmarkResourceLockError(
                "another qualification or formal benchmark batch already holds "
                f"the shared resource lock: {target}"
            ) from error
        except OSError as error:
            _close_preserving_primary(handle, error)
            raise BenchmarkResourceLockError(
                f"cannot acquire benchmark resource lock {target}: {error}"
            ) from error

        try:
            handle.seek(0)
            handle.truncate()
            json.dump(
                {"command": self.command, "pid": os.getpid()},
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except BaseException as cleanup_error:
                _attach_cleanup_note(
                    error,
                    stage="unlock after ownership-record failure",
                    cleanup_error=cleanup_error,
                )
            _close_preserving_primary(handle, error)
            raise BenchmarkResourceLockError(
                f"cannot record benchmark resource-lock ownership {target}: {error}"
            ) from error
        self._handle = handle
        return self

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        cleanup_errors: list[tuple[str, BaseException]] = []
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except BaseException as error:
            cleanup_errors.append(("unlock", error))
        try:
            handle.close()
        except BaseException as error:
            cleanup_errors.append(("close", error))
        if not cleanup_errors:
            return
        first_stage, first_error = cleanup_errors[0]
        for stage, error in cleanup_errors[1:]:
            _attach_cleanup_note(
                first_error,
                stage=stage,
                cleanup_error=error,
            )
        if isinstance(first_error, (MemoryError, KeyboardInterrupt, SystemExit)):
            raise first_error
        if not isinstance(first_error, Exception):
            raise first_error
        raise BenchmarkResourceLockError(
            f"cannot release benchmark resource lock {self.resolved_path} during "
            f"{first_stage}: {first_error}"
        ) from first_error

    def __enter__(self) -> BenchmarkResourceLock:
        return self.acquire()

    def __exit__(
        self,
        exc_type: object,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, traceback
        try:
            self.release()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            _attach_cleanup_note(
                exc,
                stage="context-manager release",
                cleanup_error=cleanup_error,
            )


__all__ = [
    "BenchmarkResourceLock",
    "BenchmarkResourceLockError",
    "DEFAULT_BENCHMARK_RESOURCE_LOCK",
]
