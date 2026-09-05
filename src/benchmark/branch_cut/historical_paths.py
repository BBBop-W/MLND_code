"""Locate preserved development evidence after source-repository integration.

This module only selects a location.  Callers must independently validate the
frozen evidence hashes and contents before using any returned files.
"""

from __future__ import annotations

from pathlib import Path


def preserved_development_root(
    repository_root: Path,
    *,
    required_relative_path: Path,
    source_repository_root: Path | None = None,
) -> Path:
    """Prefer an explicit source, then local evidence, then the saved clone.

    An explicit source is authoritative even when missing: falling back could
    silently replace the evidence the caller requested.  With no evidence
    present, return the local location so the caller's normal read fails closed.
    """

    if required_relative_path.is_absolute() or ".." in required_relative_path.parts:
        raise ValueError("required evidence path must stay within its development root")
    relative_root = Path("results/branch_cut_paper/dev")
    if source_repository_root is not None:
        return source_repository_root.expanduser().resolve() / relative_root
    root = repository_root.expanduser().resolve()
    local = root / relative_root
    saved = (
        root
        / "results/branch_cut_paper/worktrees/bnc_incumbent_reuse_dev_20260831"
        / relative_root
    )
    for candidate in (local, saved):
        if (candidate / required_relative_path).is_file():
            return candidate
    return local
