"""Content fingerprints of the actual production modules."""

from pathlib import Path
import hashlib


def source_hashes():
    """Return source-relative SHA-256 values for all current Python modules."""
    source_root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(source_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(source_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
