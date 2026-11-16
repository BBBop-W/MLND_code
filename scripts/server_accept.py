#!/usr/bin/env python3
"""Run the small K1/K2 installation checks for a freshly built batch."""
from pathlib import Path
import sys

from server_start import acceptance

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: server_accept.py RUN_DIRECTORY")
    acceptance(Path(sys.argv[1]).resolve())
