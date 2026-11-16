"""Native validation and configuration do not require a second optimizer."""

import os
from pathlib import Path
import subprocess
import sys


def test_native_validation_and_cli_import_without_gurobi():
    code = """
import importlib.abc
import sys
class NoGurobi(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'gurobipy' or fullname.startswith('gurobipy.'):
            raise AssertionError('Independent validation imported Gurobi')
sys.meta_path.insert(0, NoGurobi())
from src.experiments.native import validate_native
from src.cli import main
assert callable(validate_native)
assert main(['methods']) == 0
assert 'gurobipy' not in sys.modules
"""
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
