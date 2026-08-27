"""Versioned contract shared by the MLND solver and result readers."""

from __future__ import annotations


# Bump both values whenever a mathematical constraint or reported field changes
# in a way that makes previous results non-comparable.
FORMULATION_ID = "mlnd-unified-chi-epsilon-energy-v3"
RESULT_SCHEMA_VERSION = 5
