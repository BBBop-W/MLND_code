"""Numeric and bilingual table formatting."""

from __future__ import annotations

import numpy as np


def _tr(en, cn, lang):
    return en if lang == "en" else cn


def _n(value, digits=2):
    if value is None:
        return "—"
    value = 0.0 if abs(value) < 0.5 * 10 ** (-digits) else value
    return f"{value:,.{digits}f}"


def _flow(value):
    return "<0.001" if 0 < value < 0.001 else f"{value:,.3f}"


def _table(headers, rows):
    return "\n".join(
        [
            "|" + "|".join(headers) + "|",
            "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
        ]
        + ["|" + "|".join(map(str, row)) + "|" for row in rows]
    )


def _setting(value, digits=3):
    a = np.asarray(value, float)
    if a.ndim == 0:
        return _n(float(a), digits)
    return f"{_n(float(a.min()),digits)}–{_n(float(a.max()),digits)}"
