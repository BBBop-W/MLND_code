"""Shared scientific-figure styling and exports."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlnd-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mlnd-xdg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _style():
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=path
            ).get_name()
            break
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#22313d",
            "axes.labelcolor": "#22313d",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def _save(fig, directory, name):
    fig.savefig(
        directory / (name + ".png"), dpi=180, bbox_inches="tight", pad_inches=0.18
    )
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _grid(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#e7e9ec", linewidth=0.7)
