"""
Utility helpers for lightweight Matplotlib visualizations.
"""

from __future__ import annotations

import base64
import io
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


def encode_line_plot(x: Sequence[float], ys: Iterable[Sequence[float]], labels: Iterable[str]) -> str:
    """
    Render a simple multi-line plot and return it as a base64-encoded PNG.
    """

    fig, ax = plt.subplots(figsize=(4, 3), dpi=120)

    for y, label in zip(ys, labels):
        ax.plot(x, y, label=label)

    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend(loc="best")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")
