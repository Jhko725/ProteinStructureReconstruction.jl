from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch


def violinplot(
    ax: Axes,
    data: Sequence[np.ndarray],
    x_ticks: Sequence[float] | None = None,
    x_ticklabels: Sequence[str] | None = None,
    edgecolor: str = "black",
    facecolor: str = "tab:blue",
    alpha: float = 1.0,
    linewidth: float = 1.0,
    label: str | None = None,
    **violinplot_kwargs,
):
    parts = ax.violinplot(list(data), **violinplot_kwargs)

    body: Patch
    for body in parts["bodies"]:
        body.set_edgecolor(edgecolor)
        body.set_facecolor(facecolor)
        body.set_alpha(alpha)
        body.set_linewidth(linewidth)

    handle = Patch(
        edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth, alpha=alpha
    )

    if x_ticks is None:
        x_ticks = np.arange(1, len(data) + 1)
    ax.set_xticks(x_ticks, x_ticklabels)

    return ax, ([handle], [label])
