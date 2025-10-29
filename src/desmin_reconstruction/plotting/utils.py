from matplotlib.axes import Axes


def make_legend(
    ax: Axes, handle_and_labels: tuple[list, list], loc: str = "best", **legend_kwargs
):
    handles = []
    labels = []
    for h, l in handle_and_labels:
        handles += h
        labels += l

    set_legend = ax.figure.legend if loc.startswith("outside") else ax.legend
    set_legend(handles=handles, labels=labels, loc=loc, **legend_kwargs)
    return ax
