import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float, Int
from matplotlib.colors import ListedColormap


def clustered_scatterplot(
    ax: plt.Axes,
    points: Float[np.ndarray, "N 2"],
    labels: Int[np.ndarray, " N"],
    cmap: str | ListedColormap = cc.m_glasbey_light,
    alpha: float = 0.7,
) -> plt.Axes:
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        ax.scatter(
            *points[labels == k].T,
            s=1 if k == -1 else 2,
            marker=".",
            facecolor=tuple(col),
            edgecolor=None,
            alpha=alpha,
        )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    ax.set_title(f"Estimated number of clusters: {n_clusters_}")

    return ax
