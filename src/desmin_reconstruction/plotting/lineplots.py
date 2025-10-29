import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float


def line_and_bandplot(
    ax: plt.Axes,
    x: Float[np.ndarray, " N"],
    y: Float[np.ndarray, " N"],
    y_widths: Float[np.ndarray, " N"]
    | tuple[Float[np.ndarray, " N"], Float[np.ndarray, " N"]],
    color: str = "royalblue",
    alpha: float = 1.0,
    alpha_band: float = 0.3,
    **plot_kwargs,
) -> plt.Axes:
    ax.plot(x, y, color=color, alpha=alpha, **plot_kwargs)
    if isinstance(y_widths, tuple):
        width_lower, width_upper = y_widths
    else:
        width_lower, width_upper = y_widths, y_widths

    ax.fill_between(x, y - width_lower, y + width_upper, color=color, alpha=alpha_band)
    return ax
