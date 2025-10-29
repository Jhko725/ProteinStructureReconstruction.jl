from dataclasses import dataclass

import numpy as np
from jaxtyping import Float


@dataclass
class BoundingSquare:
    x0: float
    y0: float
    side: float

    @property
    def x1(self) -> float:
        return self.x0 + self.side

    @property
    def y1(self) -> float:
        return self.y0 + self.side


def filter_bounded(
    points: Float[np.ndarray, "N dim"], bound: BoundingSquare
) -> Float[np.ndarray, "N_bounded dim"]:
    x, y = points[:, 0], points[:, 1]
    within_x = np.logical_and(x >= bound.x0, x < bound.x1)
    within_y = np.logical_and(y >= bound.y0, y < bound.y1)
    return points[np.logical_and(within_x, within_y)]
