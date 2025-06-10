from functools import cached_property

import numpy as np
import scipy.spatial as scspatial
from jaxtyping import Float
from tqdm import tqdm


def calculate_voronoi_volumes(
    voronoi: scspatial.Voronoi,
) -> Float[np.ndarray, " {voronoi.npoints}"]:
    vol = np.zeros(voronoi.npoints)
    for i, reg_num in enumerate(tqdm(voronoi.point_region)):
        indices = voronoi.regions[reg_num]
        if -1 in indices:  # non-closed regions
            vol[i] = np.inf
        else:
            vol[i] = scspatial.ConvexHull(voronoi.vertices[indices]).volume
    return vol


class VoronoiAnalysis:
    voronoi: scspatial.Voronoi

    def __init__(self, localizations: Float[np.ndarray, "points dim"]):
        self.voronoi = scspatial.Voronoi(localizations)

    @property
    def points(self) -> Float[np.ndarray, "points dim"]:
        return self.voronoi.points

    @cached_property
    def region_volumes(self) -> Float[np.ndarray, " points"]:
        """
        Volumes of the Voronoi regions. The volume can be infinite.

        This is a cached property, meaning the first invocation will incur some
        computational cost.
        """
        return calculate_voronoi_volumes(self.voronoi)

    @property
    def region_density(self) -> Float[np.ndarray, " points"]:
        return 1 / self.region_volumes


def estimate_bounding_box(
    points: Float[np.ndarray, "points dim"],
) -> tuple[Float[np.ndarray, " dim"], Float[np.ndarray, " dim"]]:
    return np.min(points, axis=0), np.max(points, axis=0)


def random_uniform_like(
    localizations: Float[np.ndarray, "points dim"],
    rng: np.random.Generator = np.random.default_rng(),
) -> Float[np.ndarray, "points dim"]:
    """
    Create a spatially uniformly distributed localization pattern with the same number
    of points and dimensions as the given localization data.

    It is assumed that the points reside in a rectangular bounding region, which is
    estimated from the data.
    """
    b_min, b_max = estimate_bounding_box(localizations)
    points_rand = rng.uniform(size=localizations.shape)
    return b_min + (b_max - b_min) * points_rand
