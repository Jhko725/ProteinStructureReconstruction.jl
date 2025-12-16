import numpy as np
import scipy.signal as scsignal
from jaxtyping import Float


def detect_Z_lines(
    z_band_markers: Float[np.ndarray, "N dim"],
    num_spatial_bins: int = 100,
    min_peak_separation: float = 1.2,
    min_relative_prominance: float = 0.01,
) -> tuple[
    Float[np.ndarray, " Z_lines"],
    tuple[
        Float[np.ndarray, " num_spatial_bins-1"],
        Float[np.ndarray, " num_spatial_bins-1"],
    ],
]:
    """Given coordinates of the z band markers, find the locations of the Z lines.

    It is assumed that the coordinates have been rotated so that the Z lines run
    parallel to the y axis, and the sarcomere extends along the x axis.
    The function returns an array of x coordinates corresponding to the location
    of the Z lines.

    Additionally, the function also returns the histogram of x
    coordinates that was used to estimate the Z line locations."""
    x = z_band_markers[:, 0]
    counts, bins = np.histogram(x, bins=num_spatial_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Find peaks with separation more than the known sarcomere length (~1.6um)
    peaks, _ = scsignal.find_peaks(
        counts,
        distance=min_peak_separation / (bin_centers[1] - bin_centers[0]),
        prominence=(len(x) * min_relative_prominance, None),
    )
    return bin_centers[peaks], (bin_centers, counts)


def nearest_Z_line_distances(
    points: Float[np.ndarray, "N dim"],
    z_line_locations: Float[np.ndarray, " Z_lines"],
    normalize: bool = True,
) -> Float[np.ndarray, " N"]:
    """Given coordinates of points, and locations of the Z lines, calculate the
    distances of each point to the nearest Z line.

    Like the function above, it is assumed that the coordinates have been rotated so
    that the Z lines run parallel to the y axis, and the sarcomere extends along the x
    axis.
    """
    x = points[:, 0:1]
    distances = np.min(np.abs(x - z_line_locations), axis=-1)

    if normalize:
        mean_sarcomere_length = np.mean(np.diff(z_line_locations))
        distances = distances / mean_sarcomere_length
    return distances
