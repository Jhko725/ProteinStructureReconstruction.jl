from collections.abc import Callable

import numpy as np
import sklearn
from hdbscan import HDBSCAN
from jaxtyping import Float
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def calibrate_tilt(
    points: Float[np.ndarray, "N 3"],
    sweep_angles: Float[np.ndarray, " N_angles"],
    *,
    degrees: bool = True,
) -> tuple[
    Callable, tuple[Float[np.ndarray, " N_angles"], Float[np.ndarray, " N_angles"]]
]:
    """Given an array of localization coordiates, sweep over the provided angle range
    to find optimal rotation angles about the y and x axes that minimizes the z range
    of the dataset."""

    def alignment_metric(z):
        z_quantiles = np.quantile(z, (0.25, 0.75))
        return z_quantiles[1] - z_quantiles[0]

    # Note that it is not that difficult to discard the dependency on scipy.spatial,
    # as 3D rotation matrices about x and y axes are quite simple.
    heights_theta = []
    for theta in tqdm(sweep_angles):
        r = Rotation.from_rotvec(theta * np.array([0, 1, 0]), degrees=degrees)
        points_rotated = r.apply(points)
        z = points_rotated[:, 2]
        heights_theta.append(alignment_metric(z))
        # heights_theta.append(np.max(z) - np.min(z))
    theta_opt = sweep_angles[np.argmin(heights_theta)]

    r1 = Rotation.from_rotvec(theta_opt * np.array([0, 1, 0]), degrees=degrees)
    points = r1.apply(points)

    heights_phi = []
    for phi in tqdm(sweep_angles):
        r = Rotation.from_rotvec(phi * np.array([1, 0, 0]), degrees=degrees)
        points_rotated = r.apply(points)
        z = points_rotated[:, 2]
        heights_phi.append(alignment_metric(z))
        # heights_phi.append(np.max(z) - np.min(z))
    phi_opt = sweep_angles[np.argmin(heights_phi)]

    r2 = Rotation.from_rotvec(phi_opt * np.array([1, 0, 0]), degrees=degrees)
    transform = lambda pts: r2.apply(r1.apply(pts))
    return transform, (np.asarray(heights_theta), np.asarray(heights_phi))


def find_optimal_clustering_angle(
    points: Float[np.ndarray, "N 3"],
    sweep_angles: Float[np.ndarray, " N_angles"],
    *,
    degrees: bool = True,
    **hdbscan_kwargs,
):
    clusterer = HDBSCAN(**hdbscan_kwargs)

    metrics = []
    for angle in tqdm(sweep_angles):
        r = Rotation.from_rotvec(angle * np.array([0, 0, 1]), degrees=degrees)
        rotated = r.apply(points)
        xz = rotated[:, np.array([0, 2])]
        clusterer.fit(xz)

        metrics.append(sklearn.metrics.calinski_harabasz_score(xz, clusterer.labels_))

    angle_opt = sweep_angles[np.argmax(metrics)]
    r = Rotation.from_rotvec(angle_opt * np.array([0, 0, 1]), degrees=degrees)
    transform_fn = lambda pts: r.apply(pts)
    return transform_fn, np.asarray(metrics)


def align_patch(
    data: dict[str, Float[np.ndarray, "?N 3"]],
    tilt_sweep_values: np.ndarray = np.arange(-5, 5, 0.001),
    rotation_sweep_values: np.ndarray = np.arange(-3, 3, 0.1),
    clustering_species: str = "actinin",
    **clustering_kwargs,
):
    """A convenience function that performs `calibrate_tilt` and
    `find_optimal_clustering_angle` in succession.
    """
    points_all = np.concatenate(list(data.values()), axis=0)
    tilt_calibration_fn, tilt_metrics = calibrate_tilt(points_all, tilt_sweep_values)

    rotation_fn, clustering_metric = find_optimal_clustering_angle(
        tilt_calibration_fn(data[clustering_species]),
        rotation_sweep_values,
        **clustering_kwargs,
    )

    transform_fn = lambda pts: rotation_fn(tilt_calibration_fn(pts))
    metrics = [
        (angles, metric)
        for angles, metric in zip(
            (tilt_sweep_values, tilt_sweep_values, rotation_sweep_values),
            [*tilt_metrics, clustering_metric],
        )
    ]

    return transform_fn, metrics
