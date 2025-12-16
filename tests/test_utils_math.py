import math

import numpy as np
from desmin_reconstruction.utils.math import n_sphere_surface_area, n_sphere_volume


def test_n_sphere_surface_area():
    assert np.isclose(n_sphere_surface_area(2), 2 * math.pi)
    assert np.isclose(n_sphere_surface_area(3), 4 * math.pi)


def test_n_sphere_volume():
    assert np.isclose(n_sphere_volume(2), math.pi)
    assert np.isclose(n_sphere_volume(3), 4 * math.pi / 3)
