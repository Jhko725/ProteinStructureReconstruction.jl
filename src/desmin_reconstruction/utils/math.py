import math

import scipy.special as scspecial
from jaxtyping import ArrayLike


def n_sphere_surface_area(n: int, r: float | ArrayLike = 1.0) -> float | ArrayLike:
    """Calculates the surface area of an n-sphere with radius r.

    Reference: https://en.wikipedia.org/wiki/N-sphere#Volume_and_area"""
    coeff = 2 * math.pi ** (n / 2) / scspecial.gamma(n / 2)
    return coeff * r ** (n - 1)


def n_sphere_volume(n: int, r: float | ArrayLike = 1.0) -> float | ArrayLike:
    """Calculates the volume of an n-sphere with radius r.

    Reference: https://en.wikipedia.org/wiki/N-sphere#Volume_and_area"""
    coeff = math.pi ** (n / 2) / scspecial.gamma(n / 2 + 1)
    return coeff * r**n
