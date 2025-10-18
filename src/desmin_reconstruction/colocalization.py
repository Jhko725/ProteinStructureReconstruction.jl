import jax
import jax.numpy as jnp
from jax.scipy.signal import correlate
from jaxtyping import Array, ArrayLike, Float, Int


@jax.jit
def determine_characteristic_size(
    img: Float[ArrayLike, " *dims"],
) -> tuple[Int[Array, " {img.ndim}"], list[Array]]:
    """Determines the characteristic feature size of the image along each dimension in
    pixel units, as proposed by S. V. Costes et al. Biophys. J. 86, 6 (2004).

    This is done by computing the autocorrelation of the image, then computing the
    full-width-half-maximum of the autocorrelation for each axis."""
    img = jnp.asarray(img)
    img_ = img - jnp.mean(img)
    autocorr = correlate(img_, img_, mode="full", method="fft")
    sizes = []
    autocorrelations = []
    for i in range(autocorr.ndim):
        autocorr_1d = jnp.mean(
            autocorr, axis=tuple(j for j in range(autocorr.ndim) if j != i)
        )
        autocorr_1d = autocorr_1d / jnp.max(autocorr_1d)
        autocorrelations.append(autocorr_1d)
        sizes.append(jnp.sum(autocorr_1d - 0.5 >= 0))
    return jnp.asarray(sizes), autocorrelations
