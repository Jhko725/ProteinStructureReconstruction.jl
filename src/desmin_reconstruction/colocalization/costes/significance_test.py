from functools import partial, reduce
from operator import mul

import jax
import jax.numpy as jnp
from einops import rearrange
from jax.scipy.signal import correlate
from jaxtyping import Array, ArrayLike, Float, Int, PRNGKeyArray


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


@partial(jax.jit, static_argnums=1)
def scramble_image(
    img: Float[ArrayLike, " *dims"],
    chunk_size: tuple[int, ...],
    key: PRNGKeyArray,
) -> Float[Array, " *dims"]:
    """
    Given a tuple of integers specifying chunk sizes to use when scrambling the image,
    return a scrambled version of the image.

    This function is used to generate the baseline images for assessing if there is
    significant colocalization in a given image or not.
    """
    with jax.ensure_compile_time_eval():
        n_dim = img.ndim
        img_size = img.shape
        chunk_grid_shape = tuple(s // c + 1 for s, c in zip(img_size, chunk_size))

        n_samples = reduce(mul, chunk_grid_shape)
        indices = jax.random.randint(
            key,
            (n_samples, img.ndim),
            0,
            jnp.asarray(img_size) - jnp.asarray(chunk_size),
        )

        # Precompute patterns for rearrange as well
        n_str = [f"n_{i}" for i in range(n_dim)]
        x_str = [f"x_{i}" for i in range(n_dim)]
        nx_str = [f"({n} {x})" for n, x in zip(n_str, x_str)]
        kwargs = dict(zip(n_str, chunk_grid_shape))

    @jax.vmap
    def get_chunks(start_inds):
        return jax.lax.dynamic_slice(img, start_inds, chunk_size)

    img = jnp.asarray(img)
    chunks = get_chunks(indices)

    img_resampled = rearrange(
        chunks, f"({' '.join(n_str)}) {' '.join(x_str)} -> {' '.join(nx_str)}", **kwargs
    )

    img_resampled = jax.lax.dynamic_slice(img_resampled, [0] * n_dim, img_size)
    return img_resampled
