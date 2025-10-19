import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float


def deming_regression(
    x: Float[ArrayLike, " N"], y: Float[Array, " N"]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Fits a straight line y = ax+b to the given data points via deming regression
    (i.e. orthogonal regression) using the analytical formula given in
    P. Glaister. Math. Gaz. 85, 502 (2001)."""
    x, y = jnp.asarray(x), jnp.asarray(y)
    x_mean, y_mean = jnp.mean(x), jnp.mean(y)

    x_, y_ = x - x_mean, y - y_mean
    s_xx, s_yy = jnp.mean(x_ * x_), jnp.mean(y_ * y_)
    s_xy = jnp.mean(x_ * y_)
    p = (s_xx - s_yy) / (2 * s_xy)

    slope = -p + jnp.sqrt(1 + p * p)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def pearson_r(
    x: Float[Array, " N"], y: Float[Array, " N"], mask: Bool[Array, " N"] | None = None
) -> Float[Array, ""]:
    """
    Computes the Pearson correlation coefficient (r) between data points x, and y.

    If mask is given, r value is computed between x[mask] and y[mask].
    """
    if mask is None:
        mask = jnp.ones_like(x, dtype=jnp.bool)

    def masked_mean(arr: Float[Array, " N"]) -> Float[Array, ""]:
        return jnp.sum(arr * mask) / jnp.sum(mask)

    x_, y_ = x - masked_mean(x), y - masked_mean(y)
    s_xx, s_yy = masked_mean(x_ * x_), masked_mean(y_ * y_)
    s_xy = masked_mean(x_ * y_)
    return s_xy / jnp.sqrt(s_xx * s_yy)


@jax.jit
def colocalization_costes(
    channel1: Float[ArrayLike, " *dims"],
    channel2: Float[ArrayLike, " *dims"],
    *,
    n_threshold_bins: int = 2000,
    eps: float = 1e-3,
):
    """Performs the colocalization analysis as described in S. V. Costes et al.
    Biophys. J. 86, 6 (2004).
    """
    x, y = jnp.asarray(channel1).flatten(), jnp.asarray(channel2).flatten()

    # Perform orthogonal linear regression and determine thresholds to test
    a, b = deming_regression(x, y)
    threshold_max = jnp.minimum(jnp.max(x), (jnp.max(y) - b) / a)
    thresholds = jnp.linspace(0, threshold_max, n_threshold_bins)[::-1]

    def thresholded_correlation(thres: Float[Array, ""]) -> Float[Array, ""]:
        mask = jnp.logical_or(x <= thres, y <= a * thres + b)
        return pearson_r(x, y, mask)

    # Compute Pearson r for each threshold, and find the zero crossing point
    r_vals = jax.vmap(thresholded_correlation)(thresholds)
    idx_crossing = jnp.argmax(jnp.abs(r_vals) <= eps)
    thres_optimal = thresholds[idx_crossing]
    r_thres = r_vals[idx_crossing]

    # Calculate the final set of colocalization metrics
    mask = jnp.logical_and(x > thres_optimal, y > a * thres_optimal + b)
    r = pearson_r(x, y, mask)
    M1 = jnp.sum(x * mask) / jnp.sum(x)
    M2 = jnp.sum(y * mask) / jnp.sum(y)

    output_dict = {
        "pearson": r,
        "manders_1": M1,
        "manders_2": M2,
        "threshold": thres_optimal,
        "pearson_thresholded": r_thres,
        "deming_slope": a,
        "deming_intercept": b,
    }
    return output_dict
