"""Closed-form observed-norm evidence with the scale frozen at an anchor."""

from math import exp, expm1, hypot, inf, isfinite, log, sqrt


def invert_frozen_norm(count, centered_sum, progress, squared_norm_sum, alpha=0.05, stake=0.5):
    """Return residual-trace bounds, before adding the anchor's sketch trace.

    For a Gaussian Nyström path, the nonrejection set on the hard-bound domain
    is an interval. With an arbitrary residual sketch, return a conservative
    hull if the lower quadratic still allows scales below its smaller root.
    Coverage assumes exact inputs and arithmetic. The scaled binary64 roots
    avoid squaring large inputs.
    """
    if count < 0 or not 0 < alpha < 1 or not 0 < stake <= 0.5:
        raise ValueError("Expected nonnegative count, 0 < alpha < 1, and 0 < stake <= 1/2")
    if not all(isfinite(x) and x >= 0 for x in (centered_sum, progress, squared_norm_sum)):
        raise ValueError("Path summaries must be finite and nonnegative")
    if count == 0:
        return progress, inf

    threshold = (log(2.0) - log(alpha)) / stake
    norm = sqrt(squared_norm_sum)
    scale = max(centered_sum, norm)
    if scale == 0:
        return progress, 0.0 if count > threshold else inf

    total = centered_sum / scale
    scaled_norm = norm / scale
    lower_count = count + threshold
    lower_term = 2 * sqrt(lower_count * stake) * scaled_norm
    lower = progress
    if lower_term < total:
        # The factored discriminant is stable near a repeated root.
        root = sqrt((total - lower_term) * (total + lower_term))
        root_sum = total + root
        small_root = (squared_norm_sum / scale) * (2 * stake / root_sum)
        if squared_norm_sum == 0 or small_root < progress:
            lower = max(lower, scale * (root_sum / (2 * lower_count)))

    upper = inf
    upper_count = count - threshold
    if upper_count > 0:
        x = 2 * stake
        # For small x, the Taylor remainder bound avoids expm1(x) - x cancellation.
        negative_penalty = stake * exp(x) if x < 1e-4 else (expm1(x) - x) / x
        root = hypot(total, 2 * sqrt(upper_count * negative_penalty) * scaled_norm)
        upper = scale * ((total + root) / (2 * upper_count))
    return lower, upper
