"""Evaluate the same observed-norm mixtures using two scalar sums per root."""
import math
import numpy as np
from scipy.optimize import brentq

from core import NEG, POS, TraceEvidence, kappa


NEGATIVE_PENALTY = kappa(NEG) * NEG**2
POSITIVE_PENALTY = POS**2


def log_mean_exp(values):
    top = float(values.max())
    if not math.isfinite(top):
        raise ArithmeticError('Observed-norm log mixture exceeded numerical range')
    return top + math.log(float(np.exp(values - top).mean()))


class FastTraceEvidence(TraceEvidence):
    """Keep the reference append API and reuse the existing interval state."""

    def interval(self):
        if self.empty:
            return math.inf, -math.inf
        if not self.records:
            return 0.0, math.inf

        t, q, v = np.array([(record.t, record.q, record.v) for record in self.records]).T
        residual_norm = np.sqrt(v)
        count = len(q)
        domain = self.hard_lower
        lower = max(domain, self.running_lower)
        upper = self.running_upper
        threshold = self.threshold
        scale = max(domain, float(np.mean(t + q)), np.finfo(float).tiny)

        def evidence(theta, stakes, penalty, sign):
            denominator = theta - t
            if np.any(denominator <= 0):
                raise ArithmeticError('Residual trace bound lost strict positivity in floating point')
            displacement = float((q / denominator).sum()) - count
            scaled_norm = residual_norm / denominator
            variance = float(scaled_norm @ scaled_norm)
            return log_mean_exp(sign * stakes * displacement - penalty * variance) - threshold

        def positive(theta):
            return evidence(theta, POS, POSITIVE_PENALTY, 1)

        def negative(theta):
            return evidence(theta, NEG, NEGATIVE_PENALTY, -1)

        if lower > upper or negative(lower) > 0:
            self.empty = True
            return math.inf, -math.inf
        if positive(lower) > 0:
            right = max(2 * scale, 2 * lower)
            while positive(right) > 0:
                right *= 2
            root = brentq(positive, lower, right, xtol=1e-13 * scale)
            lower = max(lower, math.nextafter(root, -math.inf))
        if lower > upper or negative(lower) > 0:
            self.empty = True
            return math.inf, -math.inf
        if math.isfinite(upper):
            if negative(upper) > 0:
                root = brentq(negative, lower, upper, xtol=1e-13 * scale)
                upper = min(upper, math.nextafter(root, math.inf))
        elif log_mean_exp(count * NEG) > threshold:
            right = max(2 * scale, 2 * lower)
            while negative(right) <= 0 and right < 1e250:
                right *= 2
            if right < 1e250:
                root = brentq(negative, lower, right, xtol=1e-13 * scale)
                upper = math.nextafter(root, math.inf)
        self.running_lower, self.running_upper = lower, upper
        return lower, upper
