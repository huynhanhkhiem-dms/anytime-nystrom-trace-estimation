"""Closed-form trace confidence intervals using the sketch's trace increments.

The coverage theorem assumes exact products, arithmetic, and Gaussian draws.
The code below evaluates the formula in floating point for experiments.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Anchor:
    probe: int
    sketch_trace: float
    corrected_sum: float
    alpha: float


def invert_progress(anchor, probe, corrected_sum, sketch_trace, c=0.25):
    """Invert a pair of one-sided test supermartingales with total level alpha."""
    count = probe - anchor.probe
    if count == 0:
        return sketch_trace, math.inf
    if count < 0 or sketch_trace < anchor.sketch_trace:
        raise ArithmeticError('Probe index and sketch trace must be nondecreasing')

    progress = sketch_trace - anchor.sketch_trace
    centered_sum = corrected_sum - anchor.corrected_sum - count * anchor.sketch_trace
    risk_ratio = 2 / anchor.alpha
    log_risk_ratio = math.log(risk_ratio) if math.isfinite(risk_ratio) else math.log(2) - math.log(anchor.alpha)
    threshold = log_risk_ratio / c
    penalty = 4 * c * progress
    lower = max(sketch_trace, anchor.sketch_trace + (centered_sum - penalty) / (count + threshold))
    upper = math.inf
    if count > threshold:
        upper = anchor.sketch_trace + (centered_sum + penalty) / (count - threshold)
    return lower, upper


class ProgressSequence:
    def __init__(self, alpha=0.05, anchors=False):
        if not 0 < alpha < 1:
            raise ValueError('alpha must lie in (0, 1)')
        self.alpha = alpha
        self.use_anchors = anchors
        self.anchors = [Anchor(0, 0.0, 0.0, alpha / 2 if anchors else alpha)]
        self.next_anchor = 4
        self.lower = 0.0
        self.upper = math.inf

    def inspect(self, probe, corrected_sum, sketch_trace):
        self.lower = max(self.lower, sketch_trace)
        for anchor in self.anchors:
            lower, upper = invert_progress(anchor, probe, corrected_sum, sketch_trace)
            self.lower = max(self.lower, lower)
            self.upper = min(self.upper, upper)
        if self.use_anchors and probe == self.next_anchor:
            index = len(self.anchors)
            spend = self.alpha / (2 * index * (index + 1))
            self.anchors.append(Anchor(probe, sketch_trace, corrected_sum, spend))
            self.next_anchor *= 2
        return self.lower, self.upper


def relative_radius(lower, upper):
    if not (0 < lower <= upper < math.inf):
        return math.inf
    width, total = upper - lower, upper + lower
    if math.isfinite(total):
        return width / total
    return (width / upper) / (1 + lower / upper)


def relative_center(lower, upper):
    if not (0 < lower <= upper < math.inf):
        raise ValueError('A positive, finite, nonempty interval is required')
    twice_lower = 2 * lower
    if math.isfinite(twice_lower):
        return twice_lower / (1 + lower / upper)
    return min(upper, lower + lower * relative_radius(lower, upper))
