"""Check the inexact Galerkin interval inversion and width with rational arithmetic."""

from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from random import Random


SEED = 95274832
TRIALS = 10_000


def endpoints(count, threshold, anchor_trace, progress, z_lower, z_upper, coefficient):
    penalty = coefficient * progress
    lower = max(anchor_trace + progress, anchor_trace + (z_lower - penalty) / (count + threshold))
    upper = None
    if count > threshold:
        upper = anchor_trace + (z_upper + penalty) / (count - threshold)
    return lower, upper


def main():
    rng = Random(SEED)
    membership_checks = 0
    width_checks = 0
    residual_width_checks = 0
    for _ in range(TRIALS):
        count = rng.randrange(1, 201)
        threshold = Fraction(rng.randrange(1, 3001), 15)
        anchor_trace = Fraction(rng.randrange(0, 200), 5)
        progress = Fraction(rng.randrange(0, 300), 5)
        sigma = Fraction(rng.randrange(0, 31), 10)
        coefficient = Fraction(rng.randrange(1, 11), 10)
        gap = progress * sigma * Fraction(rng.randrange(0, 101), 100)
        z_lower = max(0, (count + threshold + coefficient) * progress + Fraction(rng.randrange(-200, 200), 10))
        z_upper = z_lower + gap
        lower, upper = endpoints(count, threshold, anchor_trace, progress, z_lower, z_upper, coefficient)

        # Probe both sides of each boundary, imposing the hard trace lower bound.
        hard_lower = anchor_trace + progress
        offset = Fraction(1, 1000)
        candidates = [hard_lower, lower, lower + offset, max(hard_lower, lower - offset)]
        if upper is not None:
            candidates.extend(max(hard_lower, upper + shift) for shift in (0, offset, -offset))
        for candidate in candidates:
            scale = candidate - anchor_trace
            if scale == 0:
                continue
            positive = (z_lower - coefficient * progress) / scale - count
            negative = count - (z_upper + coefficient * progress) / scale
            accepted = positive <= threshold and negative <= threshold
            enclosed = lower <= candidate and (upper is None or candidate <= upper)
            assert accepted == enclosed
            membership_checks += 1

        if upper is not None and upper >= lower and upper + lower > 0:
            symmetric_coefficient = coefficient + sigma / 2
            bound = (threshold + symmetric_coefficient) / (count + symmetric_coefficient)
            assert (upper - lower) / (upper + lower) <= bound
            width_checks += 1
            residual_sum = upper + lower - 2 * anchor_trace
            if residual_sum > 0:
                assert (upper - lower) / residual_sum <= bound
                residual_width_checks += 1

    # Choose the largest allowed gap and the point where the lower branches meet.
    equality_checks = 0
    for count, threshold in ((20, Fraction(10)), (100, Fraction(11, 2)), (51, Fraction(50))):
        progress = Fraction(3)
        coefficient = Fraction(1)
        sigma = Fraction(1)
        symmetric_coefficient = coefficient + sigma / 2
        midpoint = progress * (count + threshold + symmetric_coefficient)
        half_gap = sigma * progress / 2
        lower, upper = endpoints(count, threshold, 0, progress, midpoint - half_gap, midpoint + half_gap, coefficient)
        assert lower == progress
        assert (upper - lower) / (upper + lower) == (threshold + symmetric_coefficient) / (count + symmetric_coefficient)
        equality_checks += 1

    source = Path(__file__).resolve()
    result = {
        "seed": SEED,
        "generated_configurations": TRIALS,
        "signed_nonrejection_membership_checks": membership_checks,
        "total_trace_radius_bound_checks": width_checks,
        "residual_trace_radius_bound_checks": residual_width_checks,
        "sharp_radius_equality_checks": equality_checks,
        "scope": "Deterministic rational checks of the anchored formulas and width bound. No stochastic coverage was tested.",
        "source_sha256": {source.name: sha256(source.read_bytes()).hexdigest()},
    }
    report = json.dumps(result, indent=2) + "\n"
    source.with_name("inexact_endpoints_validation.json").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
