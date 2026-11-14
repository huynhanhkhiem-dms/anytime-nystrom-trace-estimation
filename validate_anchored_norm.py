"""Check the binary64 frozen-norm roots against direct evidence evaluation."""

from hashlib import sha256
import json
from math import expm1, isclose, isinf, log
from pathlib import Path

import numpy as np

from anchored_norm import invert_frozen_norm


def main():
    rng = np.random.default_rng(97108)
    alpha = 0.05
    threshold = log(2 / alpha)
    tests = 0
    for _ in range(1000):
        count = int(rng.integers(1, 100))
        stake = float(rng.uniform(0.01, 0.5))
        change = rng.lognormal(-4, 1, size=count)
        quadratic = rng.lognormal(0, 2, size=count)
        captured = float(change.sum())
        corrected = float(quadratic.sum() + np.cumsum(change)[:-1].sum())
        variance = float(change @ quadratic)
        lo, hi = invert_frozen_norm(count, corrected, captured, variance, alpha, stake)
        kappa = (expm1(2 * stake) - 2 * stake) / (2 * stake ** 2)
        candidates = captured + rng.lognormal(0, 2, size=100)
        for scale in candidates:
            linear = corrected / scale - count
            penalty = stake ** 2 * variance / scale ** 2
            passes = max(stake * linear - penalty, -stake * linear - kappa * penalty) <= threshold
            assert bool(lo <= scale <= hi) == passes
            tests += 1

    scales = [1e-140, 1e-100, 1e100, 1e140]
    root = invert_frozen_norm(80, 90, 0.8, 1)
    for scale in scales:
        actual = invert_frozen_norm(80, 90 * scale, 0.8 * scale, scale ** 2)
        assert all(isclose(x / scale, y, rel_tol=1e-14) for x, y in zip(actual, root))
    for scale in (1e-290, 1e290):
        actual = invert_frozen_norm(148, 148 * scale, 0, 0)
        unit = invert_frozen_norm(148, 148, 0, 0)
        assert all(isclose(x / scale, y, rel_tol=1e-14) for x, y in zip(actual, unit))
    assert isinf(invert_frozen_norm(7, 7, 0, 0)[1])
    assert invert_frozen_norm(0, 0, 0, 0)[0] == 0
    assert invert_frozen_norm(20, 0, 0, 0) == (0, 0)
    # On a frozen residual, the lower quadratic can have two retained branches.
    lo, hi = invert_frozen_norm(50, 100, 0.01, 10)
    assert lo == 0.01 and hi > 0.02
    # The small root can remain representable when (sqrt(V) / Z)**2 underflows.
    lo, _ = invert_frozen_norm(50, 1e300, 1e-200, 1e200)
    assert lo == 1e-200
    here = Path(__file__).resolve().parent
    result = {"pointwise_evidence_comparisons": tests, "rescaling_checks": len(scales) + 2,
              "disjoint_branch_checks": 2,
              "scope": "Deterministic checks of binary64 roots, with no stochastic coverage test.",
              "source_sha256": {name: sha256((here / name).read_bytes()).hexdigest()
                                for name in ("validate_anchored_norm.py", "anchored_norm.py")}}
    (here / "anchored_norm_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
