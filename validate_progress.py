"""Check the implemented interval inversion against the two analytic tests."""
import json
import math
from pathlib import Path
import numpy as np

from progress_certificate import Anchor, invert_progress, relative_radius


def main():
    rng = np.random.default_rng(115532)
    comparisons = 0
    cap_checks = 0
    for trial in range(1000):
        anchor_probe = trial % 17
        anchor_trace = float(rng.uniform(0.01, 8))
        anchor = Anchor(anchor_probe, anchor_trace, float(rng.uniform(0, 25)), float(rng.uniform(.001, .1)))
        count = int(rng.integers(1, 300))
        progress = float(rng.uniform(.01, 10))
        current_trace = anchor_trace + progress
        z = float(rng.uniform(0, 3 * count * progress))
        total = anchor.corrected_sum + count * anchor_trace + z
        c = .25
        h = math.log(2 / anchor.alpha)
        lo, hi = invert_progress(anchor, anchor_probe + count, total, current_trace, c)
        for residual_trace in rng.uniform(.01, 20 * progress, 100):
            candidate = anchor_trace + residual_trace
            upper_log = c * (count - (z + progress) / residual_trace)
            lower_log = c * ((z - progress) / residual_trace - count)
            assert (lo <= candidate <= hi) == (candidate >= current_trace and max(lower_log, upper_log) <= h)
            comparisons += 1
        if math.isfinite(hi) and hi >= lo:
            observed = relative_radius(lo - anchor_trace, hi - anchor_trace)
            assert observed <= (h / c + 1) / (count + 1) + 1e-12
            cap_checks += 1
    result = {'interval_inversion_checks': comparisons, 'width_cap_checks': cap_checks,
              'scope': 'implementation checks on generated states, not a probabilistic coverage study'}
    Path(__file__).with_name('progress_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
