"""High-confidence comparison under the exact flat-spectrum observation law.

For A=I_d/d and before rank exhaustion, Gaussian q_i is independent across
steps with law chi-square(d-i+1)/d, and t_i=i/d deterministically. Simulation
uses that law directly. No wall-time claim for matrix products is implied.
"""
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np


def main():
    dimension = 65536
    budget = 2048
    trials = 1000
    target = .05
    n = np.arange(1, budget + 1)
    sketch = n / dimension
    rng = np.random.default_rng(319772)
    q = rng.chisquare(dimension - n + 1, size=(trials, budget)) / dimension
    corrected_sum = np.cumsum(q + (n - 1) / dimension, axis=1)
    runs = {}
    for alpha in (.05, .001, .000001):
        a = 4 * math.log(2 / alpha)
        doob = math.sqrt(4 * math.pi / ((math.pi + 2) * alpha))
        methods = {}
        for name, threshold, penalty in [('trace_progress', a, sketch), ('doob_sharp_constant', doob, 0)]:
            lower = np.maximum.accumulate(np.maximum(sketch, (corrected_sum - penalty) / (n + threshold)), axis=1)
            upper = np.full((trials, budget), np.inf)
            valid = n > threshold
            numerator = corrected_sum + penalty
            upper[:, valid] = numerator[:, valid] / (n[valid] - threshold)
            upper = np.minimum.accumulate(upper, axis=1)
            radius = np.full((trials, budget), np.inf)
            finite = np.isfinite(upper) & (upper >= lower) & (lower > 0)
            radius[finite] = (upper[finite] - lower[finite]) / (upper[finite] + lower[finite])
            crossed = radius <= target
            all_stopped = crossed.any(axis=1)
            first = np.argmax(crossed[all_stopped], axis=1) + 1
            methods[name] = {'stopped': int(all_stopped.sum()), 'median_probe_count_among_stops': float(np.median(first)) if len(first) else None,
                             'anytime_coverage_failures': int((~np.all((lower <= 1) & (1 <= upper), axis=1)).sum())}
        runs[str(alpha)] = methods
    result = {'status': 'EXACT_LAW_SIMULATION', 'dimension': dimension, 'budget': budget, 'trials': trials, 'target': target,
              'seed': 319772, 'source_sha256': {Path(__file__).name: sha256(Path(__file__).read_bytes()).hexdigest()},
              'results': runs}
    Path(__file__).with_name('confidence_sweep_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(runs))


if __name__ == '__main__':
    main()
