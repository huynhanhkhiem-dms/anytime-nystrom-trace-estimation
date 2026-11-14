"""Matched confidence baseline with independent residual probes at each checkpoint.

All candidate sketch fractions and checkpoint times spend a portion of the
same alpha. Products are retained and shared across checkpoints and fractions.
The Loewner lower bound from the full prefix sketch is shared with the primary
procedure. No point estimate or reference trace is used in any stop decision.
"""
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from baselines import gaussian_psd_fixed_interval
from core import TraceEvidence
from progress_certificate import relative_center, relative_radius
from run_progress_pilot import DIMENSIONS, load_operator


HERE = Path(__file__).resolve().parent
CHECKPOINTS = (12, 16, 24, 32, 44, 64, 96, 128, 192, 256)
FRACTIONS = (0.25, 0.5, 0.75)
ALPHA = 0.05


def run_seed(matvec, theta, dimension, seed):
    rng = np.random.default_rng(seed)
    budget = CHECKPOINTS[-1]
    factors = np.empty((dimension, budget), order='F')
    probes = np.empty_like(factors)
    products = np.empty_like(factors)
    sketch_trace = np.zeros(budget + 1)
    level = ALPHA / (len(CHECKPOINTS) * len(FRACTIONS))
    bounds = {'split_chernoff': (0.0, np.inf), 'split_observed_v': (0.0, np.inf)}
    failures = dict.fromkeys(bounds, False)
    stops = dict.fromkeys(bounds)

    for probe_index in range(budget):
        g = rng.standard_normal(dimension)
        image = matvec(g)
        probes[:, probe_index] = g
        products[:, probe_index] = image
        previous = factors[:, :probe_index]
        residual = image - previous @ (previous.T @ g)
        q = float(g @ residual)
        v = float(residual @ residual)
        if q <= 0 or v <= 0 or not np.isfinite(q + v):
            raise ArithmeticError(f'Nonpositive or nonfinite residual at {probe_index + 1}')
        factors[:, probe_index] = residual / np.sqrt(q)
        count = probe_index + 1
        sketch_trace[count] = sketch_trace[probe_index] + v / q
        if count not in CHECKPOINTS:
            continue
        for fraction in FRACTIONS:
            split = int(count * fraction)
            basis = factors[:, :split]
            heldout = probes[:, split:count]
            r = products[:, split:count] - basis @ (basis.T @ heldout)
            q_holdout = np.sum(r * heldout, axis=0)
            v_holdout = np.sum(r * r, axis=0)
            if np.any(q_holdout <= 0) or np.any(v_holdout <= 0):
                raise ArithmeticError(f'Nonpositive holdout residual at {count}, split {split}')
            lo, hi = gaussian_psd_fixed_interval(float(q_holdout.mean()), len(q_holdout), level)
            t = sketch_trace[split]
            observed = TraceEvidence(alpha=level)
            for qi, vi in zip(q_holdout, v_holdout):
                observed.append(float(t), float(qi), float(vi))
            for method, (lower, upper) in zip(bounds, [(t + lo, t + hi), observed.interval()]):
                old_l, old_u = bounds[method]
                bounds[method] = (max(old_l, float(lower), float(sketch_trace[count])), min(old_u, float(upper)))
        for method, (lower, upper) in bounds.items():
            failures[method] |= not lower <= theta <= upper
            if stops[method] is None and relative_radius(lower, upper) <= 0.05:
                stops[method] = {'matvecs': count, 'relative_error': float(abs(relative_center(lower, upper)/theta-1))}
    return {'seed': seed, 'coverage_failures': failures, 'stops': stops}


def main():
    result = {'status': 'EXPLORATORY_PILOT', 'target': 0.05, 'alpha': ALPHA,
              'checkpoints': CHECKPOINTS, 'sketch_fractions': FRACTIONS,
              'source_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in
                 [Path(__file__), HERE/'run_progress_pilot.py', HERE/'baselines.py', HERE/'core.py']}, 'matrices': {}}
    path = HERE / 'split_checkpoint_results.json'
    with threadpool_limits(limits=1):
        for name in DIMENSIONS:
            start = perf_counter()
            matvec, theta, audit = load_operator(name)
            records = []
            for seed in range(701000, 701024):
                try:
                    records.append(run_seed(matvec, theta, audit['dimension'], seed))
                except ArithmeticError as error:
                    records.append({'seed': seed, 'numerical_failure': str(error)})
            result['matrices'][name] = {'audit': audit, 'records': records, 'elapsed_seconds': perf_counter()-start}
            path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            for method in ('split_chernoff', 'split_observed_v'):
                stops = [r['stops'][method]['matvecs'] for r in records if r.get('stops',{}).get(method)]
                print(name, method, len(stops), 'median', np.median(stops) if stops else None, flush=True)


if __name__ == '__main__':
    main()
