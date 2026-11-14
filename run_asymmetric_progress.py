"""Test asymmetric Gaussian progress certificates on fresh paired paths."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from anchored_norm import invert_frozen_norm
from progress_certificate import Anchor, invert_progress, relative_center, relative_radius
from run_anchor_norm_comparison import CASES, NEW_SOURCES, NUMERICAL_ERRORS
from run_confirmatory import SOURCES, load_case


HERE = Path(__file__).resolve().parent
ALPHA = 0.05
TARGET = 0.05
CAP = 256
PATHS = 24
FIRST_SEED = 12132000
SEED_STRIDE = 1000
PLUS_STAKE = 3 / 8
MINUS_STAKE = 5 / 8
PLUS_PENALTY = 5 * PLUS_STAKE
MINUS_PENALTY = 4 * MINUS_STAKE
METHODS = ('progress_symmetric', 'progress_asymmetric', 'frozen_norm')


@cache
def thresholds(alpha):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0, 1)')
    log_threshold = math.log(2) - math.log(alpha)
    return log_threshold / PLUS_STAKE, log_threshold / MINUS_STAKE


def invert_asymmetric_progress(count, corrected_sum, sketch_trace, alpha=ALPHA):
    """Invert the signed progress tests at the origin anchor."""
    plus, minus = thresholds(alpha)
    lower = max(sketch_trace, (corrected_sum - PLUS_PENALTY * sketch_trace) / (count + plus))
    upper = math.inf
    if count > minus:
        upper = (corrected_sum + MINUS_PENALTY * sketch_trace) / (count - minus)
    return lower, upper


def verify_constants():
    from flint import arb, ctx

    with ctx.workprec(200):
        pos, neg = arb(3) / 8, arb(5) / 8
        pos_chord = 2 / arb.pi() * (1 - (-5 * pos ** 2).exp())
        neg_chord = 2 / arb.pi() * (1 - (-4 * neg ** 2).exp())
        pos_required = -pos - (1 - 2 * pos).log() / 2
        neg_required = (1 + 2 * neg) * (neg - (1 + 2 * neg).log() / 2)
        assert pos_chord > pos_required and neg_chord > neg_required
        eps = arb(1) / 20
        threshold = ((1 - eps) * (arb(40).log() / pos + 5 * pos + 4 * neg)
                     + (1 + eps) * arb(40).log() / neg) / (2 * eps)
        symmetric = (arb(40).log() * 3 + (1 - eps) * 4 / 3) / eps
        assert 196 < threshold < 197 and 246 < symmetric < 247
        return {
            'precision_bits': 200,
            'positive_chord': str(pos_chord),
            'positive_requirement': str(pos_required),
            'negative_chord': str(neg_chord),
            'negative_requirement': str(neg_required),
            'asymmetric_threshold': str(threshold),
            'symmetric_threshold': str(symmetric),
        }


def run_path(matvec, dimension, reference, seed):
    rng = np.random.default_rng(seed)
    probes = rng.standard_normal((CAP, dimension)).T
    factor = np.empty_like(probes, order='F')
    origin = Anchor(0, 0.0, 0.0, ALPHA)
    sketch_trace = corrected_sum = squared_norm_sum = 0.0
    running = dict.fromkeys(METHODS, (0.0, math.inf))
    methods = {name: {'stop': None, 'reference_exclusions': []} for name in METHODS}
    scores = []
    error = None
    try:
        images = np.asfortranarray(matvec(probes))
        if images.shape != probes.shape or not np.isfinite(images).all():
            raise ArithmeticError('Invalid block of oracle products')
        for step, probe in enumerate(probes.T, 1):
            basis = factor[:, :step - 1]
            residual = images[:, step - 1] - basis @ (basis.T @ probe)
            q, v = float(probe @ residual), float(residual @ residual)
            if not (q > 0 and v > 0 and math.isfinite(q + v)):
                raise ArithmeticError('Nonpositive or nonfinite residual score')
            delta = v / q
            corrected_sum += sketch_trace + q
            squared_norm_sum += v
            old_trace = sketch_trace
            sketch_trace += delta
            if not (sketch_trace > old_trace and math.isfinite(sketch_trace + corrected_sum + squared_norm_sum)):
                raise ArithmeticError('Trace increment failed to advance in binary64')
            factor[:, step - 1] = residual / math.sqrt(q)
            scores.append((q, v, delta))
            current = (
                invert_progress(origin, step, corrected_sum, sketch_trace, c=1 / 3),
                invert_asymmetric_progress(step, corrected_sum, sketch_trace),
                invert_frozen_norm(step, corrected_sum, sketch_trace, squared_norm_sum, alpha=ALPHA),
            )
            if step >= 197 and current[1][1] >= current[1][0] > 0:
                assert relative_radius(*current[1]) <= TARGET
            for name, bounds in zip(METHODS, current, strict=True):
                lower = max(running[name][0], bounds[0])
                upper = min(running[name][1], bounds[1])
                running[name] = lower, upper
                method = methods[name]
                if not lower <= reference <= upper:
                    method['reference_exclusions'].append(step)
                radius = relative_radius(lower, upper)
                if method['stop'] is None and radius <= TARGET:
                    center = relative_center(lower, upper)
                    method['stop'] = {
                        'products': step, 'lower': lower, 'upper': upper, 'radius': radius,
                        'estimate': center, 'relative_error': abs(center / reference - 1),
                    }
    except NUMERICAL_ERRORS as exc:
        error = {'after_products': len(scores), 'reason': str(exc)}
    return {'seed': seed, 'methods': methods, 'scores': scores, 'numerical_error': error}


def run_case(index):
    SOURCES.update(NEW_SOURCES)
    case = CASES[index]
    with threadpool_limits(limits=1):
        matvec, reference, audit = load_case(case)
        paths = [run_path(matvec, audit['dimension'], reference, FIRST_SEED + index * SEED_STRIDE + offset)
                 for offset in range(PATHS)]
    return case.name, {'audit': audit, 'paths': paths, 'summary': summarize(paths)}


def summarize(paths):
    comparisons = Counter(earlier=0, tied=0, later=0)
    methods = {}
    for name in METHODS:
        rows = [path['methods'][name] for path in paths]
        stops = [row['stop'] for row in rows if row['stop'] is not None]
        costs = [stop['products'] for stop in stops]
        methods[name] = {
            'completions': len(stops),
            'restricted_mean_products': (sum(costs) + (len(paths) - len(stops)) * CAP) / len(paths),
            'median_products_among_stops': float(np.median(costs)) if costs else None,
            'min_products_among_stops': min(costs) if costs else None,
            'max_products_among_stops': max(costs) if costs else None,
            'reference_exclusions': sum(len(row['reference_exclusions']) for row in rows),
            'incorrect_stops': sum(stop['relative_error'] > TARGET for stop in stops),
        }
    differences = []
    for path in paths:
        rows = path['methods']
        old, new = rows['progress_symmetric']['stop'], rows['progress_asymmetric']['stop']
        old_cost, new_cost = old['products'] if old else CAP, new['products'] if new else CAP
        comparisons['earlier' if new_cost < old_cost else 'later' if new_cost > old_cost else 'tied'] += 1
        differences.append(new_cost - old_cost)
    return {
        'methods': methods,
        'asymmetric_vs_symmetric': dict(comparisons),
        'mean_paired_asymmetric_minus_symmetric_products': sum(differences) / len(differences),
        'numerical_interruptions': sum(path['numerical_error'] is not None for path in paths),
    }


def main():
    proof = verify_constants()
    with ProcessPoolExecutor(max_workers=4) as workers:
        cases = dict(workers.map(run_case, range(len(CASES))))
    paths = [path for case in cases.values() for path in case['paths']]
    summary = summarize(paths)
    sources = ('run_asymmetric_progress.py', 'asymmetric_progress_protocol.txt', 'run_anchor_norm_comparison.py',
               'run_confirmatory.py', 'progress_certificate.py', 'anchored_norm.py')
    result = {
        'status': 'COMPLETE',
        'operators': list(cases),
        'paths_per_operator': PATHS,
        'first_seed': FIRST_SEED,
        'seed_stride': SEED_STRIDE,
        'alpha': ALPHA,
        'target': TARGET,
        'cap': CAP,
        'report_every': 1,
        'analytic_caps': {'progress_symmetric': 247, 'progress_asymmetric': 197},
        'proof_check': proof,
        'summary': summary,
        'source_sha256': {name: sha256((HERE / name).read_bytes()).hexdigest() for name in sources},
        'cases': cases,
    }
    output = HERE / 'asymmetric_progress_results.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')
    temporary.replace(output)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
