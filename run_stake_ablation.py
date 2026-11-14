"""Replay archived Gaussian paths to isolate the effect of the evidence stake."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from anchored_norm import invert_frozen_norm
from progress_certificate import Anchor, invert_progress, relative_center, relative_radius
from run_anchor_norm_comparison import (
    ALPHAS, CAP, CASES, FIRST_SEED, HERE, NEW_SOURCES, NUMERICAL_ERRORS, PATHS,
    PROGRESS_STAKE, REPORT_EVERY, SEED_STRIDE, SOURCES, TARGETS, UNBOUNDED,
    file_hash, finite, intersect, load_case,
)


METHODS = ('progress', 'norm_common_stake', 'frozen_norm', 'reinforced_norm')
DESTINATION = HERE / 'stake_ablation_results.json'


def new_result():
    return {'stops': {}, 'reference_exclusions': [], 'last_checkpoint': 0, 'last_interval': None}


def update_result(result, interval, step, reference):
    lower, upper = interval
    radius = relative_radius(lower, upper)
    result['last_checkpoint'] = step
    result['last_interval'] = {
        'lower': finite(lower), 'upper': finite(upper), 'radius': finite(radius), 'empty': lower > upper,
    }
    if not lower <= reference <= upper:
        result['reference_exclusions'].append(step)
    for target in TARGETS:
        key = str(target)
        if key not in result['stops'] and radius <= target:
            center = relative_center(lower, upper)
            result['stops'][key] = {'probes': step, 'radius': radius, 'estimate': center,
                                    'relative_error': abs(center - reference) / reference}


def run_path(matvec, dimension, reference, seed):
    rng = np.random.default_rng(seed)
    factor = np.empty((dimension, CAP), order='F')
    captured = total = squared_total = 0.0
    probe_scores = []
    error = None
    rows = {str(alpha): {method: new_result() for method in METHODS} for alpha in ALPHAS}
    bounds = {alpha: dict.fromkeys(METHODS, UNBOUNDED) for alpha in ALPHAS}

    for step in range(1, CAP + 1):
        previous = captured
        try:
            vector = rng.standard_normal(dimension)
            image = matvec(vector)
            basis = factor[:, :step - 1]
            residual = image - basis @ (basis.T @ vector)
            q, v = float(vector @ residual), float(residual @ residual)
            if not (q > 0 and v > 0 and math.isfinite(q + v)):
                raise ArithmeticError('Nonpositive or nonfinite moving residual')
            delta = v / q
            captured = previous + delta
            total += previous + q
            squared_total += v
            if not (captured > previous and math.isfinite(captured + total + squared_total)):
                raise ArithmeticError('Trace progress lost positivity or exceeded numerical range')
            factor[:, step - 1] = residual / math.sqrt(q)
            probe_scores.append([q, v, delta])
        except NUMERICAL_ERRORS as exception:
            error = {'probe': step, 'reason': str(exception)}
            break

        if step % REPORT_EVERY:
            continue

        for alpha in ALPHAS:
            anchor = Anchor(0, 0.0, 0.0, alpha)
            progress = invert_progress(anchor, step, total, captured, c=PROGRESS_STAKE)
            norm_common = invert_frozen_norm(step, total, captured, squared_total, alpha, PROGRESS_STAKE)
            norm = invert_frozen_norm(step, total, captured, squared_total, alpha, 0.5)
            # Theorem 3 supplies a simultaneous lower bound on the origin scale.
            k = 4 * (math.log(3) + math.log(2 / alpha))
            norm_half = invert_frozen_norm(step, total, captured, squared_total, alpha / 2, 0.5)
            reinforced = intersect(norm_half, (math.sqrt(squared_total / k), math.inf))
            for method, interval in zip(METHODS, (progress, norm_common, norm, reinforced), strict=True):
                if method != 'reinforced_norm':
                    interval = intersect(bounds[alpha][method], interval)
                    bounds[alpha][method] = interval
                update_result(rows[str(alpha)][method], interval, step, reference)

    return {'seed': seed, 'probe_scores': probe_scores, 'shared_numerical_error': error, 'methods': rows}


def summarize(trials):
    summaries = {}
    for alpha in map(str, ALPHAS):
        summaries[alpha] = {}
        for target in map(str, TARGETS):
            methods, costs = {}, {}
            for method in METHODS:
                rows = [trial['methods'][alpha][method] for trial in trials]
                stops = [row['stops'].get(target) for row in rows]
                costs[method] = np.array([stop['probes'] if stop is not None else CAP for stop in stops])
                methods[method] = {
                    'completions': sum(stop is not None for stop in stops),
                    'incorrect_completions': sum(stop is not None and stop['relative_error'] > float(target)
                                                 for stop in stops),
                    'restricted_mean_probes': float(costs[method].mean()),
                    'reference_exclusions': sum(bool(row['reference_exclusions']) for row in rows),
                }
            contrasts = {}
            for first, second in (('norm_common_stake', 'progress'), ('frozen_norm', 'norm_common_stake'),
                                  ('reinforced_norm', 'frozen_norm')):
                differences = costs[first] - costs[second]
                contrasts[f'{first}_minus_{second}'] = {
                    'mean_difference': float(differences.mean()), 'wins': int((differences < 0).sum()),
                    'ties': int((differences == 0).sum()), 'losses': int((differences > 0).sum()),
                }
            summaries[alpha][target] = {'methods': methods, 'paired_contrasts': contrasts}
    return summaries


def compare_archive(trials, archived_trials):
    stop_count_checks = endpoint_checks = 0
    maximum_relative_endpoint_difference = 0.0
    for trial, archived in zip(trials, archived_trials, strict=True):
        if trial['seed'] != archived['seed'] or trial['shared_numerical_error'] != archived['shared_numerical_error']:
            raise ValueError(f"Probe path mismatch at seed {trial['seed']}")
        for alpha in map(str, ALPHAS):
            for method in ('progress', 'frozen_norm'):
                row, old = (path['methods'][alpha][method] for path in (trial, archived))
                if (row['last_checkpoint'], row['reference_exclusions']) != (old['last_checkpoint'],
                                                                            old['reference_exclusions']):
                    raise ValueError(f"Replay status mismatch at seed {trial['seed']} for {alpha=} {method=}")
                for target in map(str, TARGETS):
                    stop, old_stop = (entry['stops'].get(target, {}).get('probes') for entry in (row, old))
                    if stop != old_stop:
                        raise ValueError(f"Stop mismatch at seed {trial['seed']} for {alpha=} {method=} {target=}")
                    stop_count_checks += 1
                if row['last_interval'] is None:
                    continue
                for key in ('lower', 'upper', 'radius'):
                    value, previous = (entry['last_interval'][key] for entry in (row, old))
                    if value is None or previous is None:
                        if value != previous:
                            raise ValueError(f"Infinite endpoint mismatch at seed {trial['seed']}")
                        continue
                    difference = abs(value - previous) / max(abs(previous), np.finfo(float).tiny)
                    maximum_relative_endpoint_difference = max(maximum_relative_endpoint_difference, difference)
                    if not math.isclose(value, previous, rel_tol=1e-10, abs_tol=1e-13):
                        raise ValueError(f"Endpoint mismatch at seed {trial['seed']} for {alpha=} {method=} {key=}")
                    endpoint_checks += 1
    return {'stop_count_checks': stop_count_checks, 'terminal_endpoint_checks': endpoint_checks,
            'maximum_relative_terminal_endpoint_difference': maximum_relative_endpoint_difference}


def run_case(index):
    case = CASES[index]
    SOURCES.update(NEW_SOURCES)
    with threadpool_limits(limits=1):
        matvec, reference, audit = load_case(case)
        if audit['dimension'] <= CAP:
            raise ValueError(f'Dimension must exceed the replay cap for {case.name}')
        trials = [run_path(matvec, audit['dimension'], reference, FIRST_SEED + index * SEED_STRIDE + path)
                  for path in range(PATHS)]
    return case.name, {'audit': audit, 'trials': trials, 'summary': summarize(trials)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error('--jobs must be positive')
    archive = HERE / 'results' / 'anchor_norm_comparison.json'
    if not archive.is_file():
        archive = HERE / 'anchor_norm_comparison.json'
    archived = json.loads(archive.read_text(encoding='utf-8'))['cases']
    result = {'status': 'COMPLETE', 'cap': CAP, 'report_every': REPORT_EVERY, 'alphas': ALPHAS,
              'targets': TARGETS, 'paths_per_operator': PATHS, 'probe_score_columns': ['q', 'v', 'delta'],
              'archive_sha256': file_hash(archive), 'script_sha256': file_hash(Path(__file__)), 'cases': {}}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for name, case in pool.map(run_case, range(len(CASES))):
            case['archive_comparison'] = compare_archive(case['trials'], archived[name]['trials'])
            result['cases'][name] = case
            print(f'{name}: {case["archive_comparison"]}', flush=True)
    staging = DESTINATION.with_suffix('.tmp')
    staging.write_text(json.dumps(result, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')
    staging.replace(DESTINATION)


if __name__ == '__main__':
    main()
