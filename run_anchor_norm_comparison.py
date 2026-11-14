"""Replay matched trace evidence rules on repeated and additional operators."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import platform
from time import perf_counter_ns
import zipfile

import numpy as np
import requests
import scipy
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from anchored_norm import invert_frozen_norm
from chi2_comparator import ChiSquareEvidence, evidence_cutoffs
from fast_evidence import FastTraceEvidence
from progress_certificate import Anchor, invert_progress, relative_center, relative_radius
from run_confirmatory import CASES as PREVIOUS_CASES
from run_confirmatory import SOURCES, Case, file_hash, load_case, upper_failure_limit


HERE = Path(__file__).resolve().parent
PATHS = 24
CAP = 400
REPORT_EVERY = 4
SPLIT = 32
FIRST_SEED = 8100000
SEED_STRIDE = 1000
ALPHAS = (0.05, 0.01)
TARGETS = (0.05, 0.10)
PROGRESS_STAKE = 1 / 3
NORM_STAKE = 0.5
METHODS = ('progress', 'frozen_norm', 'moving_norm', 'chi_square', 'frozen_residual', 'frozen_hybrid')
NEW_SOURCES = {
    'bcsstk08': ('bcsstk08.mtx.gz',
                 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/bcsstruc1/bcsstk08.mtx.gz'),
    '1138_bus': ('1138_bus.mtx.gz',
                 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/psadmit/1138_bus.mtx.gz'),
}
CASES = PREVIOUS_CASES + tuple(Case(source, 'inverse', 0.01) for source in NEW_SOURCES)
SOURCE_FILES = ('run_anchor_norm_comparison.py', 'anchor_norm_protocol.txt', 'anchored_norm.py',
                'run_confirmatory.py', 'core.py', 'fast_evidence.py', 'chi2_comparator.py',
                'progress_certificate.py')
NUMERICAL_ERRORS = (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError)
UNBOUNDED = (0.0, math.inf)


def finite(value):
    return float(value) if math.isfinite(value) else None


def new_row():
    return {'stops': {}, 'last_interval': None, 'last_checkpoint': 0, 'reference_exclusions': [],
            'numerical_error': None, 'evidence_ns': 0}


def update_row(row, interval, step, reference):
    lower, upper = interval
    if math.isnan(lower) or math.isnan(upper):
        raise ArithmeticError('NaN confidence endpoint')
    radius = relative_radius(lower, upper)
    row['last_checkpoint'] = step
    row['last_interval'] = {
        'lower': finite(lower), 'upper': finite(upper), 'radius': finite(radius), 'empty': lower > upper,
    }
    if not lower <= reference <= upper:
        row['reference_exclusions'].append(step)
    for target in TARGETS:
        key = str(target)
        if key not in row['stops'] and radius <= target:
            center = relative_center(lower, upper)
            row['stops'][key] = {'probes': step, 'radius': radius, 'estimate': center,
                                 'relative_error': abs(center - reference) / reference,
                                 'evidence_ns': row['evidence_ns']}


def intersect(first, second):
    return max(first[0], second[0]), min(first[1], second[1])


def run_path(matvec, dimension, reference, seed):
    rng = np.random.default_rng(seed)
    factor = np.empty((dimension, CAP), order='F')
    captured = total = squared_total = fixed_q_total = fixed_v_total = split_trace = 0.0
    oracle_ns = sketch_ns = fixed_residual_ns = 0
    last_probe = 0
    common_error = fixed_error = None
    moving = {alpha: FastTraceEvidence(alpha=alpha) for alpha in ALPHAS}
    chi = {alpha: ChiSquareEvidence(alpha=alpha) for alpha in ALPHAS}
    bounds = {alpha: dict.fromkeys(METHODS, UNBOUNDED) for alpha in ALPHAS}
    rows = {alpha: {method: new_row() for method in METHODS} for alpha in ALPHAS}

    def progress(risk, step):
        return invert_progress(Anchor(0, 0.0, 0.0, risk), step, total, captured, c=PROGRESS_STAKE)

    def frozen_norm(risk, step):
        return invert_frozen_norm(step, total, captured, squared_total, alpha=risk, stake=NORM_STAKE)

    def frozen_residual(risk, step):
        if step <= SPLIT:
            return captured, math.inf
        lower, upper = invert_frozen_norm(step - SPLIT, fixed_q_total, captured - split_trace,
                                          fixed_v_total, alpha=risk, stake=NORM_STAKE)
        return max(captured, split_trace + lower), split_trace + upper

    for step in range(1, CAP + 1):
        previous = captured
        try:
            vector = rng.standard_normal(dimension)
            start = perf_counter_ns()
            image = matvec(vector)
            oracle_ns += perf_counter_ns() - start
            start = perf_counter_ns()
            basis = factor[:, :step - 1]
            residual = image - basis @ (basis.T @ vector)
            q, v = float(vector @ residual), float(residual @ residual)
            if not (q > 0 and v > 0 and math.isfinite(q + v)):
                raise ArithmeticError('Nonpositive or nonfinite moving residual')
            captured = previous + v / q
            total += previous + q
            squared_total += v
            if not (captured > previous and math.isfinite(captured + total + squared_total)):
                raise ArithmeticError('Trace progress lost positivity or exceeded numerical range')
            factor[:, step - 1] = residual / math.sqrt(q)
            sketch_ns += perf_counter_ns() - start
            last_probe = step
        except NUMERICAL_ERRORS as error:
            common_error = {'probe': step, 'reason': str(error)}
            break

        if step == SPLIT:
            split_trace = captured
        elif step > SPLIT and fixed_error is None:
            start = perf_counter_ns()
            fixed_basis = factor[:, :SPLIT]
            fixed_residual = image - fixed_basis @ (fixed_basis.T @ vector)
            fixed_q = float(vector @ fixed_residual)
            fixed_v = float(fixed_residual @ fixed_residual)
            fixed_residual_ns += perf_counter_ns() - start
            if not (fixed_q > 0 and fixed_v > 0 and math.isfinite(fixed_q + fixed_v)):
                fixed_error = {'probe': step, 'reason': 'Nonpositive or nonfinite frozen residual'}
                for alpha in ALPHAS:
                    rows[alpha]['frozen_residual']['numerical_error'] = fixed_error
            else:
                fixed_q_total += fixed_q
                fixed_v_total += fixed_v

        for alpha in ALPHAS:
            for method, process, third in (('moving_norm', moving[alpha], v),
                                            ('chi_square', chi[alpha], captured)):
                row = rows[alpha][method]
                if row['numerical_error'] is not None:
                    continue
                start = perf_counter_ns()
                try:
                    process.append(previous, q, third)
                except NUMERICAL_ERRORS as error:
                    row['numerical_error'] = {'probe': step, 'reason': str(error)}
                finally:
                    row['evidence_ns'] += perf_counter_ns() - start

        if step % REPORT_EVERY:
            continue

        def check(method, alpha):
            if method == 'progress':
                return progress(alpha, step)
            if method == 'frozen_norm':
                return frozen_norm(alpha, step)
            if method == 'moving_norm':
                return moving[alpha].interval()
            if method == 'chi_square':
                return chi[alpha].interval()
            if method == 'frozen_residual':
                return frozen_residual(alpha, step)
            return intersect(progress(alpha / 2, step), frozen_norm(alpha / 2, step))

        rotation = (seed + step // REPORT_EVERY) % len(METHODS)
        for alpha in ALPHAS:
            for method in METHODS[rotation:] + METHODS[:rotation]:
                row = rows[alpha][method]
                if row['numerical_error'] is not None:
                    continue
                start = perf_counter_ns()
                try:
                    interval = check(method, alpha)
                    bounds[alpha][method] = intersect(bounds[alpha][method], interval)
                    row['evidence_ns'] += perf_counter_ns() - start
                    update_row(row, bounds[alpha][method], step, reference)
                except NUMERICAL_ERRORS as error:
                    row['numerical_error'] = {'probe': step, 'reason': str(error)}

    if common_error is not None:
        for methods in rows.values():
            for row in methods.values():
                if row['numerical_error'] is None:
                    row['numerical_error'] = common_error
    return {'seed': seed, 'last_probe': last_probe, 'shared_numerical_error': common_error,
            'oracle_ns': oracle_ns, 'sketch_ns': sketch_ns, 'frozen_residual_extra_ns': fixed_residual_ns,
            'methods': {str(alpha): methods for alpha, methods in rows.items()}}


def summarize(trials, case_index):
    summary = {}
    for alpha_index, alpha in enumerate(ALPHAS):
        level = str(alpha)
        summary[level] = {}
        for target_index, target in enumerate(TARGETS):
            key = str(target)
            methods, costs = {}, {}
            for method in METHODS:
                rows = [trial['methods'][level][method] for trial in trials]
                complete = [row for row in rows if row['last_checkpoint'] == CAP]
                incomplete = [row for row in rows if row['last_checkpoint'] != CAP]
                stops = [row['stops'].get(key) for row in rows]
                costs[method] = np.array([stop['probes'] if stop is not None else CAP for stop in stops])
                stopped = [stop for stop in stops if stop is not None]
                terminal_radii = [row['last_interval']['radius'] for row in complete
                                  if row['last_interval']['radius'] is not None]
                exclusions = sum(bool(row['reference_exclusions']) for row in complete)
                methods[method] = {
                    'completions': len(stopped),
                    'incorrect_completions': sum(stop['relative_error'] > target for stop in stopped),
                    'restricted_mean_probes': float(costs[method].mean()),
                    'median_probes_among_stops': float(np.median([stop['probes'] for stop in stopped]))
                    if stopped else None,
                    'median_terminal_radius_among_finite': float(np.median(terminal_radii)) if terminal_radii else None,
                    'finite_terminal_radii': len(terminal_radii),
                    'complete_replays': len(complete), 'pathwise_exclusions': exclusions,
                    'one_sided_95pct_failure_rate_upper': upper_failure_limit(exclusions, len(complete)),
                    'incomplete_replay_exclusions': sum(bool(row['reference_exclusions']) for row in incomplete),
                    'numerical_failures': sum(row['numerical_error'] is not None for row in rows),
                    'median_evidence_ms_full_replay': float(np.median([row['evidence_ns'] for row in complete])) / 1e6
                    if complete else None,
                    'median_evidence_ms_at_stop': float(np.median([stop['evidence_ns'] for stop in stopped])) / 1e6
                    if stopped else None,
                }

            def paired(method, comparator):
                differences = costs[method] - costs[comparator]
                wins = int(np.sum(differences < 0))
                losses = int(np.sum(differences > 0))
                return differences, {'mean_difference': float(differences.mean()), 'wins': wins, 'losses': losses,
                                     'ties': len(differences) - wins - losses}

            differences, contrast = paired('frozen_norm', 'progress')
            bootstrap_rng = np.random.default_rng(8200000 + case_index * 1000 + alpha_index * 10 + target_index)
            draws = bootstrap_rng.choice(differences, size=(5000, len(differences)), replace=True).mean(axis=1)
            contrast['paired_percentile_bootstrap_95pct'] = np.quantile(draws, (0.025, 0.975)).tolist()
            non_ties = contrast['wins'] + contrast['losses']
            contrast['sign_test_two_sided'] = float(binomtest(contrast['wins'], non_ties).pvalue) if non_ties else 1.0
            summary[level][key] = {
                'methods': methods, 'primary_frozen_norm_minus_progress': contrast,
                'descriptive_frozen_norm_minus_control': {
                    method: paired('frozen_norm', method)[1] for method in METHODS[2:]
                },
            }
    return summary


def run_case(index):
    case = CASES[index]
    SOURCES.update(NEW_SOURCES)
    label = 'repeated operator' if index < len(PREVIOUS_CASES) else 'new source matrix'
    with threadpool_limits(limits=1):
        try:
            matvec, reference, audit = load_case(case)
            if audit['dimension'] <= CAP:
                raise ValueError('Dimension must exceed the replay cap')
        except (*NUMERICAL_ERRORS, OSError, requests.RequestException, zipfile.BadZipFile) as error:
            return case.name, {'status': 'operator_audit_failure', 'source_category': label, 'reason': str(error)}

        # The chi-square thresholds depend only on the horizon and level.
        for alpha in ALPHAS:
            for step in range(REPORT_EVERY, CAP + 1, REPORT_EVERY):
                evidence_cutoffs(step, alpha)
        trials = [run_path(matvec, audit['dimension'], reference, FIRST_SEED + index * SEED_STRIDE + path)
                  for path in range(PATHS)]
    return case.name, {'status': 'complete', 'source_category': label, 'audit': audit,
                       'summary': summarize(trials, index), 'trials': trials}


def holm(cases):
    contrasts = [values['primary_frozen_norm_minus_progress'] for case in cases.values()
                 for targets in case.get('summary', {}).values() for values in targets.values()]
    count = len(CASES) * len(ALPHAS) * len(TARGETS)
    previous = 0.0
    for index, row in enumerate(sorted(contrasts, key=lambda item: item['sign_test_two_sided'])):
        previous = min(1.0, max(previous, (count - index) * row['sign_test_two_sided']))
        row['holm_adjusted_p'] = previous


def write_output(data):
    destination = HERE / 'anchor_norm_comparison.json'
    staging = destination.with_suffix('.tmp')
    staging.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    staging.replace(destination)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error('--jobs must be positive')
    result = {
        'status': 'RUNNING', 'cap': CAP, 'report_every': REPORT_EVERY, 'paths_per_operator': PATHS,
        'planned_operators': [case.name for case in CASES], 'first_seed': FIRST_SEED, 'seed_stride': SEED_STRIDE,
        'alphas': ALPHAS, 'targets': TARGETS, 'split': SPLIT, 'progress_stake': PROGRESS_STAKE, 'norm_stake': NORM_STAKE,
        'planned_primary_contrasts': len(CASES) * len(ALPHAS) * len(TARGETS),
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'jobs': args.jobs, 'blas_threads_per_worker': 1,
        'source_sha256': {name: file_hash(HERE / name) for name in SOURCE_FILES}, 'cases': {},
    }
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_case, index): case for index, case in enumerate(CASES)}
        for job in as_completed(futures):
            name, output = job.result()
            result['cases'][name] = output
            print(f"{name}: {output['status']}", flush=True)
            write_output(result)
    result['cases'] = {case.name: result['cases'][case.name] for case in CASES}
    holm(result['cases'])
    result['status'] = 'COMPLETE'
    write_output(result)


if __name__ == '__main__':
    main()
