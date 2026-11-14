"""Replay matched-risk Gaussian NA-Hutch++ and Nyström certificates."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
import json
import math
from pathlib import Path
import platform
from time import perf_counter
from zipfile import BadZipFile

import numpy as np
import requests
import scipy
from scipy.linalg import eigvalsh, pinv
from scipy.optimize import brentq
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

from anchored_norm import invert_frozen_norm
from progress_certificate import Anchor, invert_progress, relative_center, relative_radius
from run_anchor_norm_comparison import CASES, NEW_SOURCES, NUMERICAL_ERRORS
from run_confirmatory import CHECKPOINTS, SOURCES, file_hash, load_case, upper_failure_limit


HERE = Path(__file__).resolve().parent
CAP = CHECKPOINTS[-1]
PATHS = 24
FIRST_SEED = 9060000
SEED_STRIDE = 1000
ALPHA = 0.05
TARGET = 0.05
STAKE = 1 / 3
NYSTROM = ('progress', 'frozen_norm_same_stake', 'frozen_norm')
NA = ('gaussian_na_hutchpp', 'gaussian_na_hutchpp_with_nystrom_lower')
METHODS = NYSTROM + NA
SOURCE_FILES = ('run_na_checkpoint.py', 'na_checkpoint_protocol.txt', 'run_anchor_norm_comparison.py',
                'run_confirmatory.py', 'progress_certificate.py', 'anchored_norm.py')


def finite(value):
    return float(value) if math.isfinite(value) else None


@cache
def critical_values(heldout_count):
    checkpoint_risk = ALPHA / len(CHECKPOINTS)
    norm_risk = checkpoint_risk / 4
    exponent = math.log(4 / checkpoint_risk)

    def lower_tail_equation(log_u):
        return heldout_count * (math.expm1(log_u) - log_u) / 2 - exponent

    min_log_u = -1 - 2 * exponent / heldout_count
    u = math.exp(brentq(lower_tail_equation, min_log_u, 0))
    return u, float(chi2.ppf(norm_risk, heldout_count)), exponent


def gaussian_na_checkpoint(probes, images):
    """Return Algorithm 2's point estimate and signed-residual certificate."""
    budget = probes.shape[1]
    left_count = budget // 4
    holdout_start = 3 * left_count
    left = probes[:, :left_count]
    left_images = images[:, :left_count]
    right_images = images[:, left_count:holdout_start]
    holdout = probes[:, holdout_start:]
    holdout_images = images[:, holdout_start:]
    heldout_count = holdout.shape[1]

    inverse, rank = pinv(left.T @ right_images, return_rank=True)
    factor = right_images @ inverse
    approximation_trace = float(np.einsum('ij,ij->', factor, left_images))
    residual = holdout_images - 0.5 * (
        factor @ (left_images.T @ holdout) + left_images @ (factor.T @ holdout)
    )
    point = approximation_trace + float(np.einsum('ij,ij->', holdout, residual)) / heldout_count

    u, chi_square_quantile, exponent = critical_values(heldout_count)
    residual_gram = residual.T @ residual
    top_index = heldout_count - 1
    largest_gram_eigenvalue = float(eigvalsh(residual_gram, subset_by_index=(top_index, top_index))[0])
    frobenius_upper = math.sqrt(float(np.trace(residual_gram)) / (heldout_count * u))
    spectral_upper = min(frobenius_upper, math.sqrt(max(0.0, largest_gram_eigenvalue) / chi_square_quantile))
    half_width = 2 * (frobenius_upper * math.sqrt(exponent / heldout_count)
                      + spectral_upper * exponent / heldout_count)
    if not all(map(math.isfinite, (point, half_width, point + half_width))):
        raise ArithmeticError('Nonfinite NA-Hutch++ point estimate or radius')
    return point, half_width, approximation_trace, frobenius_upper, spectral_upper, int(rank)


def nystrom_checkpoints(probes, images):
    factor = np.empty_like(probes, order='F')
    captured = corrected_sum = norm_sum = 0.0
    next_checkpoint = iter(CHECKPOINTS)
    report = next(next_checkpoint)
    for index in range(probes.shape[1]):
        probe = probes[:, index]
        basis = factor[:, :index]
        residual = images[:, index] - basis @ (basis.T @ probe)
        quadratic = float(probe @ residual)
        norm = float(residual @ residual)
        if not (quadratic > 0 and norm > 0 and math.isfinite(quadratic + norm)):
            yield index + 1, None, 'Nonpositive or nonfinite Nyström residual'
            return
        previous = captured
        captured += norm / quadratic
        corrected_sum += previous + quadratic
        norm_sum += norm
        if not (captured > previous and math.isfinite(captured + corrected_sum + norm_sum)):
            yield index + 1, None, 'Nyström trace progress lost positivity or exceeded numerical range'
            return
        factor[:, index] = residual / math.sqrt(quadratic)
        count = index + 1
        if count == report:
            yield count, (captured, corrected_sum, norm_sum), None
            report = next(next_checkpoint, None)


def report_interval(row, interval, count, reference):
    last = row['reports'][-1] if row['reports'] else None
    lower, upper = interval
    if last is not None:
        lower = max(last['lower'], lower)
        if last['upper'] is not None:
            upper = min(last['upper'], upper)
    radius = relative_radius(lower, upper)
    row['reports'].append({'products': count, 'lower': lower, 'upper': finite(upper), 'radius': finite(radius)})
    if not lower <= reference <= upper:
        row['reference_exclusions'].append(count)
    if row['stop'] is None and radius <= TARGET:
        estimate = relative_center(lower, upper)
        row['stop'] = {'products': count, 'estimate': estimate, 'radius': radius,
                       'relative_error': abs(estimate - reference) / reference}


def run_path(matvec, dimension, reference, seed):
    rng = np.random.default_rng(seed)
    probes = rng.standard_normal((CAP, dimension)).T
    rows = {name: {'reports': [], 'stop': None, 'reference_exclusions': [], 'numerical_failure': None}
            for name in METHODS}
    points = []
    result = {'seed': seed, 'methods': rows, 'na_hutchpp_point_estimates': points}

    try:
        images = np.asfortranarray(matvec(probes))
        if images.shape != probes.shape or not np.isfinite(images).all():
            raise ArithmeticError('Invalid oracle product block')
    except NUMERICAL_ERRORS as error:
        for row in rows.values():
            row['numerical_failure'] = {'products': 0, 'reason': str(error)}
        return result

    nystrom_reports = {}
    for count, scores, reason in nystrom_checkpoints(probes, images):
        if reason is not None:
            for name in (*NYSTROM, NA[1]):
                rows[name]['numerical_failure'] = {'products': count, 'reason': reason}
            break
        nystrom_reports[count] = scores

    origin = Anchor(0, 0.0, 0.0, ALPHA)
    na_failure = None
    for count in CHECKPOINTS:
        scores = nystrom_reports.get(count)
        if scores is not None:
            captured, corrected_sum, norm_sum = scores
            report_interval(rows[NYSTROM[0]], invert_progress(origin, count, corrected_sum, captured, c=STAKE),
                            count, reference)
            for name, stake in ((NYSTROM[1], STAKE), (NYSTROM[2], 0.5)):
                interval = invert_frozen_norm(count, corrected_sum, captured, norm_sum, alpha=ALPHA, stake=stake)
                report_interval(rows[name], interval, count, reference)

        if na_failure is not None:
            continue
        try:
            point, half_width, approximation_trace, frobenius_upper, spectral_upper, rank = gaussian_na_checkpoint(
                probes[:, :count], images[:, :count]
            )
        except NUMERICAL_ERRORS as error:
            na_failure = {'products': count, 'reason': str(error)}
            for name in NA:
                if rows[name]['numerical_failure'] is None:
                    rows[name]['numerical_failure'] = na_failure
            continue
        signed_error = (point - reference) / reference
        points.append({'products': count, 'point_estimate': point, 'absolute_half_width': half_width,
                       'relative_error': abs(signed_error), 'signed_relative_error': signed_error,
                       'approximation_trace': approximation_trace, 'rank': rank,
                       'frobenius_upper': frobenius_upper, 'spectral_upper': spectral_upper})
        interval = max(0.0, point - half_width), point + half_width
        report_interval(rows[NA[0]], interval, count, reference)
        if scores is not None:
            report_interval(rows[NA[1]], (max(captured, interval[0]), interval[1]), count, reference)
    return result


def summarize(trials):
    methods, costs = {}, {}
    for name in METHODS:
        rows = [trial['methods'][name] for trial in trials]
        complete = [row for row in rows if len(row['reports']) == len(CHECKPOINTS)]
        stopped = [row['stop'] for row in rows if row['stop'] is not None]
        radii = [row['reports'][-1]['radius'] for row in complete if row['reports'][-1]['radius'] is not None]
        exclusions = sum(bool(row['reference_exclusions']) for row in complete)
        costs[name] = np.asarray([row['stop']['products'] if row['stop'] is not None else CAP for row in rows])
        methods[name] = {
            'completions': len(stopped),
            'incorrect_completions': sum(stop['relative_error'] > TARGET for stop in stopped),
            'restricted_mean_products': float(costs[name].mean()),
            'median_products_among_stops': float(np.median([stop['products'] for stop in stopped])) if stopped else None,
            'median_terminal_radius_among_finite': float(np.median(radii)) if radii else None,
            'finite_terminal_radii': len(radii),
            'complete_replays': len(complete),
            'pathwise_reference_exclusions_complete': exclusions,
            'pathwise_reference_exclusions_all': sum(bool(row['reference_exclusions']) for row in rows),
            'one_sided_95pct_failure_rate_upper': upper_failure_limit(exclusions, len(complete)),
            'numerical_interruptions': sum(row['numerical_failure'] is not None for row in rows),
        }

    paired = {}
    for name in NYSTROM:
        paired[name] = {}
        for comparator in NA:
            differences = costs[name] - costs[comparator]
            paired[name][comparator] = {
                'mean_difference_products': float(differences.mean()),
                'wins': int(np.sum(differences < 0)), 'ties': int(np.sum(differences == 0)),
                'losses': int(np.sum(differences > 0)),
            }

    raw_points = {}
    for count in CHECKPOINTS:
        errors = [point['relative_error'] for trial in trials for point in trial['na_hutchpp_point_estimates']
                  if point['products'] == count]
        raw_points[count] = {
            'trials': len(errors), 'within_target': sum(error <= TARGET for error in errors),
            'median_relative_error': float(np.median(errors)) if errors else None,
            'relative_rmse': math.sqrt(np.mean(np.square(errors))) if errors else None,
        }
    return {'methods': methods, 'paired_nystrom_minus_na': paired, 'raw_na_point_errors': raw_points}


def run_case(index):
    case = CASES[index]
    SOURCES.update(NEW_SOURCES)
    with threadpool_limits(limits=1):
        try:
            matvec, reference, audit = load_case(case)
        except (*NUMERICAL_ERRORS, OSError, requests.RequestException, BadZipFile) as error:
            return case.name, {'status': 'operator_audit_failure', 'reason': str(error)}
        trials = [run_path(matvec, audit['dimension'], reference, FIRST_SEED + index * SEED_STRIDE + path)
                  for path in range(PATHS)]
    return case.name, {'status': 'complete', 'audit': audit, 'summary': summarize(trials), 'trials': trials}


def write_output(data):
    destination = HERE / 'na_checkpoint_results.json'
    staging = destination.with_suffix('.tmp')
    staging.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    staging.replace(destination)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error('--jobs must be positive')
    start = perf_counter()
    result = {
        'status': 'RUNNING', 'alpha': ALPHA, 'target': TARGET, 'cap': CAP, 'checkpoints': CHECKPOINTS,
        'paths_per_operator': PATHS, 'first_seed': FIRST_SEED, 'seed_stride': SEED_STRIDE,
        'progress_stake': STAKE, 'frozen_norm_stakes': [STAKE, 0.5], 'na_allocation': [0.25, 0.5, 0.25],
        'nominal_na_checkpoint_risk': ALPHA / len(CHECKPOINTS),
        'planned_operators': [case.name for case in CASES],
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'jobs': args.jobs, 'blas_threads_per_worker': 1,
        'source_sha256': {name: file_hash(HERE / name) for name in SOURCE_FILES}, 'cases': {},
    }
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_case, index) for index in range(len(CASES))]
        for future in as_completed(futures):
            name, case_result = future.result()
            print(f'{name}: {case_result["status"]}', flush=True)
            result['cases'][name] = case_result
            write_output(result)
    result['cases'] = {case.name: result['cases'][case.name] for case in CASES}
    result['status'] = 'COMPLETE'
    result['elapsed_seconds'] = perf_counter() - start
    write_output(result)


if __name__ == '__main__':
    main()
