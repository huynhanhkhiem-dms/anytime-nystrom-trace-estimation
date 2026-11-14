"""Compare two fixed trace-progress stakes on paired Gaussian prefixes."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from progress_certificate import Anchor, invert_progress, relative_center, relative_radius
from run_confirmatory import ALPHA, CAP, CASES, CHECKPOINTS, TARGET, load_case, upper_failure_limit


HERE = Path(__file__).resolve().parent
TRIALS = 24
SEED_START = 4_800_000
STAKES = {'quarter': 0.25, 'third': 1 / 3}
SCHEDULES = ('frozen_checkpoints', 'every_probe')


def finite(value):
    return float(value) if math.isfinite(value) else None


def nominal_cap(stake):
    threshold = math.log(2 / ALPHA) / stake
    penalty = 4 * stake
    return max(math.floor(threshold) + 1, math.ceil((threshold + penalty) / TARGET - penalty))


def record_interval(row, step, bounds, reference, save):
    lower, upper = bounds
    radius = relative_radius(lower, upper)
    excludes = not lower <= reference <= upper
    row['coverage_failure'] |= excludes
    row['last_bounds'] = [step, finite(lower), finite(upper), finite(radius), excludes]
    if save:
        row['checkpoints'].append(row['last_bounds'])
    if row['stop'] is None and radius <= TARGET:
        estimate = relative_center(lower, upper)
        row['stop'] = {'queries': step, 'estimate': estimate, 'relative_error': abs(estimate / reference - 1),
                       'relative_radius': radius, 'bounds': [lower, upper]}


def run_path(matvec, dimension, reference, seed):
    rng = np.random.default_rng(seed)
    factors = np.empty((dimension, CAP), order='F')
    captured = total = 0.0
    anchor = Anchor(0, 0.0, 0.0, ALPHA)
    states = {name: {schedule: {'coverage_failure': False, 'stop': None, 'checkpoints': [],
                               'last_bounds': None} for schedule in SCHEDULES} for name in STAKES}
    bounds = {name: dict.fromkeys(SCHEDULES, (0.0, math.inf)) for name in STAKES}
    raw_bound_exceedances = dict.fromkeys(STAKES, 0)
    raw_at_nominal_cap = dict.fromkeys(STAKES)
    completed = 0
    failure = None

    for step in range(1, CAP + 1):
        try:
            probe = rng.standard_normal(dimension)
            image = matvec(probe)
            basis = factors[:, :step - 1]
            residual = image - basis @ (basis.T @ probe)
            q, v = float(probe @ residual), float(residual @ residual)
            if q <= 0 or v <= 0 or not math.isfinite(q + v):
                raise ArithmeticError('Nonpositive or nonfinite residual')
            new_trace = captured + v / q
            if new_trace <= captured or not math.isfinite(new_trace):
                raise ArithmeticError('Positive trace increment lost in floating point')
            factors[:, step - 1] = residual / math.sqrt(q)
            total += captured + q
            captured = new_trace
            if not math.isfinite(total):
                raise ArithmeticError('Nonfinite corrected cumulative trace')
            for name, stake in STAKES.items():
                lo, hi = invert_progress(anchor, step, total, captured, c=stake)
                if math.isnan(lo) or math.isnan(hi):
                    raise ArithmeticError(f'NaN endpoint at stake {stake:g}')
                radius = relative_radius(lo, hi)
                if math.isfinite(radius):
                    threshold = math.log(2 / ALPHA) / stake
                    penalty = 4 * stake
                    theoretical_radius = (threshold + penalty) / (step + penalty)
                    raw_bound_exceedances[name] += radius > theoretical_radius + 1e-12
                if step == nominal_cap(stake):
                    raw_at_nominal_cap[name] = {'step': step, 'bounds': [finite(lo), finite(hi)],
                                                'relative_radius': finite(radius),
                                                'contains_reference': lo <= reference <= hi}
                for schedule in SCHEDULES:
                    at_checkpoint = step in CHECKPOINTS
                    if schedule == 'frozen_checkpoints' and not at_checkpoint:
                        continue
                    old_lo, old_hi = bounds[name][schedule]
                    current = (max(old_lo, lo), min(old_hi, hi))
                    bounds[name][schedule] = current
                    record_interval(states[name][schedule], step, current, reference, at_checkpoint)
            completed = step
        except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failure = {'step': step, 'error': str(error)}
            break
    return {'seed': seed, 'completed_queries': completed, 'numerical_failure': failure, 'stakes': states,
            'raw_radius_bound_exceedances': raw_bound_exceedances, 'raw_at_nominal_cap': raw_at_nominal_cap}


def summarize(trials):
    result = {}
    for schedule in SCHEDULES:
        summary = {}
        for name in STAKES:
            rows = [trial['stakes'][name][schedule] for trial in trials]
            complete = [trial['stakes'][name][schedule] for trial in trials if trial['numerical_failure'] is None]
            exclusions = sum(row['coverage_failure'] for row in complete)
            stopped = [row['stop'] for row in rows if row['stop'] is not None]
            radii = [row['last_bounds'][3] for row in complete
                     if row['last_bounds'] is not None and row['last_bounds'][3] is not None]
            summary[name] = {
                'stopped': len(stopped), 'incorrect_stops': sum(row['relative_error'] > TARGET for row in stopped),
                'median_queries_among_stops': float(np.median([row['queries'] for row in stopped])) if stopped else None,
                'restricted_mean_queries': float(np.mean([row['stop']['queries'] if row['stop'] else CAP for row in rows])),
                'median_finite_final_radius': float(np.median(radii)) if radii else None,
                'complete_paths': len(complete), 'complete_path_exclusions': exclusions,
                'all_observed_path_exclusions': sum(row['coverage_failure'] for row in rows),
                'one_sided_95pct_upper_complete_path_exclusion': upper_failure_limit(exclusions, len(complete)),
                'numerical_failures': len(rows) - len(complete),
            }

        def capped_counts(stake_name):
            rows = [trial['stakes'][stake_name][schedule] for trial in trials]
            return np.array([row['stop']['queries'] if row['stop'] else CAP for row in rows])

        quarter, third = capped_counts('quarter'), capped_counts('third')
        statuses = [(trial['stakes']['quarter'][schedule]['stop'] is not None,
                     trial['stakes']['third'][schedule]['stop'] is not None) for trial in trials]
        summary['paired_third_vs_quarter'] = {'lower_capped_count': int(np.sum(third < quarter)),
                                             'equal_capped_count': int(np.sum(third == quarter)),
                                             'higher_capped_count': int(np.sum(third > quarter)),
                                             'third_only_completion': statuses.count((False, True)),
                                             'quarter_only_completion': statuses.count((True, False)),
                                             'mean_paired_query_saving': float(np.mean(quarter - third))}
        result[schedule] = summary
    result['raw_radius_bound_exceedances'] = {
        name: sum(row['raw_radius_bound_exceedances'][name] for row in trials) for name in STAKES}
    return result


def run_case(index):
    case = CASES[index]
    start = perf_counter()
    with threadpool_limits(limits=1):
        matvec, reference, audit = load_case(case)
        trials = [run_path(matvec, audit['dimension'], reference, SEED_START + index * 1000 + row)
                  for row in range(TRIALS)]
    result = {'audit': audit, 'summary': summarize(trials), 'trials': trials,
              'seconds_including_setup': perf_counter() - start}
    return case.name, result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=2, help='Independent operators in parallel')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('Use at least one worker')
    for stake in STAKES.values():
        cap = nominal_cap(stake)
        threshold = math.log(2 / ALPHA) / stake
        b = 4 * stake
        assert (threshold + b) / (cap + b) <= TARGET < (threshold + b) / (cap - 1 + b)

    result = {'status': 'STAKE_FOLLOWUP_RUNNING', 'planned_after_inspecting_initial_holdout': True,
              'trials_per_operator': TRIALS, 'first_seed': SEED_START, 'seed_stride_by_operator': 1000,
              'relative_target': TARGET, 'nominal_alpha_per_stake': ALPHA,
              'stakes': STAKES, 'penalty_coefficient': 4,
              'nominal_exact_every_probe_caps': {name: nominal_cap(c) for name, c in STAKES.items()},
              'nominal_exact_reporting_caps': {name: next((step for step in CHECKPOINTS if step >= nominal_cap(c)), None)
                                               for name, c in STAKES.items()},
              'checkpoints': CHECKPOINTS, 'cap': CAP, 'jobs': args.jobs, 'blas_threads_per_worker': 1,
              'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
              'source_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__), HERE / 'run_confirmatory.py', HERE / 'progress_certificate.py')},
              'scope': ['Both stakes use the same Gaussian products and growing sketch on each path.',
                        'Seed ranges differ from the original frozen holdout.',
                        'Checkpoint and every-probe intersections are maintained independently.',
                        'A null stop is censored at 256 for restricted means. A stop at 256 can therefore tie a censored path.',
                        'Each fixed stake uses its own nominal risk. Post-hoc selection across stakes has no joint 95% guarantee.',
                        'The exact-model caps apply on the coverage event. These experiments use floating-point operators and reference traces.'],
              'cases': {}}
    path = HERE / 'stake_sensitivity_results.json'

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, allow_nan=False, separators=(',', ':')), encoding='utf-8')
        temporary.replace(path)

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        jobs = [executor.submit(run_case, index) for index in range(len(CASES))]
        for job in as_completed(jobs):
            name, case = job.result()
            result['cases'][name] = case
            save()
            print(name, 'complete', case['seconds_including_setup'], flush=True)
    result['status'] = 'STAKE_FOLLOWUP_COMPLETE'
    save()


if __name__ == '__main__':
    main()
