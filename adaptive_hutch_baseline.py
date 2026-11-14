"""Persson–Cortinovis–Kressner Algorithm 2 with a separate relative-error pilot.

The source is arXiv:2109.10659v3, pp. 6–11, (11), (14), Lemmas 2.1–2.3
and Algorithm 2. This uses its independent sizing and final trace probes,
positive ell, and stable-rank-free sample floor. All proofs assume exact
arithmetic. The output is a simulation using floating-point inverse products.
"""

from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaincinv
from threadpoolctl import threadpool_limits

from baselines import gaussian_psd_fixed_interval
from run_progress_pilot import DIMENSIONS, load_operator


HERE = Path(__file__).resolve().parent
TARGET = 0.05
LOWER_RISK = 0.01
STAGE_RISK = 0.02
PILOT = 16
NORM_PROBES = 16
ELL = 1.0
SEED_START = 701000


class Oracle:
    def __init__(self, matvec, dimension):
        self.matvec = matvec
        self.dimension = dimension
        self.calls = 0

    def __call__(self, vector):
        self.calls += 1
        image = np.asarray(self.matvec(vector), dtype=float)
        if image.shape != (self.dimension,) or not np.isfinite(image).all():
            raise ArithmeticError('Invalid oracle product')
        return image


def pilot_lower(oracle, dimension, rng):
    probes = rng.standard_normal((PILOT, dimension)).T
    images = np.column_stack([oracle(g) for g in probes.T])
    gaussian = float(np.sum(probes * images) / PILOT)
    # Only the lower endpoint is used. The fixed-time routine allocates half
    # its delta to that endpoint.
    lower, _ = gaussian_psd_fixed_interval(gaussian, PILOT, 2 * LOWER_RISK)
    # The same pilot products also give a pathwise PSD Nyström lower bound.
    # Taking a maximum with it does not spend any extra error probability.
    gram = probes.T @ images
    chol = np.linalg.cholesky((gram + gram.T) / 2)
    factor = solve_triangular(chol, images.T, lower=True)
    nystrom = float(np.sum(factor * factor))
    return max(lower, nystrom), lower, nystrom


def grow_projector(oracle, dimension, cap, coefficient, rng):
    """Algorithm 2 lines 1–16 with block size one and two rises to stop."""
    capacity = min(dimension, (cap - oracle.calls) // 2)
    basis = np.empty((dimension, capacity), order='F')
    trace = 0.0
    small_frob_sq = 0.0
    image_frob_sq = 0.0
    objective = []
    for index in range(capacity):
        span = basis[:, :index]
        vector = oracle(rng.standard_normal(dimension))
        for _ in range(2):
            vector -= span @ (span.T @ vector)
        length = float(np.linalg.norm(vector))
        if length <= 0.0 or not math.isfinite(length):
            raise ArithmeticError('Projection sketch lost rank')
        vector /= length
        image = oracle(vector)
        diagonal = float(vector @ image)
        cross = span.T @ image
        small_frob_sq += diagonal**2 + 2 * float(cross @ cross)
        image_frob_sq += float(image @ image)
        trace += diagonal
        basis[:, index] = vector
        rank = index + 1
        objective.append(2 * rank + coefficient * (small_frob_sq - 2 * image_frob_sq))
        if len(objective) >= 3 and objective[-1] > objective[-2] > objective[-3]:
            return basis[:, :rank], trace, objective
        if rank == dimension:
            return basis, trace, objective
    return None, trace, objective


def projected_pair(oracle, basis, vector):
    # Apply (I−QQᵀ) A (I−QQᵀ) with one counted vector product.
    projected = vector - basis @ (basis.T @ vector)
    image = oracle(projected)
    residual = image - basis @ (basis.T @ image)
    return float(projected @ residual), float(residual @ residual)


def run_trial(matvec, dimension, cap, seed, reference):
    oracle = Oracle(matvec, dimension)
    # These four batches are independent in the exact-oracle analysis.
    streams = np.random.SeedSequence(seed).spawn(4)
    pilot_rng, sketch_rng, sizing_rng, final_rng = map(np.random.default_rng, streams)
    record = {'seed': seed}
    try:
        lower, gaussian_lower, nystrom_lower = pilot_lower(oracle, dimension, pilot_rng)
        if lower <= 0 or not math.isfinite(lower):
            raise ArithmeticError('Invalid lower-bound pilot')
        tolerance = TARGET * lower
        coefficient = 4 * (1 + ELL) * math.log(2 / STAGE_RISK) / tolerance**2
        basis, captured, objective = grow_projector(oracle, dimension, cap, coefficient, sketch_rng)
        rank = len(objective)
        record.update(lower_bound=lower, gaussian_lower=gaussian_lower, nystrom_lower=nystrom_lower,
                      lower_bound_excluded_reference=lower > reference, absolute_tolerance=tolerance,
                      sketch_rank=rank, sketch_products=2 * rank, sketch_trace=captured)
        if basis is None:
            return record | {'status': 'cap_sketch', 'matvecs': oracle.calls}
        if rank == dimension:
            return record | {'status': 'complete', 'matvecs': oracle.calls, 'estimate': captured,
                             'relative_error': abs(captured / reference - 1), 'norm_products': 0,
                             'final_products': 0, 'requested_total_products': oracle.calls}
        if oracle.calls + NORM_PROBES > cap:
            return record | {'status': 'cap_sizing', 'matvecs': oracle.calls}

        # Algorithm 2 line 17, equality in the regularized incomplete gamma
        # lower-tail criterion. This is the largest allowable alpha for fixed k.
        shape = NORM_PROBES / 2
        alpha_k = float(gammaincinv(shape, STAGE_RISK) / shape)
        squared_norm = sum(projected_pair(oracle, basis, sizing_rng.standard_normal(dimension))[1]
                           for _ in range(NORM_PROBES))
        frob_sq_upper = squared_norm / (NORM_PROBES * alpha_k)
        floor = math.ceil(4 * (1 + ELL) * math.log(2 / STAGE_RISK) / ELL**2)
        final_probes = max(floor, math.ceil(coefficient * frob_sq_upper))
        required = oracle.calls + final_probes
        record.update(norm_products=NORM_PROBES, frob_sq_upper=frob_sq_upper,
                      norm_gamma_alpha=alpha_k, deterministic_final_sample_floor=floor,
                      requested_final_products=final_probes, requested_total_products=required)
        if required > cap:
            return record | {'status': 'cap_estimation', 'matvecs': oracle.calls}

        correction = sum(projected_pair(oracle, basis, final_rng.standard_normal(dimension))[0]
                         for _ in range(final_probes)) / final_probes
        estimate = captured + correction
        assert oracle.calls == required
        return record | {'status': 'complete', 'matvecs': oracle.calls, 'estimate': estimate,
                         'relative_error': abs(estimate / reference - 1), 'final_products': final_probes,
                         'absolute_error_exceeded': abs(estimate - reference) > tolerance}
    except (ArithmeticError, np.linalg.LinAlgError, ValueError, OverflowError) as error:
        return record | {'status': 'numerical_failure', 'matvecs': oracle.calls, 'error': str(error)}


def summarize(trials):
    complete = [row for row in trials if row['status'] == 'complete']
    requested = [row['requested_total_products'] for row in trials if 'requested_total_products' in row]
    return {'statuses': {status: sum(row['status'] == status for row in trials)
                         for status in ('complete', 'cap_sketch', 'cap_sizing', 'cap_estimation', 'numerical_failure')},
            'median_products_among_completed': float(np.median([row['matvecs'] for row in complete])) if complete else None,
            'median_requested_total_among_sized': float(np.median(requested)) if requested else None,
            'median_products_spent': float(np.median([row['matvecs'] for row in trials])),
            'incorrect_completed_outputs': sum(row['relative_error'] > TARGET for row in complete),
            'completed_absolute_error_exceedances': sum(row.get('absolute_error_exceeded', False) for row in complete)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--matrices', nargs='+', choices=tuple(DIMENSIONS), default=tuple(DIMENSIONS))
    parser.add_argument('--seeds', type=int, default=24)
    parser.add_argument('--cap', type=int, default=256)
    args = parser.parse_args()
    if args.seeds < 1 or args.cap < PILOT + 2:
        parser.error('Use at least one seed and a cap covering the pilot and first sketch step')
    result = {'status': 'EXPLORATORY_PILOT', 'method': 'A-Hutch++ Algorithm 2 + independent PSD lower-bound pilot',
              'source': 'https://arxiv.org/pdf/2109.10659', 'target_relative_error': TARGET,
              'confidence': 1 - LOWER_RISK - 2 * STAGE_RISK,
              'lower_pilot_failure_budget': LOWER_RISK, 'each_algorithm_2_stage_failure_budget': STAGE_RISK,
              'ell': ELL, 'lower_pilot_products': PILOT, 'norm_pilot_products': NORM_PROBES,
              'cap': args.cap, 'seed_start': SEED_START, 'seeds_per_matrix': args.seeds,
              'source_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__), HERE / 'baselines.py', HERE / 'run_progress_pilot.py')},
              'notes': ['All confidence claims assume exact PSD oracle products and exact arithmetic.',
                        'Matrix reference trace uses floating-point unit solves outside query accounting.',
                        'A cap exit emits no estimate. The unconditional probability of an inaccurate returned estimate is bounded, with no claim conditional on returning.',
                        'Lower-bound pilot, sketch, norm and final trace streams are disjoint. The pilot uses the maximum of a Chernoff lower bound and a Nyström lower bound.',
                        'Sketch rank selected using the original two-consecutive-rises heuristic. The proof conditions on the selected sketch and does not require that the heuristic optimizes cost.'],
              'matrices': {}}
    output = HERE / 'adaptive_hutch_baseline_results.json'
    with threadpool_limits(limits=1):
        for name in args.matrices:
            start = perf_counter()
            matvec, reference, audit = load_operator(name)
            print(name, 'reference ready', flush=True)
            trials = [run_trial(matvec, DIMENSIONS[name], args.cap, seed, reference)
                      for seed in range(SEED_START, SEED_START + args.seeds)]
            summary = summarize(trials)
            result['matrices'][name] = {'audit': audit, 'summary': summary, 'trials': trials,
                                        'seconds_including_reference': perf_counter() - start}
            temp = output.with_suffix('.tmp')
            temp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            temp.replace(output)
            print(name, json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
