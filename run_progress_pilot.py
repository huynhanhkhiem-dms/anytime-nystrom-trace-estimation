"""Exploratory test on inverse operators from NIST structural matrices."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen
import json
import math
import numpy as np
from scipy.io import mmread
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from core import TraceEvidence
from chi2_comparator import ChiSquareEvidence
from progress_certificate import ProgressSequence, relative_center, relative_radius


HERE = Path(__file__).resolve().parent
BASE = 'https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/bcsstruc2/'
DIMENSIONS = {'bcsstk14': 1806, 'bcsstk15': 3948, 'bcsstk16': 4884}
TARGETS = (0.10, 0.05, 0.02)


def load_operator(name):
    path = HERE / 'matrices' / (name + '.mtx.gz')
    path.parent.mkdir(exist_ok=True)
    url = BASE + path.name
    if not path.exists():
        with urlopen(url, timeout=45) as response:
            payload = response.read()
        if not payload.startswith(b'\x1f\x8b'):
            raise ValueError('Expected a gzip archive')
        path.write_bytes(payload)
    matrix = mmread(path, spmatrix=True).tocsc()
    n = DIMENSIONS[name]
    if matrix.shape != (n, n) or (matrix - matrix.T).nnz or not np.isfinite(matrix.data).all():
        raise ValueError('Unexpected shape, symmetry, or values')
    factor = splu(matrix)
    # Reference only. The probe procedure sees factor.solve as a vector oracle.
    trace = 0.0
    for start in range(0, n, 48):
        indices = np.arange(start, min(start + 48, n))
        rhs = np.zeros((n, len(indices)), order='F')
        rhs[indices, np.arange(len(indices))] = 1
        block = factor.solve(rhs)
        trace += float(block[indices, np.arange(len(indices))].sum())
    rng = np.random.default_rng(800321)
    probe = rng.standard_normal(n)
    solution = factor.solve(probe)
    diagnostic = float(np.linalg.norm(matrix @ solution - probe) / np.linalg.norm(probe))
    eigen_trace = None
    if n <= 2000:
        spectrum = eigvalsh(matrix.toarray())
        if spectrum[0] <= 0:
            raise ValueError('Independent dense eigenvalue check did not find PSD')
        eigen_trace = float((1 / spectrum).sum())
    audit = {'source': url, 'bytes': path.stat().st_size, 'sha256': sha256(path.read_bytes()).hexdigest(),
             'dimension': n, 'nnz': matrix.nnz, 'trace_from_all_unit_solves': trace,
             'trace_from_dense_eigenspectrum_if_computed': eigen_trace,
             'one_probe_relative_linear_solve_residual': diagnostic,
             'reference_is_floating_point': True}
    return factor.solve, trace, audit


def run_path(matvec, n, budget, seed, theta):
    rng = np.random.default_rng(seed)
    factors = np.empty((n, budget), order='F')
    total = 0.0
    sketch = 0.0
    observed_variance = 0.0
    single = ProgressSequence()
    anchored = ProgressSequence(anchors=True)
    empirical = TraceEvidence()
    hybrid_progress = ProgressSequence(alpha=0.025, anchors=True)
    hybrid_empirical = TraceEvidence(alpha=0.025)
    chi_square = ChiSquareEvidence()
    doob_lower, doob_upper = 0.0, math.inf
    doob_constant = math.sqrt(math.pi / 0.05)
    methods = ('progress_single', 'progress_anchored', 'residual_v', 'doob', 'chi_square', 'hybrid')
    paths = {name: [] for name in methods}
    for step in range(1, budget + 1):
        vector = rng.standard_normal(n)
        previous = factors[:, :step-1]
        residual = matvec(vector) - previous @ (previous.T @ vector)
        q = float(vector @ residual)
        v = float(residual @ residual)
        if not (q > 0 and v > 0 and math.isfinite(q) and math.isfinite(v)):
            raise ArithmeticError(f'Nonpositive or nonfinite residual at probe {step}')
        empirical.append(sketch, q, v)
        hybrid_empirical.append(sketch, q, v)
        factors[:, step-1] = residual / math.sqrt(q)
        previous_trace = sketch
        total += sketch + q
        sketch += v / q
        if sketch <= previous_trace:
            raise ArithmeticError(f'Positive trace increment lost to roundoff at probe {step}')
        chi_square.append(previous_trace, q, sketch)
        observed_variance += v
        one = single.inspect(step, total, sketch)
        many = anchored.inspect(step, total, sketch)
        hybrid_fast = hybrid_progress.inspect(step, total, sketch)
        doob_lower = max(doob_lower, sketch, total / (step + doob_constant))
        if step > doob_constant:
            doob_upper = min(doob_upper, total / (step - doob_constant))
        if step % 4 and step != budget:
            continue
        hybrid_detail = hybrid_empirical.interval()
        hybrid = (max(hybrid_fast[0], hybrid_detail[0]), min(hybrid_fast[1], hybrid_detail[1]))
        values = [one, many, empirical.interval(), (doob_lower, doob_upper), chi_square.interval(), hybrid]
        for name, (lower, upper) in zip(paths, values):
            paths[name].append((step, lower, upper, relative_radius(lower, upper)))

    metrics = {}
    for name, path in paths.items():
        stops = {}
        for target in TARGETS:
            crossing = next((r for r in path if r[3] <= target), None)
            if crossing is None:
                stops[str(target)] = None
            else:
                step, lower, upper, radius = crossing
                stops[str(target)] = {'matvecs': step, 'radius': float(radius),
                                      'relative_error': float(abs(relative_center(lower, upper) / theta - 1))}
        metrics[name] = {'coverage_failure': any(not low <= theta <= high for _, low, high, _ in path),
                         'final_radius': float(path[-1][3]) if math.isfinite(path[-1][3]) else None,
                         'stops': stops}
    return {'seed': seed, 'final_sketch_fraction': sketch / theta,
            'v_sum_relative': observed_variance / theta**2, 'metrics': metrics}


def summarize(records):
    good = [x for x in records if 'metrics' in x]
    result = {'successful_paths': len(good), 'numerical_failures': len(records) - len(good)}
    if not good:
        return result
    for name in good[0]['metrics']:
        values = [x['metrics'][name] for x in good]
        radii = [x['final_radius'] for x in values if x['final_radius'] is not None]
        stop_summary = {}
        for target in TARGETS:
            stops = [x['stops'][str(target)] for x in values if x['stops'][str(target)] is not None]
            stop_summary[str(target)] = {'stopped': len(stops), 'median_matvecs_among_stops':
                float(np.median([x['matvecs'] for x in stops])) if stops else None,
                'incorrect_stops': sum(x['relative_error'] > target for x in stops)}
        result[name] = {'pathwise_coverage_failures': sum(x['coverage_failure'] for x in values),
                        'median_finite_final_radius': float(np.median(radii)) if radii else None,
                        'finite_final_radius_count': len(radii), 'stops': stop_summary}
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument('--matrices', nargs='+', choices=list(DIMENSIONS), default=list(DIMENSIONS))
    parser.add_argument('--seeds', type=int, default=24)
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--name', default='structural_progress_pilot')
    args = parser.parse_args()
    result = {'status': 'EXPLORATORY_PILOT', 'alpha': 0.05, 'inspections_every': 4,
              'budget': args.budget, 'seed_start': 701000, 'seeds_per_matrix': args.seeds,
              'source_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), HERE/'progress_certificate.py', HERE/'core.py', HERE/'chi2_comparator.py']},
              'matrices': {}}
    path = HERE / (args.name + '.json')
    with threadpool_limits(limits=1):
        for name in args.matrices:
            start = perf_counter()
            matvec, theta, audit = load_operator(name)
            print(name, 'reference ready', audit, flush=True)
            records = []
            for index, seed in enumerate(range(701000, 701000 + args.seeds), 1):
                try:
                    records.append(run_path(matvec, audit['dimension'], args.budget, seed, theta))
                except ArithmeticError as error:
                    records.append({'seed': seed, 'numerical_failure': str(error)})
                if index % 4 == 0 or index == args.seeds:
                    result['matrices'][name] = {'data_audit': audit, 'summary': summarize(records),
                        'seconds_including_reference': perf_counter()-start, 'records': records,
                        'completed_paths': index, 'complete': index == args.seeds}
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
                    temporary.replace(path)
            print(name, json.dumps(result['matrices'][name]['summary']), flush=True)


if __name__ == '__main__':
    main()
