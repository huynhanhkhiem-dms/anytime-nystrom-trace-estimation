"""Serial timing of the estimators on one real inverse operator."""
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from chi2_comparator import ChiSquareEvidence
from core import TraceEvidence
from fast_evidence import FastTraceEvidence
from progress_certificate import ProgressSequence, relative_radius
from run_progress_pilot import load_operator


HERE = Path(__file__).resolve().parent
METHODS = ('progress_anchored', 'chi_square', 'residual_v', 'hybrid', 'residual_v_fast', 'hybrid_fast')


def measure(matvec, size, method, seed, budget=256):
    kind = method.removesuffix('_fast')
    evidence_class = FastTraceEvidence if kind != method else TraceEvidence
    rng = np.random.default_rng(seed)
    factors = np.empty((size, budget), order='F')
    progress = ProgressSequence(alpha=.025 if kind == 'hybrid' else .05, anchors=True) if kind in ('progress_anchored', 'hybrid') else None
    empirical = evidence_class(alpha=.025 if kind == 'hybrid' else .05) if kind in ('residual_v', 'hybrid') else None
    chi = ChiSquareEvidence() if kind == 'chi_square' else None
    sketch, total, oracle_time, evidence_time = 0.0, 0.0, 0.0, 0.0
    start = perf_counter()
    stopped = False
    for count in range(1, budget+1):
        g = rng.standard_normal(size)
        mark = perf_counter()
        product = matvec(g)
        oracle_time += perf_counter() - mark
        columns = factors[:, :count-1]
        r = product - columns @ (columns.T @ g)
        q, v = float(g @ r), float(r @ r)
        if q <= 0 or v <= 0:
            raise ArithmeticError('Residual computation lost positivity')
        factors[:, count-1] = r / math.sqrt(q)
        previous_trace = sketch
        total += previous_trace + q
        sketch += v / q
        mark = perf_counter()
        interval = (0.0, math.inf)
        if progress is not None:
            interval = progress.inspect(count, total, sketch)
        if empirical is not None:
            empirical.append(previous_trace, q, v)
        if chi is not None:
            chi.append(previous_trace, q, sketch)
        if count % 4 == 0:
            if chi is not None:
                interval = chi.interval()
            if empirical is not None:
                lower, upper = empirical.interval()
                interval = (max(interval[0], lower), min(interval[1], upper))
            stopped = bool(relative_radius(*interval) <= .05)
        evidence_time += perf_counter() - mark
        if stopped:
            break
    return {'method': method, 'seed': seed, 'matvecs': count, 'stopped': stopped,
            'wall_seconds': perf_counter()-start, 'oracle_seconds': oracle_time, 'evidence_seconds': evidence_time}


def main():
    records = []
    with threadpool_limits(limits=1):
        matvec, _, audit = load_operator('bcsstk14')
        backend = threadpool_info()
        for method in METHODS:
            measure(matvec, audit['dimension'], method, 700999)
        for index, seed in enumerate(range(701000, 701016)):
            for j in range(len(METHODS)):
                method = METHODS[(j + index) % len(METHODS)]
                records.append(measure(matvec, audit['dimension'], method, seed))
    summary = {}
    for method in METHODS:
        runs = [r for r in records if r['method'] == method]
        summary[method] = {k: float(np.median([x[k] for x in runs])) for k in
                           ('matvecs', 'wall_seconds', 'oracle_seconds', 'evidence_seconds')}
        summary[method]['stops'] = sum(r['stopped'] for r in runs)
    data = {'status': 'EXPLORATORY_SERIAL_TIMING', 'matrix': 'bcsstk14', 'system': platform.platform(),
            'python': platform.python_version(), 'backend': [{k: v for k, v in b.items() if k != 'filepath'} for b in backend],
            'data_audit': audit, 'source_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__), HERE/'run_progress_pilot.py', HERE/'core.py', HERE/'progress_certificate.py', HERE/'chi2_comparator.py', HERE/'fast_evidence.py']},
            'summary': summary, 'records': records}
    (HERE/'timing_check_results.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
