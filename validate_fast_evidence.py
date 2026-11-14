"""Compare the optimized scalar-root implementation to the reference."""
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from core import TraceEvidence
from fast_evidence import FastTraceEvidence


def main():
    here = Path(__file__).resolve().parent
    rng = np.random.default_rng(15703)
    checks = 0
    maximum_discrepancy = 0.0
    dimension = 64
    ranks = np.arange(1, dimension + 1)
    with threadpool_limits(limits=1):
        for shape in (0.0, 1.5, 3.0):
            for scale in (1e-6, 1.0, 1e6):
                diagonal = scale * ranks**(-shape)
                for alpha in (.025, .05):
                    slow, fast = TraceEvidence(alpha), FastTraceEvidence(alpha)
                    trace = 0.0
                    factors = np.empty((dimension, 40), order='F')
                    for index in range(40):
                        g = rng.standard_normal(dimension)
                        factor = factors[:, :index]
                        residual = diagonal * g - factor @ (factor.T @ g)
                        q, v = float(g @ residual), float(residual @ residual)
                        for method in (slow, fast):
                            method.append(trace, q, v)
                        factors[:, index] = residual / math.sqrt(q)
                        trace += v / q
                        for reference, optimized in zip(slow.interval(), fast.interval()):
                            if math.isinf(reference):
                                assert reference == optimized
                            else:
                                discrepancy = abs(reference - optimized) / diagonal.sum()
                                maximum_discrepancy = max(maximum_discrepancy, float(discrepancy))
                                assert discrepancy < 1e-9, (reference, optimized)
                            checks += 1
    subnormal = FastTraceEvidence()
    for _ in range(12):
        subnormal.append(0.0, 1e-160, 1e-320)
    lower, upper = subnormal.interval()
    assert 0 < lower <= upper < math.inf
    results = {'endpoint_checks': checks, 'max_discrepancy_relative_to_trace': maximum_discrepancy,
               'subnormal_variance_scale_checked': True,
               'source_sha256': {name: sha256((here/name).read_bytes()).hexdigest()
                                 for name in ('validate_fast_evidence.py', 'core.py', 'fast_evidence.py')}}
    (here/'fast_evidence_validation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(results)


if __name__ == '__main__':
    main()
