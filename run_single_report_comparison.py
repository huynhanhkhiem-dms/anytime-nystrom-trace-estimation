"""Scope checks for the frozen norm and progress bounds at isolated reports."""

from hashlib import sha256
import json
from math import ceil, e, exp, log, pi, sqrt
from pathlib import Path

import numpy as np

from anchored_norm import invert_frozen_norm
from progress_certificate import Anchor, invert_progress, relative_radius


DIMENSION = 65536
CAP = 2048
TRIALS = 1024
SEED = 701419
TARGET = 0.05
RISKS = (0.05, 0.001, 1e-6, 1e-9)


def norm_endpoints(n, corrected_sum, captured, variance, risk):
    """Vectorized c=1/2 inversion on the exact-law flat paths."""
    threshold = 2 * log(2 / risk)
    discriminant = corrected_sum ** 2 - 2 * (n + threshold) * variance
    lower = np.maximum(captured, np.where(discriminant >= 0,
                      (corrected_sum + np.sqrt(np.maximum(discriminant, 0))) / (2 * (n + threshold)), 0))
    upper = np.full_like(corrected_sum, np.inf)
    keep = n > threshold
    den = n[keep] - threshold
    z = corrected_sum[:, keep]
    upper[:, keep] = (z + np.sqrt(z ** 2 + 4 * den * (e - 2) * variance[:, keep])) / (2 * den)
    return lower, upper


def stop_records(lower, upper):
    lower = np.maximum.accumulate(lower, axis=1)
    upper = np.minimum.accumulate(upper, axis=1)
    finite = (lower > 0) & (lower <= upper) & np.isfinite(upper)
    radius = np.full_like(lower, np.inf)
    radius[finite] = (upper[finite] - lower[finite]) / (upper[finite] + lower[finite])
    passes = radius <= TARGET
    stopped = passes.any(axis=1)
    records = [int(i + 1) if stop else None for i, stop in zip(np.argmax(passes, axis=1), stopped)]
    exclusions = (~np.all((lower <= 1) & (1 <= upper), axis=1)).tolist()
    return {"stopped": int(stopped.sum()), "median": float(np.median([x for x in records if x])) if stopped.any() else None,
            "exclusion_count": sum(exclusions), "first_stops": records, "excluded_reference": exclusions}


def check_isolated_report():
    risk = 0.05
    examples = []
    for m in (100, 1000, 10000, 100000):
        d = 0.5 + (m - 1) / (2 * m ** 2)
        z = m + 0.75 - 7 / (4 * m) + 1 / (2 * m ** 2)
        v = m / 4 + (m - 1) / (2 * m ** 3)
        norm = invert_frozen_norm(m, z, d, v, alpha=risk)
        progress = invert_progress(Anchor(0, 0.0, 0.0, risk), m, z, d, c=1/3)
        assert norm[0] < 1 < norm[1] and progress[0] < 1 < progress[1]
        assert relative_radius(*norm) > 0.14
        examples.append({"probes": m, "dimension": m ** 2 + 1, "norm_radius": relative_radius(*norm),
                         "progress_radius": relative_radius(*progress)})
    return examples


def main():
    n = np.arange(1, CAP + 1)
    captured = n / DIMENSION
    rng = np.random.default_rng(SEED)
    q = rng.chisquare(DIMENSION - n + 1, size=(TRIALS, CAP)) / DIMENSION
    corrected = np.cumsum(q + (n - 1) / DIMENSION, axis=1)
    variance = np.cumsum(q / DIMENSION, axis=1)
    results = {}
    for risk in RISKS:
        progress_threshold = 3 * log(2 / risk)
        penalty = (4 / 3) * captured
        progress_lower = np.maximum(captured, (corrected - penalty) / (n + progress_threshold))
        progress_upper = np.full_like(corrected, np.inf)
        keep = n > progress_threshold
        progress_upper[:, keep] = (corrected[:, keep] + penalty[keep]) / (n[keep] - progress_threshold)
        norm = norm_endpoints(n, corrected, captured, variance, risk)
        rl, ru = norm_endpoints(n, corrected, captured, variance, risk / 2)
        k = 4 * (log(3) + log(2 / risk))
        reinforced_lower = np.maximum(rl, np.sqrt(variance / k))
        doob_a = sqrt(4 * pi / ((pi + 2) * risk))
        doob_lower = np.maximum(captured, corrected / (n + doob_a))
        doob_upper = np.full_like(corrected, np.inf)
        keep = n > doob_a
        doob_upper[:, keep] = corrected[:, keep] / (n[keep] - doob_a)
        results[str(risk)] = {"progress": stop_records(progress_lower, progress_upper),
                             "frozen_norm": stop_records(*norm),
                             "reinforced_norm": stop_records(reinforced_lower, ru),
                             "doob": stop_records(doob_lower, doob_upper)}
    a = 2 * log(40)
    b = e - 2
    gamma = max(exp(0.5 + b), (1 + b) / 0.5)
    caps = {"progress": ceil((3 * log(40) + 4/3) / TARGET - 4/3),
            "frozen_norm": ceil((2*a + (1 + 2*a)*(gamma - 1)) / (2*TARGET)),
            "reinforced_norm": ceil((2*log(80) + (0.5 + b) * 2 * log(120)) / TARGET)}
    here = Path(__file__).resolve().parent
    output = {"scope": "Simulation of the flat-spectrum conditional observation law, not full matrix actions.",
              "dimension": DIMENSION, "cap": CAP, "trials": TRIALS, "seed": SEED, "target": TARGET,
              "proved_95pct_caps": caps, "isolated_report_examples": check_isolated_report(), "risks": results,
              "source_sha256": {name: sha256((here/name).read_bytes()).hexdigest() for name in
                                (Path(__file__).name, "anchored_norm.py", "progress_certificate.py")}}
    (here/"single_report_results.json").write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    for risk, methods in results.items():
        print(risk, {name: (v["stopped"], v["median"], v["exclusion_count"]) for name, v in methods.items()})
    print("Proved caps", caps)
    print("Counterexample radii", output["isolated_report_examples"])


if __name__ == "__main__":
    main()
