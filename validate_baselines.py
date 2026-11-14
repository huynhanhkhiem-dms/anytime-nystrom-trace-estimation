"""Meaningful small explicit-matrix checks for fixed-budget pilot baselines."""
import json
import math
import hashlib
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest

from baselines import (gaussian_hutchinson, gaussian_psd_fixed_interval, hutchpp,
                       nystrompp, xnystrace, _xnystrace_from_probes, _xnystrace_direct)


def main():
    n, budget = 24, 12
    diagonal = np.geomspace(0.01, 2.0, n)
    truth = float(diagonal.sum())
    methods = [gaussian_hutchinson, hutchpp, nystrompp, xnystrace]
    output = {"purpose": "independent numerical sanity checks; not real-data research results",
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__},
              "baselines_sha256": hashlib.sha256(Path(__file__).with_name("baselines.py").read_bytes()).hexdigest(),
              "seed_protocol": "numpy default_rng(seed), seed=0..999, separate stream per method",
              "formula_sources": {"Hutch++": "https://arxiv.org/abs/2010.09649",
                                  "Nystrom++": "https://arxiv.org/abs/2109.10659",
                                  "XNysTrace": "https://arxiv.org/abs/2301.07825"},
              "checks": {}}
    for method in methods:
        estimates = []
        counts = []
        observed_counts = []
        for seed in range(1000):
            calls = [0]
            def matvec(x):
                calls[0] += 1
                return diagonal * x
            result = method(matvec, n, budget, np.random.default_rng(seed))
            estimates.append(result["estimate"])
            counts.append(result["matvecs"])
            observed_counts.append(calls[0])
        values = np.array(estimates)
        mean_se = float(values.std(ddof=1) / math.sqrt(len(values)))
        bias_z = float((values.mean() - truth) / mean_se) if mean_se else 0.0
        assert counts == observed_counts and max(counts) <= budget
        assert abs(bias_z) < 5.0, (method.__name__, bias_z)
        output["checks"][method.__name__ + "_mean_and_cost"] = {
            "n": n, "budget": budget, "replications": len(values), "truth": truth,
            "mean": float(values.mean()), "mean_mc_se": mean_se, "bias_mc_z": bias_z,
            "mean_squared_error": float(np.mean((values - truth)**2)),
            "reported_cost_matches_actual": True,
            "max_matvecs": max(counts), "pass": True}

    # All low-rank methods should recover rank-three PSD trace from an overcomplete sketch.
    low_diagonal = np.r_[np.array([9.0, 2.0, 0.3]), np.zeros(n-3)]
    rank_errors = {}
    for method in [hutchpp, nystrompp, xnystrace]:
        errors = [abs(method(lambda x: low_diagonal*x, n, 18,
                             np.random.default_rng(seed))["estimate"] - low_diagonal.sum())
                  for seed in range(20)]
        assert max(errors) < 1e-9, (method.__name__, max(errors))
        rank_errors[method.__name__] = float(max(errors))
    output["checks"]["rank_three_exactness"] = {"max_absolute_errors": rank_errors, "pass": True}

    zero_results = [method(lambda x: np.zeros_like(x), n, budget, np.random.default_rng(8))
                    for method in methods]
    assert all(r["estimate"] == 0 for r in zero_results)
    assert gaussian_psd_fixed_interval(0, 5) == (0.0, 0.0)
    output["checks"]["zero_matrix"] = {"estimates": [r["estimate"] for r in zero_results], "pass": True}

    # Compare efficient Gram identities against independent direct leave-one-out formulas.
    formula_checks = []
    for improved in [False, True]:
        for seed in range(10):
            rng = np.random.default_rng(20000+seed)
            omega = rng.standard_normal((n, 8))
            if improved:
                omega *= math.sqrt(n) / np.linalg.norm(omega, axis=0)[None, :]
            image = diagonal[:, None]*omega
            fast, diag = _xnystrace_from_probes(omega, image, improved)
            direct = _xnystrace_direct(omega, image, improved)
            err = float(np.max(np.abs(fast-direct)))
            assert err < 1e-9, (improved, seed, err)
            formula_checks.append({"improved": improved, "seed": seed, "max_abs_difference": err})
    output["checks"]["xnystrace_fast_vs_direct"] = {"cases": formula_checks, "pass": True}

    # Worst-case rank-one Gaussian matrix: fixed-time interval must behave conservatively.
    m, reps, delta = 16, 10000, 0.05
    rng = np.random.default_rng(314159)
    scalar_samples = np.mean(rng.standard_normal((reps,m))**2,axis=1)
    failures = 0
    for value in scalar_samples:
        lower, upper = gaussian_psd_fixed_interval(float(value), m, delta)
        failures += not (lower <= 1.0 <= upper)
    ci = binomtest(failures, reps).proportion_ci(confidence_level=0.99)
    assert ci.high < delta
    output["checks"]["rank_one_fixed_time_coverage"] = {
        "probes": m, "replications": reps, "delta": delta, "failures": failures,
        "empirical_failure_rate": failures/reps,
        "failure_rate_99pct_clopper_pearson": [ci.low, ci.high], "pass": True,
        "scope": "fixed-time Monte Carlo diagnostic; not proof and not optional-stopping validation"}

    # Full n-probe improved XNysTrace has 1-dimensional residuals and is exact in exact arithmetic.
    rng = np.random.default_rng(99)
    full = xnystrace(lambda x: diagonal*x, n, n, rng, improved=True)
    assert abs(full["estimate"]-truth) < 1e-8
    output["checks"]["xnystrace_full_dimension"] = {
        "n": n, "matvecs": full["matvecs"], "absolute_error": abs(full["estimate"]-truth), "pass": True}
    output["all_checks_passed"] = True
    path = Path(__file__).with_name("baseline_validation.json")
    path.write_text(json.dumps(output, indent=2),encoding="utf-8")
    print(json.dumps({"all_checks_passed": True, "checks": len(output["checks"]),
                      "result_file": str(path.resolve())}))


if __name__ == "__main__":
    main()
