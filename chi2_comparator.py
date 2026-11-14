"""Paired development pilot for two anytime trace evidence constructions.

For a predictable PSD residual with trace mu > 0, the Gaussian quadratic
q satisfies E exp(s q/mu) <= (1-2s)^(-1/2) for 0 < s < 1/2, and
E exp(-s q/mu) <= (1+2s)^(-1/2) for s > 0. The determinant bounds use
the nonnegative eigenvalues divided by mu, which sum to one. Multiplying
the respective normalized factors and mixing fixed stakes gives two test
supermartingales. Each sign receives alpha/2. The sketch's pathwise lower
bound is shared with the observed-variance comparator. It uses v through
the sketch increment, while the chi-square factors themselves use only q.

The theorem assumes exact PSD products. This replay is a floating-point
pilot on positive-residual paths and provides no numerical certificate.
"""

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from core import trajectory


ROOT = Path(__file__).resolve().parent
POSITIVE = np.array([1 / 32, 1 / 16, 1 / 8, 1 / 4, 7 / 16])
NEGATIVE = np.array([1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8])
TARGETS = (0.05, 0.10, 0.20)
METHODS = ("observed_v", "chi_square")
EMPTY = (math.inf, -math.inf)


def log_mixture(u, count, sign):
    """Return log evidence as a function of u = sum q_i/(theta-t_i)."""
    if sign == "positive":
        stakes = POSITIVE
        terms = stakes * u + 0.5 * count * np.log1p(-2 * stakes)
    elif sign == "negative":
        stakes = NEGATIVE
        terms = -stakes * u + 0.5 * count * np.log1p(2 * stakes)
    else:
        raise ValueError(f"Unknown evidence sign: {sign}")
    return float(logsumexp(terms) - math.log(len(stakes)))


@lru_cache(maxsize=1024)
def evidence_cutoffs(count, alpha):
    """Invert the two mixtures in u once per horizon and confidence."""
    threshold = math.log(2 / alpha)

    def crossing(sign):
        direction = 1 if sign == "positive" else -1

        def f(u):
            return direction * (log_mixture(u, count, sign) - threshold)

        right = float(max(1, count))
        while f(right) < 0:
            right *= 2
        return brentq(f, 0.0, right)

    upper_u = crossing("positive")
    lower_u = crossing("negative") if log_mixture(0, count, "negative") > threshold else 0.0
    return lower_u, upper_u


class ChiSquareEvidence:
    """Running intersection of the two inverted sign tests and sketch bound."""

    def __init__(self, alpha=0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0,1)")
        self.alpha = alpha
        self.t = []
        self.q = []
        self.hard_lower = 0.0
        self.running = (0.0, math.inf)

    def append(self, t, q, hard_lower):
        if not all(math.isfinite(x) for x in (t, q, hard_lower)) or not (0 <= t < hard_lower and q > 0):
            raise ValueError("Replay requires a positive residual and a hard lower bound above t")
        self.t.append(t)
        self.q.append(q)
        self.hard_lower = max(self.hard_lower, hard_lower)

    def interval(self):
        if self.running == EMPTY or not self.q:
            return self.running
        t, q = np.asarray(self.t), np.asarray(self.q)
        hard = self.hard_lower
        scale = max(hard, float(np.mean(t + q)))
        lower_u, upper_u = evidence_cutoffs(len(q), self.alpha)

        def u(theta):
            return float(np.sum(q / (theta - t)))

        at_hard = u(hard)
        if at_hard < lower_u or lower_u > upper_u:
            self.running = EMPTY
            return self.running

        def theta_crossing(target):
            def f(x):
                return u(scale * x) - target

            right = 2.0
            while f(right) > 0:
                right *= 2
                if not math.isfinite(scale * right):
                    raise ArithmeticError("Cannot bracket chi-square confidence endpoint")
            return scale * brentq(f, hard / scale, right, xtol=1e-13)

        lower = hard
        if at_hard > upper_u:
            lower = max(hard, math.nextafter(theta_crossing(upper_u), -math.inf))
        upper = math.inf
        if lower_u > 0:
            upper = hard if at_hard == lower_u else math.nextafter(theta_crossing(lower_u), math.inf)
        lower = max(self.running[0], lower)
        upper = min(self.running[1], upper)
        self.running = (lower, upper) if lower <= upper else EMPTY
        return self.running


def validate_inversion():
    """Check inversion against direct factors and exercise boundary branches."""
    evidence = ChiSquareEvidence()
    assert evidence.interval() == (0, math.inf)
    evidence.append(0.0, 1.0, 0.1)
    assert math.isinf(evidence.interval()[1])
    for i in range(1, 25):
        evidence.append(i / 100, 1 - i / 100, (i + 1) / 100)
    lo, hi = evidence.interval()
    assert 0.25 < lo < 1 < hi < math.inf
    threshold = math.log(2 / evidence.alpha)
    records = list(zip(evidence.t, evidence.q))

    def direct(theta, sign):
        stakes = POSITIVE if sign == "positive" else -NEGATIVE
        return float(np.log(np.mean([
            math.prod(math.exp(s * q / (theta - t)) * math.sqrt(1 - 2 * s) for t, q in records)
            for s in stakes
        ])))

    assert abs(direct(lo, "positive") - threshold) < 1e-9
    assert abs(direct(hi, "negative") - threshold) < 1e-9
    grid = np.geomspace(0.25, 100, 300)
    for theta in grid:
        accepted = all(direct(theta, sign) <= threshold for sign in ("positive", "negative"))
        assert accepted == (lo <= theta <= hi)
    evidence.append(0.25, 0.75, 0.26)
    new_lo, new_hi = evidence.interval()
    assert new_lo >= lo and new_hi <= hi
    empty = ChiSquareEvidence()
    for i in range(20):
        empty.append(float(i), 0.0001, float(i + 1))
    assert empty.interval() == EMPTY
    empty.append(20.0, 20.0, 21.0)
    assert empty.interval() == EMPTY
    return {"status": "passed", "direct_grid_candidates": len(grid),
            "checked": ["startup", "two finite roots", "direct mixture membership", "nestedness", "sticky empty"]}


def radius(interval):
    lo, hi = interval
    if interval == EMPTY:
        return None
    if not math.isfinite(hi):
        return 1.0
    return float((hi - lo) / (hi + lo))


def evaluate_case(name, matvec, eigenvalues, budgets, seeds, alpha):
    truth = float(sum(eigenvalues))
    n = len(eigenvalues)
    runs = []
    for seed in seeds:
        check_rows, _, _ = trajectory(matvec, n, max(budgets), np.random.default_rng(seed), alpha=alpha)
        chi = ChiSquareEvidence(alpha)
        results = {method: {"first_stop": {str(x): None for x in TARGETS}, "coverage_failure": False,
                            "empty_at_any_check": False, "at_budget": {}} for method in METHODS}
        for row in check_rows:
            chi.append(row["t_before"], row["q"], row["trace_sketch"])
            bounds = {"observed_v": (row["lower"], row["upper"]), "chi_square": chi.interval()}
            for method, interval in bounds.items():
                lo, hi = interval
                result = results[method]
                result["coverage_failure"] |= not lo <= truth <= hi
                result["empty_at_any_check"] |= interval == EMPTY
                relative_radius = radius(interval)
                for target in TARGETS:
                    key = str(target)
                    if result["first_stop"][key] is None and lo > 0 and math.isfinite(hi) and relative_radius <= target:
                        result["first_stop"][key] = row["step"]
                if row["step"] in budgets:
                    result["at_budget"][str(row["step"])] = relative_radius
        runs.append({"seed": seed, **results})

    summary = {}
    for method in METHODS:
        records = [run[method] for run in runs]
        summary[method] = {
            "pathwise_failures": sum(x["coverage_failure"] for x in records),
            "empty_paths": sum(x["empty_at_any_check"] for x in records),
            "median_relative_radius": {str(b): median([x["at_budget"][str(b)] for x in records]) for b in budgets},
            "stopping": {str(target): summarize_stops([x["first_stop"][str(target)] for x in records]) for target in TARGETS},
        }
    paired = {}
    for b in budgets:
        key = str(b)
        diffs = [run["observed_v"]["at_budget"][key] - run["chi_square"]["at_budget"][key]
                 for run in runs if all(run[m]["at_budget"][key] is not None for m in METHODS)]
        paired[key] = {"comparable_paths": len(diffs), "median_radius_difference_observed_v_minus_chi_square": median(diffs),
                       "observed_v_narrower": sum(x < 0 for x in diffs), "chi_square_narrower": sum(x > 0 for x in diffs)}
    return {"case": name, "dimension": n, "trace_reference": truth,
            "trace_squared_over_frobenius_squared": truth**2 / float(eigenvalues @ eigenvalues),
            "budgets": budgets, "trials": len(seeds), "summary": summary, "paired_at_budget": paired, "runs": runs}


def median(values):
    valid = [value for value in values if value is not None]
    return float(np.median(valid)) if valid else None


def summarize_stops(stops):
    return {"stopped": sum(stop is not None for stop in stops), "censored": sum(stop is None for stop in stops),
            "median_probe_count_among_stops": median(stops)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_results(suite, cases, validation, alpha, dataset=None):
    result = {"status": "EXPLORATORY_PAIRED_FLOATING_POINT_PILOT", "suite": suite, "alpha_per_method": alpha,
              "checked_every_probe": True, "relative_radius_targets": TARGETS,
              "chi_square_stakes": {"positive": POSITIVE.tolist(), "negative": NEGATIVE.tolist()},
              "width_conventions": {"infinite_upper": 1.0, "empty": None}, "inversion_validation": validation,
              "source_sha256": {p.name: digest(p) for p in [Path(__file__), ROOT / "core.py"]},
              "cases": cases,
              "limitations": ["Both intervals use the same sketch and hard lower bound on each paired path.",
                              "Equal mixture weights and fixed stakes. Positive chi-square stakes must be below 1/2.",
                              "Each method receives alpha separately. No claim of simultaneous joint coverage across methods.",
                              "One real operator and two prespecified synthetic spectra. No finite-precision guarantee."]}
    if dataset:
        result["data_sha256"] = digest(dataset)
    target = ROOT / f"chi2_comparison_{suite}.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"file": str(target), "cases": [{k: v for k, v in case.items() if k != "runs"} for case in cases]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "spambase", "synthetic"), default="all")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    validation = validate_inversion()
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return
    alpha = 0.05
    if args.suite in ("spambase", "all"):
        path = ROOT / "spambase.data"
        X = np.loadtxt(path, delimiter=",")[:, :-1]
        X = (X - X.mean(0)) / X.std(0)
        eig, U = eigh(X.T @ X / len(X))
        spectrum = 1 / (eig + 0.1)
        case = evaluate_case("Spambase inverse standardized covariance + 0.1 ridge", lambda g: U @ (spectrum * (U.T @ g)),
                             spectrum, [20, 40, 55], range(args.trials), alpha)
        write_results("spambase", [case], validation, alpha, dataset=path)
    if args.suite in ("synthetic", "all"):
        cases = []
        for name, spectrum in [("flat", np.ones(256)), ("power law j^-1.5", np.arange(1, 257, dtype=float)**-1.5)]:
            spectrum = spectrum / spectrum.sum()
            cases.append(evaluate_case(name, lambda g: spectrum * g, spectrum, [20, 40, 55, 80, 96],
                                       range(1000, 1000 + args.trials), alpha))
        write_results("synthetic", cases, validation, alpha)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
