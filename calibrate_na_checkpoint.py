"""Audit sharper finite-sample NA-Hutch++ certificates on matched paths."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from hashlib import sha256
import json
from math import exp, expm1, fsum, inf, isclose, log, log1p, sqrt
from pathlib import Path

from scipy.optimize import brentq
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

from run_na_checkpoint import (
    ALPHA, CAP, CASES, CHECKPOINTS, NEW_SOURCES, PATHS, SEED_STRIDE, SOURCES, TARGET,
    critical_values, load_case, relative_center, relative_radius, run_path,
)


HERE = Path(__file__).resolve().parent
FRESH_SEED = 11062000
WEIGHTED = (0.001,) * 7 + (0.043 / 3,) * 3
EQUAL = (ALPHA / len(CHECKPOINTS),) * len(CHECKPOINTS)
METHODS = ("frozen_norm", "original_na", "sharp_equal", "sharp_weighted")


@cache
def critical_risk(heldout, risk):
    return float(chi2.ppf(risk / 4, heldout)), log(4 / risk)


def positive_root(rate, target):
    upper = 1.0
    while rate(upper) < target:
        upper *= 2
    return brentq(lambda x: rate(x) - target, 0, upper)


def sharp_half_width(point, risk):
    heldout = point["products"] // 4
    old_u, old_quantile, _ = critical_values(heldout)
    old_f, old_b = point["frobenius_upper"], point["spectral_upper"]
    if not 0 < old_b < old_f:
        return None
    quantile, exponent = critical_risk(heldout, risk)
    observed = old_u * old_f ** 2
    spectral = old_b * sqrt(old_quantile / quantile)
    target = 2 * spectral ** 2 * exponent / heldout
    ratio_log = positive_root(lambda s: observed * (s * exp(s) - expm1(s)), target)
    frobenius_sq = observed * exp(ratio_log)
    spectral = min(spectral, sqrt(frobenius_sq))
    nu = heldout * frobenius_sq / (2 * spectral ** 2)
    scaled_tail = positive_root(lambda s: nu * (s - log1p(s)), exponent)
    return 2 * spectral * nu * scaled_tail / heldout


def replay(points, theta, spending):
    lower, upper, stop = 0.0, inf, None
    skipped = 0
    exclusions = []
    reports = []
    for point, risk in zip(points, spending, strict=True):
        budget = point["products"]
        half_width = sharp_half_width(point, risk)
        if half_width is None:
            skipped += 1
            continue
        estimate = point["point_estimate"]
        lower = max(lower, estimate - half_width)
        upper = min(upper, estimate + half_width)
        reports.append({"products": budget, "lower": lower, "upper": upper})
        if not lower <= theta <= upper:
            exclusions.append(budget)
        if stop is None and relative_radius(lower, upper) <= TARGET:
            error = abs(relative_center(lower, upper) / theta - 1)
            stop = {"products": budget, "relative_error": error}
    return {"stop": stop, "reference_exclusions": exclusions, "skipped_reports": skipped, "reports": reports}


def audit_trials(trials, theta):
    records = []
    for trial in trials:
        points = trial["na_hutchpp_point_estimates"]
        assert [row["products"] for row in points] == list(CHECKPOINTS)
        rows = trial["methods"]
        records.append({"seed": trial["seed"], "methods": {
            "frozen_norm": rows["frozen_norm"], "original_na": rows["gaussian_na_hutchpp"],
            "sharp_equal": replay(points, theta, EQUAL), "sharp_weighted": replay(points, theta, WEIGHTED),
        }, "points": points})
    return records


def fresh_case(index, source_sha256):
    SOURCES.update(NEW_SOURCES)
    case = CASES[index]
    with threadpool_limits(limits=1):
        matvec, theta, audit = load_case(case)
        assert audit["sha256"] == source_sha256
        paths = [run_path(matvec, audit["dimension"], theta, FRESH_SEED + index * SEED_STRIDE + offset)
                 for offset in range(PATHS)]
    return case.name, {"audit": audit, "records": audit_trials(paths, theta)}


def summarize(cases):
    records = [record for case in cases.values() for record in case["records"]]
    summary = {}
    for method in METHODS:
        counts, pairs = Counter(), Counter()
        totals = Counter()
        for record in records:
            methods = record["methods"]
            result = methods[method]
            stop = result["stop"]
            other_stop = methods["frozen_norm"]["stop"]
            cost = stop["products"] if stop else CAP
            other = other_stop["products"] if other_stop else CAP
            counts[str(cost) if stop else "censored"] += 1
            pairs["earlier" if cost < other else "later" if cost > other else "tie"] += 1
            totals["reference_exclusions"] += len(result["reference_exclusions"])
            totals["incorrect_stops"] += bool(stop and stop["relative_error"] > TARGET)
            totals["skipped_reports"] += result.get("skipped_reports", 0)
            totals["restricted_products"] += cost
        summary[method] = {"stops": dict(counts), "completions": len(records) - counts["censored"],
                           "restricted_mean_products": totals.pop("restricted_products") / len(records),
                           "versus_frozen_norm": dict(pairs), **totals}
    return summary


def main():
    source = HERE / "na_checkpoint_results.json"
    if not source.is_file():
        source = HERE / "results" / source.name
    archived = json.loads(source.read_bytes())
    assert archived["status"] == "COMPLETE" and len(CASES) == 10
    assert isclose(fsum(WEIGHTED), ALPHA, rel_tol=0.0, abs_tol=1e-16)
    assert len(WEIGHTED) == len(CHECKPOINTS)
    development = {
        name: {"audit": case["audit"], "records": audit_trials(case["trials"], case["audit"]["trace_reference"])}
        for name, case in archived["cases"].items()
    }
    summaries = {"development": summarize(development)}
    print("Development:", summaries["development"], flush=True)
    sources = [archived["cases"][case.name]["audit"]["sha256"] for case in CASES]
    with ProcessPoolExecutor(max_workers=4) as workers:
        fresh = dict(workers.map(fresh_case, range(len(CASES)), sources))
    summaries["fresh"] = summarize(fresh)
    print("Fresh:", summaries["fresh"], flush=True)
    files = (Path(__file__), source, HERE / "calibrate_na_checkpoint.txt", HERE / "run_na_checkpoint.py")
    output = {"status": "COMPLETE", "checkpoints": CHECKPOINTS, "equal_risk": EQUAL,
              "weighted_risk": WEIGHTED, "fresh_first_seed": FRESH_SEED, "summaries": summaries,
              "source_sha256": {file.name: sha256(file.read_bytes()).hexdigest() for file in files},
              "fresh_cases": fresh}
    (HERE / "na_calibration_results.json").write_text(
        json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
