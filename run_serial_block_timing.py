"""Time matched certificates while querying only through their first stop."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns, process_time_ns

import numpy as np
import scipy
from scipy.linalg import eigvalsh, pinv
from threadpoolctl import threadpool_info, threadpool_limits

from anchored_norm import invert_frozen_norm
from progress_certificate import relative_center, relative_radius
from calibrate_na_checkpoint import critical_risk, positive_root
from run_anchor_norm_comparison import CASES as MATCHED_CASES, NEW_SOURCES, NUMERICAL_ERRORS
from run_confirmatory import SOURCES, load_case
from run_asymmetric_progress import invert_asymmetric_progress


HERE = Path(__file__).resolve().parent
CAP = 320
STEP = 16
CHECKPOINTS = tuple(range(STEP, CAP + 1, STEP))
SPENDING = (0.001,) * 7 + (0.043 / 13,) * 13
ALPHA = TARGET = 0.05
FIRST_SEED = 11296000
PATHS = 12
METHODS = ("frozen_norm", "asymmetric_progress", "sharp_na")
MODES = ("serial", "block")
VARIANTS = tuple((method, mode) for method in METHODS for mode in MODES)
WILLIAMS = (0, 1, 5, 2, 4, 3)
CASES = tuple(MATCHED_CASES[index] for index in (0, 2, 3, 7, 8, 9))


def sharp_na_interval(probes, images, risk):
    budget = probes.shape[1]
    heldout = budget // 4
    left = probes[:, :heldout]
    left_images = images[:, :heldout]
    right_images = images[:, heldout:3 * heldout]
    holdout = probes[:, 3 * heldout:]
    holdout_images = images[:, 3 * heldout:]
    factor = right_images @ pinv(left.T @ right_images)
    approximation_trace = float(np.einsum("ij,ij->", factor, left_images))
    residual = holdout_images - 0.5 * (
        factor @ (left_images.T @ holdout) + left_images @ (factor.T @ holdout)
    )
    point = approximation_trace + float(np.einsum("ij,ij->", holdout, residual)) / heldout
    gram = residual.T @ residual
    observed = float(np.trace(gram)) / heldout
    if observed == 0:
        # In binary64, a zero residual block may also result from cancellation.
        return 0.0, math.inf
    index = heldout - 1
    eigenvalue = float(eigvalsh(gram, subset_by_index=(index, index))[0])
    quantile, exponent = critical_risk(heldout, risk)
    spectral = math.sqrt(eigenvalue / quantile)
    f_target = 2 * spectral ** 2 * exponent / heldout
    ratio_log = positive_root(lambda s: observed * (s * math.exp(s) - math.expm1(s)), f_target)
    frobenius_sq = observed * math.exp(ratio_log)
    spectral = min(spectral, math.sqrt(frobenius_sq))
    nu = heldout * frobenius_sq / (2 * spectral ** 2)
    scaled_tail = positive_root(lambda s: nu * (s - math.log1p(s)), exponent)
    half_width = 2 * spectral * nu * scaled_tail / heldout
    if not math.isfinite(point + half_width):
        raise ArithmeticError("Nonfinite sharp NA-Hutch++ interval")
    return max(0, point - half_width), point + half_width


def run_method(matvec, dimension, reference, seed, method, mode):
    started = perf_counter_ns()
    process_started = process_time_ns()
    rng = np.random.default_rng(seed)
    if method == "sharp_na":
        history = np.empty((dimension, CAP), order="F")
        products = np.empty_like(history, order="F")
    else:
        basis = np.empty((dimension, CAP), order="F")
    phases = dict.fromkeys(("draw", "oracle", "estimation_and_evidence"), 0)
    captured = total = squared = lower = 0.0
    upper = math.inf
    used = oracle_calls = 0
    reports = []
    stop = failure = None
    initialization = perf_counter_ns() - started
    for count, risk in zip(CHECKPOINTS, SPENDING, strict=True):
        try:
            previous_count = count - STEP
            mark = perf_counter_ns()
            block = rng.standard_normal((STEP, dimension)).T
            after_draw = perf_counter_ns()
            if mode == "block":
                oracle_calls += 1
                used += STEP
                images = matvec(block)
            else:
                images = np.empty_like(block, order="F")
                for index in range(STEP):
                    oracle_calls += 1
                    used += 1
                    images[:, index] = matvec(block[:, index])
            after_oracle = perf_counter_ns()
            if method == "sharp_na":
                history[:, previous_count:count] = block
                products[:, previous_count:count] = images
                interval = sharp_na_interval(history[:, :count], products[:, :count], risk)
            else:
                for index in range(STEP):
                    column = previous_count + index
                    probe = block[:, index]
                    factor = basis[:, :column]
                    residual = images[:, index] - factor @ (factor.T @ probe)
                    q = float(probe @ residual)
                    v = float(residual @ residual)
                    if not (q > 0 and v > 0 and math.isfinite(q + v)):
                        raise ArithmeticError("Nonpositive or nonfinite Nyström residual")
                    previous = captured
                    captured += v / q
                    total += previous + q
                    squared += v
                    if not (captured > previous and math.isfinite(captured + total + squared)):
                        raise ArithmeticError("Lost positivity or finite trace progress")
                    basis[:, column] = residual / math.sqrt(q)
                if method == "frozen_norm":
                    interval = invert_frozen_norm(count, total, captured, squared, alpha=ALPHA, stake=0.5)
                else:
                    interval = invert_asymmetric_progress(count, total, captured, alpha=ALPHA)
            lower, upper = max(lower, interval[0]), min(upper, interval[1])
            radius = relative_radius(lower, upper)
            reports.append({"products": count, "lower": lower, "upper": upper if math.isfinite(upper) else None})
            if radius <= TARGET:
                stop = {"products": count, "radius": radius, "estimate": relative_center(lower, upper)}
            after_evidence = perf_counter_ns()
            phases["draw"] += after_draw - mark
            phases["oracle"] += after_oracle - after_draw
            phases["estimation_and_evidence"] += after_evidence - after_oracle
            if stop is not None:
                break
        except NUMERICAL_ERRORS as exc:
            failure = str(exc)
            break
    elapsed = perf_counter_ns() - started
    cpu_elapsed = process_time_ns() - process_started
    if stop is not None:
        stop["relative_error"] = abs(stop["estimate"] - reference) / reference
    exclusions = [row["products"] for row in reports if row["lower"] > reference
                  or (row["upper"] is not None and row["upper"] < reference)]
    return {"products": used, "oracle_calls": oracle_calls, "reports": reports, "stop": stop,
            "wall_ns": elapsed, "cpu_ns": cpu_elapsed, "phase_ns": {"initialize": initialization, **phases},
            "reference_exclusions": exclusions, "numerical_failure": failure}


def summarize(records, method, mode):
    rows = [record["methods"][method][mode] for record in records]
    completed = [row for row in rows if row["stop"] is not None]
    wall = [row["wall_ns"] / 1e6 for row in rows]
    baseline = [record["methods"]["frozen_norm"][mode]["wall_ns"] for record in records]
    return {
        "completions": len(completed),
        "restricted_mean_products": sum(row["products"] if row["stop"] else CAP for row in rows) / len(rows),
        "median_products_among_completions": median(row["products"] for row in completed) if completed else None,
        "median_wall_ms": median(wall),
        "wall_ms_iqr": np.quantile(wall, [0.25, 0.75]).tolist(),
        "median_cpu_ms": median(row["cpu_ns"] / 1e6 for row in rows),
        "median_oracle_ms": median(row["phase_ns"]["oracle"] / 1e6 for row in rows),
        "median_paired_wall_ratio_to_frozen_norm": median(
            row["wall_ns"] / other for row, other in zip(rows, baseline, strict=True)
        ),
        "reference_exclusion_paths": sum(bool(row["reference_exclusions"]) for row in rows),
        "incorrect_stops": sum(row["stop"]["relative_error"] > TARGET for row in completed),
        "numerical_interruptions": sum(row["numerical_failure"] is not None for row in rows),
    }


def main():
    assert math.isclose(math.fsum(SPENDING), ALPHA, rel_tol=0, abs_tol=1e-16)
    destination = HERE / "serial_block_timing_results.json"
    files = (Path(__file__), *(HERE / name for name in (
        "serial_block_timing_protocol.txt", "run_asymmetric_progress.py", "calibrate_na_checkpoint.py",
        "run_confirmatory.py", "run_anchor_norm_comparison.py", "anchored_norm.py", "progress_certificate.py")))
    result = {
        "status": "RUNNING", "checkpoints": CHECKPOINTS, "cap": CAP, "paths_per_operator": PATHS,
        "first_seed": FIRST_SEED, "risk": ALPHA, "target": TARGET, "na_spending": SPENDING,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "processor": platform.processor(), "platform": platform.platform(),
        "visible_cpu_count": os.process_cpu_count(),
        "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in files}, "cases": {},
    }
    SOURCES.update(NEW_SOURCES)
    with threadpool_limits(limits=1):
        result["threadpools"] = threadpool_info()
        for count, risk in zip(CHECKPOINTS, SPENDING, strict=True):
            critical_risk(count // 4, risk)
        for index, case in enumerate(CASES):
            started = perf_counter_ns()
            oracle, reference, audit = load_case(case)
            setup_seconds = (perf_counter_ns() - started) / 1e9
            dimension = audit["dimension"]
            warmup_seed = FIRST_SEED + 1000 * index - 1
            for method, mode in VARIANTS:
                run_method(oracle, dimension, reference, warmup_seed, method, mode)
            records = []
            for path in range(PATHS):
                seed = FIRST_SEED + 1000 * index + path
                order = [VARIANTS[(offset + path) % len(VARIANTS)] for offset in WILLIAMS]
                methods = {name: {} for name in METHODS}
                for method, mode in order:
                    methods[method][mode] = run_method(oracle, dimension, reference, seed, method, mode)
                records.append({"seed": seed, "order": order, "methods": methods})
            summary = {method: {mode: summarize(records, method, mode) for mode in MODES}
                       for method in METHODS}
            result["cases"][case.name] = {"audit": audit, "setup_seconds": setup_seconds,
                                          "records": records, "summary": summary}
            print(case.name, {m: [(round(s["median_wall_ms"], 1), s["median_products_among_completions"],
                                  s["completions"]) for s in v.values()] for m, v in summary.items()}, flush=True)
            staging = destination.with_suffix(".tmp")
            staging.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
            staging.replace(destination)
    result["status"] = "COMPLETE"
    destination.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
