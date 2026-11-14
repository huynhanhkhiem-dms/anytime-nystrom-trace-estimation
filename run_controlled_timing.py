"""Measure complete serial, paired stopping runs on four real operators.

Each method gets its own run, with a shared seed and a warmed operator.
Online time starts before initialization and ends at the stopping
decision or cap in the binary64 prototype. Reference comparisons are performed
after the timer. Setup includes the numerical reference and is shared.
"""

from hashlib import sha256
from itertools import permutations
import json
import math
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns, process_time_ns

import numpy as np
import scipy
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_info, threadpool_limits

from anchored_norm import invert_frozen_norm
from fast_evidence import FastTraceEvidence
from progress_certificate import Anchor, invert_progress, relative_center, relative_radius
from run_anchor_norm_comparison import NEW_SOURCES
from run_confirmatory import SOURCES, Case, load_case
from run_graph_holdout import inverse_trace, load_laplacian


HERE = Path(__file__).resolve().parent
CAP = 400
REPORT_EVERY = 4
ALPHA = TARGET = 0.05
PATHS = 24
FIRST_SEED = 9_170_000
SEED_STRIDE = 1_000
METHODS = ("progress", "frozen_norm", "moving_norm")
METHOD_ORDERS = tuple(permutations(METHODS))
CASES = (
    Case("har", "log_covariance", 0.1),
    Case("email", "heat", 0.5),
    Case("bcsstk08", "inverse", 0.01),
    Case("ca-GrQc", "inverse", 0.01),
)
NUMERICAL_ERRORS = (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError)
PHASES = ("initialize", "draw", "oracle", "sketch", "evidence")


def operator(case):
    if case.source != "ca-GrQc":
        matvec, reference, audit = load_case(case)
        return matvec, reference, audit["dimension"], audit
    laplacian, audit = load_laplacian()
    dimension = laplacian.shape[0]
    factor = splu(laplacian + case.value * eye(dimension, format="csc"))
    reference = inverse_trace(factor, dimension)
    return factor.solve, reference, dimension, audit


def run_method(matvec, dimension, reference, seed, method):
    phases = dict.fromkeys(PHASES, 0)
    process_start = process_time_ns()
    start = perf_counter_ns()
    rng = np.random.default_rng(seed)
    factor = np.empty((dimension, CAP), order="F")
    progress_anchor = Anchor(0, 0.0, 0.0, ALPHA) if method == "progress" else None
    moving = FastTraceEvidence(alpha=ALPHA) if method == "moving_norm" else None
    captured = total = squared_total = lower = 0.0
    upper = math.inf
    checked = []
    estimate = radius = None
    probes = 0
    error = None
    phases["initialize"] = perf_counter_ns() - start

    try:
        for step in range(1, CAP + 1):
            previous = captured
            mark = perf_counter_ns()
            vector = rng.standard_normal(dimension)
            after_draw = perf_counter_ns()
            probes = step
            image = matvec(vector)
            after_oracle = perf_counter_ns()

            basis = factor[:, :step - 1]
            residual = image - basis @ (basis.T @ vector)
            q = float(vector @ residual)
            v = float(residual @ residual)
            if not (q > 0 and v > 0 and math.isfinite(q + v)):
                raise ArithmeticError("Nonpositive or nonfinite residual")
            captured = previous + v / q
            if not (captured > previous and math.isfinite(captured)):
                raise ArithmeticError("Nonpositive or nonfinite trace progress")
            factor[:, step - 1] = residual / math.sqrt(q)
            after_sketch = perf_counter_ns()

            if moving is None:
                total += previous + q
                if method == "frozen_norm":
                    squared_total += v
            else:
                moving.append(previous, q, v)
            if step % REPORT_EVERY == 0:
                if progress_anchor is not None:
                    interval = invert_progress(progress_anchor, step, total, captured, c=1 / 3)
                elif moving is None:
                    interval = invert_frozen_norm(step, total, captured, squared_total, alpha=ALPHA, stake=0.5)
                else:
                    interval = moving.interval()
                if any(math.isnan(endpoint) for endpoint in interval):
                    raise ArithmeticError("NaN confidence endpoint")
                lower, upper = max(lower, interval[0]), min(upper, interval[1])
                checked.append((lower, upper))
                candidate_radius = relative_radius(lower, upper)
                if candidate_radius <= TARGET:
                    radius = candidate_radius
                    estimate = relative_center(lower, upper)

            after_evidence = perf_counter_ns()
            phases["draw"] += after_draw - mark
            phases["oracle"] += after_oracle - after_draw
            phases["sketch"] += after_sketch - after_oracle
            phases["evidence"] += after_evidence - after_sketch
            if estimate is not None:
                break
    except NUMERICAL_ERRORS as exc:
        error = str(exc)

    wall_ns = perf_counter_ns() - start
    process_ns = process_time_ns() - process_start
    return {
        "probes": probes,
        "reports": len(checked),
        "completed": estimate is not None,
        "estimate": estimate,
        "stop_radius": radius,
        "relative_error": abs(estimate - reference) / reference if estimate is not None else None,
        "reference_exclusion_reports": sum(not lo <= reference <= hi for lo, hi in checked),
        "numerical_error": error,
        "phase_wall_ns": phases,
        "wall_ns": wall_ns,
        "process_cpu_ns": process_ns,
    }


def summarize(records):
    summaries = {}
    for method in METHODS:
        rows = [record["methods"][method] for record in records]
        summaries[method] = {
            "completions": sum(row["completed"] for row in rows),
            "incorrect_completions": sum(row["completed"] and row["relative_error"] > TARGET for row in rows),
            "paths_with_reference_exclusion_before_exit": sum(row["reference_exclusion_reports"] > 0 for row in rows),
            "numerical_failures": sum(row["numerical_error"] is not None for row in rows),
            "median_probes_through_exit": median(row["probes"] for row in rows),
            "median_wall_ms_through_exit": median(row["wall_ns"] for row in rows) / 1e6,
            "median_process_cpu_ms_through_exit": median(row["process_cpu_ns"] for row in rows) / 1e6,
            "median_phase_wall_ms_through_exit": {
                phase: median(row["phase_wall_ns"][phase] for row in rows) / 1e6 for phase in PHASES
            },
        }

    paired = {}
    for method in ("progress", "moving_norm"):
        complete_pairs = [
            (row["methods"]["frozen_norm"], row["methods"][method])
            for row in records
            if row["methods"]["frozen_norm"]["completed"] and row["methods"][method]["completed"]
        ]
        differences = [frozen["wall_ns"] - other["wall_ns"] for frozen, other in complete_pairs]
        saving_pairs = [(frozen, other) for frozen, other in complete_pairs if frozen["probes"] > other["probes"]]
        break_even = [
            (other["wall_ns"] - frozen["wall_ns"]) / (frozen["probes"] - other["probes"])
            for frozen, other in saving_pairs
        ]
        paired[method] = {
            "complete_pairs": len(complete_pairs),
            "wall_wins_ties_losses": [
                sum(diff < 0 for diff in differences),
                sum(diff == 0 for diff in differences),
                sum(diff > 0 for diff in differences),
            ],
            "median_frozen_minus_other_wall_ms": median(differences) / 1e6 if differences else None,
            "pairs_where_other_uses_fewer_probes": len(saving_pairs),
            "median_additional_cost_ms_per_probe_at_equal_time": median(break_even) / 1e6 if break_even else None,
        }
    return summaries, paired


def main():
    SOURCES.update(NEW_SOURCES)
    sources = (
        Path(__file__), HERE / "run_anchor_norm_comparison.py", HERE / "run_confirmatory.py",
        HERE / "run_graph_holdout.py", HERE / "core.py", HERE / "fast_evidence.py",
        HERE / "anchored_norm.py", HERE / "progress_certificate.py",
    )
    output = HERE / "controlled_timing_results.json"
    result = {
        "status": "running",
        "paths_per_operator": PATHS,
        "first_seed": FIRST_SEED,
        "seed_stride": SEED_STRIDE,
        "report_every": REPORT_EVERY,
        "cap": CAP,
        "alpha": ALPHA,
        "target": TARGET,
        "timing_scope": "Standalone serial runs, initialization through stop or exit. One BLAS thread. "
                        "Each method has one unreported full warmup. All six run orders occur equally often. "
                        "Shared loading, factorization and numerical references are timed separately.",
        "counterfactual_scope": "Break-even values add the same fixed cost to each oracle product, "
                                "holding observed phase times and stopping decisions fixed.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in sources},
        "cases": {},
    }
    with threadpool_limits(limits=1):
        result["threadpools"] = [
            {key: value for key, value in pool.items() if key != "filepath"} for pool in threadpool_info()
        ]
        for index, case in enumerate(CASES):
            setup_start = perf_counter_ns()
            matvec, reference, dimension, audit = operator(case)
            setup_ms = (perf_counter_ns() - setup_start) / 1e6
            base = FIRST_SEED + SEED_STRIDE * index
            for method in METHODS:
                warmup = run_method(matvec, dimension, reference, base - 1, method)
                if warmup["numerical_error"] is not None:
                    raise ArithmeticError(f"{case.name} {method} warmup: {warmup['numerical_error']}")
            records = []
            for path in range(PATHS):
                order = METHOD_ORDERS[path % len(METHOD_ORDERS)]
                seed = base + path
                records.append({
                    "seed": seed,
                    "order": order,
                    "methods": {method: run_method(matvec, dimension, reference, seed, method) for method in order},
                })
            summaries, paired = summarize(records)
            result["cases"][case.name] = {
                "dimension": dimension,
                "reference": reference,
                "data_audit": audit,
                "shared_setup_and_reference_ms": setup_ms,
                "methods": summaries,
                "paired_frozen_norm_vs": paired,
                "records": records,
            }
            print(case.name, json.dumps(summaries), flush=True)
            staging = output.with_suffix(".tmp")
            staging.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            staging.replace(output)
    result["status"] = "complete"
    staging = output.with_suffix(".tmp")
    staging.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(output)


if __name__ == "__main__":
    main()
