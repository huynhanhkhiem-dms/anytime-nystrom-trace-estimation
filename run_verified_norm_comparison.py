"""Compare outward inexact-inverse intervals and solver effort on ca-GrQc."""

from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
from time import perf_counter

from flint import arb, ctx
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import cg
from threadpoolctl import threadpool_limits

from run_verified_galerkin import (
    DIMENSION, PRECISION, build_operator, column, dot, dyadic, exact_matrix, normal_bin,
    reference_trace,
)


HERE = Path(__file__).resolve().parent
CAP = 126
SEEDS = range(3367000, 3367004)
TARGET = 0.1
MODES = {"loose": (1e-2,), "tight": (1e-9,), "refine": (1e-2, 1e-4, 1e-7, 1e-10)}


def projected_trial(exact_k, shift, g, h0, y, basis):
    w = column([arb(float(value)) for value in y])
    ky = exact_k * w
    kw = ky
    for direction, k_direction, energy in basis:
        coefficient = dot(direction, ky) / energy
        w = w - direction * coefficient
        kw = kw - k_direction * coefficient
    energy = dot(w, kw)
    if not energy > 0:
        eta2 = dot(h0, h0).upper() / shift
        return None, arb(0), arb(0), eta2, arb(1)

    projection = dot(g, w)
    residual = h0 - kw * (projection / energy)
    eta2 = dot(residual, residual).upper() / shift
    q0 = projection ** 2 / energy
    progress = dot(w, w) / energy
    q_floor = max(arb(0), q0.lower())
    slack = (eta2 / (q_floor + eta2)).upper().sqrt().upper() if q_floor > 0 else arb(1)
    return (w, kw, energy), q0, progress, eta2, min(arb(1), slack)


def certified_norm_interval(step, z_lower, z_upper, hard_lower, variance_upper):
    """Conservative hull of the quadratic nonrejection set for stake 1/2."""
    threshold = 2 * arb(40).log()
    lower_count = step + threshold
    lower = hard_lower
    discriminant = z_lower ** 2 - 2 * lower_count * variance_upper
    if discriminant > 0:
        root_sum = z_lower + discriminant.sqrt()
        if root_sum > 0 and (variance_upper / root_sum).upper() < hard_lower:
            lower = max(lower, (root_sum / (2 * lower_count)).lower())

    upper = None
    upper_count = step - threshold
    if upper_count > 0:
        negative_penalty = arb(1).exp() - 2
        root = (z_upper ** 2 + 4 * upper_count * negative_penalty * variance_upper).sqrt()
        upper = ((z_upper + root) / (2 * upper_count)).upper()
    return lower, upper


def run_path(matrix, exact_k, shift, reference, tolerances, seed, budget=CAP):
    rng = Random(seed)
    dimension = matrix.shape[0]
    basis = []
    sketch = zlo = zhi = variance_upper = arb(0)
    c = arb(1) / 3
    threshold = arb(40).log() / c
    envelopes = {"progress": (arb(0), None), "frozen_norm": (arb(0), None)}
    stops = dict.fromkeys(envelopes)
    exclusions = {name: 0 for name in envelopes}
    solver_seconds = proof_seconds = probe_seconds = 0.0
    inner_iterations = solve_attempts = failed_gates = skipped_pivots = 0
    full_span_enclosure = None
    probes_executed = 0

    for step in range(1, budget + 1):
        probes_executed = step
        started = perf_counter()
        probe = [normal_bin(rng) for _ in range(dimension)]
        g = column(probe)
        rhs = np.array([float(value.mid()) for value in probe])
        probe_seconds += perf_counter() - started

        started = perf_counter()
        h0 = g
        for direction, k_direction, energy in basis:
            h0 = h0 - k_direction * (dot(direction, g) / energy)
        proof_seconds += perf_counter() - started

        y = None
        for tolerance in tolerances:
            def count_iteration(_):
                nonlocal inner_iterations
                inner_iterations += 1

            started = perf_counter()
            y, _ = cg(matrix, rhs, x0=y, rtol=tolerance, maxiter=4 * dimension, callback=count_iteration)
            solver_seconds += perf_counter() - started
            solve_attempts += 1
            if not np.isfinite(y).all():
                y = np.zeros_like(rhs)
            started = perf_counter()
            pivot, q0, progress, eta2, slack = projected_trial(exact_k, shift, g, h0, y, basis)
            quality = bool(pivot is not None and eta2 <= progress.lower() and slack <= arb(3) / 80)
            proof_seconds += perf_counter() - started
            if quality:
                break
        failed_gates += int(not quality)

        started = perf_counter()
        if pivot is None:
            skipped_pivots += 1
        else:
            basis.append(pivot)
        previous_sketch = sketch
        sketch += progress
        if len(basis) == dimension:
            # Certified positive pivots form a basis. Its Galerkin inverse is K^-1.
            lower, upper = sketch.lower(), sketch.upper()
            full_span_enclosure = {"lower_dyadic": dyadic(lower), "upper_dyadic": dyadic(upper)}
            intervals = dict.fromkeys(envelopes, (lower, upper))
        else:
            zlo += previous_sketch.lower() + max(arb(0), q0.lower())
            zhi += previous_sketch.upper() + (q0 + eta2).upper()
            threshold += 4 * c * slack
            approximate_norm = max(arb(0), (q0 * progress).upper()).sqrt()
            residual_norm = (eta2 / shift).upper().sqrt()
            variance_upper += ((approximate_norm + residual_norm) ** 2).upper()

            penalty = 4 * c * sketch.upper()
            progress_lo = max(sketch.lower(), ((zlo - penalty) / (step + threshold)).lower())
            progress_hi = ((zhi + penalty) / (step - threshold)).upper() if step > threshold else None
            intervals = {"progress": (progress_lo, progress_hi),
                         "frozen_norm": certified_norm_interval(step, zlo, zhi, sketch.lower(), variance_upper)}
        for name, (lower, upper) in intervals.items():
            if full_span_enclosure is None:
                old_lower, old_upper = envelopes[name]
                lower = max(lower, old_lower)
                if old_upper is not None:
                    upper = old_upper if upper is None else min(old_upper, upper)
            envelopes[name] = lower, upper
            exclusions[name] += int(lower > reference.upper() or (upper is not None and upper < reference.lower()))
            if stops[name] is None and upper is not None and arb(0) < lower <= upper:
                radius = (upper - lower) / (upper + lower)
                if radius <= arb(1) / 10:
                    stops[name] = {"outer_probes": step, "cg_iterations": inner_iterations,
                                   "solve_attempts": solve_attempts,
                                   "lower_dyadic": dyadic(lower), "upper_dyadic": dyadic(upper)}
        proof_seconds += perf_counter() - started
        if full_span_enclosure is not None:
            break

    return {"seed": seed, "stops": stops, "reference_exclusion_checks": exclusions, "failed_solve_gates": failed_gates,
            "probes_executed": probes_executed, "full_span_trace_enclosure": full_span_enclosure,
            "skipped_pivots": skipped_pivots, "solve_attempts": solve_attempts, "cg_iterations": inner_iterations,
            "seconds": {"cg": solver_seconds, "verified_projection_and_evidence": proof_seconds,
                        "gaussian_enclosures": probe_seconds}}


def main():
    ctx.prec = PRECISION
    path = HERE / "verified_norm_comparison.json"
    result = {"status": "RUNNING", "dimension": DIMENSION, "cap": CAP, "target": TARGET,
              "alpha_per_rule": 0.05, "seeds": list(SEEDS), "mode_tolerances": MODES,
              "scope": "Seeded diagnostic with outward arithmetic. Coverage assumes ideal Gaussian bits. "
                       "Timing includes common verification and both intervals, with dense Arb matrix action.",
              "source_sha256": {file.name: sha256(file.read_bytes()).hexdigest() for file in
                                (Path(__file__), HERE / "run_verified_galerkin.py", HERE / "run_graph_holdout.py")},
              "cases": []}
    with threadpool_limits(limits=1):
        laplacian, result["data_audit"] = build_operator()
        for power in (-8, -1):
            shift = arb((1, power))
            matrix = laplacian + math.ldexp(1, power) * eye(DIMENSION, format="csr")
            exact_k = exact_matrix(laplacian, shift)
            theta = reference_trace(exact_k)
            for mode, tolerances in MODES.items():
                records = [run_path(matrix, exact_k, shift, theta, tolerances, seed) for seed in SEEDS]
                result["cases"].append({"shift_exponent": power, "mode": mode, "records": records})
                print(power, mode, [r["stops"]["progress"] and r["stops"]["progress"]["outer_probes"] for r in records],
                      [r["stops"]["frozen_norm"] and r["stops"]["frozen_norm"]["outer_probes"] for r in records],
                      flush=True)
                staging = path.with_suffix(".tmp")
                staging.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
                staging.replace(path)
    result["status"] = "COMPLETE"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
