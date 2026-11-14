"""Verify inexact-inverse trace certificates with Arb interval arithmetic."""
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
from time import perf_counter

from flint import arb, arb_mat, ctx
import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse import diags, eye
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse.linalg import cg
from threadpoolctl import threadpool_limits

from run_graph_holdout import load_laplacian


HERE = Path(__file__).resolve().parent
DIMENSION = 128
CAP = 96
SEEDS = range(626800, 626806)
SHIFT_EXPONENTS = (-8, -1)
TOLERANCES = (1e-2, 1e-9)
PRECISION = 256


def normal_bin(rng):
    """Enclose a Gaussian quantile determined by a finite uniform bit prefix."""
    bits = 64
    index = rng.getrandbits(bits)
    while index == 0 or index == (1 << bits) - 1:
        index = (index << 32) | rng.getrandbits(32)
        bits += 32
    with ctx.workprec(max(ctx.prec, bits + 48)):
        low = (arb((index, 1 - bits)) - 1).erfinv()
        high = (arb((index + 1, 1 - bits)) - 1).erfinv()
        return arb(2).sqrt() * low.union(high)


def dot(a, b):
    return (a.transpose() * b)[0, 0]


def column(values):
    return arb_mat(len(values), 1, values)


def dyadic(value):
    mantissa, exponent = value.man_exp()
    return [str(mantissa), int(exponent)]


def build_operator():
    original, provenance = load_laplacian()
    adjacency = -original.copy()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency = adjacency.tocsr()
    adjacency.sort_indices()
    order = breadth_first_order(adjacency, 0, directed=False, return_predecessors=False)
    selected = np.sort(order[:DIMENSION])
    subgraph = adjacency[selected, :][:, selected].tocsr()
    degrees = np.asarray(subgraph.sum(axis=1)).ravel()
    laplacian = diags(degrees, format="csr") - subgraph
    provenance.update(
        subgraph_rule="Induced graph of the first 128 breadth-first vertices from compact index zero in the largest component, using sorted CSR neighbors.",
        subgraph_indices=selected.tolist(),
        subgraph_edges=subgraph.nnz // 2,
    )
    return laplacian, provenance


def exact_matrix(laplacian, shift):
    entries = [arb(int(value)) for value in laplacian.toarray().ravel()]
    for i in range(DIMENSION):
        entries[i * (DIMENSION + 1)] += shift
    return arb_mat(DIMENSION, DIMENSION, entries)


def reference_trace(k):
    inverse = k.inv()
    return sum((inverse[i, i] for i in range(DIMENSION)), arb(0))


def run_path(matrix, exact_k, shift, theta, tolerance, seed):
    rng = Random(seed)
    basis = []
    trace = arb(0)
    sum_lower = arb(0)
    sum_upper = arb(0)
    threshold = 4 * arb(40).log()  # c=1/4, alpha=1/20.
    lower = arb(0)
    upper = None
    stop = None
    total_cg = 0
    inconclusive_checks = 0
    excluded_reference = []
    breakdowns = []
    max_radius = 0.0
    max_local_slack = 0.0
    for step in range(1, CAP + 1):
        probe = [normal_bin(rng) for _ in range(DIMENSION)]
        g = column(probe)
        rhs = np.array([float(component.mid()) for component in probe])
        max_radius = max(max_radius, max(float(component.rad()) for component in probe))
        iterations = 0

        def count_iteration(_):
            nonlocal iterations
            iterations += 1

        y, info = cg(matrix, rhs, rtol=tolerance, maxiter=4 * DIMENSION, callback=count_iteration)
        total_cg += iterations
        if info != 0 or not np.isfinite(y).all():
            breakdowns.append({"probe": step, "cg_info": int(info), "fallback": "zero trial direction"})
            y = np.zeros_like(rhs)
        w = column([arb(float(value)) for value in y])
        ky = exact_k * w
        kw = ky
        h0 = g
        for direction, k_direction, energy in basis:
            coefficient = dot(direction, ky) / energy
            w = w - direction * coefficient
            kw = kw - k_direction * coefficient
            h0 = h0 - k_direction * (dot(direction, g) / energy)
        energy = dot(w, kw)
        if energy > 0:
            projection = dot(g, w)
            residual = h0 - kw * (projection / energy)
            error2 = dot(residual, residual).upper() / shift
            q0 = projection ** 2 / energy
            progress = dot(w, w) / energy
            q_floor = max(arb(0), q0.lower())
            slack = (error2 / (q_floor + error2)).upper().sqrt().upper() if q_floor > 0 else arb(1)
            slack = min(arb(1), slack)
            basis.append((w, kw, energy))
        else:
            # A direction is optional; retain the previous certified subspace.
            error2 = dot(h0, h0).upper() / shift
            q0 = arb(0)
            progress = arb(0)
            slack = arb(1)
            breakdowns.append({"probe": step, "fallback": "uncertified pivot skipped"})
        max_local_slack = max(max_local_slack, float(slack))
        sum_lower += trace.lower() + max(arb(0), q0.lower())
        sum_upper += trace.upper() + (q0 + error2).upper()
        trace += progress
        threshold += slack
        lower = max(lower, trace.lower(), ((sum_lower - trace.upper()) / (step + threshold)).lower())
        if step > threshold:
            candidate = ((sum_upper + trace.upper()) / (step - threshold)).upper()
            upper = candidate if upper is None else min(upper, candidate)

        if lower > theta.upper() or (upper is not None and upper < theta.lower()):
            excluded_reference.append(step)
        elif lower > theta.lower() or (upper is not None and upper < theta.upper()):
            inconclusive_checks += 1
        if stop is None and upper is not None and arb(0) < lower <= upper:
            radius = (upper - lower) / (upper + lower)
            if radius <= arb(1) / 10:
                stop = {"probes": step, "cg_iterations": total_cg, "radius_upper_dyadic": dyadic(radius.upper()),
                        "lower_dyadic": dyadic(lower), "upper_dyadic": dyadic(upper)}
    return {"seed": seed, "stop": stop, "certified_pivots": len(basis), "fallbacks": breakdowns,
            "excluded_reference_at": excluded_reference, "inconclusive_checks": inconclusive_checks,
            "maximum_gaussian_bin_radius": max_radius, "maximum_step_slack": max_local_slack,
            "cg_iterations_at_cap": total_cg, "final_lower_dyadic": dyadic(lower),
            "final_upper_dyadic": dyadic(upper) if upper is not None else None}


def main():
    ctx.prec = PRECISION
    path = HERE / "verified_galerkin_results.json"
    output = {
        "scope": "Arb arithmetic enclosures with seeded uniform prefixes for reproducibility; the Gaussian-law guarantee assumes ideal unbiased independent bits and continuous tails.",
        "dimension": DIMENSION, "cap": CAP, "target": 0.10, "alpha": 0.05,
        "precision_bits": PRECISION, "seeds": list(SEEDS), "cases": [],
        "source_sha256": {file.name: sha256(file.read_bytes()).hexdigest()
                          for file in [Path(__file__), HERE / "run_graph_holdout.py"]},
    }
    with threadpool_limits(limits=1):
        laplacian, output["provenance"] = build_operator()
        for power in SHIFT_EXPONENTS:
            shift = arb((1, power))
            exact_k = exact_matrix(laplacian, shift)
            theta = reference_trace(exact_k)
            matrix = laplacian + math.ldexp(1, power) * eye(DIMENSION, format="csr")
            alternate = float((1 / eigvalsh(matrix.toarray())).sum())
            assert abs(alternate / float(theta.mid()) - 1) < 1e-10
            for tolerance in TOLERANCES:
                start = perf_counter()
                records = [run_path(matrix, exact_k, shift, theta, tolerance, seed) for seed in SEEDS]
                stops = [item["stop"]["probes"] for item in records if item["stop"]]
                case = {"shift_power_of_two": power, "cg_rtol": tolerance,
                        "reference_lower_dyadic": dyadic(theta.lower()), "reference_upper_dyadic": dyadic(theta.upper()),
                        "floating_spectrum_reference": alternate, "stops": len(stops),
                        "median_probes_among_stops": float(np.median(stops)) if stops else None,
                        "paths_excluding_reference": sum(bool(record["excluded_reference_at"]) for record in records),
                        "seconds": perf_counter() - start, "records": records}
                output["cases"].append(case)
                staging = path.with_suffix(".tmp")
                staging.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
                staging.replace(path)
                print(power, tolerance, len(stops), case["median_probes_among_stops"], case["seconds"], flush=True)


if __name__ == "__main__":
    main()
