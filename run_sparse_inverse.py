"""Time independent outward Galerkin certificates on two sparse graph inverses."""

import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
from time import perf_counter

from flint import arb, arb_mat, ctx
import numpy as np
import requests
from scipy.linalg import eigvalsh
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.sparse.linalg import cg
from threadpoolctl import threadpool_limits

from run_verified_galerkin import PRECISION, column, dot, dyadic, normal_bin
from run_verified_norm_comparison import MODES, certified_norm_interval, projected_trial


HERE = Path(__file__).resolve().parent
CAP = 144
TARGET = 0.10
CASES = (
    ("ca-GrQc", "ca-GrQc.txt.gz", 512, -5, 11304000,
     "a254442cdf5d684712578b630c2e0d7543518ab154ef2341cabb607572ce7230"),
    ("Email-Eu-core", "email-Eu-core.txt.gz", 768, -1, 11305000,
     "4b47acdb80197b085fe63c819c357ae488131ee904ed93d1b219a68b0f9e245f"),
)
METHODS = ("progress", "frozen_norm")


class SparseShiftedLaplacian:
    def __init__(self, shift, dimension, edge_rows, edge_cols):
        self.shift = shift
        self.dimension = dimension
        self.edges = list(zip(edge_rows.tolist(), edge_cols.tolist(), strict=True))

    def __mul__(self, vector):
        entries = vector.entries()
        image = [self.shift * entry for entry in entries]
        for first, second in self.edges:
            difference = entries[first] - entries[second]
            image[first] += difference
            image[second] -= difference
        return column(image)


def build_case(filename, dimension, power, expected_sha):
    path = HERE / "matrices" / filename
    path.parent.mkdir(exist_ok=True)
    url = f"https://snap.stanford.edu/data/{filename}"
    if path.is_file():
        payload = path.read_bytes()
    else:
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        payload = response.content
    if sha256(payload).hexdigest() != expected_sha:
        raise ValueError(f"Unexpected SHA-256 for {filename}")
    if not path.exists():
        path.write_bytes(payload)

    with gzip.open(path, "rt") as handle:
        edges = np.loadtxt(handle, dtype=np.int64, usecols=(0, 1))
    labels, compact = np.unique(edges, return_inverse=True)
    compact = compact.reshape(edges.shape)
    adjacency = coo_matrix((np.ones(len(compact), dtype=np.int64), (compact[:, 0], compact[:, 1])),
                           shape=(len(labels), len(labels))).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data.fill(1)
    _, groups = connected_components(adjacency, directed=False)
    keep = groups == np.bincount(groups).argmax()
    component = adjacency[keep, :][:, keep].tocsr()
    component.sort_indices()
    order = breadth_first_order(component, 0, directed=False, return_predecessors=False)
    selected = np.sort(order[:dimension])
    assert len(selected) == dimension
    subgraph = component[selected, :][:, selected].tocsr()
    subgraph.sort_indices()
    assert connected_components(subgraph, directed=False, return_labels=False) == 1
    rows, cols = subgraph.nonzero()
    upper = rows < cols
    shift = arb((1, power))
    sparse_arb = SparseShiftedLaplacian(shift, dimension, rows[upper], cols[upper])
    degrees = np.asarray(subgraph.sum(axis=1)).ravel()
    laplacian = diags(degrees, dtype=np.int64, format="csr") - subgraph
    sparse = diags(degrees + math.ldexp(1.0, power), format="csr") - subgraph
    audit = {"source": url, "sha256": expected_sha, "source_bytes": len(payload),
             "raw_vertices": len(labels), "source_edge_rows": len(edges), "dimension": dimension,
             "subgraph_labels": labels[keep][selected].tolist(), "undirected_edges": int(upper.sum()),
             "max_degree": int(degrees.max()), "matrix_nnz": sparse.nnz, "shift_exponent": power,
             "condition_upper": 1 + 2 * int(degrees.max()) / math.ldexp(1.0, power)}
    return laplacian, sparse, sparse_arb, audit


def audit_reference(laplacian, sparse, sparse_arb):
    dimension = sparse.shape[0]
    entries = laplacian.toarray().ravel().tolist()
    for i in range(dimension):
        entries[i * (dimension + 1)] = arb(entries[i * (dimension + 1)]) + sparse_arb.shift
    dense = arb_mat(dimension, dimension, entries)
    for multiplier in (5, 11):
        probe = column([arb(((multiplier * i % 19) - 9, -3)) for i in range(dimension)])
        assert (sparse_arb * probe).entries() == (dense * probe).entries()
    inverse = dense.inv()
    reference = sum((inverse[i, i] for i in range(dimension)), arb(0))
    spectrum = eigvalsh(sparse.toarray(), driver="evd")
    approximate = float(np.reciprocal(spectrum).sum())
    assert abs(approximate / float(reference.mid()) - 1) < 1e-10
    return reference, {"reference_lower_dyadic": dyadic(reference.lower()),
                       "reference_upper_dyadic": dyadic(reference.upper()),
                       "spectral_reference_binary64": approximate,
                       "condition_estimate_binary64": float(spectrum[-1] / spectrum[0]),
                       "sparse_actions_equal_dense_on_dyadic_checks": 2}


def run_path(matrix, sparse_arb, method, seed, budget):
    start = perf_counter()
    rng = Random(seed)
    shift = sparse_arb.shift
    dimension = matrix.shape[0]
    basis = []
    history = []
    cg_codes = []
    timings = dict.fromkeys(("gaussian", "cg", "verification"), 0.0)
    sketch = zlo = zhi = variance = lower = arb(0)
    upper = None
    c = arb(1) / 3
    threshold = arb(40).log() / c
    cg_iterations = solve_attempts = failed_gates = skipped_pivots = 0
    stop = None
    termination = "cap"
    for step in range(1, budget + 1):
        phase_start = perf_counter()
        probe = [normal_bin(rng) for _ in range(dimension)]
        g = column(probe)
        rhs = np.array([float(value.mid()) for value in probe])
        timings["gaussian"] += perf_counter() - phase_start

        phase_start = perf_counter()
        h0 = g
        for direction, k_direction, energy in basis:
            h0 = h0 - k_direction * (dot(direction, g) / energy)
        timings["verification"] += perf_counter() - phase_start

        def count_iteration(_):
            nonlocal cg_iterations
            cg_iterations += 1

        y = None
        for tolerance in MODES["refine"]:
            phase_start = perf_counter()
            y, info = cg(matrix, rhs, x0=y, rtol=tolerance, maxiter=4 * dimension, callback=count_iteration)
            timings["cg"] += perf_counter() - phase_start
            solve_attempts += 1
            if info or not np.isfinite(y).all():
                cg_codes.append({"probe": step, "tolerance": tolerance, "cg_info": int(info),
                                 "finite": bool(np.isfinite(y).all())})
                if not np.isfinite(y).all():
                    y = np.zeros_like(rhs)
            phase_start = perf_counter()
            pivot, q0, progress, eta2, slack = projected_trial(sparse_arb, shift, g, h0, y, basis)
            quality = pivot is not None and eta2 <= progress.lower() and slack <= arb(3) / 80
            timings["verification"] += perf_counter() - phase_start
            if quality:
                break

        phase_start = perf_counter()
        failed_gates += int(not quality)
        if pivot is None:
            skipped_pivots += 1
        else:
            basis.append(pivot)
        zlo += sketch.lower() + max(arb(0), q0.lower())
        zhi += sketch.upper() + (q0 + eta2).upper()
        sketch += progress
        full_span = len(basis) == dimension
        if full_span:
            lower, upper = sketch.lower(), sketch.upper()
            termination = "full_span"
        else:
            if method == "progress":
                threshold += 4 * c * slack
                penalty = 4 * c * sketch.upper()
                current_lower = max(sketch.lower(), ((zlo - penalty) / (step + threshold)).lower())
                current_upper = ((zhi + penalty) / (step - threshold)).upper() if step > threshold else None
            else:
                approximate_norm = max(arb(0), (q0 * progress).upper()).sqrt()
                residual_norm = (eta2 / shift).upper().sqrt()
                variance += ((approximate_norm + residual_norm) ** 2).upper()
                current_lower, current_upper = certified_norm_interval(step, zlo, zhi, sketch.lower(), variance)
            lower = max(lower, current_lower)
            if current_upper is not None:
                upper = current_upper if upper is None else min(upper, current_upper)
        lower_dyadic = dyadic(lower)
        upper_dyadic = dyadic(upper) if upper is not None else None
        history.append({"lower_dyadic": lower_dyadic, "upper_dyadic": upper_dyadic})
        if upper is not None and arb(0) < lower <= upper:
            radius = (upper - lower) / (upper + lower)
            if radius <= arb(1) / 10:
                stop = {"probes": step, "lower_dyadic": lower_dyadic, "upper_dyadic": upper_dyadic,
                        "relative_radius_upper_dyadic": dyadic(radius.upper())}
                termination = "full_span" if full_span else "certificate"
        timings["verification"] += perf_counter() - phase_start
        if stop is not None or full_span:
            break
    wall = perf_counter() - start
    return {"seed": seed, "method": method, "probes_executed": len(history), "stop": stop,
            "termination": termination, "certified_pivots": len(basis), "cg_iterations": cg_iterations,
            "solve_attempts": solve_attempts, "failed_gates": failed_gates, "skipped_pivots": skipped_pivots,
            "cg_status_exceptions": cg_codes, "seconds": {"wall": wall, **timings}, "interval_history": history}


def arb_from_dyadic(parts):
    return arb((int(parts[0]), parts[1]))


def verify_path(record, reference):
    exclusions = []
    inconclusive = []
    for step, item in enumerate(record["interval_history"], start=1):
        lower = arb_from_dyadic(item["lower_dyadic"])
        upper = arb_from_dyadic(item["upper_dyadic"]) if item["upper_dyadic"] is not None else None
        if lower > reference.upper() or (upper is not None and upper < reference.lower()):
            exclusions.append(step)
        elif lower > reference.lower() or (upper is not None and upper < reference.upper()):
            inconclusive.append(step)
    record["excluded_reference_at"] = exclusions
    record["inconclusive_reference_at"] = inconclusive


def main():
    ctx.prec = PRECISION
    path = HERE / "sparse_inverse_results.json"
    sources = (Path(__file__), HERE / "run_verified_galerkin.py", HERE / "run_verified_norm_comparison.py",
               HERE / "sparse_inverse_protocol.txt")
    output = {"status": "RUNNING", "precision_bits": PRECISION, "target": TARGET, "cap": CAP,
              "alpha_per_rule": 0.05, "modes": METHODS, "tolerances": MODES["refine"],
              "scope": "Outward arithmetic with a sparse K action. Coverage uses ideal independent Gaussian bits.",
              "source_sha256": {source.name: sha256(source.read_bytes()).hexdigest() for source in sources},
              "cases": []}
    with threadpool_limits(limits=1):
        for case_index, (operator, filename, dimension, power, first_seed, expected_sha) in enumerate(CASES):
            setup_start = perf_counter()
            laplacian, matrix, sparse_arb, audit = build_case(filename, dimension, power, expected_sha)
            setup = perf_counter() - setup_start
            reference_start = perf_counter()
            reference, reference_audit = audit_reference(laplacian, matrix, sparse_arb)
            reference_seconds = perf_counter() - reference_start
            print(operator, dimension, audit["matrix_nnz"], audit["condition_upper"], reference_seconds, flush=True)
            for method in METHODS:
                run_path(matrix, sparse_arb, method, first_seed - 1, 4)
            case = {"operator": operator, **audit, **reference_audit, "setup_seconds": setup,
                    "reference_seconds": reference_seconds, "records": []}
            output["cases"].append(case)
            for index in range(3):
                seed = first_seed + index
                order = METHODS if (case_index + index) % 2 == 0 else METHODS[::-1]
                for method in order:
                    record = run_path(matrix, sparse_arb, method, seed, CAP)
                    verify_path(record, reference)
                    case["records"].append(record)
                    staging = path.with_suffix(".tmp")
                    staging.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
                    staging.replace(path)
                    print(operator, seed, method, record["termination"], record["probes_executed"],
                          record["cg_iterations"], round(record["seconds"]["wall"], 3), flush=True)
    output["status"] = "COMPLETE"
    path.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
