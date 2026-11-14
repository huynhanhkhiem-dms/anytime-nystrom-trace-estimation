"""Frozen follow-up on held-out real matrix functions."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import gzip
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import platform
from time import perf_counter
import zipfile

import numpy as np
import requests
import scipy
from scipy.io import mmread
from scipy.linalg import cholesky, eigh, eigvalsh
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply, splu
from scipy.stats import beta, binomtest
from threadpoolctl import threadpool_limits

from adaptive_hutch_baseline import run_trial as run_adaptive
from baselines import gaussian_psd_fixed_interval
from chi2_comparator import ChiSquareEvidence
from fast_evidence import FastTraceEvidence
from progress_certificate import Anchor, invert_progress, relative_center, relative_radius


HERE = Path(__file__).resolve().parent
ALPHA = 0.05
TARGET = 0.05
PATHS = 96
CHECKPOINTS = (12, 16, 24, 32, 44, 64, 96, 128, 192, 256)
CAP = CHECKPOINTS[-1]
FRACTIONS = (0.25, 0.5, 0.75)
SPLIT_LEVEL = ALPHA / (len(FRACTIONS) * len(CHECKPOINTS))
METHODS = ("progress_anchored", "progress_single", "residual_v", "hybrid", "chi_square",
           "split_observed_v", "split_chernoff")
CONTRASTS = ("chi_square", "split_observed_v")
REFERENCE_RTOL = 1e-8
SOURCES = {
    "email": ("email-Eu-core.txt.gz", "https://snap.stanford.edu/data/email-Eu-core.txt.gz"),
    "yeast": ("bio-yeast.zip", "https://nrvis.com/download/data/bio/bio-yeast.zip"),
    "494_bus": ("494_bus.mtx.gz",
                "https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/psadmit/494_bus.mtx.gz"),
    "nos3": ("nos3.mtx.gz",
             "https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/lanpro/nos3.mtx.gz"),
    "har": ("har-240.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"),
}


@dataclass(frozen=True)
class Case:
    source: str
    transform: str
    value: float

    @property
    def name(self):
        return f"{self.source}_{self.transform}_{self.value:g}"


CASES = (
    Case("email", "heat", 0.5), Case("email", "heat", 5),
    Case("yeast", "heat", 0.5), Case("yeast", "heat", 5),
    Case("494_bus", "inverse", 0.01), Case("nos3", "inverse", 0.01),
    Case("har", "log_covariance", 0.1), Case("har", "log_covariance", 10),
)


def finite(value):
    return float(value) if math.isfinite(value) else None


def relative_difference(value, reference):
    return float(abs(value - reference) / abs(reference))


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def source_payload(key):
    filename, url = SOURCES[key]
    path = HERE / "matrices" / filename
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        payload = path.read_bytes()
    else:
        response = requests.get(url, timeout=(15, 180))
        response.raise_for_status()
        payload = response.content
        expected_magic = b"PK\x03\x04" if filename.endswith(".zip") else b"\x1f\x8b"
        if not payload.startswith(expected_magic):
            raise ValueError(f"Unexpected download format for {key}")
        return payload, {"source": url, "bytes": len(payload), "sha256": sha256(payload).hexdigest()}, path
    return payload, {"source": url, "bytes": len(payload), "sha256": sha256(payload).hexdigest()}, None


def graph_laplacian(source, payload):
    if source == "email":
        edges = np.loadtxt(io.BytesIO(gzip.decompress(payload)), usecols=(0, 1), dtype=np.int64)
        labels, indices = np.unique(edges.ravel(), return_inverse=True)
        indices = indices.reshape(-1, 2)
        dimension = len(labels)
        rows = len(edges)
        adjacency = coo_matrix((np.ones(rows), (indices[:, 0], indices[:, 1])),
                               shape=(dimension, dimension)).tocsr()
    else:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [name for name in archive.namelist() if name.endswith(".mtx")]
            if len(names) != 1:
                raise ValueError(f"Expected a single yeast Matrix Market file, found {names!r}")
            content = archive.read(names[0])
        adjacency = mmread(io.BytesIO(content), spmatrix=True).tocsr().astype(float)
        dimension = adjacency.shape[0]
        rows = int(content.splitlines()[1].split()[2])
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data.fill(1)
    _, groups = connected_components(adjacency, directed=False)
    keep = groups == np.bincount(groups).argmax()
    adjacency = adjacency[keep, :][:, keep]
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise ValueError("Largest connected component has isolated vertices")
    scale = diags(1 / np.sqrt(degree), format="csr")
    normalized = scale @ adjacency @ scale
    laplacian = eye(len(degree), format="csr") - (normalized + normalized.T) * 0.5
    return laplacian, {"raw_vertices": dimension, "raw_edge_rows": rows,
                       "undirected_edges_in_component": adjacency.nnz // 2}


def heat_operator(case, payload):
    laplacian, audit = graph_laplacian(case.source, payload)
    matrix = laplacian.toarray()
    spectrum, vectors = eigh(matrix, driver="evr")
    reference_spectrum = eigvalsh(matrix, driver="evd")
    if spectrum[0] < -1e-10 or spectrum[-1] > 2 + 1e-10:
        raise ValueError("Normalized Laplacian has eigenvalues outside [0, 2]")
    weights = np.exp(-case.value * spectrum)
    dense = (vectors * weights) @ vectors.T
    dense = 0.5 * (dense + dense.T)
    reference = float(np.exp(-case.value * reference_spectrum).sum())
    rng = np.random.default_rng(600419)
    probes = rng.standard_normal((len(weights), 4))
    checks = expm_multiply(-case.value * laplacian, probes)
    discrepancy = np.linalg.norm(dense @ probes - checks, axis=0) / np.linalg.norm(checks, axis=0)
    return dense.__matmul__, reference, weights, audit | {
        "reference_method": "independent dense eigvalsh evd, exponential of eigenvalues",
        "oracle_action_check": "sparse expm_multiply on four independent fixed audit probes",
        "secondary_trace": float(np.trace(dense)),
        "max_audit_product_relative_discrepancy": float(discrepancy.max()),
    }


def inverse_operator(case, payload):
    matrix = mmread(io.BytesIO(gzip.decompress(payload)), spmatrix=True).tocsc().astype(float)
    diagonal = float(np.median(matrix.diagonal()))
    if matrix.shape[0] != matrix.shape[1] or (matrix - matrix.T).nnz or diagonal <= 0:
        raise ValueError("Inverse source is not a real symmetric square matrix with positive scale")
    dimension = matrix.shape[0]
    shifted = matrix / diagonal + case.value * eye(dimension, format="csc")
    spectrum = eigvalsh(shifted.toarray(), driver="evd")
    if spectrum[0] <= 0:
        raise ValueError("Shifted inverse source is not numerically positive definite")
    factor = splu(shifted)
    weights = 1 / spectrum
    trace = 0.0
    for start in range(0, dimension, 64):
        coordinates = np.arange(start, min(start + 64, dimension))
        positions = np.arange(len(coordinates))
        rhs = np.zeros((dimension, len(coordinates)), order="F")
        rhs[coordinates, positions] = 1
        trace += float(factor.solve(rhs)[coordinates, positions].sum())
    rng = np.random.default_rng(600419)
    probes = rng.standard_normal((dimension, 4))
    discrepancy = np.linalg.norm(shifted @ factor.solve(probes) - probes, axis=0) / np.linalg.norm(probes, axis=0)
    return factor.solve, float(weights.sum()), weights, {
        "reference_method": "dense eigvalsh evd, sum of reciprocal eigenvalues",
        "oracle_action_check": "relative linear system residual on four fixed audit probes",
        "secondary_trace": trace,
        "max_audit_product_relative_discrepancy": float(discrepancy.max()),
        "median_diagonal_scale": diagonal, "source_nnz": matrix.nnz,
    }


def har_features(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = [name for name in archive.namelist() if name.endswith("/train/X_train.txt")]
        if len(files) == 1:
            return np.loadtxt(io.BytesIO(archive.read(files[0])))
        nested = [name for name in archive.namelist() if name.endswith("UCI HAR Dataset.zip")]
        if len(nested) == 1:
            return har_features(archive.read(nested[0]))
        raise ValueError("Published training features were not found")


def log_covariance_operator(case, payload):
    samples = har_features(payload)
    if samples.shape != (7352, 561) or not np.isfinite(samples).all():
        raise ValueError("Unexpected UCI HAR training matrix")
    samples -= samples.mean(axis=0)
    covariance = samples.T @ samples / len(samples)
    scale = float(np.trace(covariance) / len(covariance))
    covariance /= scale
    spectrum, vectors = eigh(covariance, driver="evr")
    if spectrum[0] < -1e-10:
        raise ValueError("Gram covariance is not numerically positive semidefinite")
    weights = np.log1p(case.value * np.maximum(spectrum, 0))
    dense = (vectors * weights) @ vectors.T
    dense = 0.5 * (dense + dense.T)
    positive = np.eye(len(covariance)) + case.value * covariance
    reference = 2 * float(np.log(cholesky(positive, lower=True).diagonal()).sum())
    return dense.__matmul__, reference, weights, {
        "reference_method": "Cholesky log determinant of I + s C",
        "secondary_trace": float(np.trace(dense)),
        "mean_diagonal_scale": scale, "feature_rows": len(samples),
        "numerically_negative_gram_eigenvalues_clipped_to_zero": int(np.sum(spectrum < 0)),
    }


def load_case(case):
    payload, source_audit, cache = source_payload(case.source)
    loader = {"heat": heat_operator, "inverse": inverse_operator, "log_covariance": log_covariance_operator}
    matvec, reference, weights, operator_audit = loader[case.transform](case, payload)
    audit = source_audit | operator_audit
    audit["reference_relative_disagreement"] = relative_difference(audit["secondary_trace"], reference)
    if (audit["reference_relative_disagreement"] > REFERENCE_RTOL or
            audit.get("max_audit_product_relative_discrepancy", 0) > REFERENCE_RTOL):
        raise ArithmeticError(f"Operator reference checks failed: {audit}")
    if len(weights) <= CAP or not np.isfinite(weights).all():
        raise ValueError("Operator dimension or transformed spectrum invalid")
    if cache is not None:
        # Distinct case jobs may request one data source at the same time.
        from os import getpid
        staging = cache.with_suffix(f".{getpid()}.tmp")
        staging.write_bytes(payload)
        staging.replace(cache)
    audit.update(
        dimension=len(weights), trace_reference=reference,
        transformed_min_eigenvalue=float(weights.min()),
        transformed_max_eigenvalue=float(weights.max()),
        participation_ratio=float(weights.sum() ** 2 / (weights @ weights)),
        reference_is_floating_point=True,
    )
    return matvec, reference, audit


class CheckpointProgress:
    """Record the original anchors and only intersect tests at shared checkpoints."""

    def __init__(self, alpha, anchored):
        self.alpha = alpha
        self.anchors = [Anchor(0, 0.0, 0.0, alpha / 2 if anchored else alpha)]
        self.next_anchor = 4 if anchored else CAP + 1
        self.bounds = (0.0, math.inf)

    def update(self, step, total, captured):
        if step in CHECKPOINTS:
            lower, upper = self.bounds
            for anchor in self.anchors:
                lo, hi = invert_progress(anchor, step, total, captured)
                lower, upper = max(lower, lo), min(upper, hi)
            self.bounds = lower, upper
        if step == self.next_anchor:
            count = len(self.anchors)
            spend = self.alpha / (2 * count * (count + 1))
            self.anchors.append(Anchor(step, captured, total, spend))
            self.next_anchor *= 2
        return self.bounds


def intersect(first, second):
    return max(first[0], second[0]), min(first[1], second[1])


def split_intervals(factors, probes, products, sketch_trace, step):
    observed, chernoff = (sketch_trace[step], math.inf), (sketch_trace[step], math.inf)
    for fraction in FRACTIONS:
        split = int(step * fraction)
        basis = factors[:, :split]
        heldout = probes[:, split:step]
        residual = products[:, split:step] - basis @ (basis.T @ heldout)
        q = (residual * heldout).sum(axis=0)
        v = (residual * residual).sum(axis=0)
        if np.any(q <= 0) or np.any(v <= 0) or not np.isfinite(q + v).all():
            raise ArithmeticError(f"Invalid held-out residual at {step}, split {split}")
        t = sketch_trace[split]
        lo, hi = gaussian_psd_fixed_interval(float(q.mean()), len(q), SPLIT_LEVEL)
        chernoff = intersect(chernoff, (t + lo, t + hi))
        evidence = FastTraceEvidence(alpha=SPLIT_LEVEL)
        for qi, vi in zip(q, v):
            evidence.append(float(t), float(qi), float(vi))
        observed = intersect(observed, evidence.interval())
    return observed, chernoff


def run_path(matvec, dimension, seed, reference):
    rng = np.random.default_rng(seed)
    factors = np.empty((dimension, CAP), order="F")
    probes = np.empty_like(factors)
    products = np.empty_like(factors)
    sketch_trace = np.zeros(CAP + 1)
    progress = CheckpointProgress(ALPHA, True)
    single = CheckpointProgress(ALPHA, False)
    hybrid_progress = CheckpointProgress(ALPHA / 2, True)
    evidence = FastTraceEvidence()
    hybrid_observed = FastTraceEvidence(alpha=ALPHA / 2)
    chi = ChiSquareEvidence()
    cumulative = 0.0
    split_old = [(0.0, math.inf), (0.0, math.inf)]
    states = {name: {"coverage_failure": False, "stop": None, "numerical_failure": None,
                     "checkpoints": []} for name in METHODS}
    common_errors = (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError)

    def report(name, interval):
        row = states[name]
        if row["numerical_failure"] is not None:
            return
        try:
            lower, upper = interval()
            if math.isnan(lower) or math.isnan(upper):
                raise ArithmeticError("Interval endpoint was NaN")
            radius = relative_radius(lower, upper)
            excluded = bool(not lower <= reference <= upper)
            row["checkpoints"].append([step, finite(lower), finite(upper), finite(radius), excluded])
            row["coverage_failure"] |= excluded
            if row["stop"] is None and radius <= TARGET:
                estimate = relative_center(lower, upper)
                row["stop"] = {"queries": step, "estimate": estimate, "radius": radius,
                               "relative_error": relative_difference(estimate, reference)}
        except common_errors as error:
            row["numerical_failure"] = {"step": step, "error": str(error)}

    for step in range(1, CAP + 1):
        try:
            g = rng.standard_normal(dimension)
            image = matvec(g)
            basis = factors[:, :step - 1]
            r = image - basis @ (basis.T @ g)
            q, v = float(g @ r), float(r @ r)
            previous_trace = float(sketch_trace[step - 1])
            if q <= 0 or v <= 0 or not math.isfinite(q + v):
                raise ArithmeticError("Nonpositive or nonfinite residual")
            factors[:, step - 1] = r / math.sqrt(q)
            probes[:, step - 1], products[:, step - 1] = g, image
            captured = previous_trace + v / q
            if not captured > previous_trace or not math.isfinite(captured):
                raise ArithmeticError("Positive trace increment lost in floating point")
            sketch_trace[step] = captured
            cumulative += previous_trace + q
            if not math.isfinite(cumulative):
                raise ArithmeticError("Nonfinite corrected cumulative trace")
        except common_errors as error:
            for row in states.values():
                if row["numerical_failure"] is None:
                    row["numerical_failure"] = {"step": step, "error": str(error)}
            break
        lo_progress = progress.update(step, cumulative, captured)
        lo_single = single.update(step, cumulative, captured)
        lo_hybrid = hybrid_progress.update(step, cumulative, captured)
        for name, process in (("residual_v", evidence), ("hybrid", hybrid_observed), ("chi_square", chi)):
            if states[name]["numerical_failure"] is not None:
                continue
            try:
                process.append(previous_trace, q, captured if name == "chi_square" else v)
            except common_errors as error:
                states[name]["numerical_failure"] = {"step": step, "error": str(error)}
        if step not in CHECKPOINTS:
            continue
        report("progress_anchored", lambda: lo_progress)
        report("progress_single", lambda: lo_single)
        report("residual_v", evidence.interval)
        report("hybrid", lambda: intersect(lo_hybrid, hybrid_observed.interval()))
        report("chi_square", chi.interval)
        try:
            split_new = split_intervals(factors, probes, products, sketch_trace, step)
            split_old = [intersect(old, new) for old, new in zip(split_old, split_new)]
            report("split_observed_v", lambda: split_old[0])
            report("split_chernoff", lambda: split_old[1])
        except common_errors as error:
            for name in ("split_observed_v", "split_chernoff"):
                if states[name]["numerical_failure"] is None:
                    states[name]["numerical_failure"] = {"step": step, "error": str(error)}
    return states


def upper_failure_limit(failures, trials):
    if trials == 0:
        return None
    if failures == trials:
        return 1.0
    return float(beta.ppf(0.95, failures + 1, trials - failures))


def cost(row):
    return CAP if row["stop"] is None else row["stop"]["queries"]


def summarize(trials, index):
    output = {}
    for name in METHODS:
        rows = [trial["shared"][name] for trial in trials]
        complete = [row for row in rows if row["numerical_failure"] is None]
        incomplete = [row for row in rows if row["numerical_failure"] is not None]
        stops = [row["stop"] for row in rows if row["stop"] is not None]
        exclusions = sum(row["coverage_failure"] for row in complete)
        output[name] = {
            "completions": len(stops),
            "incorrect_completions": sum(row["relative_error"] > TARGET for row in stops),
            "median_queries_among_completions": float(np.median([r["queries"] for r in stops])) if stops else None,
            "restricted_mean_queries": float(np.mean([cost(row) for row in rows])),
            "complete_replays": len(complete),
            "complete_replay_exclusions": exclusions,
            "complete_replay_exclusion_probability_one_sided_95pct_upper": upper_failure_limit(exclusions, len(complete)),
            "incomplete_replay_exclusions": sum(row["coverage_failure"] for row in incomplete),
            "numerical_failures": len(incomplete),
        }
    adaptive = [trial["adaptive_hutch"] for trial in trials]
    estimates = [row for row in adaptive if row["status"] == "complete"]
    output["adaptive_hutch"] = {
        "statuses": {status: sum(row["status"] == status for row in adaptive)
                     for status in ("complete", "cap_sketch", "cap_sizing", "cap_estimation", "numerical_failure")},
        "completions": len(estimates),
        "incorrect_completions": sum(row["relative_error"] > TARGET for row in estimates),
        "median_queries_among_completions": float(np.median([r["matvecs"] for r in estimates])) if estimates else None,
        "restricted_mean_queries": float(np.mean([row["matvecs"] if row["status"] == "complete" else CAP
                                                 for row in adaptive])),
    }
    contrasts = {}
    for number, comparator in enumerate(CONTRASTS):
        differences = np.array([cost(trial["shared"]["progress_anchored"]) - cost(trial["shared"][comparator])
                                for trial in trials])
        wins, losses = int(np.sum(differences < 0)), int(np.sum(differences > 0))
        rng = np.random.default_rng(7200000 + index * 1000 + number)
        sample = rng.choice(differences, size=(5000, len(differences)), replace=True)
        contrasts[comparator] = {
            "mean_difference_progress_minus_comparator": float(np.mean(differences)),
            "paired_percentile_bootstrap_95pct": np.quantile(sample.mean(axis=1), (0.025, 0.975)).tolist(),
            "progress_wins": wins, "progress_losses": losses, "ties": len(differences) - wins - losses,
            "sign_test_two_sided": float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0,
        }
    return output, contrasts


def run_case(index, audit_only):
    case = CASES[index]
    start = perf_counter()
    with threadpool_limits(limits=1):
        try:
            matvec, reference, audit = load_case(case)
        except (ArithmeticError, ValueError, RuntimeError, OSError, requests.RequestException, zipfile.BadZipFile) as error:
            print(case.name, "operator audit failed:", error, flush=True)
            return case.name, {"status": "operator_audit_failure", "error": str(error)}
        print(case.name, "audit passed", audit, flush=True)
        if audit_only:
            return case.name, {"status": "audit_only", "audit": audit}
        trials = []
        for path in range(PATHS):
            seed = 2500000 + 1000 * index + path
            adaptive_seed = seed + 1000000
            trials.append({
                "seed": seed, "adaptive_seed": adaptive_seed,
                "shared": run_path(matvec, audit["dimension"], seed, reference),
                "adaptive_hutch": run_adaptive(matvec, audit["dimension"], CAP, adaptive_seed, reference),
            })
            if (path + 1) % 24 == 0:
                print(case.name, f"{path + 1}/{PATHS} paths, {perf_counter() - start:.1f}s", flush=True)
        methods, contrasts = summarize(trials, index)
        result = {"status": "complete", "audit": audit, "summary": methods, "contrasts": contrasts,
                  "trials": trials, "elapsed_seconds_including_setup": perf_counter() - start}
        return case.name, result


def holm_adjust(cases):
    entries = sorted((value["sign_test_two_sided"], name, comparator)
                     for name, case in cases.items() for comparator, value in case.get("contrasts", {}).items())
    previous = 0.0
    tests = len(CASES) * len(CONTRASTS)
    for index, (pvalue, name, comparator) in enumerate(entries):
        previous = min(1.0, max(previous, (tests - index) * pvalue))
        cases[name]["contrasts"][comparator]["sign_test_holm_adjusted"] = previous


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2, help="Parallel independent operators, each with one BLAS thread")
    parser.add_argument("--audit-only", action="store_true", help="Load sources and check operators without trials")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    filenames = ("run_confirmatory.py", "confirmatory_protocol.txt", "adaptive_hutch_baseline.py",
                 "baselines.py", "chi2_comparator.py", "core.py", "fast_evidence.py", "progress_certificate.py")
    result = {
        "status": "AUDIT_ONLY" if args.audit_only else "IN_PROGRESS_FROZEN_HOLDOUT",
        "confidence": 1 - ALPHA, "relative_target": TARGET, "path_count_per_operator": PATHS,
        "checkpoints": CHECKPOINTS, "split_fractions": FRACTIONS,
        "source_sha256": {name: file_hash(HERE / name) for name in filenames},
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "jobs": args.jobs, "cases": {},
    }
    path = HERE / ("confirmatory_operator_audit.json" if args.audit_only else "confirmatory_results.json")

    def save():
        staging = path.with_suffix(".tmp")
        staging.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")), encoding="utf-8")
        staging.replace(path)

    save()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [pool.submit(run_case, index, args.audit_only) for index in range(len(CASES))]
        for job in as_completed(jobs):
            name, case = job.result()
            result["cases"][name] = case
            save()
    if not args.audit_only:
        holm_adjust(result["cases"])
        failed = sum(case["status"] != "complete" for case in result["cases"].values())
        result["status"] = "FROZEN_HOLDOUT_WITH_AUDIT_FAILURES" if failed else "FROZEN_HOLDOUT_COMPLETE"
    save()


if __name__ == "__main__":
    main()
