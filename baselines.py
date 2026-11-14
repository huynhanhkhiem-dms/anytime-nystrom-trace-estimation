"""Independent fixed-budget trace baselines for the exploratory pilot.

All public estimators have signature (matvec, n, budget, rng, ...)->dict.
``matvec`` receives a length-n vector; ``rng`` is numpy.random.Generator.
Counts are scalar-vector oracle products, including products on a learned basis.
These are fixed-budget estimates, not anytime-valid procedures.

Sources (formula implementations, not copies of the authors' source code):
  Hutch++: Meyer et al., SOSA 2021, arXiv:2010.09649, Algorithm 1.
  Nystrom++: Persson et al., SIMAX 2022, arXiv:2109.10659, Algorithm 5.
  XNysTrace: Epperly et al., SIMAX 2024, arXiv:2301.07825, (1.9),
            test-vector normalization in section 2.3, posterior SE in (3.1).
  Gaussian fixed-time interval: elementary weighted-chi-square Chernoff bound;
            background Avron--Toledo, JACM 2011, doi:10.1145/1944345.1944349.

Floating-point low-rank computations use a symmetric Gram eigendecomposition.
Tiny eigenvalues are truncated. No finite-precision certificate is claimed.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.linalg import eigh, qr
from scipy.optimize import brentq
from scipy.stats import norm

VectorOracle = Callable[[np.ndarray], np.ndarray]


class _CountedOracle:
    def __init__(self, matvec: VectorOracle, n: int):
        self.matvec = matvec
        self.n = n
        self.count = 0

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            if x.shape[1] == 0:
                return np.empty((self.n, 0))
            return np.column_stack([self.apply(x[:, j]) for j in range(x.shape[1])])
        if x.shape != (self.n,):
            raise ValueError("The oracle input must have shape (n,).")
        y = np.asarray(self.matvec(x), dtype=float)
        self.count += 1
        if y.shape != (self.n,) or not np.all(np.isfinite(y)):
            raise ValueError("The oracle must return a finite vector of shape (n,).")
        return y


def _validate(n: int, budget: int, minimum: int = 1) -> None:
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(budget, (int, np.integer)) or budget < minimum:
        raise ValueError(f"budget must be an integer at least {minimum}")


def _se(samples: np.ndarray) -> float:
    return float(np.std(samples, ddof=1) / math.sqrt(samples.size)) if samples.size > 1 else math.nan


def gaussian_psd_fixed_interval(estimate: float, probes: int, delta: float = 0.05) -> tuple[float, float]:
    """Fixed-time 1-delta confidence interval for exact Gaussian PSD probing.

    Let H be the mean of m independent g'Ag, A PSD, tau=tr(A). For s<1/2,
    E exp(s*g'Ag/tau) <= (1-2s)^(-1/2), since
    product_j(1-2s*lambda_j/tau) >= 1-2s. This holds for s<0 too.
    Chernoff gives P(H/tau>=u) <= exp[-m*(u-1-log u)/2] for u>1,
    and the analogous lower-tail bound for 0<u<1. Allocate delta/2 per
    tail, solve the scalar equations, and invert H/tau to obtain this CI.
    The guarantee is not simultaneous over probe counts or adaptive stops.
    It assumes fixed A, exact oracle products and unnormalized Gaussian probes.
    """
    if probes < 1 or not 0 < delta < 1:
        raise ValueError("probes >= 1 and 0 < delta < 1 are required")
    if estimate < 0 or not math.isfinite(estimate):
        raise ValueError("A PSD Gaussian estimate must be finite and nonnegative")
    if estimate == 0:
        return (0.0, 0.0)
    c = 2.0 * (math.log(2.0) - math.log(delta)) / probes
    # Parameterization u=exp(y) avoids near-zero lower-root underflow.
    upper_y = 1.0
    while math.expm1(upper_y) - upper_y < c:
        upper_y *= 2.0
    y_hi = brentq(lambda y: math.expm1(y) - y - c, 0.0, upper_y)
    y_lo = brentq(lambda y: math.expm1(-y) + y - c, 0.0, c + 1.0)
    lower = estimate * math.exp(-y_hi)
    upper = estimate * math.exp(y_lo) if y_lo < math.log(np.finfo(float).max) - math.log(estimate) else math.inf
    return (float(lower), float(upper))


def gaussian_hutchinson(matvec: VectorOracle, n: int, budget: int, rng: np.random.Generator,
                        delta: float = 0.05) -> dict:
    """Gaussian mean; nominal normal SE interval is explicitly heuristic."""
    _validate(n, budget)
    oracle = _CountedOracle(matvec, n)
    omega = rng.standard_normal((n, budget))
    image = oracle.apply(omega)
    samples = np.sum(omega * image, axis=0)
    estimate = float(samples.mean())
    se = _se(samples)
    radius = float(norm.ppf(1.0 - delta / 2.0)) * se
    rigorous = gaussian_psd_fixed_interval(max(0.0, estimate), budget, delta) if estimate >= 0 else None
    return {"method": "Gaussian Hutchinson", "estimate": estimate, "se": se,
            "matvecs": oracle.count, "requested_budget": int(budget),
            "nominal_normal_interval": [estimate - radius, estimate + radius],
            "nominal_normal_interval_kind": "heuristic normal approximation; not optional-stopping valid",
            "rigorous_fixed_interval": list(rigorous) if rigorous is not None else None,
            "rigorous_fixed_interval_kind": "Gaussian PSD Chernoff fixed-time only; exact products assumed",
            "delta": float(delta)}


def _orthogonal_basis(y: np.ndarray) -> np.ndarray:
    """Rank-revealing basis; avoids inventing columns after rank exhaustion."""
    if y.shape[1] == 0:
        return np.empty((y.shape[0], 0))
    q, r, _ = qr(y, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0 or diagonal[0] == 0:
        return q[:, :0]
    tolerance = np.finfo(float).eps * max(y.shape) * diagonal[0]
    rank = int(np.count_nonzero(diagonal > tolerance))
    return q[:, :rank]


def _nystrom_factor(omega: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, dict]:
    """F satisfying B=FF' ~= Y(omega'Y)^dagger Y', using only stored Y."""
    if omega.shape[1] == 0:
        return np.empty((omega.shape[0], 0)), {"rank": 0, "discarded_gram_eigenvalues": 0}
    h = omega.T @ image
    h = (h + h.T) / 2.0
    eig, vectors = eigh(h)
    scale = float(np.max(np.abs(eig), initial=0.0))
    tol = np.finfo(float).eps * max(omega.shape) * scale
    if eig[0] < -32.0 * max(tol, np.finfo(float).tiny):
        raise ValueError("Nyström Gram matrix is materially indefinite; PSD assumption failed")
    active = eig > tol
    factor = (image @ vectors[:, active]) / np.sqrt(eig[active])[None, :]
    return factor, {"rank": int(active.sum()), "discarded_gram_eigenvalues": int((~active).sum()),
                    "gram_rank_tolerance": float(tol)}


def hutchpp(matvec: VectorOracle, n: int, budget: int, rng: np.random.Generator) -> dict:
    """Hutch++: k range probes, r basis products, remaining residual probes.

    For a full-rank k-column sketch r=k=floor(budget/3), matching the paper's
    equal three-way split when budget is divisible by three. Rank exhaustion
    reduces the second-stage cost; any remaining budget goes to residual probes.
    """
    _validate(n, budget, 3)
    oracle = _CountedOracle(matvec, n)
    k = min(n, budget // 3)
    omega = rng.standard_normal((n, k))
    basis = _orthogonal_basis(oracle.apply(omega))
    basis_image = oracle.apply(basis)
    low_trace = float(np.sum(basis * basis_image))
    residual_probes = budget - oracle.count
    probes = rng.standard_normal((n, residual_probes))
    probes -= basis @ (basis.T @ probes)
    residual_image = oracle.apply(probes)
    samples = np.sum(probes * residual_image, axis=0)
    estimate = low_trace + float(samples.mean())
    return {"method": "Hutch++", "estimate": estimate, "se": _se(samples),
            "se_kind": "conditional fixed-sketch posterior SE; not a confidence guarantee",
            "matvecs": oracle.count, "requested_budget": int(budget),
            "sketch_probes": k, "basis_rank": int(basis.shape[1]),
            "basis_products": int(basis.shape[1]), "residual_probes": residual_probes}


def nystrompp(matvec: VectorOracle, n: int, budget: int, rng: np.random.Generator) -> dict:
    """Nyström++ (Algorithm 5), Gaussian sketch and independent residual block.

    An odd budget uses floor(budget/2) sketch probes and ceil(budget/2)
    residual probes, with the residual mean divided by its actual probe count.
    """
    _validate(n, budget, 2)
    oracle = _CountedOracle(matvec, n)
    k = min(n, budget // 2)
    omega = rng.standard_normal((n, k))
    image = oracle.apply(omega)
    factor, diagnostics = _nystrom_factor(omega, image)
    low_trace = float(np.sum(factor * factor))
    residual_probes = budget - oracle.count
    probes = rng.standard_normal((n, residual_probes))
    residual_image = oracle.apply(probes) - factor @ (factor.T @ probes)
    samples = np.sum(probes * residual_image, axis=0)
    return {"method": "Nyström++", "estimate": low_trace + float(samples.mean()), "se": _se(samples),
            "se_kind": "conditional fixed-sketch posterior SE; not a confidence guarantee",
            "matvecs": oracle.count, "requested_budget": int(budget),
            "sketch_probes": k, "residual_probes": residual_probes, "nystrom": diagnostics}


def _xnystrace_from_probes(omega: np.ndarray, image: np.ndarray, improved: bool = True) -> tuple[np.ndarray, dict]:
    """Compute leave-one-out estimates using stored products, with no oracle calls.

    The full-Gram inverse identity gives B_{-i}=B-s_i s_i', where
    s_i=Y H^{-1}e_i/sqrt((H^{-1})_ii). The residual quadratic is
    1/(H^{-1})_ii. Improved normalization multiplies it by
    (n-m+1)*(G^{-1})_ii, G=omega'omega. This is the paper's
    projected-probe normalization expressed through Gram inverses.
    A direct leave-one-out fallback handles numerically rank-deficient sketches.
    """
    n, m = omega.shape
    if m > n:
        raise ValueError("XNysTrace uses at most n probes")
    h = omega.T @ image
    h = (h + h.T) / 2.0
    factor, diagnostics = _nystrom_factor(omega, image)
    gram_eig = np.linalg.eigvalsh(h)
    threshold = np.finfo(float).eps * max(omega.shape) * np.max(np.abs(gram_eig), initial=0.0)
    full_rank = bool(np.min(gram_eig, initial=math.inf) > 64.0 * threshold)
    if full_rank:
        inverse_h = np.linalg.solve(h, np.eye(m))
        inverse_diag = np.diag(inverse_h)
        columns = image @ inverse_h
        removed_trace = np.sum(columns * columns, axis=0) / inverse_diag
        residual_quad = 1.0 / inverse_diag
        if improved:
            gram = omega.T @ omega
            inverse_g = np.linalg.solve(gram, np.eye(m))
            residual_quad *= (n - m + 1) * np.diag(inverse_g)
        estimates = np.sum(factor * factor) - removed_trace + residual_quad
        diagnostics["implementation"] = "full-Gram leave-one-out identity"
    else:
        estimates = _xnystrace_direct(omega, image, improved)
        diagnostics["implementation"] = "direct leave-one-out rank-deficient fallback"
    return np.asarray(estimates), diagnostics


def _xnystrace_direct(omega: np.ndarray, image: np.ndarray, improved: bool = True) -> np.ndarray:
    """Independent direct formula for fallback and numerical validation."""
    n, m = omega.shape
    estimates = np.empty(m)
    for i in range(m):
        keep = np.arange(m) != i
        old_probes = omega[:, keep]
        factor, _ = _nystrom_factor(old_probes, image[:, keep])
        low_trace = float(np.sum(factor * factor))
        vector = omega[:, i]
        residual_quad = float(vector @ image[:, i] - np.sum((factor.T @ vector) ** 2))
        if improved:
            previous_basis = _orthogonal_basis(old_probes)
            projected = vector - previous_basis @ (previous_basis.T @ vector)
            dimension = n - previous_basis.shape[1]
            residual_quad *= dimension / float(projected @ projected)
        estimates[i] = low_trace + residual_quad
    return estimates


def xnystrace(matvec: VectorOracle, n: int, budget: int, rng: np.random.Generator,
             improved: bool = True) -> dict:
    """XNysTrace, with optional projected-sphere normalization (recommended).

    The leave-one-out posterior SE is heuristic because terms are correlated.
    At most n probes are used; extra budget beyond n is reported as unused.
    """
    _validate(n, budget)
    oracle = _CountedOracle(matvec, n)
    m = min(n, budget)
    omega = rng.standard_normal((n, m))
    if improved:
        omega *= math.sqrt(n) / np.linalg.norm(omega, axis=0)[None, :]
    image = oracle.apply(omega)
    samples, diagnostics = _xnystrace_from_probes(omega, image, improved=improved)
    return {"method": "XNysTrace improved" if improved else "XNysTrace Gaussian",
            "estimate": float(np.mean(samples)), "se": _se(samples),
            "se_kind": "heuristic correlated leave-one-out posterior SE; no confidence guarantee",
            "matvecs": oracle.count, "requested_budget": int(budget),
            "unused_budget": int(budget - m), "nystrom": diagnostics,
            "probe_distribution": "independent sphere, projected normalization" if improved else "Gaussian"}


# Spelling aliases for scripts using descriptive names.
hutchinson_gaussian = gaussian_hutchinson
nystrom_plus_plus = nystrompp

