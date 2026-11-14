"""Floating-point diagnostics for the inexact-inverse Galerkin certificate.

Each approximate solve contributes a K-orthogonal sketch update and a residual
envelope. The theoretical certificate requires exact operations or outward
rounding with a validated spectral lower bound. This program uses ordinary
floating point and a sparse LU reference for diagnostics.
"""

from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import LinearOperator, cg, splu
from threadpoolctl import threadpool_limits

from run_graph_holdout import inverse_trace, load_laplacian


HERE = Path(__file__).resolve().parent
SHIFTS = (0.01, 1.0)
TARGETS = (0.10, 0.05)
MODES = ('loose_cg', 'tight_cg', 'lu_reference')
TOLERANCES = {'loose_cg': 2e-2, 'tight_cg': 1e-7}
SEED_START = 930100
STAKE = 0.25
ALPHA = 0.05
CG_CAP = 800


def finite(value):
    return float(value) if math.isfinite(value) else None


class GalerkinPath:
    def __init__(self, matrix, spectral_lower, budget):
        self.matrix = matrix
        self.spectral_lower = spectral_lower
        self.basis = np.empty((matrix.shape[0], budget), order='F')
        self.images = np.empty_like(self.basis)
        self.rank = 0
        self.count = 0
        self.trace = 0.0
        self.z_lower = 0.0
        self.z_upper = 0.0
        self.angle_sum = 0.0
        self.interval = (0.0, math.inf)

    def observe(self, probe, approximation, exact_product=None, ideal_lu=False):
        """Append one observation. `exact_product` is reserved for diagnostics."""
        basis = self.basis[:, :self.rank]
        images = self.images[:, :self.rank]
        coefficients = basis.T @ probe
        t_before = self.trace
        # Reorthogonalize in the K inner product. Recompute the action on the
        # final w for an independent residual evaluation.
        image = self.matrix @ approximation
        w = approximation - basis @ (basis.T @ image)
        Kw = self.matrix @ w
        w -= basis @ (basis.T @ Kw)
        Kw = self.matrix @ w
        tau = float(w @ Kw)
        norm_sq = float(w @ w)
        if not (math.isfinite(tau) and math.isfinite(norm_sq)) or tau < 0:
            raise ArithmeticError('Nonfinite or negative K-energy')
        input_energy = float(approximation @ image)
        discarded = bool(tau <= (64 * np.finfo(float).eps)**2 * input_energy and norm_sq > 0)
        if discarded:
            # Choose the valid zero trial instead of normalizing a direction
            # lost to projection roundoff. It never implies exact exhaustion.
            w.fill(0.0)
            Kw.fill(0.0)
            tau = norm_sq = 0.0
        residual = probe - images @ coefficients - Kw
        delta = norm_sq / tau if tau > 0 else 0.0
        eta_sq = float(residual @ residual) / self.spectral_lower
        phi = 2 * float(probe @ w) - tau
        xi = min(1.0, math.sqrt(eta_sq / tau)) if tau > 0 else 1.0
        q_lower = max(0.0, phi)
        q_upper = phi + eta_sq

        reference_q = reference_delta = None
        if exact_product is not None:
            shadow_residual = exact_product - basis @ coefficients
            reference_q = float(probe @ shadow_residual)
            if reference_q > 0:
                reference_delta = float(shadow_residual @ shadow_residual) / reference_q
        if ideal_lu:
            # Benchmark the exact-oracle formula using the computed LU solve.
            if reference_q is None or reference_q < 0:
                raise ArithmeticError('Invalid LU reference quadratic form')
            q_lower = q_upper = reference_q
            xi = 0.0
        if not all(math.isfinite(x) for x in (delta, eta_sq, q_lower, q_upper, xi)) or q_upper < q_lower:
            raise ArithmeticError('Invalid residual enclosure')
        self.count += 1
        self.z_lower += t_before + q_lower
        self.z_upper += t_before + q_upper
        self.angle_sum += xi
        if tau > 0:
            scale = math.sqrt(tau)
            self.basis[:, self.rank] = w / scale
            self.images[:, self.rank] = Kw / scale
            self.trace += delta
            self.rank += 1
        return {'trace_before': t_before, 'delta': delta, 'eta_sq': eta_sq, 'tau': tau, 'xi': xi,
                'q_lower': q_lower, 'q_upper': q_upper, 'q_reference': reference_q,
                'delta_reference': reference_delta, 'discarded_direction': discarded, 'k_products': 3}

    def inspect(self):
        # Sticky running intersection of the anchor-zero intervals.
        a = math.log(2 / ALPHA) / STAKE + 4 * STAKE * self.angle_sum
        lower = max(self.trace, (self.z_lower - 4 * STAKE * self.trace) / (self.count + a))
        upper = ((self.z_upper + 4 * STAKE * self.trace) / (self.count - a)
                 if self.count > a else math.inf)
        self.interval = (max(self.interval[0], lower), min(self.interval[1], upper))
        return self.interval

    def orthogonality_error(self):
        basis = self.basis[:, :self.rank]
        images = self.images[:, :self.rank]
        return float(np.linalg.norm(basis.T @ images - np.eye(self.rank), ord='fro'))


def summarize_path(matrix, shift, theta, probes, lu_products, lu_times, mode):
    n, budget = probes.shape
    path = GalerkinPath(matrix, shift, budget)
    inverse_diag = 1 / matrix.diagonal()
    preconditioner = LinearOperator((n, n), matvec=lambda x: inverse_diag * x)
    iterations = []
    solver_calls = 0
    envelope_calls = 0
    lu_solves = 0
    elapsed = 0.0
    capped_cg = 0
    strict_q = material_q = strict_delta = material_delta = 0
    q_violation = delta_violation = 0.0
    q_widths = []
    angles = []
    limited_angles = 0
    failed_quality = 0
    zero_updates = 0
    coverage_failure = False
    stops = {str(target): None for target in TARGETS}
    ideal_lu = mode == 'lu_reference'
    for index, probe in enumerate(probes.T):
        step = index + 1
        reference = lu_products[:, index]
        start = perf_counter()
        if ideal_lu:
            approximation = reference
            lu_solves += 1
            elapsed += lu_times[index]
        else:
            applications = 0

            def k_action(x):
                nonlocal applications
                applications += 1
                return matrix @ x

            operator = LinearOperator((n, n), matvec=k_action, dtype=float)
            approximation, info = cg(operator, probe, rtol=TOLERANCES[mode], atol=0.0,
                                     maxiter=CG_CAP, M=preconditioner)
            if info < 0:
                raise ArithmeticError(f'CG failed with status {info}')
            capped_cg += info > 0
            solver_calls += applications
            iterations.append(applications)
        row = path.observe(probe, approximation, exact_product=reference, ideal_lu=ideal_lu)
        envelope_calls += row['k_products']
        if step % 4 == 0 or step == budget:
            lower, upper = path.inspect()
        elapsed += perf_counter() - start

        q = row['q_reference']
        q_gap = max(row['q_lower'] - q, q - row['q_upper'], 0.0)
        strict_q += q_gap > 0
        # LU and the float Galerkin computations are numerical references.
        material_q += q_gap > 1e-10 * max(1.0, abs(q), theta)
        q_violation = max(q_violation, q_gap / theta)
        if row['delta_reference'] is not None and not ideal_lu:
            allowance = max(0.0, theta - row['trace_before']) * row['xi']
            delta_gap = max(0.0, row['delta_reference'] - row['delta'] - allowance)
            strict_delta += delta_gap > 0
            material_delta += delta_gap > 1e-10 * max(1.0, row['delta_reference'], theta)
            delta_violation = max(delta_violation, delta_gap / theta)
        q_widths.append((row['q_upper'] - row['q_lower']) / theta)
        angles.append(row['xi'])
        limited_angles += row['xi'] == 1.0
        failed_quality += row['xi'] > 0.05 / (8 * STAKE) or row['eta_sq'] > row['delta']
        zero_updates += row['tau'] == 0.0
        if step % 4 and step != budget:
            continue
        coverage_failure |= not lower <= theta <= upper
        radius = (upper - lower) / (upper + lower) if 0 < lower <= upper < math.inf else math.inf
        for target in TARGETS:
            if stops[str(target)] is None and radius <= target:
                estimate = 2 * lower / (1 + lower / upper)
                stops[str(target)] = {'probes': step, 'cg_inner_matvecs': solver_calls,
                                      'envelope_matvecs': envelope_calls, 'lu_solves': lu_solves,
                                      'seconds': elapsed, 'relative_error': abs(estimate / theta - 1),
                                      'relative_radius': radius}
    return {'stops': stops, 'coverage_failure': coverage_failure, 'probes': budget,
            'cg_inner_matvecs': solver_calls, 'envelope_matvecs': envelope_calls, 'lu_solves': lu_solves,
            'median_cg_iterations': float(np.median(iterations)) if iterations else None,
            'max_cg_iterations': max(iterations) if iterations else None, 'capped_cg_solves': capped_cg,
            'diagnostic_seconds_excluding_reference': elapsed,
            'strict_q_enclosure_exceedances': strict_q, 'material_q_enclosure_exceedances': material_q,
            'strict_delta_enclosure_exceedances': strict_delta, 'material_delta_enclosure_exceedances': material_delta,
            'max_q_exceedance_divided_by_trace': q_violation, 'max_delta_exceedance_divided_by_trace': delta_violation,
            'median_q_enclosure_width_divided_by_trace': float(np.median(q_widths)),
            'max_q_enclosure_width_divided_by_trace': max(q_widths),
            'median_angle_bound': float(np.median(angles)), 'max_angle_bound': max(angles),
            'angle_cap_hits': limited_angles, 'strong_stopping_quality_gate_failures': failed_quality if not ideal_lu else 0,
            'zero_updates': zero_updates, 'orthogonality_frobenius_error': path.orthogonality_error(),
            'final_radius': finite(radius), 'final_sketch_fraction': path.trace / theta}


def diagnostic_cases():
    diagonal = np.array([0.01, 1.0, 30.0, 1000.0])
    matrix = np.diag(diagonal)
    g = np.array([1.0, -1.5, 2.0, -0.8])
    lu = g / diagonal
    path = GalerkinPath(matrix, diagonal.min(), budget=3)
    zero = path.observe(g, np.zeros_like(g), exact_product=lu)
    zero_interval = path.inspect()
    orthogonal_direction = np.array([0.0, 0.0, 1.0, 0.0])
    angle_cap = path.observe(g, orthogonal_direction, exact_product=lu)
    nearly_dependent = path.observe(g, orthogonal_direction, exact_product=lu)
    assert zero['tau'] == zero['delta'] == 0 and zero['xi'] == 1
    assert zero_interval[0] == 0 and math.isinf(zero_interval[1])
    assert angle_cap['xi'] == nearly_dependent['xi'] == 1
    assert nearly_dependent['tau'] == 0 and path.rank == 1
    return {'purpose': 'Deterministic algebra checks only. Repeated probes here are not a coverage experiment.',
            'zero_initial_solve': zero, 'angle_bound_saturates': angle_cap,
            'repeated_sketch_direction': nearly_dependent, 'zero_initial_update_stops': False}


def summarize(records, mode):
    rows = [r[mode] for r in records if mode in r and 'error' not in r[mode]]
    def median(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.median(values)) if values else None

    stops = {}
    for target in TARGETS:
        successful = [r['stops'][str(target)] for r in rows if r['stops'][str(target)] is not None]
        stops[str(target)] = {'count': len(successful), 'incorrect': sum(r['relative_error'] > target for r in successful)}
        for k in ('probes', 'cg_inner_matvecs', 'envelope_matvecs', 'lu_solves', 'seconds'):
            stops[str(target)][f'median_{k}'] = float(np.median([r[k] for r in successful])) if successful else None
    summary = {'completed_paths': len(rows), 'numerical_failures': len(records) - len(rows), 'stops': stops}
    for key in ('final_radius', 'median_cg_iterations', 'median_angle_bound',
                'median_q_enclosure_width_divided_by_trace', 'diagnostic_seconds_excluding_reference', 'cg_inner_matvecs'):
        summary[f'median_path_{key}'] = median(key)
    for key in ('coverage_failure', 'capped_cg_solves', 'strict_q_enclosure_exceedances',
                'material_q_enclosure_exceedances', 'strict_delta_enclosure_exceedances',
                'material_delta_enclosure_exceedances', 'angle_cap_hits',
                'strong_stopping_quality_gate_failures', 'zero_updates'):
        summary[f'total_{key}'] = sum(r[key] for r in rows)
    for key in ('max_cg_iterations', 'max_angle_bound', 'max_q_exceedance_divided_by_trace',
                'max_delta_exceedance_divided_by_trace', 'orthogonality_frobenius_error'):
        values = [r[key] for r in rows if r[key] is not None]
        summary[f'largest_path_{key}'] = max(values) if values else None
    return summary


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--budget', type=int, default=320)
    args = parser.parse_args()
    if args.trials < 1 or args.budget < 4:
        parser.error('Use at least one trial and four probes')
    result = {'status': 'EXPLORATORY_FLOAT_DIAGNOSTIC', 'target_alpha': ALPHA, 'stake': STAKE,
              'trials_per_shift': args.trials, 'budget': args.budget, 'seeds_start': SEED_START,
              'cg_relative_tolerances': TOLERANCES, 'cg_iteration_cap': CG_CAP,
              'source_sha256': {f.name: sha256(f.read_bytes()).hexdigest()
                                for f in (Path(__file__), HERE / 'run_graph_holdout.py')},
              'scope': ['An exact arithmetic envelope is evaluated in ordinary floating point.',
                        'Both operators are shifted inverse Laplacians of the same SNAP graph.',
                        'Ideal LU mode treats the sparse LU result as an exact product for its control interval.',
                        'A diagnostic LU product is shared across all methods for each probe and omitted from inexact costs.',
                        'Reported timing includes projection and envelope diagnostics but excludes offline reference solves for inexact paths.',
                        'CG inner matvec count includes every linear-operator callback, and three K products per probe are counted separately for the Galerkin envelope.',
                        'The formal per-probe quality gate is measured, without rejecting probes that fail it. Those paths still use the wider evidence envelope.',
                        'Product count, residual/reference computations and in-host timing are distinct metrics.'],
              'diagnostic_cases': diagnostic_cases(), 'shifts': {}}
    output = HERE / 'inexact_galerkin_results.json'
    with threadpool_limits(limits=1):
        laplacian, audit = load_laplacian()
        result['data_audit'] = audit
        n = laplacian.shape[0]
        if args.budget >= n:
            parser.error('Choose a budget below the operator dimension for this comparison')
        for shift in SHIFTS:
            matrix = laplacian + shift * eye(n, format='csc')
            lu_factor = splu(matrix)
            theta = inverse_trace(lu_factor, n)
            records = []
            print(f'shift {shift:g}, trace reference {theta:g}', flush=True)
            for offset in range(args.trials):
                seed = SEED_START + offset
                probes = np.random.default_rng(seed).standard_normal((args.budget, n)).T
                lu_products = np.empty_like(probes)
                lu_times = []
                for index, probe in enumerate(probes.T):
                    start = perf_counter()
                    lu_products[:, index] = lu_factor.solve(probe)
                    lu_times.append(perf_counter() - start)
                record = {'seed': seed}
                for j in range(len(MODES)):
                    mode = MODES[(offset + j) % len(MODES)]
                    try:
                        record[mode] = summarize_path(matrix, shift, theta, probes, lu_products, lu_times, mode)
                    except ArithmeticError as error:
                        record[mode] = {'error': str(error)}
                records.append(record)
                summaries = {mode: summarize(records, mode) for mode in MODES}
                result['shifts'][str(shift)] = {'trace_reference': theta, 'spectral_lower_exact_model': shift,
                                               'records': records, 'summary': summaries,
                                               'complete': offset + 1 == args.trials}
                temp = output.with_suffix('.tmp')
                temp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
                temp.replace(output)
                print(f'  {offset+1}/{args.trials}: ' + ', '.join(f'{mode} {record[mode].get("final_radius")}' for mode in MODES), flush=True)


if __name__ == '__main__':
    main()
