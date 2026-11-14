"""Exact-rational checks of the Galerkin and residual-angle formulas."""
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from flint import fmpq, fmpq_mat


def dot(x, y):
    return (x.transpose() * y)[0, 0]


def trace(matrix):
    return sum(matrix[i, i] for i in range(matrix.nrows()))


def main():
    rng = Random(38801)
    tests = {"positive_pivots": 0, "skipped_pivots": 0, "zero_residuals": 0,
             "positive_angles": 0, "comparison_with_exact_solve": 0}
    n = 5
    for _ in range(160):
        x = fmpq_mat(n, n, [rng.randrange(-3, 4) for _ in range(n * n)])
        ell = fmpq(1, rng.randrange(1, 500))
        identity = fmpq_mat(n, n, [int(i == j) for i in range(n) for j in range(n)])
        k = x.transpose() * x + ell * identity
        a = k.inv()
        sketch = fmpq_mat(n, n)
        basis = []
        for step in range(n + 2):
            g = fmpq_mat(n, 1, [rng.randrange(-7, 8) for _ in range(n)])
            ideal = (a - sketch) * g
            q = dot(g, ideal)
            assert q == dot(ideal, k * ideal) >= 0
            delta = dot(ideal, ideal) / q if q else fmpq(0)
            scale = trace(a - sketch)
            assert 0 <= delta <= scale
            tests["zero_residuals"] += int(q == 0)
            if step == 0:
                trial = fmpq_mat(n, 1)
            elif step % 2:
                error = fmpq_mat(n, 1, [rng.randrange(-2, 3) for _ in range(n)])
                trial = a * g + fmpq(1, rng.randrange(1, 25)) * error
            else:
                trial = a * g

            w = trial - sketch * k * trial
            kw = k * w
            energy = dot(w, kw)
            assert energy >= 0
            old_sketch = sketch
            if energy:
                tests["positive_pivots"] += 1
                assert all(dot(direction, kw) == 0 for direction in basis)
                basis.append(w)
                sketch = sketch + w * w.transpose() / energy
                q_lower = dot(g, w) ** 2 / energy
                progress = dot(w, w) / energy
                if step % 2 == 0:
                    assert w == ideal and progress == delta and q_lower == q
                    tests["comparison_with_exact_solve"] += 1
            else:
                tests["skipped_pivots"] += 1
                q_lower = fmpq(0)
                progress = fmpq(0)
            assert trace(sketch) - trace(old_sketch) == progress >= 0
            rho = g - k * sketch * g
            error2 = dot(rho, rho) / ell
            assert q_lower <= q <= q_lower + error2
            if q and energy:
                angle2 = (q - q_lower) / q
                slack2 = error2 / (q_lower + error2)
                assert 0 <= angle2 <= slack2 <= 1
                tests["positive_angles"] += int(angle2 > 0)
                if delta > progress:
                    assert (delta - progress) ** 2 <= scale ** 2 * angle2
            else:
                assert delta <= progress + scale
        if len(basis) == n:
            assert sketch == a
    path = Path(__file__)
    payload = {"cases": tests, "scope": "Exact deterministic algebra checks on integer probes and rational matrices.",
               "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest()}}
    path.with_name("inexact_galerkin_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(tests)


if __name__ == "__main__":
    main()
