"""Check that verified Galerkin runs stop spending probes at full span."""

from hashlib import sha256
import json
from pathlib import Path

from flint import arb, arb_mat, ctx
from scipy.sparse import csr_matrix

from run_verified_norm_comparison import PRECISION, run_path


HERE = Path(__file__).resolve().parent
CASES = (
    ([[1]], 1, 1),
    ([[1, 0, 0], [0, 2, 0], [0, 0, 4]], 7, 4),
    ([[3, 1, 0], [1, 3, 0], [0, 0, 2]], 5, 4),
)


def endpoint(dyadic):
    mantissa, exponent = dyadic
    return arb((int(mantissa), exponent))


def main():
    ctx.prec = PRECISION
    records = []
    for entries, numerator, denominator in CASES:
        dimension = len(entries)
        exact = arb_mat(entries)
        reference = arb(numerator) / denominator
        for seed in (7, 19, 114):
            for budget in (dimension - 1, 2 * dimension + 2):
                result = run_path(csr_matrix(entries, dtype=float), exact, arb(1), reference,
                                  (1e-9,), seed, budget=budget)
                full_span = budget >= dimension
                assert result["probes_executed"] == min(budget, dimension)
                assert (result["full_span_trace_enclosure"] is not None) == full_span
                assert result["solve_attempts"] == result["probes_executed"]
                assert not any(result["reference_exclusion_checks"].values())
                if full_span:
                    enclosure = result["full_span_trace_enclosure"]
                    lower = endpoint(enclosure["lower_dyadic"])
                    upper = endpoint(enclosure["upper_dyadic"])
                    assert 0 < lower <= reference <= upper
                    assert (upper - lower) / (upper + lower) <= arb(1) / 10
                    for stop in result["stops"].values():
                        assert stop["outer_probes"] == dimension
                        assert stop["lower_dyadic"] == enclosure["lower_dyadic"]
                        assert stop["upper_dyadic"] == enclosure["upper_dyadic"]
                else:
                    assert all(stop is None for stop in result["stops"].values())
                del result["seconds"]
                records.append({"matrix": entries, "budget": budget, "result": result})
    # Low precision may prove full span while leaving a wide trace enclosure.
    ctx.prec = 5
    identity = [[1, 0], [0, 1]]
    try:
        wide = run_path(csr_matrix(identity, dtype=float), arb_mat(identity), arb(1), arb(2), (1e-9,), 7, budget=12)
        enclosure = wide["full_span_trace_enclosure"]
        assert 2 <= wide["probes_executed"] < 12 and enclosure is not None
        assert endpoint(enclosure["lower_dyadic"]) <= 2 <= endpoint(enclosure["upper_dyadic"])
        assert all(stop is None for stop in wide["stops"].values())
        del wide["seconds"]
    finally:
        ctx.prec = PRECISION
    sources = (Path(__file__), HERE / "run_verified_norm_comparison.py", HERE / "run_verified_galerkin.py")
    summary = {"status": "PASS", "precision_bits": PRECISION, "records": records,
               "low_precision": {"precision_bits": 5, "budget": 12, "result": wide},
               "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in sources}}
    (HERE / "full_span_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Passed {len(records)} full-span and pre-span checks, plus the low-precision check.")


if __name__ == "__main__":
    main()
