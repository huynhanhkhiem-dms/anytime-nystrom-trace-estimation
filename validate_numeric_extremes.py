"""Regression checks for overflow in the binary64 reporting helpers."""

from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path

from progress_certificate import Anchor, invert_progress, relative_center, relative_radius


HERE = Path(__file__).resolve().parent


def main():
    records = []
    for lower, upper in ((6e307, 1.5e308), (1e308, 1.5e308), (1.7e308, 1.7e308)):
        left, right = Fraction(lower), Fraction(upper)
        radius = relative_radius(lower, upper)
        center = relative_center(lower, upper)
        assert math.isclose(radius, float((right - left) / (right + left)), rel_tol=2e-16)
        assert math.isclose(center, float(2 * left * right / (left + right)), rel_tol=2e-16)
        records.append({"interval": [lower, upper], "relative_radius": radius, "harmonic_center": center})
    alpha = 1e-310
    lower, upper = invert_progress(Anchor(0, 0.0, 0.0, alpha), 5000, 5000.0, 0.5, c=1/3)
    assert 0 < lower < 1 < upper < math.inf
    sources = (Path(__file__), HERE / "progress_certificate.py")
    result = {"status": "PASS", "records": records, "subnormal_risk": {"alpha": alpha, "interval": [lower, upper]},
              "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in sources}}
    (HERE / "numeric_extremes_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Passed four overflow regression checks.")


if __name__ == "__main__":
    main()
