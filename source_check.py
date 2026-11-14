"""Resolve source provenance for the recorded experiments."""

from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def source_matches(name, expected):
    candidate = HERE / name
    if not candidate.is_file():
        candidate = HERE / "results" / name
    if not candidate.is_file():
        return False
    observed = sha256(candidate.read_bytes()).hexdigest()
    if observed == expected:
        return True
    mapping = json.loads((HERE / "source_provenance.json").read_text(encoding="utf-8"))
    entry = mapping.get(candidate.relative_to(HERE).as_posix(), {})
    return observed == entry.get("delivered_sha256") and expected in entry.get("recorded_sha256", ())
