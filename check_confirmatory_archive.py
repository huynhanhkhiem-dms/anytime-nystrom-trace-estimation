"""Independent integrity and endpoint replay check for the held-out study."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from threadpoolctl import threadpool_limits

from source_check import source_matches

import run_confirmatory as frozen
from core import TraceEvidence


ROOT = Path(__file__).resolve().parent


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_statistics(observed, recorded):
    if isinstance(recorded, dict):
        return observed.keys() == recorded.keys() and all(
            equal_statistics(observed[key], value) for key, value in recorded.items()
        )
    if isinstance(recorded, list):
        return len(observed) == len(recorded) and all(
            equal_statistics(left, right) for left, right in zip(observed, recorded, strict=True)
        )
    if isinstance(recorded, float):
        return math.isclose(observed, recorded, rel_tol=1e-12, abs_tol=1e-15)
    return observed == recorded


def validate_case(raw, case, index, checks):
    audit = raw["audit"]
    trials = raw["trials"]
    assert len(trials) == frozen.PATHS
    assert [row["seed"] for row in trials] == list(range(2500000 + index * 1000, 2500000 + index * 1000 + frozen.PATHS))
    assert all(row["adaptive_seed"] - row["seed"] == 1000000 for row in trials)
    assert audit["reference_relative_disagreement"] <= frozen.REFERENCE_RTOL
    assert audit.get("max_audit_product_relative_discrepancy", 0) <= frozen.REFERENCE_RTOL
    for name in frozen.METHODS:
        path_rows = [row["shared"][name] for row in trials]
        paid = []
        crossing_count = 0
        complete_replay_exclusion_count = 0
        incomplete_replay_exclusion_count = 0
        for row in path_rows:
            checkpoint_rows = row["checkpoints"]
            assert [item[0] for item in checkpoint_rows] == list(frozen.CHECKPOINTS)[:len(checkpoint_rows)]
            if row["numerical_failure"] is None:
                assert len(checkpoint_rows) == len(frozen.CHECKPOINTS)
            qualified = []
            for k, lo, hi, r, exclusion in checkpoint_rows:
                if lo is not None and hi is not None:
                    assert bool(not lo <= audit["trace_reference"] <= hi) == exclusion
                    if 0 < lo <= hi:
                        assert math.isclose((hi - lo) / (hi + lo), r, abs_tol=1e-15, rel_tol=1e-12)
                        if r <= frozen.TARGET:
                            qualified.append((k, lo, hi))
                checks["recorded_checkpoints_checked"] += 1
            assert row["coverage_failure"] == any(item[4] for item in checkpoint_rows)
            if row["numerical_failure"] is None:
                complete_replay_exclusion_count += row["coverage_failure"]
            else:
                incomplete_replay_exclusion_count += row["coverage_failure"]
            if qualified:
                k, lo, hi = qualified[0]
                assert row["stop"]["queries"] == k
                center = 2 * lo / (1 + lo / hi)
                assert math.isclose(center, row["stop"]["estimate"], rel_tol=1e-12)
                error = abs(center / audit["trace_reference"] - 1)
                assert math.isclose(error, row["stop"]["relative_error"], abs_tol=1e-12)
                paid.append(k)
                crossing_count += 1
            else:
                assert row["stop"] is None
                paid.append(frozen.CAP)
        summary = raw["summary"][name]
        assert summary["completions"] == crossing_count
        assert summary["complete_replay_exclusions"] == complete_replay_exclusion_count
        assert summary["incomplete_replay_exclusions"] == incomplete_replay_exclusion_count
        assert math.isclose(summary["restricted_mean_queries"], mean(paid), abs_tol=1e-12)
        checks["cost_summaries_checked"] += 1
    matvec, theta, replay_audit = frozen.load_case(case)
    assert audit["sha256"] == replay_audit["sha256"]
    slow = frozen.run_path(matvec, audit["dimension"], trials[0]["seed"], theta)
    for name in frozen.METHODS:
        stored = trials[0]["shared"][name]
        replayed = slow[name]
        assert (stored["stop"] is None) == (replayed["stop"] is None)
        if stored["stop"] is not None:
            assert stored["stop"]["queries"] == replayed["stop"]["queries"]
        assert stored["coverage_failure"] == replayed["coverage_failure"]
        assert stored["numerical_failure"] == replayed["numerical_failure"]
        for old, new in zip(stored["checkpoints"], replayed["checkpoints"], strict=True):
            assert old[0] == new[0] and old[-1] == new[-1]
            for left, right in zip(old[1:3], new[1:3]):
                assert (left is None) == (right is None)
                if left is not None:
                    discrepancy = abs(left - right) / theta
                    assert discrepancy < 1e-9
                    checks["maximum_replay_endpoint_discrepancy_over_trace"] = max(
                        checks["maximum_replay_endpoint_discrepancy_over_trace"], discrepancy)
                    checks["finite_replay_endpoints_checked"] += 1
    checks["replayed_cases"].append(case.name)


def main():
    data_dir = ROOT / "results" if (ROOT / "results").is_dir() else ROOT
    path = data_dir / "confirmatory_results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "FROZEN_HOLDOUT_COMPLETE"
    checks = {
        "source_hashes_checked": len(result["source_sha256"]),
        "results_sha256": source_hash(path),
        "recorded_checkpoints_checked": 0,
        "finite_replay_endpoints_checked": 0,
        "cost_summaries_checked": 0,
        "maximum_replay_endpoint_discrepancy_over_trace": 0.0,
        "replayed_cases": [],
        "one_shared_path_per_case_replayed_with_slow_reference_evidence": True,
    }
    for filename, expected in result["source_sha256"].items():
        assert source_matches(filename, expected), filename
    fresh = {}
    for index, case in enumerate(frozen.CASES):
        archived = result["cases"][case.name]
        summary, contrasts = frozen.summarize(archived["trials"], index)
        assert equal_statistics(summary, archived["summary"])
        fresh[case.name] = {"contrasts": contrasts}
    frozen.holm_adjust(fresh)
    for name, case in fresh.items():
        assert equal_statistics(case["contrasts"], result["cases"][name]["contrasts"])
    checks["paired_contrasts_recomputed"] = 2 * len(fresh)
    checks["source_sha256"] = {Path(__file__).name: source_hash(Path(__file__))}
    frozen.FastTraceEvidence = TraceEvidence
    with threadpool_limits(limits=1):
        for index, case in enumerate(frozen.CASES):
            validate_case(result["cases"][case.name], case, index, checks)
    (ROOT / "confirmatory_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
