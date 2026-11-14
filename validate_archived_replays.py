"""Rebuild archived stopping decisions from archived sufficient statistics."""

from hashlib import sha256
import json
import math
from pathlib import Path

from source_check import source_matches

from anchored_norm import invert_frozen_norm
from progress_certificate import Anchor, invert_progress
from run_controlled_timing import summarize as summarize_timing
from run_stake_ablation import METHODS, new_result, summarize, update_result


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" if (HERE / "results").is_dir() else HERE


def read(name):
    return json.loads((RESULTS / name).read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def replay(trial, reference, levels, gap):
    actual = {str(alpha): {method: new_result() for method in METHODS} for alpha in levels}
    running = {alpha: dict.fromkeys(METHODS, (0, math.inf)) for alpha in levels}
    captured = total = squared = 0.0
    reports = 0
    for step, (q, v, delta) in enumerate(trial["probe_scores"], 1):
        assert q > 0 and v > 0 and delta == v / q
        total += captured + q
        captured += delta
        squared += v
        if step % gap:
            continue
        reports += 1
        for alpha in levels:
            variance_limit = 4 * (math.log(3) + math.log(2 / alpha))
            lo, hi = invert_frozen_norm(step, total, captured, squared, alpha / 2)
            intervals = {
                "progress": invert_progress(Anchor(0, 0.0, 0.0, alpha), step, total, captured, c=1/3),
                "norm_common_stake": invert_frozen_norm(step, total, captured, squared, alpha, 1/3),
                "frozen_norm": invert_frozen_norm(step, total, captured, squared, alpha),
                "reinforced_norm": (max(lo, math.sqrt(squared / variance_limit)), hi),
            }
            for name, (lo, hi) in intervals.items():
                if name != "reinforced_norm":
                    past_lo, past_hi = running[alpha][name]
                    lo, hi = running[alpha][name] = max(lo, past_lo), min(hi, past_hi)
                update_result(actual[str(alpha)][name], (lo, hi), step, reference)
    assert actual == trial["methods"], trial["seed"]
    return reports


def main():
    ablation = read("stake_ablation_results.json")
    original = RESULTS / "anchor_norm_comparison.json"
    assert source_matches(original.name, ablation["archive_sha256"])
    assert source_matches("run_stake_ablation.py", ablation["script_sha256"])
    timing = read("controlled_timing_results.json")
    flat = read("single_report_results.json")
    for result in (timing, flat):
        for name, expected in result["source_sha256"].items():
            assert source_matches(name, expected), name
    paths = reports = 0
    for case in ablation["cases"].values():
        for trial in case["trials"]:
            reports += replay(trial, case["audit"]["trace_reference"], ablation["alphas"], ablation["report_every"])
            paths += 1
        assert summarize(case["trials"]) == case["summary"]
    timing_runs = 0
    for case in timing["cases"].values():
        assert summarize_timing(case["records"]) == (case["methods"], case["paired_frozen_norm_vs"])
        timing_runs += sum(len(record["methods"]) for record in case["records"])
    output = {"paths_reconstructed": paths, "report_points_reconstructed": reports,
              "method_interval_checks": reports * len(METHODS) * len(ablation["alphas"]),
              "timing_runs_reanalyzed": timing_runs,
              "source_sha256": {Path(__file__).name: digest(Path(__file__))}}
    (HERE / "archived_replay_validation.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
