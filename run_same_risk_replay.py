"""Compare four frozen-norm reporting rules on the stored archived probe scores."""

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations
import json
from math import ceil, exp, expm1, inf, isfinite, log, sqrt
from pathlib import Path
from statistics import mean, median

from anchored_norm import invert_frozen_norm
from progress_certificate import relative_center, relative_radius


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" if (HERE / "results").is_dir() else HERE
STAKE = 0.5
B_MINUS = (expm1(2 * STAKE) - 2 * STAKE) / (2 * STAKE)
B_SUM = STAKE + B_MINUS
GAMMA = max(exp(B_SUM), (1 + B_MINUS) / (1 - STAKE))
RAW = "raw_current"
SAME = "same_risk_current"
SPLIT = "split_risk_current"
RUNNING = "running_norm"
METHODS = (RAW, SAME, SPLIT, RUNNING)
UNBOUNDED = (0.0, inf)


def prop1_limit(alpha, gap):
    a = log(2 / alpha) / STAKE
    return (gap + 2 * a) * (GAMMA - 1) / B_SUM


def split_limit(alpha):
    return 4 * (log(3) + log(2 / alpha))


def width_cap(target, norm_risk, variance_limit):
    a = log(2 / norm_risk) / STAKE
    return max(int(a) + 1, ceil((2 * a + B_SUM * variance_limit) / (2 * target)))


def intersect(first, second):
    return max(first[0], second[0]), min(first[1], second[1])


def current_intervals(step, total, captured, squared, alpha, same_limit, split_limit_):
    raw = invert_frozen_norm(step, total, captured, squared, alpha, STAKE)
    split_raw = invert_frozen_norm(step, total, captured, squared, alpha / 2, STAKE)
    same_floor, split_floor = sqrt(squared / same_limit), sqrt(squared / split_limit_)
    same = intersect(raw, (same_floor, inf))
    split = intersect(split_raw, (split_floor, inf))
    return raw, same, split, same_floor, split[0] > split_raw[0]


def interval_record(interval):
    lower, upper = interval
    radius = relative_radius(lower, upper)
    return {
        "lower": lower,
        "upper": upper if isfinite(upper) else None,
        "radius": radius if isfinite(radius) else None,
        "empty": lower > upper,
    }


@dataclass
class Row:
    interval: tuple = UNBOUNDED
    last_checkpoint: int = 0
    stops: dict = field(default_factory=dict)
    reference_exclusions: list = field(default_factory=list)
    empty_reports: int = 0
    reports_different_from_raw: int = 0
    variance_floor_active_reports: int = 0
    max_floor_over_raw_lower: float = 0.0

    def update(self, interval, raw, step, targets, reference):
        self.interval = interval
        self.last_checkpoint = step
        self.reports_different_from_raw += interval != raw
        lower, upper = interval
        self.empty_reports += lower > upper
        if not lower <= reference <= upper:
            self.reference_exclusions.append(step)
        radius = relative_radius(lower, upper)
        for target in targets:
            key = str(target)
            if key not in self.stops and radius <= target:
                center = relative_center(lower, upper)
                self.stops[key] = {
                    "probes": step,
                    "radius": radius,
                    "estimate": center,
                    "relative_error": abs(center - reference) / reference,
                }

    def archived_record(self):
        return {
            "stops": self.stops,
            "reference_exclusions": self.reference_exclusions,
            "last_checkpoint": self.last_checkpoint,
            "last_interval": interval_record(self.interval),
        }


def replay(trial, reference, alphas, targets, gaps, cap, archive_gap):
    scores = trial["probe_scores"]
    if len(scores) != cap or trial["shared_numerical_error"] is not None:
        raise ValueError(f"Expected a complete source trajectory at seed {trial['seed']}")
    rows = {
        gap: {alpha: {method: Row() for method in METHODS} for alpha in alphas}
        for gap in gaps
    }
    variance_limits = {alpha: (prop1_limit(alpha, 1), split_limit(alpha)) for alpha in alphas}
    captured = total = squared = 0.0
    for step, (q, v, delta) in enumerate(scores, 1):
        if not (0 < q < inf and 0 < v < inf and 0 < delta < inf and delta == v / q):
            raise ValueError(f"Invalid score at seed {trial['seed']}, probe {step}")
        total += captured + q
        captured += delta
        squared += v
        reporting_gaps = [gap for gap in gaps if step % gap == 0]
        for alpha in alphas:
            raw, same, split, same_floor, split_active = current_intervals(
                step, total, captured, squared, alpha, *variance_limits[alpha]
            )
            for gap in reporting_gaps:
                methods = rows[gap][alpha]
                running = intersect(methods[RUNNING].interval, raw)
                for method, interval in zip(METHODS, (raw, same, split, running), strict=True):
                    methods[method].update(interval, raw, step, targets, reference)
                methods[SAME].variance_floor_active_reports += same[0] > raw[0]
                methods[SAME].max_floor_over_raw_lower = max(
                    methods[SAME].max_floor_over_raw_lower, same_floor / raw[0]
                )
                methods[SPLIT].variance_floor_active_reports += split_active

    for alpha in alphas:
        for method, archived_method in ((RUNNING, "frozen_norm"), (SPLIT, "reinforced_norm")):
            old = trial["methods"][str(alpha)][archived_method]
            if rows[archive_gap][alpha][method].archived_record() != old:
                raise ValueError(f"Archived record mismatch at seed {trial['seed']}, {alpha=}, {method=}")
    return rows


def summarize(rows, targets, cap):
    grouped = {method: [path[method] for path in rows] for method in METHODS}
    report_outcomes = {}
    for method, paths in grouped.items():
        report_outcomes[method] = {
            "empty_reports": sum(path.empty_reports for path in paths),
            "paths_with_any_reference_exclusion": sum(bool(path.reference_exclusions) for path in paths),
            "reports_different_from_raw": sum(path.reports_different_from_raw for path in paths),
            "paths_with_a_report_different_from_raw": sum(
                path.reports_different_from_raw > 0 for path in paths
            ),
        }
        if method in (SAME, SPLIT):
            report_outcomes[method]["variance_floor_active_reports"] = sum(
                path.variance_floor_active_reports for path in paths
            )
    report_outcomes[SAME]["max_variance_floor_over_raw_lower"] = max(
        path.max_floor_over_raw_lower for path in grouped[SAME]
    )

    target_outcomes = {}
    for target in targets:
        key = str(target)
        methods, costs, completions = {}, {}, {}
        for method, paths in grouped.items():
            stops = [path.stops.get(key) for path in paths]
            completed = [stop for stop in stops if stop is not None]
            costs[method] = [cap if stop is None else stop["probes"] for stop in stops]
            completions[method] = [stop is not None for stop in stops]
            methods[method] = {
                "completions": len(completed),
                "completions_outside_target_at_reference": sum(
                    stop["relative_error"] > target for stop in completed
                ),
                "restricted_mean_probes": mean(costs[method]),
                "median_probes_among_completions": median(
                    stop["probes"] for stop in completed
                ) if completed else None,
            }
        contrasts = {}
        for baseline, comparator in combinations(METHODS, 2):
            differences = [new - old for new, old in zip(costs[comparator], costs[baseline], strict=True)]
            status_pairs = list(zip(completions[comparator], completions[baseline], strict=True))
            contrasts[f"{comparator}_minus_{baseline}"] = {
                "restricted_mean_difference": mean(differences),
                "lower_restricted_cost": sum(diff < 0 for diff in differences),
                "equal_restricted_cost": sum(diff == 0 for diff in differences),
                "higher_restricted_cost": sum(diff > 0 for diff in differences),
                "completed_only_by_comparator": status_pairs.count((True, False)),
                "completed_only_by_baseline": status_pairs.count((False, True)),
            }
        target_outcomes[key] = {"methods": methods, "paired_contrasts": contrasts}
    return {"reports": report_outcomes, "targets": target_outcomes}


def summarize_group(paths, gaps, alphas, targets, cap):
    return {
        str(gap): {
            str(alpha): summarize([path[gap][alpha] for path in paths], targets, cap)
            for alpha in alphas
        }
        for gap in gaps
    }


def isolated_report_fixture():
    # Diagonal construction from the archived counterexample with adjusted probe lengths.
    step, alpha, reference = 400, 0.05, 1.0
    leading_q, trailing_q = 60.0, 0.3
    trailing_eigenvalue = 1 / (2 * step**2)
    captured = 0.5 + (step - 1) * trailing_eigenvalue
    total = leading_q + (step - 1) * (0.5 + trailing_q)
    total += (step - 1) * (step - 2) * trailing_eigenvalue / 2
    squared = leading_q / 2 + (step - 1) * trailing_q * trailing_eigenvalue
    raw, same, split, _, _ = current_intervals(
        step, total, captured, squared, alpha, prop1_limit(alpha, 1), split_limit(alpha)
    )
    first = invert_frozen_norm(1, leading_q, 0.5, leading_q / 2, alpha, STAKE)
    two_report_running = intersect(first, raw)
    assert raw[0] < same[0] < reference < raw[1] == same[1]
    assert relative_radius(*same) < 0.05 < relative_radius(*raw)
    assert split[0] > split[1] and two_report_running[0] > two_report_running[1]
    return {
        "construction": (
            "A is diagonal with eigenvalues 1/2 and 1/(2 m^2) repeated m^2 times. "
            "Use scaled distinct coordinate probes so q is 60 at probe one and 0.3 thereafter. "
            "Each probe removes its coordinate eigenvalue."
        ),
        "scope": (
            "A deterministic diagnostic with trace one, excluded from the archive summaries. "
            "The running interval at a first report equals raw_current. "
            "The two-report interval retains the additional constraint at probe one."
        ),
        "probes": step,
        "dimension": step**2 + 1,
        "alpha": alpha,
        "target": 0.05,
        "trace": reference,
        "first_report_only": {
            method: interval_record(interval)
            for method, interval in zip(METHODS, (raw, same, split, raw), strict=True)
        },
        "running_with_reports_at_one_and_final_probe": interval_record(two_report_running),
    }


def main():
    source = RESULTS / "stake_ablation_results.json"
    source_bytes = source.read_bytes()
    archive = json.loads(source_bytes)
    if archive["status"] != "COMPLETE" or archive["probe_score_columns"] != ["q", "v", "delta"]:
        raise ValueError("Expected the completed archived stake ablation with q, v, delta columns")
    cap, archive_gap = archive["cap"], archive["report_every"]
    alphas, targets = archive["alphas"], archive["targets"]
    gaps = sorted({1, archive_gap, 16, 80})
    all_paths, cases = [], {}
    for name, case in archive["cases"].items():
        reference = case["audit"]["trace_reference"]
        paths = [
            replay(trial, reference, alphas, targets, gaps, cap, archive_gap)
            for trial in case["trials"]
        ]
        if len(paths) != archive["paths_per_operator"]:
            raise ValueError(f"Unexpected path count for {name}")
        all_paths.extend(paths)
        cases[name] = {
            "paths": len(paths),
            "trace_reference": reference,
            "report_gaps": summarize_group(paths, gaps, alphas, targets, cap),
        }

    caps = {}
    for alpha in alphas:
        k1, k_split = prop1_limit(alpha, 1), split_limit(alpha)
        caps[str(alpha)] = {
            "same_risk_K1": k1,
            "split_risk_K": k_split,
            "targets": {
                str(target): {
                    "same_risk_current_sufficient_report": width_cap(target, alpha, k1),
                    "split_risk_current_sufficient_report": width_cap(target, alpha / 2, k_split),
                    "running_sufficient_report_by_max_gap": {
                        str(gap): width_cap(target, alpha, prop1_limit(alpha, gap)) for gap in gaps
                    },
                }
                for target in targets
            },
        }
    result = {
        "scope": (
            "Descriptive paired binary64 replay of archived trajectories. No new probes, matrix actions, "
            "timing data, independent paths, or empirical coverage claims. Reference comparisons use "
            "the stored operator trace references. The fixed settings each have nominal risk alpha."
        ),
        "methods": {
            RAW: "Current frozen-norm interval with alpha assigned to the two signed tests.",
            SAME: "Current frozen-norm interval at alpha with the Proposition 1 floor sqrt(V/K1).",
            SPLIT: "Current frozen-norm interval at alpha/2 with the Theorem 3 floor sqrt(V/K) at risk alpha/2.",
            RUNNING: "Running intersection of current frozen-norm intervals at alpha on the indicated grid.",
        },
        "same_risk_justification": (
            "On the original simultaneous coverage event the true scale survives every hypothetical "
            "probe-by-probe report. Proposition 1 then bounds V/T_true^2 by K1 at every probe. "
            "The floor holds on that same event even when earlier intervals were never computed."
        ),
        "stopping_rule": (
            "First scheduled positive finite nonempty interval with radius at most the target. "
            "A cap exit has restricted cost equal to the cap and counts as no completion. "
            "Reference exclusions and interval differences use the full grid through the cap."
        ),
        "pathwise_width_caps_scope": (
            "Exact-arithmetic bounds at any report reaching the indicated probe count with a nonempty interval. "
            "For the running rule, the first and subsequent gaps must be at most the specified gap. "
            "A grid must actually report at or after the sufficient count, and counts beyond the replay cap "
            "are theoretical. An empty interval provides no certificate. Raw current has no pathwise "
            "vanishing width bound at an isolated report. Its simultaneous coverage event permits a separate bound."
        ),
        "archive_cap": cap,
        "archived_gap": archive_gap,
        "report_steps_by_gap": {str(gap): list(range(gap, cap + 1, gap)) for gap in gaps},
        "stake": STAKE,
        "alphas": alphas,
        "targets": targets,
        "operators": len(cases),
        "paths": len(all_paths),
        "archived_records_verified": len(all_paths) * len(alphas) * 2,
        "source_sha256": {
            source.name: sha256(source_bytes).hexdigest(),
            **{
                name: sha256((HERE / name).read_bytes()).hexdigest()
                for name in (Path(__file__).name, "anchored_norm.py", "progress_certificate.py")
            },
        },
        "pathwise_width_caps": caps,
        "isolated_report_diagnostic": isolated_report_fixture(),
        "pooled_report_gaps": summarize_group(all_paths, gaps, alphas, targets, cap),
        "cases": cases,
    }
    destination = RESULTS / "same_risk_replay_results.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Replayed {result['paths']} paths and verified {result['archived_records_verified']} archived records.")


if __name__ == "__main__":
    main()
