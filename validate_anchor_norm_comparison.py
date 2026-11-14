"""Check the recorded paired comparison independently of the replay driver."""

from hashlib import sha256
import json
from math import comb, inf, isclose
from pathlib import Path
from statistics import median
from urllib.parse import unquote, urlparse

import numpy as np

from source_check import source_matches


HERE = Path(__file__).resolve().parent
RESULTS_NAME = 'anchor_norm_comparison.json'


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def check_close(value, expected):
    assert isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12), (value, expected)


def wins_ties_losses(differences):
    return (sum(diff < 0 for diff in differences), sum(diff == 0 for diff in differences),
            sum(diff > 0 for diff in differences))


def two_sided_sign_test(wins, losses):
    count = wins + losses
    if count == 0:
        return 1.0
    tail = sum(comb(count, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2**count)


def without_timings(value):
    if isinstance(value, dict):
        return {key: without_timings(item) for key, item in value.items()
                if '_ns' not in key and '_ms' not in key}
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    return value


def check_method(rows, target, cap, grid, reference, summary):
    stops = []
    complete = []
    incomplete = []
    costs = []
    for row in rows:
        last = row['last_checkpoint']
        exclusions = row['reference_exclusions']
        assert last == 0 or last in grid
        assert exclusions == sorted(set(exclusions))
        assert all(step in grid and step <= last for step in exclusions)
        if last:
            interval = row['last_interval']
            lower = interval['lower'] if interval['lower'] is not None else inf
            upper = interval['upper'] if interval['upper'] is not None else inf
            empty = interval['empty']
            assert (last in exclusions) == (empty or not lower <= reference <= upper)
            if interval['radius'] is not None:
                assert 0 < lower <= upper < inf and not empty
                check_close(interval['radius'], (upper - lower) / (upper + lower))
        if last == cap:
            complete.append(row)
        else:
            incomplete.append(row)

        stop = row['stops'].get(str(target))
        if stop is None:
            costs.append(cap)
        else:
            assert stop['probes'] in grid and stop['probes'] <= last
            assert 0 <= stop['radius'] <= target
            check_close(stop['relative_error'], abs(stop['estimate'] - reference) / reference)
            stops.append(stop)
            costs.append(stop['probes'])

    assert summary['completions'] == len(stops)
    assert summary['incorrect_completions'] == sum(stop['relative_error'] > target for stop in stops)
    assert summary['complete_replays'] == len(complete)
    assert summary['numerical_failures'] == sum(row['numerical_error'] is not None for row in rows)
    exclusions = sum(bool(row['reference_exclusions']) for row in complete)
    assert summary['pathwise_exclusions'] == exclusions
    assert summary['incomplete_replay_exclusions'] == sum(bool(row['reference_exclusions']) for row in incomplete)
    check_close(summary['restricted_mean_probes'], sum(costs) / len(costs))
    if stops:
        check_close(summary['median_probes_among_stops'], median(stop['probes'] for stop in stops))
        check_close(summary['median_evidence_ms_at_stop'], median(stop['evidence_ns'] for stop in stops) / 1e6)
    if complete:
        check_close(summary['median_evidence_ms_full_replay'], median(row['evidence_ns'] for row in complete) / 1e6)
        if exclusions == 0:
            check_close(summary['one_sided_95pct_failure_rate_upper'], 1 - 0.05**(1 / len(complete)))
    return costs


def check_contrast(left, right, summary):
    differences = [a - b for a, b in zip(left, right, strict=True)]
    counts = wins_ties_losses(differences)
    assert counts == (summary['wins'], summary['ties'], summary['losses'])
    check_close(summary['mean_difference'], sum(differences) / len(differences))
    return differences, counts


def check_source_data(case, checked, unavailable):
    audit = case['audit']
    assert audit['reference_relative_disagreement'] <= 1e-8
    assert audit.get('max_audit_product_relative_discrepancy', 0) <= 1e-8
    filename = unquote(Path(urlparse(audit['source']).path).name)
    if filename == 'UCI HAR Dataset.zip':
        filename = 'har-240.zip'
    if filename in checked or filename in unavailable:
        return
    source = HERE / 'matrices' / filename
    if source.is_file():
        assert file_hash(source) == audit['sha256'], filename
        checked.add(filename)
    else:
        unavailable.add(filename)


def main():
    path = HERE / RESULTS_NAME
    if not path.is_file():
        path = HERE / 'results' / RESULTS_NAME
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['status'] == 'COMPLETE'
    assert list(result['cases']) == result['planned_operators']
    for name, expected in result['source_sha256'].items():
        assert source_matches(name, expected), name

    cap = result['cap']
    stride = result['report_every']
    grid = set(range(stride, cap + 1, stride))
    contrasts = []
    checked_data, unavailable_data = set(), set()
    methods_checked = paths_checked = 0
    for index, case in enumerate(result['cases'].values()):
        if case['status'] != 'complete':
            assert case['status'] == 'operator_audit_failure' and case['reason']
            continue
        check_source_data(case, checked_data, unavailable_data)
        trials = case['trials']
        base = result['first_seed'] + index * result['seed_stride']
        assert [trial['seed'] for trial in trials] == list(range(base, base + result['paths_per_operator']))
        paths_checked += len(trials)
        reference = case['audit']['trace_reference']
        for alpha_index, alpha in enumerate(result['alphas']):
            for target_index, target in enumerate(result['targets']):
                level, key = str(alpha), str(target)
                setting = case['summary'][level][key]
                costs = {}
                for method, summary in setting['methods'].items():
                    rows = [trial['methods'][level][method] for trial in trials]
                    costs[method] = check_method(rows, target, cap, grid, reference, summary)
                    methods_checked += 1
                primary = setting['primary_frozen_norm_minus_progress']
                differences, (wins, _, losses) = check_contrast(costs['frozen_norm'], costs['progress'], primary)
                p = two_sided_sign_test(wins, losses)
                check_close(primary['sign_test_two_sided'], p)
                contrasts.append((p, primary['holm_adjusted_p']))
                bootstrap_seed = 8200000 + 1000 * index + 10 * alpha_index + target_index
                rng = np.random.default_rng(bootstrap_seed)
                draws = rng.choice(differences, size=(5000, len(differences)), replace=True).mean(axis=1)
                for observed, expected in zip(primary['paired_percentile_bootstrap_95pct'],
                                              np.quantile(draws, (0.025, 0.975)), strict=True):
                    check_close(observed, expected)
                for method, summary in setting['descriptive_frozen_norm_minus_control'].items():
                    check_contrast(costs['frozen_norm'], costs[method], summary)

    count = result['planned_primary_contrasts']
    assert count == len(result['cases']) * len(result['alphas']) * len(result['targets'])
    previous = 0.0
    for index, (p, adjusted) in enumerate(sorted(contrasts)):
        previous = min(1.0, max(previous, (count - index) * p))
        check_close(adjusted, previous)

    fingerprint = sha256(json.dumps(without_timings(result['cases']), sort_keys=True).encode()).hexdigest()
    validation = {
        'status': 'passed', 'results_sha256': file_hash(path), 'nontiming_case_sha256': fingerprint,
        'validator_sha256': file_hash(Path(__file__)), 'source_file_hashes_checked': len(result['source_sha256']),
        'cached_operator_sources_checked': sorted(checked_data), 'uncached_operator_sources': sorted(unavailable_data),
        'paths_checked': paths_checked, 'method_setting_summaries_checked': methods_checked,
        'primary_contrasts_checked': len(contrasts),
    }
    (HERE / 'validate_anchor_norm_comparison.json').write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
