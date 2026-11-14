"""Regenerate the figures and paired analysis from the archived experiments."""
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from source_check import source_matches


PILOT = Path(__file__).resolve().parent
OUTPUTS = PILOT


MATRICES = ('bcsstk14', 'bcsstk15', 'bcsstk16')
METHODS = {
    'progress_single': 'Trace progress, one start',
    'progress_anchored': 'Trace progress, doubling starts',
    'doob': 'Doob variance bound',
    'chi_square': 'Chi-square, same sketch',
    'residual_v': 'Observed residual norm',
    'hybrid': 'Hybrid, checks every 4',
    'split_chernoff': 'Split Nyström, Chernoff checkpoints',
    'split_observed_v': 'Split Nyström, observed-norm checkpoints',
    'hybrid_matched': 'Hybrid, reporting at the same checkpoints',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_sources(document):
    for name, expected in document.get('source_sha256', {}).items():
        assert source_matches(name, expected), name


def structural_statistics(structural, split):
    checkpoints = split['checkpoints']
    assert checkpoints[-1] == 256
    result = {}
    rng = np.random.default_rng(804412)
    for name in MATRICES:
        matrix = structural[name]
        heldout = split['matrices'][name]
        paths = matrix['records']
        split_paths = heldout['records']
        assert matrix['complete'] and len(paths) == len(split_paths) == 24
        assert [p['seed'] for p in paths] == [p['seed'] for p in split_paths]
        assert heldout['audit']['sha256'] == matrix['data_audit']['sha256']
        assert sha256((PILOT / 'matrices' / (name + '.mtx.gz')).read_bytes()).hexdigest() == matrix['data_audit']['sha256']
        entries = {}
        for method in METHODS:
            if method.startswith('split_'):
                stops = [p.get('stops', {}).get(method) for p in split_paths]
                misses = sum(p.get('coverage_failures', {}).get(method, False) for p in split_paths)
                errors = sum(p is not None and p['relative_error'] > .05 for p in stops)
            else:
                source = 'hybrid' if method == 'hybrid_matched' else method
                stops = [p.get('metrics', {}).get(source, {}).get('stops', {}).get('0.05') for p in paths]
                misses = matrix['summary'][source]['pathwise_coverage_failures']
                errors = sum(p is not None and p['relative_error'] > .05 for p in stops)
            counts = [s['matvecs'] if s else None for s in stops]
            if method == 'hybrid_matched':
                counts = [next(c for c in checkpoints if c >= k) if k is not None else None for k in counts]
            observed = [n for n in counts if n is not None]
            entries[method] = {'counts': counts, 'median': float(np.median(observed)) if observed else None,
                               'stopped': len(observed), 'coverage_failures': misses, 'incorrect_stops': errors}
        new = np.array([x if x is not None else 260 for x in entries['hybrid']['counts']])
        comparator = np.array([x if x is not None else 260 for x in entries['chi_square']['counts']])
        paired = comparator - new
        better = int((paired > 0).sum())
        worse = int((paired < 0).sum())
        p = binomtest(better, better + worse).pvalue if better + worse else 1.0
        bootstrap = np.median(rng.choice(paired, size=(10000, len(paired)), replace=True), axis=1)
        result[name] = {'dimension': matrix['data_audit']['dimension'], 'methods': entries,
                        'median_paired_saving': float(np.median(paired)),
                        'paired_bootstrap_95_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
                        'wins': better, 'losses': worse, 'ties': len(paired)-better-worse, 'sign_test_p': p}
        matched = np.array([x if x is not None else 260 for x in entries['hybrid_matched']['counts']])
        holdout = np.array([x if x is not None else 260 for x in entries['split_observed_v']['counts']])
        result[name]['matched_checkpoint_pairs'] = {
            'wins': int((matched < holdout).sum()),
            'ties': int((matched == holdout).sum()),
            'losses': int((matched > holdout).sum()),
        }
    maximum = 0.0
    for rank, case in enumerate(sorted(result, key=lambda name: result[name]['sign_test_p'])):
        maximum = max(maximum, min(1.0, (len(result)-rank) * result[case]['sign_test_p']))
        result[case]['holm_p'] = maximum
    return result


def draw_figure(stats):
    methods = ['chi_square', 'hybrid', 'hybrid_matched', 'split_observed_v', 'progress_anchored']
    labels = ['Chi-square', 'Hybrid, every 4', 'Hybrid, matched checkpoints', 'Split, observed norm', 'Progress, doubling starts']
    colors = ['#637083', '#007D83', '#56B4AD', '#B06A36', '#B0BAC9']
    fig, ax = plt.subplots(figsize=(9.1, 4.6), layout='constrained')
    indices = np.arange(len(MATRICES))
    for j, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = [stats[name]['methods'][method]['median'] for name in MATRICES]
        bars = ax.bar(indices + (j-2)*.158, values, width=.15, color=color, label=label, zorder=3)
        ax.bar_label(bars, labels=[str(int(value)) for value in values], padding=3, fontsize=9, color='#263342')
    ax.set_xticks(indices, [f'{name.upper()}\n{stats[name]["dimension"]:,} dimensions' for name in MATRICES])
    ax.set_ylabel('Median matrix–vector products to stop')
    ax.set_ylim(0, 168)
    ax.set_title('Stopping at a 5% relative accuracy certificate', loc='left', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=.24, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', ncols=2, frameon=False, fontsize=8.5)
    path = OUTPUTS / 'nystrom-trace-figure.png'
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    return path


NORM_METHODS = {'progress': 'Progress', 'frozen_norm': 'Frozen scale', 'moving_norm': 'Moving scale',
              'chi_square': 'Chi-square', 'frozen_residual': 'Frozen residual', 'frozen_hybrid': 'Hybrid'}


def draw_anchor_norm_figure(result):
    cases = list(result['cases'].values())
    colors = ('#496789', '#007a76', '#b55b2a', '#76737b', '#7d698e', '#566d38')
    fig, ax = plt.subplots(figsize=(7.6, 3.1), layout='constrained')
    for (method, label), color in zip(NORM_METHODS.items(), colors):
        timings = [trial['methods']['0.05'][method]['evidence_ns'] / 1e6
                   for case in cases for trial in case['trials']]
        cost = np.mean([case['summary']['0.05']['0.05']['methods'][method]['restricted_mean_probes']
                        for case in cases])
        ax.scatter(np.median(timings), cost, color=color, s=56, zorder=3)
        offset = (6, -15) if method == 'frozen_norm' else (6, 5)
        ax.annotate(label, (np.median(timings), cost), xytext=offset, textcoords='offset points',
                    fontsize=8, color='#1e2d3c')
    ax.set(xscale='log', xlabel='Median evidence time through 400 probes (ms, log scale)',
           ylabel='Mean restricted products', xlim=(0.58, 270), ylim=(95, 412))
    ax.grid(axis='y', color='#e2e8ef', linewidth=.8)
    for edge in ('top', 'right'):
        ax.spines[edge].set_visible(False)
    figure = OUTPUTS / 'nystrom-frozen-norm-figure.png'
    fig.savefig(figure, dpi=185, facecolor='white')
    plt.close(fig)
    return figure


def analyze_archive(directory):
    global PILOT, OUTPUTS
    PILOT = directory
    OUTPUTS = directory
    structural = {}
    for name in ('structural_progress_pilot.json', 'structural_progress_remainder.json'):
        data = read(directory / 'results' / name)
        verify_sources(data)
        structural.update(data['matrices'])
    split = read(directory / 'results' / 'split_checkpoint_results.json')
    verify_sources(split)
    stats = structural_statistics(structural, split)
    draw_figure(stats)
    matched = read(directory / 'results' / 'anchor_norm_comparison.json')
    verify_sources(matched)
    draw_anchor_norm_figure(matched)
    (directory / 'paired_analysis_recomputed.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')


if __name__ == '__main__':
    analyze_archive(Path(__file__).resolve().parent)
