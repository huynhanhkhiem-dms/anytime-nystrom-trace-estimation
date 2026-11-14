"""Frozen follow-up on a coauthorship graph, with two shifts and fresh seeds."""
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import requests
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from run_progress_pilot import run_path, summarize


HERE = Path(__file__).resolve().parent
URL = 'https://snap.stanford.edu/data/ca-GrQc.txt.gz'
SHIFTS = (0.01, 1.0)
SEEDS = range(912000, 912064)
BUDGET = 192


def load_laplacian():
    path = HERE / 'matrices' / 'ca-GrQc.txt.gz'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        response = requests.get(URL, timeout=45)
        response.raise_for_status()
        if not response.content.startswith(b'\x1f\x8b'):
            raise ValueError('SNAP download was not gzip')
        path.write_bytes(response.content)
    with gzip.open(path, 'rt') as handle:
        edges = np.loadtxt(handle, dtype=np.int64)
    identifiers, compact = np.unique(edges, return_inverse=True)
    compact = compact.reshape(edges.shape)
    n = len(identifiers)
    adjacency = coo_matrix((np.ones(len(compact)), (compact[:, 0], compact[:, 1])), shape=(n, n)).tocsc()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data.fill(1)
    _, groups = connected_components(adjacency, directed=False)
    keep = groups == np.bincount(groups).argmax()
    adjacency = adjacency[keep, :][:, keep].tocsc()
    dimension = adjacency.shape[0]
    assert n == 5242 and dimension == 4158
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = diags(degree, format='csc') - adjacency
    return laplacian, {'source': URL, 'sha256': sha256(path.read_bytes()).hexdigest(),
                       'vertices_raw': n, 'vertices_connected_component': dimension,
                       'undirected_edges_after_deduplication': adjacency.nnz // 2}


def inverse_trace(factor, dimension):
    total = 0.0
    for begin in range(0, dimension, 64):
        indices = np.arange(begin, min(begin + 64, dimension))
        rhs = np.zeros((dimension, len(indices)), order='F')
        rhs[indices, np.arange(len(indices))] = 1
        total += float(factor.solve(rhs)[indices, np.arange(len(indices))].sum())
    return total


def main():
    root = {'status': 'FROZEN_GRAPH_FOLLOWUP', 'operator': 'inverse shifted combinatorial Laplacian on largest component',
            'seed_start': SEEDS.start, 'seeds_per_shift': len(SEEDS), 'budget': BUDGET,
            'source_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in
                              [Path(__file__), HERE/'run_progress_pilot.py', HERE/'core.py', HERE/'progress_certificate.py', HERE/'chi2_comparator.py']},
            'cases': {}}
    path = HERE / 'graph_holdout_results.json'
    with threadpool_limits(limits=1):
        laplacian, data = load_laplacian()
        root['data_audit'] = data
        d = laplacian.shape[0]
        for shift in SHIFTS:
            start = perf_counter()
            matrix = laplacian + shift * eye(d, format='csc')
            factor = splu(matrix)
            theta = inverse_trace(factor, d)
            print('shift', shift, 'reference', theta, 'dimension', d, flush=True)
            records = []
            for index, seed in enumerate(SEEDS, 1):
                try:
                    records.append(run_path(factor.solve, d, BUDGET, seed, theta))
                except ArithmeticError as error:
                    records.append({'seed': seed, 'numerical_failure': str(error)})
                if index % 4 == 0 or index == len(SEEDS):
                    root['cases'][str(shift)] = {'trace_reference': theta, 'completed_paths': index,
                        'complete': index == len(SEEDS), 'summary': summarize(records),
                        'records': records, 'seconds': perf_counter()-start}
                    staging = path.with_suffix('.tmp')
                    staging.write_text(json.dumps(root, indent=2, allow_nan=False), encoding='utf-8')
                    staging.replace(path)
            print('shift', shift, json.dumps(root['cases'][str(shift)]['summary']), flush=True)


if __name__ == '__main__':
    main()
