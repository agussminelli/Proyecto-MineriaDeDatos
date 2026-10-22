"""Reproducible DBSCAN correctness and performance benchmark (Windows/Linux/macOS)."""
import os
# Apply before importing numerical libraries, including in fresh workers.
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import sklearn
from sklearn.cluster import DBSCAN as ReferenceDBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dbscan import DBSCAN

VARIANTS = ('own_on_demand', 'own_cached', 'sklearn_brute')


def reference(X, eps, minimum, metric, p):
    return ReferenceDBSCAN(eps=eps, min_samples=minimum, metric=metric,
                           p=p, algorithm='brute', n_jobs=1).fit_predict(X)


def agreement(a, b):
    return {'ari': float(adjusted_rand_score(a, b)),
            'noise_equal': bool(np.array_equal(a == -1, b == -1))}


def correctness():
    blobs = make_blobs(n_samples=120, centers=[[-3, -3], [3, 3]],
                       cluster_std=.2, random_state=42)[0]
    moons = make_moons(n_samples=150, noise=.02, random_state=42)[0]
    cases = [
        ('blobs', blobs, .65, 4), ('moons', moons, .18, 4),
        ('all_noise', np.array([[0., 0.], [3., 0.], [6., 0.]]), .2, 2),
        ('one_cluster', np.array([[0., 0.], [.1, 0.], [.2, 0.]]), .3, 2),
        ('duplicates', np.array([[0., 0.]] * 4 + [[3., 3.]] * 4), .1, 3),
        # First point has two neighbours including itself, but is reachable
        # from a later core point with at least three neighbours.
        ('noise_to_border', np.array([[0., 0.], [.09, 0.], [.18, 0.], [.19, 0.]]), .11, 3),
        ('self_counts', np.array([[0., 0.], [3., 0.]]), .1, 1),
        ('inclusive_eps', np.array([[0., 0.], [1., 0.], [2., 0.]]), 1., 2),
    ]
    rows = []
    for name, X, eps, minimum in cases:
        for metric, p in [('euclidean', 2), ('manhattan', 1), ('minkowski', 3)]:
            expected = reference(X, eps, minimum, metric, p)
            for cached in (False, True):
                actual = DBSCAN(eps, minimum, metric, p).fit_predict(X, precompute=cached)
                result = agreement(actual, expected)
                rows.append(dict(case=name, metric=metric, cached=cached, **result))
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal([1, 0, 0], .03, size=(30, 3)),
                   rng.normal([0, 1, 0], .03, size=(30, 3)), [[0, 0, 1]]])
    for cached in (False, True):
        actual = DBSCAN(.02, 4, 'cosine').fit_predict(X, precompute=cached)
        rows.append(dict(case='cosine_directions', metric='cosine', cached=cached,
                         **agreement(actual, reference(X, .02, 4, 'cosine', 2))))
    # Independent expectations prevent shared agreement from hiding trivial errors.
    border = DBSCAN(.11, 3).fit_predict(cases[5][1])
    if not np.all(border == border[0]) or border[0] == -1:
        raise AssertionError('Initially noisy border point was not reassigned')
    if not all(r['ari'] == 1.0 and r['noise_equal'] for r in rows):
        raise AssertionError(f'Correctness mismatch: {rows}')
    return rows


def peak_process_memory_bytes():
    """Return OS process peak resident memory in bytes, including startup."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)

    import resource
    scale = 1 if sys.platform == 'darwin' else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def worker(spec):
    X = np.load(spec['features'], allow_pickle=False)
    indices = np.load(spec['indices'], allow_pickle=False)
    X = np.ascontiguousarray(X[indices], dtype=np.float64)
    # Dataset loading, imports and metrics are outside the timer.
    baseline_peak = peak_process_memory_bytes()
    started = perf_counter()
    assignments = []
    model = None
    matrix_seconds = 0.0
    for eps, minimum in spec['grid']:
        if spec['variant'] == 'sklearn_brute':
            labels = reference(X, eps, minimum, spec['metric'], spec['p'])
        else:
            if model is None:
                model = DBSCAN(eps, minimum, spec['metric'], spec['p'])
                labels = model.fit_predict(X, precompute=spec['variant'] == 'own_cached')
                matrix_seconds = getattr(model, 'matrix_build_seconds_', 0.0)
            else:
                model.eps, model.min_samples = eps, minimum
                labels = model.fit_predict()
        assignments.append(labels)
    elapsed = perf_counter() - started
    peak = peak_process_memory_bytes()
    # Keep all assignments in each variant for equal output retention costs.
    return dict(seconds=elapsed, process_peak_rss_bytes=peak,
                before_fit_peak_rss_bytes=baseline_peak,
                matrix_build_seconds=matrix_seconds,
                cached_matrix_bytes=0 if model is None or model.distance_matrix_ is None
                else model.distance_matrix_.nbytes,
                assignments=[a.tolist() for a in assignments])


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--features', type=Path, help='Saved, numeric 2D features.npy; no embedding generation')
    parser.add_argument('--dataset', choices=['blobs', 'moons', 'smiley'], default='blobs')
    parser.add_argument('--sizes', type=int, nargs='+', default=[250, 500, 1000])
    parser.add_argument('--eps', type=float, nargs='+', default=[.2, .35, .5])
    parser.add_argument('--min-samples', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--metric', choices=['euclidean', 'manhattan', 'minkowski', 'cosine'], default='euclidean')
    parser.add_argument('--p', type=float, default=2.)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float, default=300.)
    parser.add_argument('--max-matrix-mib', type=float, default=512., help='Fail before allocating a larger cached matrix')
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('benchmark_results'))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if args.repeats < 1 or min(args.sizes) < 2:
        parser.error('Use positive repeats and sizes >= 2')
    if min(args.eps) <= 0 or min(args.min_samples) < 1:
        parser.error('eps must be positive and min-samples >= 1')
    args.output.mkdir(parents=True, exist_ok=True)
    checks = correctness()
    write_csv(args.output / 'correctness.csv', checks)
    print(f'Passed {len(checks)} reference comparisons plus independent border assertion.', flush=True)
    if args.correctness_only:
        return
    if args.features:
        X = np.load(args.features, allow_pickle=False)
    elif args.dataset == 'smiley':
        from generate_testdata import generate_smiley_data
        X = generate_smiley_data()
    elif args.dataset == 'moons':
        X = make_moons(n_samples=max(args.sizes), noise=.03, random_state=args.seed)[0]
    else:
        X = make_blobs(n_samples=max(args.sizes), centers=4, n_features=2,
                       cluster_std=.35, random_state=args.seed)[0]
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all() or not all(X.shape):
        parser.error('Features must be a nonempty finite numeric 2D array')
    if args.metric == 'cosine' and np.any(np.linalg.norm(X, axis=1) == 0):
        parser.error('Cosine input contains zero vectors')
    sizes = sorted(set(args.sizes))
    if max(sizes) > len(X):
        parser.error(f'Largest sample size exceeds available {len(X)} rows')
    if 8 * max(sizes)**2 > args.max_matrix_mib * 1024**2:
        parser.error('Cached matrix exceeds --max-matrix-mib; reduce sizes or explicitly increase budget')
    grid = [(float(e), int(m)) for e in args.eps for m in args.min_samples]
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    python=sys.version, platform=platform.platform(), numpy=np.__version__,
                    sklearn=sklearn.__version__, shape=list(X.shape),
                    feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                    threads=1, memory='Process high-water RSS including imports and input; not isolated fit allocation',
                    sampling='Nested seeded subsets; fixed across repetitions; not stratified',
                    dbscan_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / 'dbscan.py').read_bytes()).hexdigest())
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    rows, comparisons = [], []
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(X))
    with tempfile.TemporaryDirectory() as temp:
        features = Path(temp) / 'features.npy'
        indices = Path(temp) / 'indices.npy'
        np.save(features, X)
        for n in sizes:
            np.save(indices, order[:n])
            for scenario, current_grid in [('single_fit', grid[:1]), ('parameter_sweep', grid)]:
                for repeat in range(args.repeats):
                    outputs = {}
                    for variant in rng.permutation(VARIANTS):
                        spec = dict(features=str(features), indices=str(indices), grid=current_grid,
                                    metric=args.metric, p=args.p, variant=str(variant))
                        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                              input=json.dumps(spec), text=True, capture_output=True,
                                              timeout=args.timeout, check=True)
                        result = json.loads(proc.stdout)
                        outputs[variant] = [np.asarray(a) for a in result.pop('assignments')]
                        rows.append(dict(n=n, dimensions=X.shape[1], scenario=scenario, repeat=repeat,
                                         variant=variant, configurations=len(current_grid), **result))
                        write_csv(args.output / 'timings.csv', rows)
                    for variant in VARIANTS[:2]:
                        for i, (actual, expected) in enumerate(zip(outputs[variant], outputs['sklearn_brute'])):
                            a = agreement(actual, expected)
                            comparisons.append(dict(n=n, scenario=scenario, repeat=repeat, variant=variant,
                                                    eps=current_grid[i][0], min_samples=current_grid[i][1],
                                                    n_clusters=len(set(actual) - {-1}),
                                                    noise_fraction=float(np.mean(actual == -1)), **a))
                    write_csv(args.output / 'agreement.csv', comparisons)
                    print(f'Completed n={n}, {scenario}, repetition {repeat+1}/{args.repeats}', flush=True)
    summaries = []
    for n in sizes:
        for scenario in ('single_fit', 'parameter_sweep'):
            for variant in VARIANTS:
                subset = [r for r in rows if r['n'] == n and r['scenario'] == scenario and r['variant'] == variant]
                summaries.append(dict(n=n, scenario=scenario, variant=variant,
                    median_seconds=float(np.median([r['seconds'] for r in subset])),
                    min_seconds=min(r['seconds'] for r in subset), max_seconds=max(r['seconds'] for r in subset),
                    median_process_peak_rss_mib=float(np.median([r['process_peak_rss_bytes'] for r in subset]))/1024**2))
    write_csv(args.output / 'summary.csv', summaries)
    mismatches = sum(r['ari'] != 1.0 or not r['noise_equal'] for r in comparisons)
    print(f'Done. {mismatches} benchmark partition mismatches; inspect agreement.csv before interpreting performance.')


if __name__ == '__main__':
    main()
