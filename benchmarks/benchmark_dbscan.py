"""Reproducible DBSCAN correctness and performance benchmark (Windows/Linux/macOS)."""

import os

# Apply before importing numerical libraries, including in fresh workers.
for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
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

VARIANTS = ("own_on_demand", "own_cached", "sklearn_brute")


def reference(X, eps, minimum, metric, p):
    """Fit sklearn DBSCAN using exhaustive search and one worker.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Numeric feature matrix in the metric's input space.
    eps : float
        Inclusive neighborhood radius.
    minimum : int
        Minimum neighborhood size, including the point itself.
    metric : str
        Distance metric supported by the benchmark.
    p : float
        Minkowski exponent; forwarded to sklearn.

    Returns
    -------
    ndarray of shape (n_samples,)
        Cluster labels, with -1 marking noise.
    """
    return ReferenceDBSCAN(
        eps=eps,
        min_samples=minimum,
        metric=metric,
        p=p,
        algorithm="brute",
        n_jobs=1,
    ).fit_predict(X)


def agreement(a, b):
    """Compare partitions independently of their cluster ID numbering.

    Parameters
    ----------
    a, b : ndarray of shape (n_samples,)
        Labels for the same points in the same order; noise uses -1.

    Returns
    -------
    dict
        ``ari`` compares partitions; ``noise_equal`` checks identical noise
        masks separately. Agreement here is not agreement with death causes.
    """
    return {
        "ari": float(adjusted_rand_score(a, b)),
        "noise_equal": bool(np.array_equal(a == -1, b == -1)),
    }


def correctness(distance_backend="baseline", block_size=256):
    """Validate a backend against sklearn and a border-point assertion.

    Parameters
    ----------
    distance_backend : {"baseline", "optimized"}
        Custom distance implementation to check.
    block_size : int
        Row-block size passed to the custom implementation.

    Returns
    -------
    list of dict
        Fifty comparison records, including ARI and noise-mask agreement.

    Raises
    ------
    AssertionError
        If partitions disagree or the independent border check fails.

    Notes
    -----
    Uses deterministic small fixtures. These checks do not prove correctness
    for every input or resolve every ambiguous border-point assignment.
    """
    # Deterministic reference fixtures.
    blobs = make_blobs(
        n_samples=120,
        centers=[[-3, -3], [3, 3]],
        cluster_std=0.2,
        random_state=42,
    )[0]
    moons = make_moons(n_samples=150, noise=0.02, random_state=42)[0]

    cases = [
        ("blobs", blobs, 0.65, 4),
        ("moons", moons, 0.18, 4),
        ("all_noise", np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]]), 0.2, 2),
        (
            "one_cluster",
            np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]),
            0.3,
            2,
        ),
        ("duplicates", np.array([[0.0, 0.0]] * 4 + [[3.0, 3.0]] * 4), 0.1, 3),
        # First point has two neighbours including itself, but is reachable
        # from a later core point with at least three neighbours.
        (
            "noise_to_border",
            np.array([[0.0, 0.0], [0.09, 0.0], [0.18, 0.0], [0.19, 0.0]]),
            0.11,
            3,
        ),
        ("self_counts", np.array([[0.0, 0.0], [3.0, 0.0]]), 0.1, 1),
        (
            "inclusive_eps",
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            1.0,
            2,
        ),
    ]

    rows = []
    for name, X, eps, minimum in cases:
        for metric, p in [
            ("euclidean", 2),
            ("manhattan", 1),
            ("minkowski", 3),
        ]:
            expected = reference(X, eps, minimum, metric, p)
            for cached in (False, True):
                actual = DBSCAN(
                    eps, minimum, metric, p, distance_backend, block_size
                ).fit_predict(X, precompute=cached)
                result = agreement(actual, expected)
                rows.append(
                    dict(case=name, metric=metric, cached=cached, **result)
                )

    rng = np.random.default_rng(42)

    X = np.vstack(
        [
            rng.normal([1, 0, 0], 0.03, size=(30, 3)),
            rng.normal([0, 1, 0], 0.03, size=(30, 3)),
            [[0, 0, 1]],
        ]
    )

    for cached in (False, True):
        actual = DBSCAN(
            0.02, 4, "cosine", distance_backend=distance_backend
        ).fit_predict(X, precompute=cached)
        rows.append(
            dict(
                case="cosine_directions",
                metric="cosine",
                cached=cached,
                **agreement(actual, reference(X, 0.02, 4, "cosine", 2)),
            )
        )

    # Independent expectations prevent shared agreement from hiding trivial errors.
    border = DBSCAN(0.11, 3, distance_backend=distance_backend).fit_predict(
        cases[5][1]
    )
    if not np.all(border == border[0]) or border[0] == -1:
        raise AssertionError("Initially noisy border point was not reassigned")
    if not all(r["ari"] == 1.0 and r["noise_equal"] for r in rows):
        raise AssertionError(f"Correctness mismatch: {rows}")
    return rows


def peak_process_memory_bytes():
    """Return the process high-water resident-memory estimate in bytes.

    Uses PeakWorkingSetSize on Windows and ru_maxrss on Unix (converting
    Linux KiB to bytes). Includes imports and input loading, not just fitting.
    Subtracting two high-water marks is not an exact fit-allocation measure.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
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

    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def worker(spec):
    """Measure one variant over a parameter grid in a fresh subprocess.

    Parameters
    ----------
    spec : dict
        JSON-compatible settings: features and indices are .npy file paths;
        grid contains (eps, min_samples) pairs; variant selects on-demand,
        cached or sklearn fitting. metric, p, distance_backend and block_size
        specify distance computation.

    Returns
    -------
    dict
        Elapsed seconds, process-memory peaks in bytes, matrix-build seconds,
        cached-matrix bytes, and JSON-compatible assignments for every fit.

    Notes
    -----
    Loading and conversion precede the timer. The first cached fit includes
    matrix construction; later custom fits reuse the same model. sklearn is
    refitted for each pair. All label arrays are retained through measurement.
    Imports, process startup, loading and agreement metrics are not timed.
    """
    X = np.load(spec["features"], allow_pickle=False)
    indices = np.load(spec["indices"], allow_pickle=False)
    X = np.ascontiguousarray(X[indices], dtype=np.float64)

    # Dataset loading, imports and metrics are outside the timer.
    baseline_peak = peak_process_memory_bytes()
    started = perf_counter()
    assignments = []
    model = None
    matrix_seconds = 0.0

    for eps, minimum in spec["grid"]:
        if spec["variant"] == "sklearn_brute":
            labels = reference(X, eps, minimum, spec["metric"], spec["p"])
        else:
            if model is None:
                model = DBSCAN(
                    eps,
                    minimum,
                    spec["metric"],
                    spec["p"],
                    spec["distance_backend"],
                    spec["block_size"],
                )
                labels = model.fit_predict(
                    X, precompute=spec["variant"] == "own_cached"
                )
                matrix_seconds = getattr(model, "matrix_build_seconds_", 0.0)
            else:
                model.eps, model.min_samples = eps, minimum
                labels = model.fit_predict()
        assignments.append(labels)

    elapsed = perf_counter() - started
    peak = peak_process_memory_bytes()

    # Keep all assignments in each variant for equal output retention costs.
    return dict(
        seconds=elapsed,
        process_peak_rss_bytes=peak,
        before_fit_peak_rss_bytes=baseline_peak,
        matrix_build_seconds=matrix_seconds,
        cached_matrix_bytes=(
            0
            if model is None or model.distance_matrix_ is None
            else model.distance_matrix_.nbytes
        ),
        assignments=[a.tolist() for a in assignments],
    )


def write_csv(path, rows):
    """Write nonempty, consistently keyed records to a CSV file.

    path must be a pathlib.Path with an existing parent directory. Column
    order comes from the first record. Replaces the file, allowing the driver
    to checkpoint all completed measurements after each worker.
    """
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run correctness checks, isolated measurements and aggregate reporting.

    Reads CLI arguments, or reads a JSON specification from stdin in worker
    mode. Uses fixed seeded subsets across repetitions and randomized method
    order. Failed workers are recorded while remaining measurements continue.
    Writes raw results, agreement checks, metadata and available summaries.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Saved, numeric 2D features.npy; no embedding generation",
    )
    parser.add_argument(
        "--dataset", choices=["blobs", "moons", "smiley"], default="blobs"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[250, 500, 1000]
    )
    parser.add_argument(
        "--eps", type=float, nargs="+", default=[0.2, 0.35, 0.5]
    )
    parser.add_argument("--min-samples", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--metric",
        choices=["euclidean", "manhattan", "minkowski", "cosine"],
        default="euclidean",
    )
    parser.add_argument(
        "--distance-backend",
        choices=["baseline", "optimized"],
        default="baseline",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--max-matrix-mib",
        type=float,
        default=512.0,
        help="Fail before allocating a larger cached matrix",
    )
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/dbscan")
    )

    # Validate the requested run before allocating datasets.
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return

    if args.repeats < 1 or min(args.sizes) < 2:
        parser.error("Use positive repeats and sizes >= 2")
    if min(args.eps) <= 0 or min(args.min_samples) < 1:
        parser.error("eps must be positive and min-samples >= 1")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(
            "Output directory is not empty; choose a new name to preserve previous results"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    # Correctness checks precede performance measurements.
    checks = correctness(args.distance_backend, args.block_size)
    write_csv(args.output / "correctness.csv", checks)
    print(
        f"Passed {len(checks)} reference comparisons plus independent border assertion.",
        flush=True,
    )

    if args.correctness_only:
        return
    if args.features:
        X = np.load(args.features, allow_pickle=False)
    elif args.dataset == "smiley":
        from benchmarks.generate_testdata import generate_smiley_data

        X = generate_smiley_data()
    elif args.dataset == "moons":
        X = make_moons(
            n_samples=max(args.sizes), noise=0.03, random_state=args.seed
        )[0]
    else:
        X = make_blobs(
            n_samples=max(args.sizes),
            centers=4,
            n_features=2,
            cluster_std=0.35,
            random_state=args.seed,
        )[0]

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all() or not all(X.shape):
        parser.error("Features must be a nonempty finite numeric 2D array")
    if args.metric == "cosine" and np.any(np.linalg.norm(X, axis=1) == 0):
        parser.error("Cosine input contains zero vectors")

    sizes = sorted(set(args.sizes))
    if max(sizes) > len(X):
        parser.error(f"Largest sample size exceeds available {len(X)} rows")
    if 8 * max(sizes) ** 2 > args.max_matrix_mib * 1024**2:
        parser.error(
            "Cached matrix exceeds --max-matrix-mib; reduce sizes or explicitly increase budget"
        )

    grid = [(float(e), int(m)) for e in args.eps for m in args.min_samples]

    # Record the environment and input fingerprints for reproduction.
    metadata = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        shape=list(X.shape),
        feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
        threads=1,
        memory="Process high-water RSS including imports and input; not isolated fit allocation",
        sampling="Nested seeded subsets; fixed across repetitions; not stratified",
        dbscan_sha256=hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "dbscan.py").read_bytes()
        ).hexdigest(),
    )

    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    rows, comparisons, failures = [], [], []

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(X))

    # Each measured variant runs in a fresh process.
    with tempfile.TemporaryDirectory() as temp:
        features = Path(temp) / "features.npy"
        indices = Path(temp) / "indices.npy"
        np.save(features, X)

        for n in sizes:
            np.save(indices, order[:n])
            for scenario, current_grid in [
                ("single_fit", grid[:1]),
                ("parameter_sweep", grid),
            ]:
                for repeat in range(args.repeats):
                    outputs = {}
                    for variant in rng.permutation(VARIANTS):
                        spec = dict(
                            features=str(features),
                            indices=str(indices),
                            grid=current_grid,
                            metric=args.metric,
                            p=args.p,
                            variant=str(variant),
                            distance_backend=args.distance_backend,
                            block_size=args.block_size,
                        )
                        print(
                            (
                                f"Starting n={n}, {scenario}, repetition "
                                f"{repeat+1}/{args.repeats}: {variant}"
                            ),
                            flush=True,
                        )

                        try:
                            proc = subprocess.run(
                                [
                                    sys.executable,
                                    str(Path(__file__).resolve()),
                                    "--worker",
                                ],
                                input=json.dumps(spec),
                                text=True,
                                capture_output=True,
                                timeout=args.timeout,
                                check=True,
                            )
                        except (
                            subprocess.TimeoutExpired,
                            subprocess.CalledProcessError,
                        ) as error:
                            failures.append(
                                dict(
                                    n=n,
                                    scenario=scenario,
                                    repeat=repeat,
                                    variant=variant,
                                    status=(
                                        "timeout"
                                        if isinstance(
                                            error, subprocess.TimeoutExpired
                                        )
                                        else "error"
                                    ),
                                    worker_timeout_seconds=args.timeout,
                                    detail=str(error),
                                )
                            )
                            write_csv(args.output / "failures.csv", failures)
                            print(
                                f"  Failed: {variant}; recorded in failures.csv, continuing.",
                                flush=True,
                            )
                            continue

                        result = json.loads(proc.stdout)
                        outputs[variant] = [
                            np.asarray(a) for a in result.pop("assignments")
                        ]
                        rows.append(
                            dict(
                                n=n,
                                dimensions=X.shape[1],
                                scenario=scenario,
                                repeat=repeat,
                                variant=variant,
                                distance_backend=args.distance_backend,
                                configurations=len(current_grid),
                                **result,
                            )
                        )
                        write_csv(args.output / "timings.csv", rows)

                    for variant in VARIANTS[:2]:
                        if (
                            variant not in outputs
                            or "sklearn_brute" not in outputs
                        ):
                            continue

                        for i, (actual, expected) in enumerate(
                            zip(outputs[variant], outputs["sklearn_brute"])
                        ):
                            a = agreement(actual, expected)
                            comparisons.append(
                                dict(
                                    n=n,
                                    scenario=scenario,
                                    repeat=repeat,
                                    variant=variant,
                                    eps=current_grid[i][0],
                                    min_samples=current_grid[i][1],
                                    n_clusters=len(set(actual) - {-1}),
                                    noise_fraction=float(
                                        np.mean(actual == -1)
                                    ),
                                    **a,
                                )
                            )

                    if comparisons:
                        write_csv(args.output / "agreement.csv", comparisons)

                    print(
                        (
                            f"Finished n={n}, {scenario}, repetition "
                            f"{repeat+1}/{args.repeats}; {len(outputs)}/3 "
                            f"successful"
                        ),
                        flush=True,
                    )

    # Aggregate completed repetitions; failed workers remain recorded.
    summaries = []
    for n in sizes:
        for scenario in ("single_fit", "parameter_sweep"):
            for variant in VARIANTS:
                subset = [
                    r
                    for r in rows
                    if r["n"] == n
                    and r["scenario"] == scenario
                    and r["variant"] == variant
                ]
                if not subset:
                    continue

                summaries.append(
                    dict(
                        n=n,
                        scenario=scenario,
                        variant=variant,
                        completed_repeats=len(subset),
                        median_seconds=float(
                            np.median([r["seconds"] for r in subset])
                        ),
                        min_seconds=min(r["seconds"] for r in subset),
                        max_seconds=max(r["seconds"] for r in subset),
                        median_process_peak_rss_mib=float(
                            np.median(
                                [r["process_peak_rss_bytes"] for r in subset]
                            )
                        )
                        / 1024**2,
                    )
                )

    if summaries:
        write_csv(args.output / "summary.csv", summaries)

    mismatches = sum(
        r["ari"] != 1.0 or not r["noise_equal"] for r in comparisons
    )
    print(
        (
            f"Done. {len(failures)} failed measurements. "
            f"{mismatches} benchmark partition mismatches; "
            f"inspect agreement.csv before interpreting "
            f"performance."
        )
    )


if __name__ == "__main__":
    main()
