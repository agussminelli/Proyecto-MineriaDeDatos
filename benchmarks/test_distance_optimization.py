"""Run from the repository root: python benchmarks/test_distance_optimization.py."""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dbscan import DBSCAN

if __package__:
    from .benchmark_dbscan import correctness, agreement
else:
    from benchmark_dbscan import correctness, agreement


def main():
    """Check optimized distances, cache invalidation and partition equivalence.

    Also prints an illustrative synthetic timing comparison; it is not a
    replacement for isolated repeated benchmark measurements.
    """
    rng = np.random.default_rng(42)
    datasets = {
        "random_384d": rng.normal(size=(80, 384)),
        "large_offset": 1e10 + rng.normal(size=(35, 8)),
        "duplicates": np.repeat(rng.normal(size=(8, 10)), 3, axis=0),
        "near_duplicates": np.vstack(
            [np.ones((3, 5)), np.ones((3, 5)) + 1e-12, np.zeros((3, 5))]
        ),
        "threshold": np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [np.nextafter(1.0, 2.0), 1.0]]
        ),
    }

    for backend in ("baseline", "optimized"):
        checks = correctness(backend)
        print(f"{backend}: {len(checks)} sklearn comparisons passed")

    count = 0
    for name, X in datasets.items():
        base = DBSCAN(0.5, 2).fit(X, precompute=True)

        for block in (1, 7, 256):
            fast = DBSCAN(
                0.5, 2, distance_backend="optimized", block_size=block
            ).fit(X, precompute=True)
            np.testing.assert_allclose(
                fast.distance_matrix_,
                base.distance_matrix_,
                atol=1e-10,
                rtol=1e-10,
            )
            np.testing.assert_array_equal(np.diag(fast.distance_matrix_), 0)

            # Includes exact and adjacent floating-point distance thresholds.
            for eps in (
                0.000000000001,
                0.5,
                1.0,
                np.nextafter(1.0, 0.0),
                np.nextafter(1.0, 2.0),
                3.0,
                30.0,
            ):
                base.eps = fast.eps = eps
                a = base.fit_predict()
                b = fast.fit_predict()
                assert agreement(a, b) == {"ari": 1.0, "noise_equal": True}, (
                    name,
                    block,
                    eps,
                )
                direct = DBSCAN(
                    eps, 2, distance_backend="optimized", block_size=block
                ).fit_predict(X)
                assert agreement(a, direct) == {
                    "ari": 1.0,
                    "noise_equal": True,
                }, (name, block, eps, "on-demand")
                count += 2

    X = rng.normal(size=(12, 3))
    model = DBSCAN(0.8, 2, distance_backend="optimized").fit(
        X, precompute=True
    )
    old = model.distance_matrix_
    model.distance_backend = "baseline"
    model.fit(precompute=True)
    assert model.distance_matrix_ is not old
    model.distance_backend = "optimized"
    model.fit(X * 3, precompute=True)
    expected = DBSCAN(0.8, 2).fit(X * 3, precompute=True)
    np.testing.assert_allclose(
        model.distance_matrix_, expected.distance_matrix_, atol=1e-12
    )
    print(
        (
            f"{count} baseline/optimized partition comparisons "
            f"passed; matrix accuracy and cache invalidation "
            f"passed."
        )
    )

    X = rng.normal(size=(1000, 384))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for backend in ("baseline", "optimized"):
        times = []
        for _ in range(3):
            m = DBSCAN(0.8, 9, distance_backend=backend)
            t = time.perf_counter()
            m.fit(X, precompute=True)
            times.append(time.perf_counter() - t)
        print(
            (
                f"Synthetic 1000 x 384 cached fit, {backend}: "
                f"median={np.median(times):.4f}s, "
                f"range={min(times):.4f}–{max(times):.4f}s"
            )
        )


if __name__ == "__main__":
    main()
