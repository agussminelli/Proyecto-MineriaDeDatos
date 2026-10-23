"""Verify cosine norm caching: python benchmarks/test_cosine_norm_cache.py."""
if __package__:
    from . import benchmark_dbscan as bench
else:
    import benchmark_dbscan as bench  # Sets single-thread environment before NumPy imports.
import numpy as np
from unittest.mock import patch
from dbscan import DBSCAN


def main():
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal([1, 0, 0], .03, size=(30, 3)),
                   rng.normal([0, 1, 0], .03, size=(30, 3)), [[0, 0, 1]]])
    X = np.vstack([X, X[:3]])  # Include exact duplicates.

    comparisons = 0
    for cached in (False, True):
        base = DBSCAN(.02, 4, 'cosine').fit(X, precompute=cached)
        fast = DBSCAN(.02, 4, 'cosine', distance_backend='optimized').fit(X, precompute=cached)
        if cached:
            np.testing.assert_array_equal(base.distance_matrix_, fast.distance_matrix_)
        for eps in (.005, .02, .3, 1.1):
            for minimum in (2, 4, 8):
                base.eps = fast.eps = eps
                base.min_samples = fast.min_samples = minimum
                a, b = base.fit_predict(), fast.fit_predict()
                np.testing.assert_array_equal(a, b)
                ref = bench.reference(X, eps, minimum, 'cosine', 2)
                assert bench.agreement(b, ref) == {'ari': 1., 'noise_equal': True}
                comparisons += 1

    # Count actual norm calls across repeated fits, queries, and matrix building.
    original_norm = np.linalg.norm
    with patch.object(np.linalg, 'norm', wraps=original_norm) as counter:
        model = DBSCAN(.02, 4, 'cosine', distance_backend='optimized').fit(X)
        assert counter.call_count == 1

        for _ in range(3):
            model.fit()
            model._distances_from_point(0)

        model._ensure_distance_matrix()
        model.eps = .3
        model.fit()
        assert counter.call_count == 1, counter.call_count

        # Metric switching with unchanged X may safely retain cached norms.
        model.metric = 'euclidean'; model.fit()
        model.metric = 'cosine'; model.fit()
        assert counter.call_count == 1

        # New data with unchanged shape must invalidate the cache as well.
        Y = X * rng.uniform(.5, 2., size=(len(X), 1))
        model.fit(Y, precompute=True)
        assert counter.call_count == 2

    np.testing.assert_array_equal(model._cosine_norms_, original_norm(Y, axis=1))
    expected = DBSCAN(.3, 4, 'cosine').fit(Y, precompute=True)
    np.testing.assert_array_equal(model.distance_matrix_, expected.distance_matrix_)

    for cached in (False, True):
        try:
            model.fit(np.vstack([X, np.zeros((1, 3))]), precompute=cached)
        except ValueError as error:
            assert 'zero vectors' in str(error)
        else:
            raise AssertionError('Zero vector was accepted')
        model.fit(X, precompute=cached)  # Recovery after rejected input.
        
    print(f'Passed {comparisons} cosine configurations against baseline and sklearn.')
    print('Exact cached-distance agreement passed. Norms computed once per dataset; reuse, invalidation, metric switching and zero-vector handling passed.')


if __name__ == '__main__':
    main()
