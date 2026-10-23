"""Test the explicit nearest-core extension to custom DBSCAN."""
if __package__:
    from . import benchmark_dbscan as bench
else:
    import benchmark_dbscan as bench
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN as SKDBSCAN
from dbscan import DBSCAN


def main():
    X, _ = make_blobs(n_samples=100, centers=[[-2.,0.],[2.,0.]],
                      cluster_std=.22, random_state=42)
    unseen = np.array([[-2.05,.02],[2.02,-.04],[20.,20.]])
    checked=0
    for backend in ('baseline','optimized'):
        for metric,eps,p in [('euclidean',.45,2),('manhattan',.6,2),('minkowski',.5,3),('cosine',.015,2)]:
            # Cosine test points must have nonzero norms; include a zero case separately.
            model=DBSCAN(eps,4,metric,p,distance_backend=backend).fit(X,precompute=True)
            predicted=model.predict(unseen)
            assert predicted[0]>0 and predicted[1]>0 and predicted[2]==-1,(backend,metric,predicted)
            assert np.all(model.nearest_core_indices_[:2]>=0)
            assert model.nearest_core_indices_[2]==-1
            core=model.core_sample_indices_
            assert np.all(model.core_sample_mask_[core])
            # Each assigned label comes from the nearest fitted core point.
            for j in range(2):
                assert predicted[j] == model.labels_[model.nearest_core_indices_[j]]
            texts=[f'training narrative {i}' for i in range(len(X))]
            examples=model.nearest_examples(unseen[0],texts,n=5)
            assert len(examples)==5 and all(e['cluster']==predicted[0] for e in examples)
            assert all(examples[i]['distance'] <= examples[i+1]['distance'] for i in range(4))
            assert examples[0]['index']==int(model.nearest_core_indices_[0]) or model.labels_[examples[0]['index']]==predicted[0]
            assert model.nearest_examples(unseen[2],texts,n=5)==[]
            checked+=1
    try:
        DBSCAN(.2,2,'cosine').fit(X).predict([[0.,0.]])
    except ValueError as exc:
        assert 'zero vectors' in str(exc)
    else:
        raise AssertionError('Cosine prediction accepted a zero vector')
    try:
        DBSCAN().predict([[1.,2.]])
    except ValueError as exc:
        assert 'Call fit' in str(exc)
    else:
        raise AssertionError('Prediction before fit was accepted')
    print(f'Passed nearest-core/noise prediction checks for {checked} backend/metric combinations, plus zero-vector and fit-state validation.')


if __name__=='__main__':main()
