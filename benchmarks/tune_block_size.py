"""Tune cached Euclidean block size; requires the updated benchmark_dbscan.py."""
# Import this first: it sets numerical-library threads to one before NumPy loads.
import benchmark_dbscan as bench
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--blocks', nargs='+', type=int, default=[128, 256, 512, 1024])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--eps', nargs='+', type=float, default=[.8, .7, .9])
    parser.add_argument('--min-samples', nargs='+', type=int, default=[9, 5])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--max-matrix-mib', type=float, default=512)
    parser.add_argument('--output', type=Path, default=Path('benchmark_results/block_size_tuning'))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(bench.worker(json.load(sys.stdin))))
        return
    if not args.features or args.repeats < 1 or min(args.blocks) < 1:
        parser.error('Provide --features, positive --blocks and positive --repeats')
    if min(args.eps) <= 0 or min(args.min_samples) < 1:
        parser.error('eps and min-samples must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new output directory to preserve existing results')
    X = np.load(args.features, allow_pickle=False)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
        parser.error('Features must be finite, nonempty and two-dimensional')
    if X.shape[0]**2 * 8 > args.max_matrix_mib * 2**20:
        parser.error('Distance matrix exceeds memory budget')
    blocks = sorted(set(args.blocks) | {256})
    args.output.mkdir(parents=True)
    grid = [(float(e), int(m)) for e in args.eps for m in args.min_samples]
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    tested_blocks=blocks, shape=list(X.shape), threads=1,
                    feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                    numpy=np.__version__, sklearn=bench.sklearn.__version__,
                    python=sys.version, platform=bench.platform.platform(),
                    dbscan_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'dbscan.py').read_bytes()).hexdigest(),
                    protocol='One fresh worker per cached six-configuration sweep; no slow baseline or on-demand fits. All assignments checked against one sklearn reference sweep. Blocks randomized each repetition.',
                    memory='OS peak process resident memory, including imports and input; not isolated allocations')
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rows, comparisons, failures = [], [], []
    rng = np.random.default_rng(args.seed)
    with tempfile.TemporaryDirectory() as temp:
        feature_path=Path(temp)/'features.npy'; index_path=Path(temp)/'indices.npy'
        np.save(feature_path,X);np.save(index_path,rng.permutation(len(X)))
        spec=dict(features=str(feature_path),indices=str(index_path),grid=grid,metric='euclidean',p=2,
                  variant='sklearn_brute',distance_backend='optimized',block_size=256)
        def run(spec):
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],
                                input=json.dumps(spec),text=True,capture_output=True,
                                timeout=args.timeout,check=True)
            return json.loads(proc.stdout)
        print('Running one sklearn reference sweep for correctness...',flush=True)
        try:
            reference=run(spec)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
            (args.output/'reference_error.txt').write_text(str(error))
            raise SystemExit('Reference failed; details saved. No candidates were timed.')
        ref_labels=[np.asarray(a) for a in reference.pop('assignments')]
        (args.output/'reference.json').write_text(json.dumps(reference,indent=2))
        spec['variant']='own_cached'
        for repeat in range(args.repeats):
            for block in rng.permutation(blocks):
                block=int(block);spec['block_size']=block
                print(f'Starting block={block}, repetition {repeat+1}/{args.repeats}',flush=True)
                try:
                    result=run(spec)
                except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
                    failures.append(dict(block_size=block,repeat=repeat,status=type(error).__name__,detail=str(error)))
                    bench.write_csv(args.output/'failures.csv',failures)
                    print('  Failed; recorded, continuing.',flush=True)
                    continue
                labels=[np.asarray(a) for a in result.pop('assignments')]
                all_equal=True
                for (eps,minimum),actual,expected in zip(grid,labels,ref_labels):
                    a=bench.agreement(actual,expected)
                    all_equal &= a['ari']==1 and a['noise_equal']
                    comparisons.append(dict(block_size=block,repeat=repeat,eps=eps,min_samples=minimum,**a))
                rows.append(dict(block_size=block,repeat=repeat,all_agree=bool(all_equal),**result))
                bench.write_csv(args.output/'timings.csv',rows)
                bench.write_csv(args.output/'agreement.csv',comparisons)
                print(f"  Done: sweep={result['seconds']:.3f}s; matrix={result['matrix_build_seconds']:.3f}s; agreement={all_equal}",flush=True)
    summary=[]
    for block in blocks:
        rr=[r for r in rows if r['block_size']==block]
        if not rr: continue
        times=[r['seconds'] for r in rr]
        summary.append(dict(block_size=block,completed_repeats=len(rr),
            all_agree=all(r['all_agree'] for r in rr),median_sweep_seconds=float(np.median(times)),
            min_sweep_seconds=min(times),max_sweep_seconds=max(times),
            median_matrix_seconds=float(np.median([r['matrix_build_seconds'] for r in rr])),
            median_process_peak_mib=float(np.median([r['process_peak_rss_bytes'] for r in rr]))/2**20))
    if summary: bench.write_csv(args.output/'summary.csv',summary)
    eligible=[r for r in summary if r['completed_repeats']==args.repeats and r['all_agree']]
    baseline=next((r for r in eligible if r['block_size']==256),None)
    print(f'Finished. {len(failures)} failed measurements.',flush=True)
    if eligible and baseline:
        best=min(eligible,key=lambda r:r['median_sweep_seconds'])
        gain=1-best['median_sweep_seconds']/baseline['median_sweep_seconds']
        recommendation=best['block_size'] if gain>=.05 else 256
        result=dict(fastest_observed_block=best['block_size'],reduction_vs_256_percent=100*gain,
                    provisional_recommended_block=recommendation,
                    rule='Retain 256 unless a fully validated candidate reduces median total sweep time by at least 5%. This is a practical threshold, not statistical significance; inspect ranges and memory before changing the default.')
        (args.output/'recommendation.json').write_text(json.dumps(result,indent=2))
        print(f'Fastest median: block {best["block_size"]}; reduction vs 256: {100*gain:.1f}%. Provisional choice: {recommendation}.')
    else:
        print('No recommendation: complete validated candidate results including block 256 are required.')


if __name__=='__main__':
    main()
