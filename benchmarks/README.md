# DBSCAN correctness and computational cost

Run commands from the repository root. This benchmark compares the custom implementation in on-demand and cached modes with sklearn DBSCAN using exhaustive (`brute`) search. It does not train embeddings or tune against death-cause labels.

## Setup and first run

```bash
python -m pip install -r benchmarks/requirements.txt
python benchmarks/benchmark_dbscan.py --correctness-only --output outputs/benchmarks/checks
python benchmarks/benchmark_dbscan.py --sizes 100 250 500 --repeats 3 --output outputs/benchmarks/blobs
python benchmarks/plot_results.py outputs/benchmarks/blobs
```

The benchmark supports Windows through the process-memory API and Linux/macOS through `resource` peak-memory measurement. It sets numerical thread counts to one. The core DBSCAN imports only NumPy; matplotlib is loaded only for its plotting method.

## Synthetic examples

```bash
python benchmarks/benchmark_dbscan.py --dataset moons --sizes 250 500 1000 --eps .1 .15 .2 --min-samples 4 8 --output outputs/benchmarks/moons
python benchmarks/benchmark_dbscan.py --dataset smiley --sizes 200 400 790 --eps .08 .12 .16 --min-samples 4 8 --output outputs/benchmarks/smiley
```

The existing smiley generator has 790 points and a fixed seed of 42. Blobs and moons use `--seed`. Each run constructs one dataset and takes nested subsets using a seeded permutation. Repetitions use the same subsets; they measure timing variation, not sampling stability. Sparse subsets may have more noise at fixed parameters. This is why assignments and noise fractions are recorded. Synthetic generator labels are not used.

## Verbal-autopsy features

Use an existing `features.npy` produced by `run_experiment.py`. No raw narratives or death-cause labels are needed. Preserve the run's configuration and `rows.csv` separately to establish representation and row provenance.

Example only: replace the path and parameters with your actual experiment settings. Choose sizes no larger than the available row count.

```bash
python benchmarks/benchmark_dbscan.py --features experiments/YOUR_RUN/features.npy --sizes 250 500 1000 --metric cosine --eps .2 .3 .4 --min-samples 5 9 --repeats 3 --output outputs/benchmarks/child
python benchmarks/plot_results.py outputs/benchmarks/child
```

A matrix containing multiple modules is not automatically filtered. Supply a matrix containing the intended module if reporting module-specific results. The feature file is validated as finite, numeric and two-dimensional; cosine rejects zero vectors. Each cached matrix is limited to 512 MiB by default, but the process can use more memory. Increase `--max-matrix-mib` only when appropriate. Each worker has a 300-second timeout, adjustable with `--timeout`.

## Protocol and interpretation

- **Correctness:** 50 comparisons cover blobs, moons, all noise, one cluster, duplicates, noise-to-border reassignment, self-neighbour counting, inclusive epsilon, Euclidean/Manhattan/Minkowski distances, and cosine directional clusters. A separate assertion verifies the expected border reassignment. These are finite checks, not a proof of correctness.
- **Partition agreement:** ARI compares custom labels against sklearn labels, not against death causes. Cluster IDs are arbitrary. Noise masks are compared separately. Correctness fixtures must pass before benchmarking. Any benchmark mismatch is recorded, not hidden: inspect ambiguous border points and distance-threshold rounding before attributing it to a bug.
- **Single fit:** first epsilon/min-samples combination supplied. Cached timing includes matrix construction.
- **Parameter sweep:** Cartesian product of all supplied values. Cached mode reuses one matrix. On-demand mode reuses input but calculates neighbourhood distances again. sklearn fits a fresh estimator per combination. Every method retains all label arrays. Selection, plotting, loading, and comparison metrics are outside the timer.
- **Repeats:** three by default; execution order is seeded and randomized per repetition. Every measurement starts a new Python process. Import time is excluded. Worker startup, input reading and process teardown are excluded.
- **Memory:** `process_peak_rss_bytes` is OS-reported process high-water resident memory. It includes imports, loading, input data and the fit. `before_fit_peak_rss_bytes` is the high-water mark before timing, not current RSS. Do not subtract these fields and call the result exact algorithm memory. Loading a large full feature file can dominate the peak even for a small subset. The same input-loading protocol is used for all methods.
- **Matrix accounting:** `cached_matrix_bytes` is exact for the custom matrix: 8*n*n bytes for float64. It is not total peak memory. A small dataset's process-memory baseline can hide the difference between variants.
- **Fairness:** exhaustive sklearn search controls the neighbour-search strategy, but sklearn has optimized internals. These measurements compare implementations, not identical low-level operations. A default/auto sklearn benchmark would be a separate practical comparison.

## Output files

| File | Meaning |
| --- | --- |
| `correctness.csv` | Individual small-data reference checks |
| `timings.csv` | Raw repeated timings and memory observations |
| `agreement.csv` | Per-configuration partition agreement, cluster count and noise fraction |
| `summary.csv` | Median time, min/max time and median peak RSS |
| `metadata.json` | Arguments, versions, system, feature hash and implementation hash |
| `runtime.png`, `memory.png` | Figures created by the plotting script |

Use a distinct output directory for each run; files in a reused directory are overwritten. A failed or timed-out run may leave partial raw CSVs; only a completed run produces a new summary. Check the command's exit status and completion message before plotting.

## Complexity discussion for the report

For n points with d features, exhaustive neighbourhood calculations have an O(n²d) distance-work upper bound per fit. This implementation can also inspect O(n²) neighbour relations. On-demand distance calculation avoids storing an n-by-n matrix, but holds the O(nd) input, O(nd) temporary differences for some metrics, and O(n) clustering/queue arrays. The baseline cosine backend recalculates norms during queries; the optimized backend caches training-vector norms.

Cached mode stores an additional O(n²) matrix and performs its O(n²d) construction once. Across G parameter combinations, distance work can therefore be amortized, while each fit still scans matrix rows and expands neighbourhoods. Worst-case sweep work is O(n²d + G*n²), versus O(G*n²d) exhaustive on-demand work. These are upper bounds; report observed costs alongside them and do not infer asymptotic scaling from only a few small measurements. Sklearn brute search can also consume quadratic memory through neighbourhood storage.

Caching is an established technique. The project's evidence is its verified implementation and measured time-memory trade-off, not a claim to invent caching.
