# Proyecto-MineriaDeDatos

## Loading the Dataset
Locate the dataset in a folder named "verbal_autopsy" parallel to the git-repository folder, not _*IN*_ the git folder.

## DBSCAN validation and performance

See [the benchmark guide](benchmarks/README.md) for reproducible correctness checks, runtime and memory measurements, and evaluation on saved verbal-autopsy feature matrices.
# DBSCAN clustering of verbal-autopsy narratives

Custom DBSCAN implementation, reproducible representation experiments, runtime
and memory benchmarks, and nearest-core assignment of unseen Child narratives.
Clusters describe narrative similarity; they are not predicted diagnoses.

## Setup

Run commands from the repository root. Keep the dataset in a sibling folder
named `verbal_autopsy`. The loader expects:

- `cleaned_PHMRC_VAI_redacted_free_text.train.csv`
- `cleaned_PHMRC_VAI_redacted_free_text.test_blind.csv`
- `VA_ConsistentRelabelling.txt` (the project's edited grouping file)

```powershell
py -m pip install -r requirements.txt
```

The dependency list is not version-locked. Preserve the working environment
used for the reported experiments rather than upgrading it before submission.
The embedding model must be cached when `embedding.local_files_only=true`.
For initial model setup, set that option to `false`. Cleaning requires NLTK
resources; set `preprocess.download=true` for initial setup when needed.
These setup operations require internet access.

## Main entry points

| File | Purpose |
|---|---|
| `dbscan.py` | Custom clustering and nearest-core prediction |
| `run_experiment.py` | Configured representation and parameter experiments |
| `conf/config.yaml` | Hydra defaults; these are not the selected Child model |
| `evaluation.py` | Library metrics and cluster-example export |
| `compare_coverage.py` | Reselect existing grids without refitting |
| `predict_child.py` | Apply a saved Child configuration to unseen data |
| `prediction_reporting.py` | Local inspection and aggregate test metrics |
| `summary.py` | Tables comparing saved experiments |
| `benchmarks/` | Correctness checks and computational-cost experiments |

`project.py`, `compare_cleaning_dense.py`, `plot_clusters_dense.py`,
`inspect_narratives.py` and `silhouette_dense.py` support earlier exploratory
analyses. They may contain experiment-specific settings; use the entry points
above for the final workflow. Files remain in their original locations to
preserve imports.

## Experiments

Example: the refined Child grid used for the final comparison:

```powershell
py run_experiment.py dataset.module=Child representation=dense `
    text_column=open_response_clean pca.enabled=false `
    clustering.metric=euclidean `
    "clustering.eps_values=[0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825]" `
    "clustering.min_samples_values=[3,4,5,6,7,9,12]" `
    selection.min_coverage=0.7
```

Results go to timestamped `experiments/` directories. Preserve the resolved
configuration, metrics, assignments, features, row alignment and summary.
Do not rerun completed experiments merely to reorganize the repository.

Compare the original 70% coverage floor with the later 50% sensitivity analysis:

```powershell
py compare_coverage.py --root experiments `
    --thresholds 0.7 0.5 --output coverage_review_final
```

The script uses saved metrics and assignments. It maximizes finite DBCV with
at least two clusters at each coverage floor; it does not use death-cause labels
for selection. Allowing 50% noise does not force a particular noise fraction.

## Unseen Child narratives

Use the same saved experiment for both configurations:

```powershell
$run = "experiments\2026-10-21\17-52-39-553828"

py predict_child.py --run-dir "$run" `
    --eps 0.725 --min-samples 5 --output predictions_child_primary

py predict_child.py --run-dir "$run" `
    --eps 0.675 --min-samples 3 --output predictions_child_comparison
```

The script validates saved training data, fits the selected training model once,
and predicts on the blind Child split using fixed preprocessing. Unseen points
join the nearest training core's cluster only within eps. Empty inputs are
separate from noise. The first configuration is the training-DBCV selection;
the second is an exploratory comparison with less training-cluster dominance.

Choose a fresh output directory. Add `--dataset-dir` for another dataset
location, or `--examples-per-cluster 0` for all local inspection examples.
`test_metrics.json` contains aggregate results; example files contain narratives.
Training labels attached to nearby examples are not predicted test outcomes.

## Validation and benchmarks

See [the benchmark guide](benchmarks/README.md). A quick correctness check:

```powershell
py benchmarks/benchmark_dbscan.py --correctness-only `
    --distance-backend optimized --output benchmark_results/checks
py benchmarks/test_unseen_prediction.py
```

Silhouette and DBCV use existing libraries. DBCV reuses the project's numerical
stabilization. Test metrics describe fixed assignments, not test-data refitting.

## Files to keep local

Datasets, narrative exports, embeddings, fitted artifacts and generated
experiment directories are ignored. Ignore rules do not remove files already
tracked by Git. Inspect `git status` and `git diff --cached` before committing.
Only reviewed aggregate results and plots should be selected for publication.
