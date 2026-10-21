import json
from pathlib import Path

import numpy as np
import pandas as pd


def create_cost_table(runs, min_coverage=0.70):
    """Compare recorded matrix and cached-fit costs.

    Uses the same parameter-selection rule as the quality table.
    Missing timings remain NaN.
    """
    records = []

    for setup, folder in runs.items():
        folder = Path(folder)
        metrics = pd.read_csv(folder / "metrics.csv")

        with (folder / "summary.json").open(
            encoding="utf-8"
        ) as file:
            summary = json.load(file)

        runtime = summary.get("runtime", {})

        X = np.load(
            folder / "features.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        n_samples, dimensions = X.shape
        vector_mb = X.nbytes / 1_000_000
        del X

        eligible = metrics.loc[
            (metrics["coverage"] >= min_coverage)
            & (metrics["n_clusters"] >= 2)
            & np.isfinite(metrics["dbcv"])
        ]

        record = {
            "Setup": setup,
            "Instances": n_samples,
            "Dimensions": dimensions,
            "eps": np.nan,
            "min_samples": np.nan,
            "Matrix construction (s)": runtime.get(
                "matrix_build_seconds", np.nan
            ),
            "Selected fit, cached (s)": np.nan,
            "Vectors (MB)": vector_mb,
            # Your DBSCAN stores the full matrix as float64.
            "Distance matrix (MB)": n_samples**2 * 8 / 1_000_000,
            "Status": "no eligible configuration",
        }

        if not eligible.empty:
            best = eligible.sort_values(
                ["dbcv", "coverage", "eps", "min_samples"],
                ascending=[False, False, True, True],
            ).iloc[0]

            record.update({
                "eps": best["eps"],
                "min_samples": int(best["min_samples"]),
                "Selected fit, cached (s)": best.get(
                    "fit_seconds_cached", np.nan
                ),
                "Status": "selected",
            })

        records.append(record)

    return pd.DataFrame(records)


def create_comparison_table(runs, min_coverage=0.70):
    """Summarise explicitly selected experiment folders.

    runs: dictionary mapping descriptive setup names to run directories.
    """
    records = []

    for setup, folder in runs.items():
        folder = Path(folder)
        metrics = pd.read_csv(folder / "metrics.csv")

        # Read shape without loading the whole feature matrix.
        X = np.load(
            folder / "features.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        dimensions = X.shape[1]
        del X

        eligible = metrics.loc[
            (metrics["coverage"] >= min_coverage)
            & (metrics["n_clusters"] >= 2)
            & np.isfinite(metrics["dbcv"])
        ]

        row = {
            "Setup": setup,
            "Dimensions": dimensions,
            "Status": "no eligible configuration",
        }

        if not eligible.empty:
            best = eligible.sort_values(
                ["dbcv", "coverage", "eps", "min_samples"],
                ascending=[False, False, True, True],
            ).iloc[0]

            row.update({
                "Status": "selected",
                "eps": best["eps"],
                "min_samples": int(best["min_samples"]),
                "Clusters": int(best["n_clusters"]),
                "Coverage (%)": 100 * best["coverage"],
                "DBCV": best["dbcv"],
                "Silhouette": best["silhouette"],
                "ARI detailed": best["ari_all"],
                "ARI grouped": best.get("ari_grouped_all", np.nan),
            })

        records.append(row)

    return pd.DataFrame(records)


if __name__ == "__main__":
    
    root = Path("experiments/2026-10-21")

    runs = {
        "Dense noPrep, noPCA": root / "08-41-34-152913", 
        "Dense noPCA": root / "09-55-24-950284",
        "Dense PCA95": root / "10-43-12-589093",
        "Dense PCA80": root / "10-45-48-029192",
        "LSA 50": root / "11-03-47-830903" / "0",
        "LSA 100": root / "11-03-47-830903" / "1",
        "Hybrid PCA95, alpha=0.25": root / "11-32-21-928658" / "0",
        "Hybrid PCA95, alpha=0.50": root / "11-32-21-928658" / "1",
        "Hybrid PCA95, alpha=0.75": root / "11-32-21-928658" / "2",
    }

    comparison = create_comparison_table(runs)

    comparison.to_csv("experiment_comparison.csv", index=False)

    print(comparison.round({
        "Coverage (%)": 1,
        "DBCV": 3,
        "Silhouette": 3,
        "ARI detailed": 4,
        "ARI grouped": 4,
    }).to_string(index=False))

    costs = create_cost_table(runs, min_coverage=0.70)

    costs.to_csv("experiment_costs.csv", index=False)

    print(costs.round(3).to_string(index=False))