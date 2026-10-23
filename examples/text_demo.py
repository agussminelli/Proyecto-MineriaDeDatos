"""Plot synthetic text clusters or a saved verbal-autopsy experiment.

Run from the repository root with --dataset synthetic or --dataset
verbal-autopsy. The latter requires --run-dir, --eps and --min-samples.
Only synthetic mode fits DBSCAN. Saved experiments are never refitted.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dbscan import DBSCAN
from analysis.plot_text_clusters import plot_text_clusters


def load_saved_run(run_dir, eps, min_samples):
    """Load one saved grid partition and validate its alignment with metrics."""
    required = ["features.npy", "metrics.csv", "assignments.npz"]
    missing = [name for name in required if not (run_dir / name).is_file()]

    if missing:
        raise ValueError(
            "Missing saved artifacts: "
            + ", ".join(missing)
            + ". No synthetic substitution was made. For the public demo, "
            "run: py -m examples.text_demo --dataset synthetic"
        )

    metrics = pd.read_csv(run_dir / "metrics.csv")
    selected = metrics.loc[
        np.isclose(metrics["eps"], eps, rtol=0, atol=1e-10)
        & (metrics["min_samples"] == min_samples)
    ]

    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one saved result for those parameters"
        )

    row = selected.iloc[0]
    key = str(row["run_key"])
    features = np.load(run_dir / "features.npy", allow_pickle=False)

    with np.load(run_dir / "assignments.npz", allow_pickle=False) as archive:
        labels = archive[key].copy()

    if (
        labels.ndim != 1
        or len(labels) != len(features)
        or not np.isfinite(labels).all()
        or np.any(labels != labels.astype(int))
        or np.any((labels != -1) & (labels <= 0))
    ):
        raise ValueError(
            "Saved labels must align with rows and use -1 or positive IDs"
        )

    clusters = np.unique(labels[labels > 0])

    if len(clusters) != int(row["n_clusters"]) or not np.isclose(
        np.mean(labels > 0), row["coverage"], rtol=0, atol=1e-8
    ):
        raise ValueError("Saved assignments disagree with recorded metrics")

    details = {}
    summary_path = run_dir / "summary.json"

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        details = summary.get("representation", {})

    metadata = {
        "dataset": "verbal-autopsy",
        "source_run": str(run_dir),
        "run_key": key,
        "metric": str(row.get("metric", "see resolved experiment config")),
        "text_column": details.get(
            "text_column", "see resolved experiment config"
        ),
        "refitted": False,
    }
    return features, labels.astype(int), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["synthetic", "verbal-autopsy"], required=True
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-clusters", type=int, default=12)
    args = parser.parse_args()

    if not 1 <= args.max_clusters <= 20:
        parser.error("max-clusters must be between 1 and 20")

    if args.dataset == "synthetic":
        if args.run_dir is not None:
            parser.error("run-dir only applies to verbal-autopsy mode")

        from benchmarks.generate_testdata import generate_easy_text

        eps = 0.7 if args.eps is None else args.eps
        minimum = 3 if args.min_samples is None else args.min_samples
        features, _ = generate_easy_text()
        model = DBSCAN(
            eps=eps,
            min_samples=minimum,
            metric="euclidean",
            distance_backend="optimized",
        )
        labels = model.fit_predict(features, precompute=True)
        metadata = {
            "dataset": "synthetic",
            "representation": "TF-IDF",
            "metric": "euclidean",
            "refitted": True,
        }
    else:
        if (
            args.run_dir is None
            or args.eps is None
            or args.min_samples is None
        ):
            parser.error("verbal-autopsy needs run-dir, eps and min-samples")

        eps, minimum = args.eps, args.min_samples

        try:
            features, labels, metadata = load_saved_run(
                args.run_dir, eps, minimum
            )
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))

    metadata.update({"eps": eps, "min_samples": minimum})
    output_dir = args.output_dir or (
        Path("outputs/examples") / f"text_{args.dataset}_{eps:g}_{minimum}"
    )
    title = (
        "Synthetic text: 13 phrases (TF-IDF)"
        if args.dataset == "synthetic"
        else "Verbal autopsy: saved clustering"
    )
    paths = plot_text_clusters(
        features,
        labels,
        output_dir,
        title=title,
        metadata=metadata,
        max_clusters=args.max_clusters,
        filename="text_demo",
    )

    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
