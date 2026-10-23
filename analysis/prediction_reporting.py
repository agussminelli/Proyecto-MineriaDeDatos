"""Local narrative inspection and evaluation of fixed predictions."""

import numpy as np


def evaluate_predictions(vectors, labels):
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    assigned = labels > 0
    cluster_ids, sizes = np.unique(
        labels[assigned],
        return_counts=True,
    )

    result = {
        "evaluation": "fixed predictions on valid unseen inputs only; no refitting",
        "n_valid": len(labels),
        "n_assigned": int(assigned.sum()),
        "coverage": float(assigned.mean()) if len(labels) else None,
        "n_clusters_represented": len(cluster_ids),
        "cluster_sizes": {
            str(int(cluster)): int(size)
            for cluster, size in zip(cluster_ids, sizes)
        },
        "largest_cluster_share_assigned": (
            float(sizes.max() / sizes.sum()) if len(sizes) else None
        ),
        "silhouette": None,
        "dbcv": None,
    }

    # Silhouette describes assigned points only; noise is not a cluster.
    if 2 <= len(cluster_ids) < assigned.sum():
        score = float(silhouette_score(
            vectors[assigned],
            labels[assigned],
            metric="euclidean",
        ))

        result["silhouette"] = score if np.isfinite(score) else None
        result["silhouette_status"] = "assigned points only"
    else:
        result["silhouette_status"] = (
            "undefined: requires 2 <= clusters < assigned points"
        )

    # Reuse training's stabilized library DBCV implementation.
    if len(cluster_ids) >= 2 and np.all(sizes >= 2):
        try:
            from analysis.evaluation import dbcv_module, prepare_dbcv_labels
            from sklearn.metrics import pairwise_distances

            distances = pairwise_distances(vectors, metric="euclidean")
            np.fill_diagonal(distances, 0)

            score = float(dbcv_module.validity_index(
                distances,
                prepare_dbcv_labels(labels),
                metric="precomputed",
                d=vectors.shape[1],
            ))

            result["dbcv"] = score if np.isfinite(score) else None
            result["dbcv_status"] = (
                "includes noise in total-size weighting"
                if np.isfinite(score)
                else "undefined/nonfinite"
            )

        except (
            ImportError,
            ValueError,
            RuntimeError,
            FloatingPointError,
        ) as error:
            result["dbcv_status"] = (
                f"unavailable: {type(error).__name__}: {error}"
            )
    else:
        result["dbcv_status"] = (
            "not calculated: requires >=2 clusters, each with >=2 points"
        )

    return result


def add_training_example(lines, row):
    for field in ["gs_text34", "gs_grouped"]:
        value = row.get(field, "<missing>")
        lines.append(f"{field}: {value}")

    lines.append(str(row["open_response"]))
    lines.append("")


def add_label_distributions(lines, members):
    for field in ["gs_text34", "gs_grouped"]:
        lines.append(f"Training {field} distribution:")

        if field not in members:
            continue

        counts = members[field].fillna("<missing>").value_counts()

        for value, count in counts.items():
            share = count / len(members)
            lines.append(f"  {value}: {count} ({share:.1%})")


def add_unseen_example(lines, entry):
    lines.extend(["\n[Unseen narrative]", entry["text"]])
    core = entry.get("nearest_core")

    if core is None:
        lines.append("No training core points available.")
        return

    lines.extend([
        (
            f"[Nearest training core: cluster {core['cluster']}, "
            f"distance {core['distance']:.6f}]"
        ),
        f"gs_text34: {core['gs_text34']}",
        f"gs_grouped: {core['gs_grouped']}",
        core["text"],
    ])

    if entry["cluster"] == -1:
        lines.append(
            "Outside eps: this core is a comparison, not an assignment."
        )


def write_inspection(path, training, model, examples, limit=5):
    """Write sampled examples; limit=0 includes all unseen narratives."""
    rng = np.random.default_rng(42)

    lines = [
        "UNSEEN CHILD ASSIGNMENTS",
        "Local report: contains narratives. No original IDs exported.",
        "Reference labels describe TRAINING examples, not unseen outcomes.",
        "Sampling seed: 42. Cluster counts use complete groups.",
    ]

    cluster_ids = list(np.unique(model.labels_[model.labels_ > 0]))

    for cluster in cluster_ids + [-1]:
        members = training.loc[model.labels_ == cluster]
        new = [entry for entry in examples if entry["cluster"] == cluster]

        lines.extend([
            "\n" + "=" * 60,
            f"Cluster {cluster}" if cluster != -1 else "NOISE",
            f"Training narratives: {len(members)}",
            f"Unseen narratives: {len(new)}",
        ])
        add_label_distributions(lines, members)

        if len(members):
            lines.append("\nTRAINING EXAMPLES (up to 5)")
            training_sample = rng.choice(
                len(members),
                size=min(5, len(members)),
                replace=False,
            )

            for index in training_sample:
                add_training_example(lines, members.iloc[index])

        if limit == 0:
            selected = np.arange(len(new))
        else:
            selected = np.sort(rng.choice(
                len(new),
                size=min(limit, len(new)),
                replace=False,
            ))

        for index in selected:
            add_unseen_example(lines, new[index])

    path.write_text("\n".join(lines), encoding="utf-8")
