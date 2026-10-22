"""Assign unseen Child narratives to a saved DBSCAN clustering.

Example:
    py predict_child.py --run-dir experiments/... --eps 0.675 \
        --min-samples 3 --output predictions_child_0675_3
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from dbscan import DBSCAN
from loading_dataset import load_dataset
from prediction_reporting import evaluate_predictions, write_inspection
from preprocess import preprocess_dataset


def fingerprint(value):
    payload = json.dumps(value,ensure_ascii=False,sort_keys=True,).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def children(df):
    required = {"module", "open_response", "newid"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    modules = df["module"].astype(str).str.strip().str.casefold()
    return df.loc[modules == "child"].reset_index(drop=True)


def clean_texts(df, text_column):
    cleaned = preprocess_dataset(df,text_cols=["open_response"],download=False,)

    return cleaned[text_column].fillna("").astype(str)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--eps", type=float, default=0.725)
    parser.add_argument("--min-samples", type=int, default=5)

    parser.add_argument("--input",type=Path,help="Optional CSV. Otherwise use the blind test split.",)
    parser.add_argument("--output",type=Path,default=Path("predictions_child"),)
    parser.add_argument("--examples-per-cluster",type=int,default=5,help="Use 0 to export all unseen narratives.",)

    args = parser.parse_args()

    if not np.isfinite(args.eps) or args.eps <= 0:
        parser.error("eps must be finite and positive")

    if args.min_samples < 1 or args.examples_per_cluster < 0:
        parser.error("Invalid min-samples or example limit")

    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")

    return args


def check_config(cfg):
    supported = (
        str(cfg.dataset.module).casefold() == "child"
        and cfg.representation == "dense"
        and not cfg.pca.enabled
        and cfg.text_column == "open_response_clean"
        and cfg.clustering.metric == "euclidean"
    )

    if not supported:
        raise ValueError(
            "Requires Child, cleaned dense embeddings, "
            "no PCA and Euclidean distance"
        )


def fit_saved_model(run, X, eps, min_samples):
    metrics = pd.read_csv(run / "metrics.csv")

    matching = metrics.loc[
        np.isclose(metrics["eps"], eps, rtol=0, atol=1e-10)
        & (metrics["min_samples"] == min_samples)
    ]

    if len(matching) != 1:
        raise ValueError(
            f"Expected one saved result for eps={eps}, "
            f"min_samples={min_samples}"
        )

    key = str(matching.iloc[0]["run_key"])

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        distance_backend="optimized",
        block_size=256,
    )

    if not callable(getattr(model, "predict", None)):
        raise ValueError("Install your tested DBSCAN.predict first")

    model.fit(X, precompute=True)

    with np.load(run / "assignments.npz", allow_pickle=False) as saved:
        if not np.array_equal(model.labels_, saved[key]):
            raise ValueError("Refitted labels differ from saved labels")

    return model, key


def describe_prediction(point, label, training, X, model):
    core_indices = np.asarray(model.core_sample_indices_)
    nearest_core = None

    if len(core_indices):
        distances = np.linalg.norm(X[core_indices] - point, axis=1)
        best = int(np.argmin(distances))
        index = int(core_indices[best])
        reference = training.iloc[index]

        nearest_core = {
            "cluster": int(model.labels_[index]),
            "distance": float(distances[best]),
            "text": str(reference["open_response"]),
            "gs_text34": str(reference.get("gs_text34", "<missing>")),
            "gs_grouped": str(reference.get("gs_grouped", "<missing>")),
        }

    neighbors = []

    if label != -1:
        members = np.flatnonzero(model.labels_ == label)
        distances = np.linalg.norm(X[members] - point, axis=1)

        for position in np.argsort(distances, kind="stable")[:3]:
            reference = training.iloc[members[position]]
            neighbors.append({
                "distance": float(distances[position]),
                "text": str(reference["open_response"]),
            })

    return nearest_core, neighbors


def main():
    args = parse_arguments()
    run = args.run_dir
    cfg = OmegaConf.load(run / "config_resolved.yaml")
    check_config(cfg)

    # Validate the saved training data and preprocessing.
    train, test, headers, test_headers, _ = load_dataset(
        args.dataset_dir,
        apply_grouping=True,
    )

    training = children(pd.DataFrame(train, columns=headers))
    saved_ids = pd.read_csv(run / "rows.csv", dtype=str,)["newid"].to_numpy()

    training_ids = training["newid"].astype(str).to_numpy()

    if not np.array_equal(training_ids, saved_ids):
        raise ValueError("Training IDs/order differ from saved experiment")

    train_texts = clean_texts(training, cfg.text_column).tolist()
    summary = json.loads(
        (run / "summary.json").read_text(encoding="utf-8")
    )
    saved_fingerprint = summary["representation"]["text_fingerprint"]

    if fingerprint(train_texts) != saved_fingerprint:
        raise ValueError(
            "Training text/preprocessing changed. "
            "Restore the experiment's preprocessing before prediction."
        )

    X = np.load(run / "features.npy", allow_pickle=False)

    if X.shape != (len(training), 384) or not np.isfinite(X).all():
        raise ValueError("Expected finite training embeddings, shape (n, 384)")

    model, key = fit_saved_model(
        run,
        X,
        args.eps,
        args.min_samples,
    )

    # Transform unseen inputs without fitting preprocessing on them.
    if args.input:
        incoming = pd.read_csv(args.input, dtype={"newid": str})
    else:
        incoming = pd.DataFrame(test, columns=test_headers)

    incoming = children(incoming)

    if incoming.empty:
        raise ValueError("No Child narratives in input")

    if set(incoming["newid"].astype(str)) & set(saved_ids):
        raise ValueError("Input IDs overlap training IDs")

    raw = incoming["open_response"].fillna("").astype(str)
    texts = clean_texts(incoming, cfg.text_column)
    valid = raw.str.strip().ne("") & texts.str.strip().ne("")

    output = pd.DataFrame({
        "input_row": np.arange(1, len(incoming) + 1),
        "cluster": -1,
        "status": "empty_text",
        "nearest_core_distance": np.nan,
        "same_cleaned_text_in_training": texts.isin(set(train_texts)),
    })

    examples = []
    vectors = np.empty((0, X.shape[1]))
    predictions = np.empty(0, dtype=int)

    if valid.any():
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            cfg.embedding.model_name,
            local_files_only=bool(cfg.embedding.local_files_only),
        )
        vectors = np.asarray(
            encoder.encode(
                texts[valid].tolist(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float64,
        )
        predictions = model.predict(vectors)

        for row, point, label in zip(
            np.flatnonzero(valid), vectors, predictions
        ):
            output.loc[row, "cluster"] = int(label)
            output.loc[row, "status"] = (
                "noise" if label == -1 else "assigned"
            )

            core, neighbors = describe_prediction(
                point, label, training, X, model
            )

            if core is not None:
                output.loc[row, "nearest_core_distance"] = core["distance"]

            examples.append({
                "input_row": int(row + 1),
                "text": str(raw.iloc[row]),
                "cluster": int(label),
                "nearest_core": core,
                "training_examples": neighbors,
            })

    # Save local inspection files and narrative-free aggregate metrics.
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "assignments.csv", index=False)

    np.savez_compressed(
        args.output / "prediction_vectors.npz",
        vectors=vectors,
        labels=predictions,
        input_rows=np.flatnonzero(valid) + 1,
    )

    write_inspection(
        args.output / "prediction_examples.txt",
        training,
        model,
        examples,
        args.examples_per_cluster,
    )

    scores = evaluate_predictions(vectors, predictions)
    metadata = {
        "training_run": str(run.resolve()),
        "run_key": key,
        "eps": args.eps,
        "min_samples": args.min_samples,
        "selection_min_coverage": 0.5,
        "original_config_min_coverage": float(cfg.selection.min_coverage),
        "rule": "nearest training core within eps; otherwise noise",
        "training_clusters_frozen": True,
        "embedding_model": str(cfg.embedding.model_name),
        "normalized_embeddings": True,
        "preprocessing_text_fingerprint": fingerprint(train_texts),
        "n_input": len(output),
        "counts": output["status"].value_counts().to_dict(),
        "note": (
            "input_row is 1-based within filtered Child input. "
            "empty_text is not a DBSCAN noise decision. "
            "Example files contain narratives; review before sharing."
        ),
    }

    for filename, content in [
        ("test_metrics.json", scores),
        ("prediction_summary.json", metadata),
        ("examples.json", examples),
    ]:
        (args.output / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    print(output["status"].value_counts().to_string())
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
