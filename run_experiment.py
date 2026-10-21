import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import hydra
import joblib
import numpy as np
import pandas as pd

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize

from dbscan import DBSCAN
from evaluation import evaluate_parameters, save_cluster_examples
from loading_dataset import load_dataset
from preprocess import preprocess_dataset
from vectorization.dense import load_or_embed_texts
from vectorization.sparse import apply_ngrams_and_vectorization


def fingerprint(value):
    """Hash JSON-serialisable inputs, preserving list order."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def select_best(results, min_coverage):
    """Maximise DBCV among eligible evaluated configurations."""
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1")

    eligible = results.loc[
        (results["coverage"] >= min_coverage)
        & (results["n_clusters"] >= 2)
        & np.isfinite(results["dbcv"])
    ]

    if eligible.empty:
        return None

    return eligible.sort_values(
        ["dbcv", "coverage", "eps", "min_samples"],
        ascending=[False, False, True, True],
    ).iloc[0]


def load_dense(texts, cfg, cache_dir):
    """Cache by actual ordered texts and embedding settings."""
    cache_spec = {
        "texts": texts,
        "model": cfg.embedding.model_name,
        "normalize_embeddings": True,
        "cache_schema": 1,
    }
    cache_key = fingerprint(cache_spec)

    X = load_or_embed_texts(
        texts=texts,
        file_path=cache_dir / f"dense_{cache_key}.npy",
        model_name=cfg.embedding.model_name,
        recalculate=cfg.embedding.recalculate,
        local_files_only=cfg.embedding.local_files_only,
    )

    return X, cache_key


def build_representation(df, cfg, cache_dir):
    """Return vectors, fitted transformers and representation details."""
    representation = cfg.representation

    if representation not in {"dense", "lsa", "hybrid"}:
        raise ValueError("representation must be dense, lsa or hybrid")

    if representation != "dense" and cfg.pca.enabled:
        raise ValueError("This runner applies PCA only to dense embeddings")

    if cfg.text_column not in df.columns:
        raise ValueError(f"Missing text column: {cfg.text_column}")

    texts = df[cfg.text_column].fillna("").astype(str).tolist()

    transformers = {}
    details = {
        "text_column": cfg.text_column,
        "text_fingerprint": fingerprint(texts),
    }

    if representation in {"dense", "hybrid"}:
        dense, cache_key = load_dense(texts, cfg, cache_dir)
        details["embedding_cache_key"] = cache_key

    if representation in {"lsa", "hybrid"}:
        tfidf, _, vectorizer = apply_ngrams_and_vectorization(
            texts,
            n_min=cfg.tfidf.n_min,
            n_max=cfg.tfidf.n_max,
            min_docs=cfg.tfidf.min_docs,
            max_feat=cfg.tfidf.max_feat,
        )

        components = int(cfg.lsa.n_components)

        if not 1 <= components < min(tfidf.shape):
            raise ValueError(
                "LSA components must be positive and smaller than "
                "both the number of rows and TF-IDF features"
            )

        svd = TruncatedSVD(
            n_components=components,
            random_state=cfg.seed,
        )
        reduced = normalize(
            svd.fit_transform(tfidf),
            norm="l2",
        )

        transformers["tfidf"] = vectorizer
        transformers["svd"] = svd

        details["svd_retained_variance"] = float(
            svd.explained_variance_ratio_.sum()
        )
        details["lsa_normalized"] = True

    if representation == "dense":
        X = dense

        if cfg.pca.enabled:
            threshold = float(cfg.pca.variance_threshold)

            if not 0 < threshold < 1:
                raise ValueError("PCA variance threshold must be between 0 and 1")

            pca = PCA(
                n_components=threshold,
                svd_solver="full",
            )
            X = pca.fit_transform(X)
            transformers["pca"] = pca

            details["pca_retained_variance"] = float(
                pca.explained_variance_ratio_.sum()
            )
            details["normalized_after_pca"] = False

    elif representation == "lsa":
        X = reduced

    else:
        alpha = float(cfg.hybrid.alpha)

        if not 0 <= alpha <= 1:
            raise ValueError("Hybrid alpha must be between 0 and 1")

        X = np.hstack((
            np.sqrt(1 - alpha) * normalize(dense, norm="l2"),
            np.sqrt(alpha) * reduced,
        ))

    X = np.asarray(X, dtype=np.float64)

    if not np.isfinite(X).all():
        raise ValueError("Representation contains nonfinite values")

    details["shape"] = list(X.shape)

    return X, transformers, details


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="config",
)
def main(cfg: DictConfig):
    started = perf_counter()
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    OmegaConf.save(
        OmegaConf.to_container(cfg, resolve=True),
        run_dir / "config_resolved.yaml",
    )

    train, _, headers, _, data_dir = load_dataset()
    df = pd.DataFrame(train, columns=headers).reset_index(drop=True)

    # Original text does not require the cleaning pipeline.
    if cfg.text_column != "open_response":
        df = preprocess_dataset(
            df=df,
            text_cols=["open_response"],
            download=cfg.preprocess.download,
        )

    cache_dir = Path(data_dir) / "output" / "embedding_cache"

    X, transformers, details = build_representation(
        df, cfg, cache_dir
    )

    # Save representation and row alignment before expensive evaluation.
    np.save(run_dir / "features.npy", X)

    df[["newid"]].to_csv(
        run_dir / "rows.csv",
        index=False,
    )

    joblib.dump(
        transformers,
        run_dir / "transformers.joblib",
    )

    eps_values = list(cfg.clustering.eps_values)
    min_samples_values = list(cfg.clustering.min_samples_values)

    if not eps_values or not min_samples_values:
        raise ValueError("Parameter grids must not be empty")

    model = DBSCAN(
        eps=float(eps_values[0]),
        min_samples=int(min_samples_values[0]),
        metric=cfg.clustering.metric,
        p=cfg.clustering.p,
    )

    model.fit(X, precompute=True)

    results, assignments = evaluate_parameters(
        model,
        eps_values=eps_values,
        min_samples_values=min_samples_values,
        y_true=df["gs_text34"].to_numpy(),
        return_assignments=True,
    )

    results.to_csv(run_dir / "metrics.csv", index=False)

    np.savez_compressed(
        run_dir / "assignments.npz",
        **assignments,
    )

    best = select_best(
        results,
        min_coverage=cfg.selection.min_coverage,
    )

    if best is None:
        selection = {
            "status": "no_eligible_configuration",
            "min_coverage": float(cfg.selection.min_coverage),
        }
        print("No configuration meets the selection constraints.")

    else:
        selection = {
            "status": "selected",
            "objective": "maximise_dbcv",
            "min_coverage": float(cfg.selection.min_coverage),
            "run_key": best["run_key"],
            "eps": float(best["eps"]),
            "min_samples": int(best["min_samples"]),
            "dbcv": float(best["dbcv"]),
            "coverage": float(best["coverage"]),
        }

        selected_rows = df[
            ["newid", "gs_text34"] #"open_response" should not be uploaded
        ].copy()

        selected_rows["cluster"] = assignments[best["run_key"]]

        selected_rows.to_csv(
            run_dir / "selected_clusters.csv",
            index=False,
        )

        print("Selected configuration:")
        print(selection)

    if best is not None:
        inspection = best
        report_name = "selected_cluster_examples.txt"
    else:
        candidates = results.loc[
            (results["n_clusters"] >= 2)
            & np.isfinite(results["dbcv"])
        ]

        inspection = (
            candidates.sort_values("dbcv", ascending=False).iloc[0]
            if not candidates.empty
            else None
        )
        report_name = "diagnostic_cluster_examples.txt"

    if inspection is not None:
        report_path = save_cluster_examples(
            df=df,
            labels=assignments[inspection["run_key"]],
            distance_matrix=model.distance_matrix_,
            output_path=run_dir / report_name,
            seed=cfg.seed,
        )

        print(
            f"Inspecting eps={inspection['eps']}, "
            f"min_samples={inspection['min_samples']}"
        )
        print("Text examples saved to:", report_path.resolve())
    else:
        print("No valid multi-cluster configuration available for inspection.")

    packages = [
        "numpy", "pandas", "scipy", "scikit-learn",
        "hdbscan", "sentence-transformers", "hydra-core",
    ]

    runtime = {
        "elapsed_seconds": perf_counter() - started,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in packages},
        "dbcv_core_distance": "logsumexp_patch",
    }

    summary = {
        "representation": details,
        "selection": selection,
        "runtime": runtime,
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    print(results.sort_values("dbcv", ascending=False).to_string(index=False))
    print(f"\nSaved experiment: {run_dir}")


if __name__ == "__main__":
    main()