from time import perf_counter

import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5,
                 metric="euclidean", p=2):
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.p = p

        self.distance_matrix_ = None
        self._distance_config_ = None

    def _region_query(self, i):
        if self.distance_matrix_ is not None:
            distances = self.distance_matrix_[i]
        else:
            distances = self._distances_from_point(i)

        return np.flatnonzero(distances <= self.eps)

    def _ensure_distance_matrix(self):
        self._prepare_distances()

        # Time spent constructing the matrix in this call.
        self.matrix_build_seconds_ = 0.0

        if self.distance_matrix_ is None:
            started = perf_counter()
            n = len(self.X_)
            D = np.empty((n, n), dtype=float)

            for i in range(n):
                D[i] = self._distances_from_point(i)

            self.distance_matrix_ = D
            self.matrix_build_seconds_ = perf_counter() - started

    def _distances_from_point(self, i):
        point = self.X_[i]

        if self.metric == "cosine":
            norms = np.linalg.norm(self.X_, axis=1)
            similarities= (self.X_@point)/(norms*norms[i])

            distances = 1.0 - np.clip(similarities, -1.0, 1.0)
        else:
            differences = np.abs(self.X_ - point)

            if self.metric =="euclidean":
                distances = np.sqrt(np.sum(differences**2, axis=1))
            elif self.metric == "manhattan":
                distances = np.sum(differences, axis=1)
            elif self.metric == "minkowski":
                distances = np.sum(differences**self.p, axis=1)**(1.0/self.p)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        distances[i]=0.0
        return distances

    def _prepare_distances(self):
        """Validate distance settings and clear an incompatible cache."""
        allowed = {"euclidean", "manhattan", "cosine", "minkowski"}

        if self.metric not in allowed:
            raise ValueError(f"metric must be one of {sorted(allowed)}")

        if self.metric == "minkowski":
            if not np.isfinite(self.p) or self.p < 1:
                raise ValueError("Minkowski p must be finite and >= 1")

        if not hasattr(self, "X_"):
            raise ValueError("Provide data with fit(X) first")

        if self.metric == "cosine":
            if np.any(np.linalg.norm(self.X_, axis=1) == 0):
                raise ValueError(
                    "Cosine distance is undefined for zero vectors. "
                    "Remove empty-vector rows while preserving their "
                    "matching IDs, or choose another metric."
                )

        config = (
            self.metric,
            self.p if self.metric == "minkowski" else None,
        )

        if config != self._distance_config_:
            self.distance_matrix_ = None
            self._distance_config_ = config

    def _initialise(self):
        self.labels_ = np.zeros(len(self.X_), dtype=int)

    def _find_clusters(self):
        self._initialise()
        cluster_id = 0

        for i in range(len(self.X_)):
            if self.labels_[i] != 0:
                continue

            neighbors = self._region_query(i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                cluster_id += 1
                self._expand_cluster(i, neighbors, cluster_id)


    def _expand_cluster(self, i, neighbors, cluster_id):
        self.labels_[i] = cluster_id

        queue = list(neighbors)
        queued = set(queue)  # Prevent duplicate queue entries
        position = 0

        while position < len(queue):
            j = queue[position]
            position += 1

            if self.labels_[j] == -1:
                self.labels_[j] = cluster_id
                continue  # Already examined; not a core point

            if self.labels_[j] != 0:
                continue  # Already assigned to a cluster

            self.labels_[j] = cluster_id
            neighbors_j = self._region_query(j)

            if len(neighbors_j) >= self.min_samples:
                for neighbor in neighbors_j:
                    if neighbor not in queued:
                        queue.append(neighbor)
                        queued.add(neighbor)

        
    def fit(self, X=None, precompute=False):
        """Cluster new or previously stored data.

        Parameters
        ----------
        X : array-like of shape (n_points, n_features), optional
            New data. Supplying X replaces stored data and clears the cache.
            If omitted, reuse previously stored data.
        precompute : bool
            Whether to build the full distance matrix before clustering.

        Returns
        -------
        self
            Cluster labels use -1 for noise and positive integers for clusters.

        Example Usage:      #model.fit(X)                  # Calculate distances on demand
                            #model.fit(X, precompute=True) # Cache all distances
                            #model.fit()                   # Reuse stored data and any cached distances
        """

        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")

        if (
            not isinstance(self.min_samples, (int, np.integer))
            or self.min_samples < 1
        ):
            raise ValueError("min_samples must be a positive integer")

        if X is not None:
            X = np.asarray(X, dtype=float)

            if X.ndim != 2 or 0 in X.shape:
                raise ValueError("X must be a nonempty 2D array")

            if not np.isfinite(X).all():
                raise ValueError("X must contain only finite values")

            self.X_ = X.copy()
            self.distance_matrix_ = None

        elif not hasattr(self, "X_"):
            raise ValueError("Provide X on the first call to fit")

        self._prepare_distances()
        if precompute:
            self._ensure_distance_matrix()

        self._find_clusters()
        return self

    def fit_predict(self, X=None, precompute=False):
        """Fit and return a copy of the cluster assignments."""
        self.fit(X, precompute=precompute)
        return self.labels_.copy()

    # Visualisation Evaluation Function
    def examine_parameters(self, eps_values, min_samples_values, y_true=None):
        """Fit DBSCAN for each parameter pair and plot k-distance, clusters and noise.
        Calculates Silhouette Score and DBVC per pair, as well as ARI if provided GT.
        Requires data already stored through fit(data=X).
        Returns:
            cluster_counts: shape (n_min_samples, n_eps).
            noise_fractions: same shape, values between 0 and 1.
            eps_candidates: for every given min_samples_value
            y_true: provide with the true labels for ARI
        """
        import matplotlib.pyplot as plt

        eps_values = list(eps_values)
        min_samples_values = list(min_samples_values)

        self._ensure_distance_matrix()
        D = self.distance_matrix_
        n = len(self.X_)

        if not eps_values or not min_samples_values:
            raise ValueError("Parameter ranges must not be empty")

        for eps in eps_values:
            if not np.isfinite(eps) or eps <= 0:
                raise ValueError("All eps values must be finite and positive")

        for value in min_samples_values:
            if (
                not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError("min_samples values must be positive integers")

        shape = (len(min_samples_values), len(eps_values))
        cluster_counts = np.zeros(shape, dtype=int)
        noise_fractions = np.zeros(shape, dtype=float)
        eps_candidates = {}

        fig, axes = plt.subplots(
            1, 3,
            figsize=(19, 6),
            constrained_layout=True,
        )
        ax_distance, ax_clusters, ax_noise = axes

        # Overlay k-distance curves and mark candidate knees.
        for min_samples in min_samples_values:
            k = min_samples - 1
            radii = np.sort(np.partition(D, k, axis=1)[:, k])

            line, = ax_distance.plot(
                radii,
                label=f"min_samples={min_samples}",
            )

            candidate = None
            span = radii[-1] - radii[0]

            if span > 0:
                x = np.linspace(0, 1, n)
                y = (radii - radii[0]) / span
                scores = x - y
                knee = int(np.argmax(scores))

                if 0 < knee < n - 1 and scores[knee] > 0:
                    candidate = float(radii[knee])

                    ax_distance.scatter(
                        knee,
                        candidate,
                        color=line.get_color(),
                        marker="x",
                        s=60,
                    )

            eps_candidates[min_samples] = candidate

        ax_distance.set_title("K-distance curves; × = candidate knee")
        ax_distance.set_xlabel("Points ordered separately for each curve")
        ax_distance.set_ylabel(f"Distance ({self.metric})")
        ax_distance.grid(alpha=0.3)
        ax_distance.legend(fontsize=8)

        # separate model so experiments preserve current fit.
        trial = DBSCAN(metric=self.metric, p=self.p)
        trial.X_ = self.X_
        trial._prepare_distances()
        trial.distance_matrix_ = self.distance_matrix_

        for row, min_samples in enumerate(min_samples_values):
            for col, eps in enumerate(eps_values):
                trial.eps = float(eps)
                trial.min_samples = int(min_samples)
                trial.fit()

                labels = trial.labels_

                if np.any(labels == 0):
                    raise RuntimeError("Some points remain unassigned")

                cluster_counts[row, col] = np.unique(
                    labels[labels > 0]
                ).size
                noise_fractions[row, col] = np.mean(labels == -1)

        # 3. Draw both heatmaps.
        panels = [
            (ax_clusters, cluster_counts, "Number of clusters", "viridis", False),
            (ax_noise, noise_fractions, "Noise fraction", "magma", True),
        ]

        for ax, values, title, cmap, is_noise in panels:
            options = {"vmin": 0, "vmax": 1} if is_noise else {}

            image = ax.imshow(
                values,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                **options,
            )

            ax.set_xticks(
                range(len(eps_values)),
                labels=[f"{eps:.3g}" for eps in eps_values],
                rotation=45,
            )
            ax.set_yticks(
                range(len(min_samples_values)),
                labels=min_samples_values,
            )
            ax.set_xlabel("eps")
            ax.set_ylabel("min_samples")
            ax.set_title(title)
            fig.colorbar(image, ax=ax)

            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    value = values[row, col]
                    label = f"{value:.0%}" if is_noise else str(int(value))

                    # Choose contrasting text for the cell colour.
                    rgba = image.cmap(image.norm(value))
                    brightness = (
                        0.299 * rgba[0]
                        + 0.587 * rgba[1]
                        + 0.114 * rgba[2]
                    )

                    ax.text(
                        col, row, label,
                        ha="center",
                        va="center",
                        color="black" if brightness > 0.5 else "white",
                        fontsize=8,
                    )
        print("Candidate eps values for min_samples:", eps_candidates,min_samples_values)
        plt.show()

        return cluster_counts, noise_fractions, eps_candidates


    

if __name__ == "__main__":
    from generate_testdata import generate_smiley_data, generate_easy_text
    X_text, texts=generate_easy_text()

    # EXAMPLE USAGE
    model = DBSCAN(eps=0.5,min_samples=3,metric="cosine")

    # Store data and build the reusable distance matrix,
    # if you want to examin params.
    model.fit(X_text, precompute=True)

    model.examine_parameters(
        eps_values=[0.3, 0.5, 0.7, 0.9, 1.1],
        min_samples_values=[2, 3, 4],
    )

    # Example setting to investigate
    model.eps = 0.3 
    model.min_samples = 3
    labels = model.fit_predict()

    for cluster_id in sorted(set(labels)):
        heading = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"\n{heading}")

        for i in np.flatnonzero(labels == cluster_id):
            print(" -", texts[i])
