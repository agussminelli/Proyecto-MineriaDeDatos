import numpy as np
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")

        self.eps = eps
        self.min_samples = min_samples
        self.distance_matrix_ = None

    def _region_query(self, i):
        if self.distance_matrix_ is not None:
            distances = self.distance_matrix_[i]
        else:
            distances = self._distances_from_point(i)

        return np.flatnonzero(distances <= self.eps)

    def _ensure_distance_matrix(self):
        if not hasattr(self, "X_"):
            raise ValueError("Provide data with fit(X) first")

        if self.distance_matrix_ is None:
            n = len(self.X_)

            # Assign the cache only after calculation succeeds.
            D = np.empty((n, n), dtype=float)

            for i in range(n):
                D[i] = self._distances_from_point(i)

            self.distance_matrix_ = D

    def _distances_from_point(self, i):
        return np.linalg.norm(self.X_ - self.X_[i], axis=1)

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
        """
        # Validate again because parameters may change between fits.
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

        if precompute:
            self._ensure_distance_matrix()

        # This method already resets labels through _initialise().
        self._find_clusters()
        return self

    def fit_predict(self, X=None, precompute=False):
        """Fit and return a copy of the cluster assignments."""
        self.fit(X, precompute=precompute)
        return self.labels_.copy()

    # Evaluation Functions

    def plot_k_distance(self, min_samples=None): #TODO: Delete
        """Plot neighbourhood radii, counting the point itself.

        Requires stored data. Creates the distance cache if necessary.
        Returns the sorted radius values.
        """
        if min_samples is None:
            min_samples = self.min_samples

        self._ensure_distance_matrix()

        if (
            not isinstance(min_samples, (int, np.integer))
            or not 2 <= min_samples <= len(self.X_)
        ):
            raise ValueError("min_samples must be an integer between 2 and n_points")

        k = min_samples - 1
        kth_distances = np.partition(
            self.distance_matrix_, k, axis=1
        )[:, k]
        sorted_distances = np.sort(kth_distances)

        fig, ax = plt.subplots()
        ax.plot(sorted_distances)
        ax.set_xlabel("Points ordered by neighbourhood radius")
        ax.set_ylabel(f"Neighbourhood radius (min_samples={min_samples})")
        ax.set_title("K-distance graph")
        ax.grid(alpha=0.3)
        plt.show()

        return sorted_distances

    def estimate_eps(self, min_samples=None): #TODO: look over algorithm
        """Estimate eps from the bend in the sorted k-distance curve."""
        if min_samples is None:
            min_samples = self.min_samples

        self._ensure_distance_matrix()
        n = len(self.X_)

        if (
            not isinstance(min_samples, (int, np.integer))
            or not 2 <= min_samples <= n
        ):
            raise ValueError("min_samples must be between 2 and n_points")

        k = min_samples - 1
        distances = np.sort(
            np.partition(self.distance_matrix_, k, axis=1)[:, k]
        )

        span = distances[-1] - distances[0]
        if span == 0:
            raise ValueError("The curve is constant; no knee can be estimated")

        x = np.linspace(0, 1, n)
        y = (distances - distances[0]) / span

        # For an increasing curve with a flat region and steep tail:
        scores = x - y
        knee_index = int(np.argmax(scores))

        if knee_index in (0, n - 1) or scores[knee_index] <= 0:
            raise ValueError("No interior knee found by this heuristic")

        eps_candidate = float(distances[knee_index])

        fig, ax = plt.subplots()
        ax.plot(distances)
        ax.scatter(
            knee_index, eps_candidate,
            color="red", label=f"Candidate eps = {eps_candidate:.4f}",
        )
        ax.axhline(eps_candidate, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Points ordered by neighbourhood radius")
        ax.set_ylabel(f"Neighbourhood radius (min_samples={min_samples})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.show()

        return eps_candidate

    def examine_parameters(self, eps_values, min_samples_values):
        """Fit DBSCAN for each parameter pair and plot clusters and noise.

        Returns:
            cluster_counts: shape (n_min_samples, n_eps).
            noise_fractions: same shape, values between 0 and 1.
        """
        eps_values = list(eps_values)
        min_samples_values = list(min_samples_values)

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

        self._ensure_distance_matrix()

        shape = (len(min_samples_values), len(eps_values))
        cluster_counts = np.zeros(shape, dtype=int)
        noise_fractions = np.zeros(shape, dtype=float)

        # Use a separate model so experiments preserve your current fit.
        trial = DBSCAN()
        trial.X_ = self.X_
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

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax, values, title, cmap in [
            (axes[0], cluster_counts, "Number of clusters", "viridis"),
            (axes[1], noise_fractions, "Noise fraction", "magma"),
        ]:
            options = {"vmin": 0, "vmax": 1} if title == "Noise fraction" else {}

            image = ax.imshow(
                values,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                **options,
            )

            ax.set_xticks(
                range(len(eps_values)),
                labels=[f"{eps:.3f}" for eps in eps_values],
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
                    text = (
                        str(int(values[row, col]))
                        if title == "Number of clusters"
                        else f"{values[row, col]:.0%}"
                    )
                    ax.text(
                        col, row, text,
                        ha="center", va="center",
                        color="white",
                        fontsize=8,
                    )

        fig.tight_layout()
        plt.show()

        return cluster_counts, noise_fractions


    

if __name__ == "__main__":
    from generate_testdata import generate_smiley_data, generate_easy_text

    # X_smiley = generate_smiley_data()

    # model = DBSCAN(eps=0.1, min_samples=5)

    # # First fit: on-demand distances.
    # labels = model.fit_predict(X_smiley)
    # print(labels)

    # # Builds the distance matrix for reuse.
    # model.plot_k_distance(min_samples=5)
    # eps_candidate = model.estimate_eps(min_samples=5)

    # # Call on the object, not on the DBSCAN class.
    # cluster_counts, noise_fractions = model.examine_parameters(
    #     eps_values=np.linspace(0.04, 0.20, 9),
    #     min_samples_values=[3, 5, 8, 12, 20],
    # )

    # # After inspecting the plots, choose parameters and refit.
    # model.eps = 0.1
    # model.min_samples = 5
    # labels = model.fit_predict()

    X_text, texts=generate_easy_text()


    model = DBSCAN(min_samples=3)

    # Store data and build the reusable distance matrix.
    model.fit(X_text, precompute=True)

    eps_candidate = model.estimate_eps(min_samples=3)
    print("Candidate eps:", eps_candidate)

    model.examine_parameters(
        eps_values=[0.3, 0.5, 0.7, 0.9, 1.1],
        min_samples_values=[2, 3, 4],
    )

    model.eps = 0.7  # Example setting to investigate
    model.min_samples = 3
    labels = model.fit_predict()

    for cluster_id in sorted(set(labels)):
        heading = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"\n{heading}")

        for i in np.flatnonzero(labels == cluster_id):
            print(" -", texts[i])
