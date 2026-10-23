"""Visualize custom DBSCAN fitting and fixed-cluster prediction on two moons.

Run from the repository root:
    py -m examples.dbscan_demo

Writes a PNG and SVG under outputs/examples. No private data or embeddings
are used. This is a visual demonstration, not a performance benchmark.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.datasets import make_moons

from dbscan import DBSCAN

SEED = 42
EPS = 0.16
MIN_SAMPLES = 8
NOISE_COLOR = "#777777"
PALETTE = ["#007C91", "#C45D28", "#7754A3", "#54913A"]


def create_data():
    """Return deterministic training points and illustrative unseen queries.

    Six isolated training points are added to make noise visible. New query
    points are generated separately; they do not participate in fitting.
    Ground-truth moon labels are deliberately unused.
    """
    moons, _ = make_moons(
        n_samples=250,
        noise=0.055,
        random_state=SEED,
    )
    isolated = np.array(
        [
            [-1.35, -0.55],
            [-0.85, 1.45],
            [0.40, 1.48],
            [1.15, 1.40],
            [2.35, 0.95],
            [2.30, -0.65],
        ]
    )

    angles = np.array([0.35, 1.35, 2.55])
    upper = np.column_stack((np.cos(angles), np.sin(angles)))
    lower = np.column_stack((1 - np.cos(angles), 0.5 - np.sin(angles)))
    far_away = np.array([[-1.25, 0.75], [1.35, 1.12]])

    return np.vstack((moons, isolated)), np.vstack((upper, lower, far_away))


def plot_training(ax, points, labels, cores, colors, faded=False):
    """Draw core circles, hollow border circles and noise crosses."""
    alpha = 0.28 if faded else 0.9

    for cluster in sorted(colors):
        members = labels == cluster
        core_members = members & cores
        border_members = members & ~cores

        ax.scatter(
            *points[core_members].T,
            s=23,
            color=colors[cluster],
            alpha=alpha,
            linewidths=0,
        )
        ax.scatter(
            *points[border_members].T,
            s=35,
            facecolors="none",
            edgecolors=colors[cluster],
            alpha=alpha,
            linewidths=1.3,
        )

    ax.scatter(
        *points[labels == -1].T,
        s=28,
        marker="x",
        color=NOISE_COLOR,
        alpha=alpha,
        linewidths=1.2,
    )


def main():
    """Fit once, predict unseen points and save a three-panel figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/examples"),
        help="Destination for dbscan_demo.png and dbscan_demo.svg.",
    )
    args = parser.parse_args()

    training, unseen = create_data()
    model = DBSCAN(
        eps=EPS,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        distance_backend="optimized",
        block_size=256,
    )
    model.fit(training, precompute=True)

    labels = model.labels_.copy()
    cores = model.core_sample_mask_.copy()
    predicted = model.predict(unseen)
    nearest = model.nearest_core_indices_.copy()

    # Prediction must leave the fitted training partition unchanged.
    np.testing.assert_array_equal(model.labels_, labels)

    clusters = sorted(set(labels) - {-1})
    colors = {
        cluster: PALETTE[index % len(PALETTE)]
        for index, cluster in enumerate(clusters)
    }
    noise_count = int(np.sum(labels == -1))
    border_count = int(np.sum((labels > 0) & ~cores))
    assigned_count = int(np.sum(predicted > 0))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5.7),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.04, right=0.99, top=0.77, bottom=0.23, wspace=0.1
    )
    fig.suptitle(
        "DBSCAN: from points to clusters to unseen assignments",
        fontsize=17,
        y=0.97,
    )
    fig.text(
        0.5,
        0.89,
        f"Two moons + isolated points | Euclidean distance | "
        f"eps = {EPS} | min_samples = {MIN_SAMPLES} | seed = {SEED}",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    axes[0].scatter(*training.T, s=20, color="#526579", linewidths=0)
    axes[0].set_title("1  Input\n256 training points", fontsize=12, pad=12)

    plot_training(axes[1], training, labels, cores, colors)
    axes[1].set_title(
        f"2  Fitted DBSCAN\n{len(clusters)} clusters · "
        f"{noise_count} noise · {border_count} border",
        fontsize=12,
        pad=12,
    )

    plot_training(axes[2], training, labels, cores, colors, faded=True)

    for point, label, core_index in zip(unseen, predicted, nearest):
        color = colors.get(label, NOISE_COLOR)

        if label > 0:
            core = training[core_index]
            axes[2].plot(
                [point[0], core[0]],
                [point[1], core[1]],
                color=color,
                linewidth=1.2,
                zorder=3,
            )
            axes[2].scatter(
                *core,
                s=50,
                facecolors="none",
                edgecolors=color,
                linewidths=1.1,
                zorder=4,
            )

        axes[2].scatter(
            *point,
            marker="*",
            s=190,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )

    axes[2].set_title(
        f"3  Unseen points (stars)\n{assigned_count} assigned · "
        f"{len(unseen) - assigned_count} rejected",
        fontsize=12,
        pad=12,
    )

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.55, 2.55)
        ax.set_ylim(-0.85, 1.65)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#DDDDDD")

    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=color,
            label=f"Cluster {cluster}",
        )
        for cluster, color in colors.items()
    ]
    legend.extend(
        [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="#444444",
                label="Core: filled",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="#444444",
                markerfacecolor="none",
                label="Border: hollow",
            ),
            Line2D(
                [],
                [],
                marker="x",
                linestyle="none",
                color=NOISE_COLOR,
                label="Training noise",
            ),
            Line2D(
                [],
                [],
                marker="*",
                markersize=12,
                linestyle="none",
                color=NOISE_COLOR,
                label="Unseen point",
            ),
        ]
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=6,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.065,
        "Prediction extension: join the nearest training core's cluster only "
        "within eps; otherwise remain noise. Training clusters stay fixed.",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.025,
        "Short lines connect assigned queries to their nearest cores. "
        "Synthetic illustration; not evidence of verbal-autopsy clustering quality.",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "svg"):
        destination = args.output_dir / f"dbscan_demo.{extension}"
        fig.savefig(destination, dpi=180, facecolor="white")
        print(f"Saved: {destination}")

    plt.close(fig)
    print(
        f"Training: {len(clusters)} clusters, {noise_count} noise, {border_count} border"
    )
    print(
        f"Unseen: {assigned_count} assigned, {len(unseen) - assigned_count} noise"
    )


if __name__ == "__main__":
    main()
