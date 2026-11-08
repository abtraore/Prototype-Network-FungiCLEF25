"""Figure 3: UMAP of the 2,427 class prototypes of the best prototypical network.

    python -m protonet.predict --data-root DATA --weights protonet.pt \
        --save-prototypes prototypes.pt
    python -m analysis.umap_prototypes prototypes.pt
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans

DENSE = (-5, -2, 2.5, 4)  # x_min, x_max, y_min, y_max (read off the paper's UMAP run)
MIXED = (0, 4, 4, 6.5)

if __name__ == "__main__":
    protos = torch.load(sys.argv[1])["prototypes"].numpy()
    xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine").fit_transform(protos)
    species = np.arange(len(xy))

    kmeans = KMeans(n_clusters=3, random_state=2024, algorithm="elkan").fit(xy)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=species, cmap="viridis", s=30, alpha=0.3)
    regions = [(DENSE, "red", "Dense Cluster Region", "Dense\nCluster"),
               (MIXED, "orange", "Mixed Region", "Mixed\nRegion")]
    for (x0, x1, y0, y1), color, label, tag in regions:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color,
                               linewidth=2.5, linestyle="--", label=label))
        ax.text(x0 + 0.5, y1 - 0.2, tag,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10, weight="bold", color=color)

    ax.scatter(*kmeans.cluster_centers_.T, c="white", s=300, marker="x", linewidth=3)
    for i, (color, marker) in enumerate(zip(["darkred", "darkorange", "gold"], ["*", "s", "^"])):
        pts = xy[kmeans.labels_ == i]
        closest = pts[np.argsort(np.linalg.norm(pts - kmeans.cluster_centers_[i], axis=1))[:3]]
        ax.scatter(*closest.T, color=color, s=200, marker=marker, edgecolor="black",
                   linewidth=2, label=f"Cluster {i} Representative Species")

    ax.set_title("UMAP Embedding Space with Zoom Regions and Representative Species", fontsize=14)
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    plt.colorbar(sc, label="Species ID", ax=ax)
    plt.tight_layout()
    plt.savefig("umap_with_regions_and_samples.pdf", dpi=300, bbox_inches="tight")
