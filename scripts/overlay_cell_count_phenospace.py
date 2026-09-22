#!/usr/bin/env python3
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "organoid_phenotype_groups"
OUT = SOURCE / "cell_count_overlay"


def category(values):
    bins = [4.5, 7.5, 9.5, 11.5, 14.5]
    labels = ["5–7 cells", "8–9 cells", "10–11 cells", "12–14 cells"]
    return pd.cut(values, bins=bins, labels=labels), labels


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(SOURCE / "group_assignments.csv")
    profiles = pd.read_csv(SOURCE / "context_distribution_profiles.csv")
    merged = assignments.merge(profiles, on="sample", suffixes=("", "_profile"))
    counts = merged["cell_count"].to_numpy()
    cats, labels = category(counts)

    # Sensitivity phenospace: recompute the same standardize/PCA step without cell count.
    feature_cols = [c for c in profiles.columns if c not in {"sample", "cell_count"}]
    x_without_count = StandardScaler().fit_transform(merged[feature_cols])
    pca_without_count = PCA(n_components=.95, random_state=0)
    z_without_count = pca_without_count.fit_transform(x_without_count)

    ks = range(2, 9)
    sils = []
    labels_by_k = {}
    for k in ks:
        group = KMeans(n_clusters=k, random_state=0, n_init=100).fit_predict(z_without_count)
        labels_by_k[k] = group
        sils.append(silhouette_score(z_without_count, group))
    best_k = list(ks)[int(np.argmax(sils))]

    rho1, p1 = spearmanr(counts, assignments["context_pc1"])
    rho2, p2 = spearmanr(counts, assignments["context_pc2"])
    rho1_nc, p1_nc = spearmanr(counts, z_without_count[:, 0])
    rho2_nc, p2_nc = spearmanr(counts, z_without_count[:, 1])
    group_counts = [counts[assignments.context_group.to_numpy() == g] for g in sorted(assignments.context_group.unique())]
    kw = kruskal(*group_counts)

    palette = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for g in sorted(assignments.context_group.unique()):
        use = assignments.context_group == g
        axes[0].scatter(assignments.loc[use, "context_pc1"], assignments.loc[use, "context_pc2"],
                        s=70, alpha=.85, label=f"Phenotype group {g}")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Existing context groups")

    for label, color in zip(labels, palette):
        use = np.asarray(cats == label)
        axes[1].scatter(assignments.loc[use, "context_pc1"], assignments.loc[use, "context_pc2"],
                        s=75, alpha=.9, color=color, label=f"{label} (n={use.sum()})")
    axes[1].legend(fontsize=8)
    axes[1].set_title(f"Same phenospace, colored by cell count\nSpearman count–PC1: {rho1:.2f}")

    for label, color in zip(labels, palette):
        use = np.asarray(cats == label)
        axes[2].scatter(z_without_count[use, 0], z_without_count[use, 1],
                        s=75, alpha=.9, color=color, label=f"{label} (n={use.sum()})")
    axes[2].legend(fontsize=8)
    axes[2].set_title(f"Recomputed without cell-count variable\nSpearman count–PC1: {rho1_nc:.2f}")

    for ax, xlab, ylab in [
        (axes[0], "Context PC1 (31.3%)", "Context PC2 (25.5%)"),
        (axes[1], "Context PC1 (31.3%)", "Context PC2 (25.5%)"),
        (axes[2], f"Count-excluded PC1 ({pca_without_count.explained_variance_ratio_[0]:.1%})",
         f"Count-excluded PC2 ({pca_without_count.explained_variance_ratio_[1]:.1%})")]:
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.axhline(0, color="0.85", lw=.7); ax.axvline(0, color="0.85", lw=.7)
    fig.suptitle("Does cell count explain the context-rich organoid phenospace?")
    fig.tight_layout()
    fig.savefig(OUT / "cell_count_on_context_phenospace.png", dpi=200)
    plt.close(fig)

    table = assignments[["sample", "cell_count", "context_group", "context_pc1", "context_pc2"]].copy()
    table["cell_count_category"] = cats.astype(str)
    table["count_excluded_pc1"] = z_without_count[:, 0]
    table["count_excluded_pc2"] = z_without_count[:, 1]
    table.to_csv(OUT / "cell_count_overlay_coordinates.csv", index=False)

    stats = {
        "organoids": int(len(merged)),
        "cell_count_categories": {label: int((cats == label).sum()) for label in labels},
        "existing_space_spearman": {"pc1_rho": float(rho1), "pc1_p": float(p1), "pc2_rho": float(rho2), "pc2_p": float(p2)},
        "count_excluded_space_spearman": {"pc1_rho": float(rho1_nc), "pc1_p": float(p1_nc), "pc2_rho": float(rho2_nc), "pc2_p": float(p2_nc)},
        "cell_count_by_existing_context_group": assignments.groupby("context_group").cell_count.agg(["count", "min", "median", "mean", "max"]).to_dict("index"),
        "kruskal_cell_count_across_existing_groups": {"H": float(kw.statistic), "p": float(kw.pvalue)},
        "count_excluded_best_k": int(best_k),
        "count_excluded_best_silhouette": float(max(sils)),
        "count_excluded_silhouettes": {str(k): float(s) for k, s in zip(ks, sils)},
    }
    (OUT / "statistics.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
