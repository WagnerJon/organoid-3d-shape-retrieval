"""Compare held-out retrieval methods, estimate chance, and render retrieval examples."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from evaluate_shape_retrieval import retrieval_metrics
from train_shape_retrieval import ROOT, SHAPE_FEATURES


def load_named(path, ids):
    frame = pd.read_csv(path).set_index("cell_id").loc[ids]
    columns = [c for c in frame if c.startswith("embedding_")]
    value = frame[columns].to_numpy(float)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def main():
    out = ROOT / "results/shape_retrieval"
    manifest = pd.read_csv(out / "manifest.csv"); test = manifest[manifest.split == "test"].copy()
    train = manifest[manifest.split == "train"]
    ids = test.cell_id.tolist(); samples = test["sample"].to_numpy()
    mean = train[SHAPE_FEATURES].to_numpy(float).mean(0)
    std = train[SHAPE_FEATURES].to_numpy(float).std(0).clip(1e-8)
    truth = (test[SHAPE_FEATURES].to_numpy(float) - mean) / std
    methods = {
        "Shape student (raw)": np.load(out / "test_student_embeddings.npy"),
        "Mask teacher": np.load(out / "test_teacher_embeddings.npy"),
        "Context SimCLR": load_named(ROOT / "results/simclr3d_raw/embeddings.csv", ids),
        "Context IConE": load_named(ROOT / "results/icone3d_raw_context/embeddings.csv", ids),
    }
    metrics = {name: retrieval_metrics(value, truth, samples) for name, value in methods.items()}
    rng = np.random.default_rng(20260922); chance = []
    for _ in range(200):
        random_embedding = rng.normal(size=methods["Shape student (raw)"].shape)
        chance.append(retrieval_metrics(random_embedding, truth, samples))
    metrics["Chance (mean)"] = {key: float(np.mean([x[key] for x in chance])) for key in chance[0]}
    (out / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2))

    names = list(metrics); colors = ["#2563eb", "#16a34a", "#f59e0b", "#9333ea", "#9ca3af"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, title in zip(axes, ["recall_at_5", "ndcg_at_5", "distance_spearman"],
                              ["Recall@5", "NDCG@5", "Distance correlation"]):
        values = [metrics[n][key] for n in names]
        ax.bar(range(len(names)), values, color=colors); ax.set_title(title); ax.set_ylim(0, max(values)*1.18)
        ax.set_xticks(range(len(names)), [n.replace(" ", "\n", 1) for n in names], fontsize=8)
        for i, value in enumerate(values): ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Held-out cross-organoid shape retrieval (132 cells, 16 stacks)")
    fig.tight_layout(); fig.savefig(out / "retrieval_comparison.png", dpi=200); plt.close(fig)

    # Four fixed, spread-out queries; all retrieved cells must come from other organoids.
    student = methods["Shape student (raw)"]; teacher = methods["Mask teacher"]
    sd = pairwise_distances(student, metric="cosine"); td = pairwise_distances(teacher, metric="cosine")
    shape_d = pairwise_distances(truth)
    queries = np.linspace(0, len(test)-1, 4, dtype=int)
    raw_all = np.load(ROOT / "results/simclr3d_raw/raw64.npy", mmap_mode="r")
    masks_all = np.load(out / "masks_fixed64.npy", mmap_mode="r")
    fig, axes = plt.subplots(len(queries), 7, figsize=(13, 8))
    for row, q in enumerate(queries):
        allowed = np.flatnonzero(samples != samples[q])
        sr = allowed[np.argsort(sd[q, allowed])[:3]]; tr = allowed[np.argsort(td[q, allowed])[:3]]
        order = [q, *sr, *tr]
        for col, index in enumerate(order):
            raw_index = int(test.iloc[index]["index"])
            image = np.asarray(raw_all[raw_index]).max(0)
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=np.percentile(image, 99.5))
            axes[row, col].contour(np.asarray(masks_all[int(test.index[index])]).max(0), levels=[.5], colors="#00e5ff", linewidths=.6)
            title = "Query" if col == 0 else (f"Student {col}\nΔshape={shape_d[q,index]:.2f}" if col <= 3 else f"Teacher {col-3}\nΔshape={shape_d[q,index]:.2f}")
            axes[row, col].set_title(title, fontsize=8); axes[row, col].axis("off")
    fig.suptitle("Raw maximum projections: query and nearest neighbors from other test organoids\nCyan: mask used only for evaluation")
    fig.tight_layout(); fig.savefig(out / "retrieval_examples.png", dpi=200); plt.close(fig)

    report = f'''# Shape-specialized raw-image retrieval results

The held-out benchmark contains {len(test)} cells from {test['sample'].nunique()} complete test stacks. Candidate matches always come from a different held-out organoid.

| Representation | Recall@5 | NDCG@5 | Shape-distance Spearman |
|---|---:|---:|---:|
''' + "\n".join(f"| {n} | {metrics[n]['recall_at_5']:.3f} | {metrics[n]['ndcg_at_5']:.3f} | {metrics[n]['distance_spearman']:.3f} |" for n in names) + '''

The specialized raw student is substantially above chance and is numerically better than the earlier context-rich SimCLR and IConE representations on all three shape-retrieval measures. The gain over context SimCLR is modest and has not yet been tested across repeated data splits or training seeds. The binary-mask teacher remains markedly stronger, particularly for preserving the complete ranking of pairwise shape distances. This shows successful transfer of shape information to raw images, with remaining room for better raw-to-mask alignment.

Recall@5 is the fraction of the five morphology-nearest cells recovered in the embedding's top five. NDCG@5 rewards placing morphologically closer cells earlier. Shape-distance Spearman compares all eligible embedding distances with distances across eight standardized conventional shape measurements.

The cyan outlines in the retrieval examples are evaluation overlays and were not supplied to the raw student.
'''
    (out / "REPORT.md").write_text(report)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
