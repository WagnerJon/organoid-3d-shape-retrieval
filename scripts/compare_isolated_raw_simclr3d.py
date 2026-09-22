"""Compare isolated raw, context-rich raw, and binary-mask representations."""
from pathlib import Path
import os, json
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/simclr3d_raw_isolated"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    m = pd.read_csv(OUT / "manifest.csv")
    binary_manifest = pd.read_csv(ROOT / "results/simclr3d/manifest.csv")
    assert m.cell_id.equals(binary_manifest.cell_id) and m.split.equals(binary_manifest.split)
    morphology = pd.read_csv(ROOT / "results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id").loc[m.cell_id]
    train = m.split.eq("train").to_numpy()
    validation = m.split.eq("validation").to_numpy()
    isolated = np.load(OUT / "embeddings.npy")
    untrained = np.load(OUT / "untrained_embeddings.npy")
    untrained /= np.maximum(np.linalg.norm(untrained, axis=1, keepdims=True), 1e-12)
    binary = np.load(ROOT / "results/simclr3d/embeddings.npy")
    context = np.load(ROOT / "results/simclr3d_raw/embeddings.npy")
    intensity = m[["intensity_mean", "intensity_std", "intensity_p10",
                   "intensity_p50", "intensity_p90", "intensity_p99"]].to_numpy()
    representations = {
        "Intensity statistics": intensity,
        "Untrained isolated raw CNN": untrained,
        "Context-rich raw CNN": context,
        "Binary-mask CNN": binary,
        "Isolated raw CNN": isolated,
        "Isolated raw + binary CNN": np.concatenate([isolated, binary], axis=1),
    }
    targets = ["sphericity", "aspect_ratio", "elongation", "flatness",
               "mesh_solidity", "volume_um3"]
    rows = []
    groups_train = m.loc[train, "sample"]
    groups_validation = m.loc[validation, "sample"].to_numpy()
    for name, x in representations.items():
        for target in targets:
            y = morphology[target].to_numpy()
            model = GridSearchCV(
                make_pipeline(StandardScaler(), Ridge()),
                {"ridge__alpha": [.1, 1, 10, 100, 1000]},
                cv=GroupKFold(5), scoring="neg_mean_squared_error", n_jobs=2)
            model.fit(x[train], y[train], groups=groups_train)
            prediction = model.predict(x[validation])
            observed_centered = y[validation].copy()
            predicted_centered = prediction.copy()
            for sample in np.unique(groups_validation):
                use = groups_validation == sample
                observed_centered[use] -= observed_centered[use].mean()
                predicted_centered[use] -= predicted_centered[use].mean()
            rows.append(dict(representation=name, target=target,
                r2=float(r2_score(y[validation], prediction)),
                within_stack_r2=float(r2_score(observed_centered, predicted_centered)),
                mae=float(mean_absolute_error(y[validation], prediction)),
                alpha=model.best_params_["ridge__alpha"]))
        print("Evaluated " + name, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "morphology_probe_comparison.csv", index=False)
    agreements = {
        "isolated_vs_binary": float(spearmanr(
            pdist(isolated[validation], metric="cosine"),
            pdist(binary[validation], metric="cosine")).statistic),
        "isolated_vs_context_raw": float(spearmanr(
            pdist(isolated[validation], metric="cosine"),
            pdist(context[validation], metric="cosine")).statistic),
    }
    payload = dict(validation_cells=int(validation.sum()), agreements=agreements,
                   results=rows)
    (OUT / "comparison.json").write_text(json.dumps(payload, indent=2))
    pivot = result.pivot(index="target", columns="representation", values="r2").reindex(targets)
    ax = pivot.plot.bar(figsize=(14, 6))
    ax.axhline(0, color="black", lw=.6)
    ax.set(ylabel="Validation R²", title="Morphology recovered from isolated raw intensities")
    ax.legend(fontsize=8)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUT / "comparison.png", dpi=160)
    plt.close()
    lines = [
        "# Segmentation-guided raw-intensity experiment", "",
        "Raw intensity values inside each predicted cell are retained; all voxels outside that cell are zero. The binary mask is used as a preprocessing gate and is not supplied as a separate CNN channel. Inputs use the same cell-specific isotropic cube geometry and split as the binary model.", "",
        f"Validation cells: {validation.sum()}. Isolated-versus-binary pair-distance Spearman: {agreements['isolated_vs_binary']:.3f}. Isolated-versus-context-rich-raw: {agreements['isolated_vs_context_raw']:.3f}.", "",
        "| Representation | Target | Validation R² | Within-stack R² | MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['representation']} | {row['target']} | {row['r2']:.3f} | {row['within_stack_r2']:.3f} | {row['mae']:.3f} |")
    lines += ["",
        "The untrained isolated CNN is a necessary control: zeroing outside the predicted cell makes the segmentation boundary visible implicitly. Improvement over that control measures what contrastive training adds, but cannot show that morphology was learned independently of segmentation.", "",
        "Morphology probes were fitted on training cells only; ridge regularization used whole-stack cross-validation within training. Reported values use the same diagnostic validation stacks that selected encoder checkpoints. Biological independence remains unknown, and an external organoid-level test is required for biological claims."]
    (OUT / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    print("Comparison completed", flush=True)


if __name__ == "__main__":
    main()
