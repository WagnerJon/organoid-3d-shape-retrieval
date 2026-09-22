"""Matched morphology probes for IConE versus the completed SimCLR encoders."""
from pathlib import Path
import json, os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/icone3d_raw_isolated"
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,GroupKFold
from sklearn.metrics import r2_score,mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    source=ROOT/"results/simclr3d_raw_isolated"
    m=pd.read_csv(source/"manifest.csv")
    morphology=pd.read_csv(ROOT/"results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id").loc[m.cell_id]
    train=m.split.eq("train").to_numpy();validation=m.split.eq("validation").to_numpy()
    embeddings={
        "Binary-mask SimCLR":np.load(ROOT/"results/simclr3d/embeddings.npy"),
        "Isolated-raw SimCLR":np.load(source/"embeddings.npy"),
        "Isolated-raw IConE":np.load(OUT/"embeddings.npy"),
    }
    embeddings["IConE + binary SimCLR"]=np.concatenate(
        [embeddings["Isolated-raw IConE"],embeddings["Binary-mask SimCLR"]],axis=1)
    targets=["sphericity","aspect_ratio","elongation","flatness","mesh_solidity","volume_um3"]
    rows=[]
    for name,x in embeddings.items():
        for target in targets:
            y=morphology[target].to_numpy()
            probe=GridSearchCV(make_pipeline(StandardScaler(),Ridge()),
                {"ridge__alpha":[.1,1,10,100,1000]},cv=GroupKFold(5),
                scoring="neg_mean_squared_error",n_jobs=2)
            probe.fit(x[train],y[train],groups=m.loc[train,"sample"])
            prediction=probe.predict(x[validation]);observed=y[validation].copy();centered=prediction.copy()
            groups=m.loc[validation,"sample"].to_numpy()
            for sample in np.unique(groups):
                use=groups==sample;observed[use]-=observed[use].mean();centered[use]-=centered[use].mean()
            rows.append(dict(representation=name,target=target,
                r2=float(r2_score(y[validation],prediction)),
                within_stack_r2=float(r2_score(observed,centered)),
                mae=float(mean_absolute_error(y[validation],prediction)),
                alpha=probe.best_params_["ridge__alpha"]))
        print("Evaluated "+name,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/"morphology_probe_comparison.csv",index=False)
    icone=embeddings["Isolated-raw IConE"]
    agreements={name:float(spearmanr(pdist(icone[validation],metric="cosine"),
        pdist(value[validation],metric="cosine")).statistic)
        for name,value in embeddings.items() if name!="Isolated-raw IConE" and "+" not in name}
    (OUT/"comparison.json").write_text(json.dumps(dict(validation_cells=int(validation.sum()),
        agreements=agreements,results=rows),indent=2))
    pivot=frame.pivot(index="target",columns="representation",values="r2").reindex(targets)
    ax=pivot.plot.bar(figsize=(13,6));ax.axhline(0,color="black",lw=.6)
    ax.set(ylabel="Validation R²",title="IConE versus SimCLR on isolated raw cell intensities")
    ax.legend(fontsize=8);plt.xticks(rotation=20);plt.tight_layout();plt.savefig(OUT/"comparison.png",dpi=160);plt.close()
    status=json.loads((OUT/"status.json").read_text())
    lines=["# IConE on segmentation-guided raw cell intensities","",
        "Controlled comparison using the same 1,088 prepared objects, 674 training cells, 267 validation cells, augmentations, compact 3D encoder family and 50-epoch budget as the isolated-raw SimCLR experiment. IConE replaces the projector and NT-Xent objective with direct backbone embeddings, a persistent training-instance anchor table, and the paper's three equally weighted losses.","",
        f"The exported encoder is the fixed-budget epoch 50 checkpoint. Best diagnostic augmented-view retrieval occurred at epoch {status['diagnostic_best_retrieval_epoch']} ({status['best_validation_pair_retrieval']:.1%}).", "",
        "| Representation | Target | Validation R² | Within-stack R² | MAE |","|---|---|---:|---:|---:|"]
    for row in rows:lines.append(f"| {row['representation']} | {row['target']} | {row['r2']:.3f} | {row['within_stack_r2']:.3f} | {row['mae']:.3f} |")
    lines += ["",f"Pair-distance agreement between IConE and the other representations: {json.dumps(agreements)}.","",
        "This is a controlled practical comparison, not an exact reproduction of the paper: it retains our compact encoder, cell-specific augmentations and 50 epochs, while using the paper's AdamW learning rate 1e-4, weight decay 0.05, cosine schedule and unweighted IConE objective. The paper used a 3D ResNet-18 with 512-dimensional embeddings for 100 epochs.","",
        "The cell mask remains an implicit source of boundary information because it gates the raw intensities. Validation stacks were used for checkpoint diagnostics and may not be biologically independent. External organoid-level validation and multiple seeds are required before claiming superiority."]
    (OUT/"COMPARISON.md").write_text("\n".join(lines)+"\n")
    print("Comparison completed",flush=True)


if __name__=="__main__":main()
