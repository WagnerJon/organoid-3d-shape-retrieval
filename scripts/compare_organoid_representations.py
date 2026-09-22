"""Compare organoids as distributions of their quality-passing cell representations."""
from pathlib import Path
import json, os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/organoid_representation_comparison"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def split_half_retrieval(x, samples, trials=200, seed=20260921):
    """Match one half's centroid to the other half among all sample candidates."""
    rng=np.random.default_rng(seed);unique=np.unique(samples);accuracies=[];margins=[]
    for _ in range(trials):
        left=[];right=[]
        for sample in unique:
            ids=np.where(samples==sample)[0];ids=rng.permutation(ids);cut=len(ids)//2
            left.append(normalize(x[ids[:cut]].mean(0,keepdims=True))[0])
            right.append(normalize(x[ids[cut:]].mean(0,keepdims=True))[0])
        left=np.stack(left);right=np.stack(right);similarity=left@right.T
        correct=np.diag(similarity);other=similarity.copy();np.fill_diagonal(other,-np.inf)
        accuracies.append(np.mean(similarity.argmax(1)==np.arange(len(unique))))
        margins.extend(correct-np.max(other,axis=1))
    return dict(mean_top1=float(np.mean(accuracies)),sd_top1=float(np.std(accuracies)),
        mean_cosine_margin=float(np.mean(margins)),median_cosine_margin=float(np.median(margins)),
        trials=trials,organoids=len(unique),chance_top1=1/len(unique))


def dispersion(x,samples):
    centroids={s:normalize(x[samples==s].mean(0,keepdims=True))[0] for s in np.unique(samples)}
    within=np.mean([1-x[i]@centroids[s] for i,s in enumerate(samples)])
    c=np.stack(list(centroids.values()));d=1-c@c.T;between=d[np.triu_indices(len(c),1)].mean()
    return dict(mean_within_cosine_distance=float(within),mean_between_centroid_distance=float(between),
                between_within_ratio=float(between/max(within,1e-12)))


def main():
    manifest=pd.read_csv(ROOT/"results/simclr3d/manifest.csv")
    morphology=pd.read_csv(ROOT/"results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id").loc[manifest.cell_id]
    qc=manifest.shape_qc_pass.to_numpy();counts=manifest[qc].groupby("sample").size()
    eligible=set(counts[counts>=4].index);use=qc & manifest["sample"].isin(eligible).to_numpy()
    samples=manifest.loc[use,"sample"].to_numpy()
    traditional_columns=["volume_um3","sphericity","elongation","flatness","mesh_solidity",
                         "surface_to_volume_per_um","major_axis_um","middle_axis_um","minor_axis_um"]
    traditional=StandardScaler().fit_transform(morphology[traditional_columns].to_numpy())[use]
    binary=np.load(ROOT/"results/simclr3d/embeddings.npy")[use]
    isolated=np.load(ROOT/"results/simclr3d_raw_isolated/embeddings.npy")[use]
    context=np.load(ROOT/"results/simclr3d_raw/embeddings.npy")[use]
    icone=np.load(ROOT/"results/icone3d_raw_isolated/embeddings.npy")[use]
    reps={
        "Traditional cell features":normalize(traditional),
        "Binary-mask SimCLR":binary,
        "Isolated-raw SimCLR":isolated,
        "Context-rich raw SimCLR":context,
        "Isolated-raw IConE":icone,
        "Binary + isolated-raw SimCLR":normalize(np.concatenate([binary,isolated],axis=1)),
    }
    context_icone_path=ROOT/"results/icone3d_raw_context/embeddings.npy"
    if context_icone_path.exists():
        reps["Context-rich raw IConE"]=np.load(context_icone_path)[use]
    rows=[]
    for name,x in reps.items():
        row=dict(representation=name,**split_half_retrieval(x,samples),**dispersion(x,samples))
        rows.append(row);print(name,json.dumps(row),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/"split_half_retrieval.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].barh(frame.representation,frame.mean_top1,xerr=frame.sd_top1,color="#527aa3")
    axes[0].axvline(frame.chance_top1.iloc[0],color="black",ls="--",label="Chance")
    axes[0].set(xlabel="Same-organoid top-1 retrieval",xlim=(0,1));axes[0].legend()
    axes[1].barh(frame.representation,frame.between_within_ratio,color="#5ca47a")
    axes[1].axvline(1,color="black",ls="--");axes[1].set(xlabel="Between / within cosine distance")
    fig.suptitle("Do independent cell subsets preserve organoid identity?")
    fig.tight_layout();fig.savefig(OUT/"comparison.png",dpi=160);plt.close(fig)
    # Export organoid centroids and conventional distribution summaries for downstream work.
    exports=[]
    for sample in sorted(eligible):
        ids=np.where(samples==sample)[0];row={"sample":sample,"cells":len(ids)}
        for name,x in reps.items():
            key=name.lower().replace(" ","_").replace("+","plus").replace("-","_")
            centroid=normalize(x[ids].mean(0,keepdims=True))[0]
            for j,value in enumerate(centroid):row[f"{key}_{j:03d}"]=value
        for column in traditional_columns:
            values=morphology.loc[manifest.loc[use,"cell_id"],column].to_numpy()[ids]
            row[f"{column}_mean"]=values.mean();row[f"{column}_sd"]=values.std()
        exports.append(row)
    pd.DataFrame(exports).to_csv(OUT/"organoid_features.csv",index=False)
    best=frame.sort_values("mean_top1",ascending=False).iloc[0]
    lines=["# Organoids represented by their constituent cells","",
        f"Analysis includes {len(eligible)} image stacks with at least four quality-passing cells ({use.sum()} cells). Each trial randomly divides every stack's cells in half. A centroid from one half retrieves the closest centroid among the other halves of all stacks. Results average 200 random partitions; chance is {frame.chance_top1.iloc[0]:.1%}.","",
        "| Cell representation | Same-organoid top-1 | SD | Cosine margin | Between/within distance |","|---|---:|---:|---:|---:|"]
    for _,r in frame.iterrows():lines.append(f"| {r.representation} | {r.mean_top1:.1%} | {r.sd_top1:.1%} | {r.mean_cosine_margin:.3f} | {r.between_within_ratio:.2f} |")
    lines += ["",f"The strongest split-half identity signal is {best.representation} ({best.mean_top1:.1%}). This tests reproducibility of stack-specific cell distributions, not biological correctness.","",
        "A high score can arise from biological organoid differences, acquisition differences, segmentation behavior, or—especially for context-rich crops—shared background and neighbourhood structure. The two halves are drawn from the same image stack, so they are not independent biological replicates.","",
        "The filename sequence may contain related observations or time points, and no animal/organoid grouping metadata has been established. Consequently, these results show that representations distinguish image stacks through their cells; they do not yet prove differences between independent organoids. A biological comparison requires organoid identity, condition, animal and time-point metadata, followed by group-held-out evaluation."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":main()
