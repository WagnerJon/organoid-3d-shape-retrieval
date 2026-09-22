"""Evaluate distribution-aware organoid comparisons with split-half retrieval."""
from pathlib import Path
import json, os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/organoid_distribution_comparison"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist,pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES=["volume_um3","surface_area_um2","sphericity","aspect_ratio",
          "elongation","flatness","mesh_solidity","surface_to_volume_per_um"]


def centroid_distance(a,b):
    ma=a.mean(0);mb=b.mean(0)
    ma/=max(np.linalg.norm(ma),1e-12);mb/=max(np.linalg.norm(mb),1e-12)
    return 1-float(ma@mb)


def sliced_wasserstein(a,b):
    # Quantile-grid approximation avoids millions of small scipy sorting calls.
    q=np.linspace(0,1,17)
    return float(np.mean(np.abs(np.quantile(a,q,axis=0)-np.quantile(b,q,axis=0))))


def energy_distance(a,b):
    value=2*cdist(a,b).mean()-cdist(a,a).mean()-cdist(b,b).mean()
    return float(np.sqrt(max(value,0)))


def mmd_distance(a,b,gamma):
    kaa=np.exp(-gamma*cdist(a,a,"sqeuclidean")).mean()
    kbb=np.exp(-gamma*cdist(b,b,"sqeuclidean")).mean()
    kab=np.exp(-gamma*cdist(a,b,"sqeuclidean")).mean()
    return float(np.sqrt(max(kaa+kbb-2*kab,0)))


def profile(a):
    q=np.quantile(a,[.1,.25,.5,.75,.9],axis=0)
    return np.concatenate([q.ravel(),a.std(0)])


def profile_distance(a,b):return float(np.linalg.norm(profile(a)-profile(b))/np.sqrt(len(profile(a))))


def evaluate(values,samples,prefix,gamma,trials=20,seed=20260922):
    rng=np.random.default_rng(seed);organoids=np.unique(samples);records=[]
    for trial in range(trials):
        left={};right={}
        for sample in organoids:
            ids=rng.permutation(np.where(samples==sample)[0]);cut=len(ids)//2
            left[sample]=values[ids[:cut]];right[sample]=values[ids[cut:]]
        left_centroid=np.stack([normalize_rows(left[x].mean(0,keepdims=True))[0] for x in organoids])
        right_centroid=np.stack([normalize_rows(right[x].mean(0,keepdims=True))[0] for x in organoids])
        left_profile=np.stack([profile(left[x]) for x in organoids]);right_profile=np.stack([profile(right[x]) for x in organoids])
        q=np.linspace(0,1,17)
        left_quantile=np.stack([np.quantile(left[x],q,axis=0) for x in organoids])
        right_quantile=np.stack([np.quantile(right[x],q,axis=0) for x in organoids])
        distance_matrices={
            prefix+(" PCA centroid" if prefix=="Context" else " centroid"):1-left_centroid@right_centroid.T,
            prefix+" quantiles + spread":cdist(left_profile,right_profile)/np.sqrt(left_profile.shape[1]),
            prefix+" sliced Wasserstein":np.mean(np.abs(left_quantile[:,None]-right_quantile[None,:]),axis=(2,3)),
        }
        for method_name in [prefix+" energy distance",prefix+" kernel MMD"]:
            matrix=np.empty((len(organoids),len(organoids)))
            distance=energy_distance if method_name.endswith("energy distance") else lambda a,b:mmd_distance(a,b,gamma)
            for i,a in enumerate(organoids):
                for j,b in enumerate(organoids):matrix[i,j]=distance(left[a],right[b])
            distance_matrices[method_name]=matrix
        for method_name,matrix in distance_matrices.items():
            correct=0;ranks=[];margins=[]
            for target,sample in enumerate(organoids):
                candidates=matrix[target];order=np.argsort(candidates)
                correct+=int(order[0]==target);ranks.append(int(np.where(order==target)[0][0])+1)
                other=np.delete(candidates,target);margins.append(float(other.min()-candidates[target]))
            records.append(dict(trial=trial,method=method_name,top1=correct/len(organoids),
                median_rank=float(np.median(ranks)),mean_margin=float(np.mean(margins))))
    return pd.DataFrame(records)


def normalize_rows(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def main():
    manifest=pd.read_csv(ROOT/"results/simclr3d/manifest.csv")
    cells=pd.read_csv(ROOT/"results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id").loc[manifest.cell_id]
    qc=manifest.shape_qc_pass.to_numpy();samples=manifest.loc[qc,"sample"].to_numpy()
    traditional=StandardScaler().fit_transform(cells.loc[qc,FEATURES])
    context=np.load(ROOT/"results/simclr3d_raw/embeddings.npy")[qc]
    # Retain stable variation, then give retained axes equal units for distribution distances.
    pca=PCA(n_components=min(8,context.shape[1]),svd_solver="full").fit(context)
    context_reduced=StandardScaler().fit_transform(pca.transform(context))
    traditional_gamma=1/max(2*np.median(pdist(traditional,"sqeuclidean")),1e-12)
    context_gamma=1/max(2*np.median(pdist(context_reduced,"sqeuclidean")),1e-12)
    result=pd.concat([evaluate(traditional,samples,"Traditional",traditional_gamma),
                      evaluate(context_reduced,samples,"Context",context_gamma)],ignore_index=True)
    result.to_csv(OUT/"split_half_trials.csv",index=False)
    summary=result.groupby("method").agg(mean_top1=("top1","mean"),sd_top1=("top1","std"),
        median_rank=("median_rank","mean"),mean_margin=("mean_margin","mean")).reset_index()
    summary["chance_top1"]=1/len(np.unique(samples));summary.to_csv(OUT/"method_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,5.5));ordered=summary.sort_values("mean_top1")
    colors=["#577fa6" if x.startswith("Traditional") else "#58a17b" for x in ordered.method]
    ax.barh(ordered.method,ordered.mean_top1,xerr=ordered.sd_top1,color=colors)
    ax.axvline(1/len(np.unique(samples)),color="black",ls="--",label="Chance")
    ax.set(xlabel="Same-organoid top-1 retrieval",title="Centroids versus cell-distribution comparisons")
    ax.legend();fig.tight_layout();fig.savefig(OUT/"method_comparison.png",dpi=180);plt.close(fig)
    best_traditional=summary[summary.method.str.startswith("Traditional")].sort_values("mean_top1",ascending=False).iloc[0]
    best_context=summary[summary.method.str.startswith("Context")].sort_values("mean_top1",ascending=False).iloc[0]
    diagnostics=dict(organoids=int(len(np.unique(samples))),cells=int(qc.sum()),trials=20,
        context_pca_components=int(pca.n_components_),context_pca_variance=float(pca.explained_variance_ratio_.sum()),
        best_traditional=best_traditional.to_dict(),best_context=best_context.to_dict())
    (OUT/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    lines=["# Capturing organoid heterogeneity as cell distributions","",
        f"The analysis uses {qc.sum()} quality-passing cells across {len(np.unique(samples))} stacks. Every trial divides each stack's cells into two random halves and asks whether one half retrieves the other among all stacks. Results average 20 seeded partitions; top-1 chance is {1/len(np.unique(samples)):.1%}.","",
        "| Method | Top-1 retrieval | SD | Mean median rank | Margin |","|---|---:|---:|---:|---:|"]
    for _,row in summary.sort_values("mean_top1",ascending=False).iterrows():
        lines.append(f"| {row.method} | {row.mean_top1:.1%} | {row.sd_top1:.1%} | {row.median_rank:.1f} | {row.mean_margin:.3f} |")
    lines += ["",f"Best traditional method: **{best_traditional.method}** ({best_traditional.mean_top1:.1%}). Best context method: **{best_context.method}** ({best_context.mean_top1:.1%}).","",
        "Centroids retain only the average cell. Quantile profiles add tails and spread; sliced Wasserstein compares every marginal cell distribution using a 17-point quantile-grid approximation; energy distance and kernel MMD compare multivariate cell sets. Context embeddings were reduced to their first eight principal components, then standardized before distribution comparison.","",
        "The small number of cells per stack is the main limitation: random halves contain only 2–7 cells. Estimates of rare subpopulations, covariance and distribution tails are therefore noisy. Higher retrieval indicates a more reproducible stack signature, but can still reflect acquisition or segmentation differences rather than biological phenotype.","",
        "For downstream work, retain cell count, robust traditional quantiles, feature spread, the original context centroid, and the complete cell-level embedding set. Use the centroid as the stable core and add heterogeneity terms with shrinkage toward the dataset average; weight them only as strongly as split-half reliability supports. Report bootstrap intervals for every organoid. Direct full-distribution distances are currently too noisy to replace the centroid.","",
        "A practical organoid record should contain: cell count; mean, SD and 10/25/50/75/90% quantiles of traditional features; original context-rich embedding centroid; dispersion along a small number of stable embedding PCs; and links to all constituent cell rows. Rare-cell fractions or a learned DeepSets/Set Transformer model become appropriate after substantially more cells per organoid, organoid-level labels, and independent biological replicates are available."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(summary.sort_values("mean_top1",ascending=False).to_string(index=False))


if __name__=="__main__":main()
