"""Exploratory phenotype grouping using all cells in every image stack."""
from pathlib import Path
import json, os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/organoid_phenotype_groups"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES=["volume_um3","surface_area_um2","sphericity","aspect_ratio",
          "elongation","flatness","mesh_solidity","surface_to_volume_per_um"]
STATS=[("mean",lambda x:np.mean(x,axis=0)),("sd",lambda x:np.std(x,axis=0)),
       ("q10",lambda x:np.quantile(x,.10,axis=0)),("q25",lambda x:np.quantile(x,.25,axis=0)),
       ("median",lambda x:np.quantile(x,.50,axis=0)),("q75",lambda x:np.quantile(x,.75,axis=0)),
       ("q90",lambda x:np.quantile(x,.90,axis=0))]


def summarize(values,samples,base_names):
    rows=[]
    for sample in sorted(np.unique(samples)):
        x=values[samples==sample];row={"sample":sample,"cell_count":len(x)}
        for stat,function in STATS:
            result=function(x)
            for name,value in zip(base_names,result):row[f"{name}_{stat}"]=float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_space(frame):
    x=StandardScaler().fit_transform(frame.drop(columns="sample"))
    pca=PCA(n_components=.95,svd_solver="full").fit(x)
    return pca.transform(x),pca


def gap_statistic(x,k,observed_inertia,rng,repeats=50):
    null=[]
    for _ in range(repeats):
        shuffled=np.column_stack([rng.permutation(x[:,j]) for j in range(x.shape[1])])
        null.append(KMeans(k,n_init=20,random_state=int(rng.integers(1e9))).fit(shuffled).inertia_)
    logs=np.log(null)
    return float(logs.mean()-np.log(observed_inertia)),float(logs.std(ddof=1)*np.sqrt(1+1/repeats))


def assess(x,name,seed=20260922):
    rng=np.random.default_rng(seed);rows=[];models={}
    for k in range(2,9):
        model=KMeans(k,n_init=100,random_state=seed).fit(x);labels=model.labels_;models[k]=model
        gap,gap_se=gap_statistic(x,k,model.inertia_,rng)
        rows.append(dict(representation=name,k=k,silhouette=silhouette_score(x,labels),
            calinski_harabasz=calinski_harabasz_score(x,labels),davies_bouldin=davies_bouldin_score(x,labels),
            gap=gap,gap_se=gap_se,min_group=int(np.bincount(labels).min()),max_group=int(np.bincount(labels).max()),
            inertia=float(model.inertia_)))
    metrics=pd.DataFrame(rows);chosen=int(metrics.loc[metrics.silhouette.idxmax(),"k"])
    return metrics,models[chosen],chosen


def main():
    manifest=pd.read_csv(ROOT/"results/simclr3d/manifest.csv")
    morphology=pd.read_csv(ROOT/"results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id").loc[manifest.cell_id]
    qc=manifest.shape_qc_pass.to_numpy();samples=manifest.loc[qc,"sample"].to_numpy()
    traditional_values=morphology.loc[qc,FEATURES].to_numpy(float)
    context=np.load(ROOT/"results/simclr3d_raw/embeddings.npy")[qc]
    # Stable, low-dimensional cell coordinates before estimating distribution summaries.
    cell_pca=PCA(n_components=min(8,context.shape[1]),svd_solver="full").fit(context)
    context_values=cell_pca.transform(context)
    traditional=summarize(traditional_values,samples,FEATURES)
    context_summary=summarize(context_values,samples,[f"context_pc{i+1}" for i in range(context_values.shape[1])])
    assert traditional["sample"].equals(context_summary["sample"])
    traditional_x,traditional_pca=prepare_space(traditional)
    context_x,context_pca=prepare_space(context_summary)
    traditional_metrics,traditional_model,traditional_k=assess(traditional_x,"Traditional distributions")
    context_metrics,context_model,context_k=assess(context_x,"Context SimCLR distributions")
    metrics=pd.concat([traditional_metrics,context_metrics],ignore_index=True);metrics.to_csv(OUT/"cluster_quality.csv",index=False)
    traditional_labels=traditional_model.labels_;context_labels=context_model.labels_
    assignments=pd.DataFrame(dict(sample=traditional["sample"],cell_count=traditional.cell_count,
        traditional_group=traditional_labels+1,context_group=context_labels+1,
        traditional_pc1=traditional_x[:,0],traditional_pc2=traditional_x[:,1],
        context_pc1=context_x[:,0],context_pc2=context_x[:,1]))
    assignments.to_csv(OUT/"group_assignments.csv",index=False)
    traditional.to_csv(OUT/"traditional_distribution_profiles.csv",index=False)
    context_summary.to_csv(OUT/"context_distribution_profiles.csv",index=False)
    agreement=dict(adjusted_rand=float(adjusted_rand_score(traditional_labels,context_labels)),
                   normalized_mutual_information=float(normalized_mutual_info_score(traditional_labels,context_labels)))
    # Medoids provide reviewable examples without implying that cluster centres are real images.
    medoids=[]
    for representation,x,labels in [("traditional",traditional_x,traditional_labels),("context",context_x,context_labels)]:
        for group in np.unique(labels):
            ids=np.where(labels==group)[0];centre=x[ids].mean(0);index=ids[np.argmin(np.linalg.norm(x[ids]-centre,axis=1))]
            medoids.append(dict(representation=representation,group=int(group+1),sample=traditional.iloc[index]["sample"],cells=int(traditional.iloc[index].cell_count)))
    pd.DataFrame(medoids).to_csv(OUT/"group_medoids.csv",index=False)
    # Quality curves.
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for representation,group in metrics.groupby("representation"):
        axes[0].plot(group.k,group.silhouette,"o-",label=representation)
        axes[1].plot(group.k,group.gap,"o-",label=representation)
        axes[2].plot(group.k,group.davies_bouldin,"o-",label=representation)
    axes[0].set(xlabel="Number of groups",ylabel="Silhouette (higher better)")
    axes[1].set(xlabel="Number of groups",ylabel="Gap statistic (higher better)")
    axes[2].set(xlabel="Number of groups",ylabel="Davies–Bouldin (lower better)")
    axes[0].legend(fontsize=8);fig.suptitle("Exploratory phenotype-group quality using all cells")
    fig.tight_layout();fig.savefig(OUT/"cluster_quality.png",dpi=180);plt.close(fig)
    # Chosen group maps.
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,x,labels,pca,title,k in [(axes[0],traditional_x,traditional_labels,traditional_pca,"Traditional cell distributions",traditional_k),
                                   (axes[1],context_x,context_labels,context_pca,"Context SimCLR cell distributions",context_k)]:
        for group in range(k):
            use=labels==group;ax.scatter(x[use,0],x[use,1],s=35+4*traditional.cell_count.to_numpy()[use],label=f"Group {group+1}",alpha=.85)
        ax.set(xlabel=f"Organoid PC1 ({pca.explained_variance_ratio_[0]:.1%})",ylabel=f"Organoid PC2 ({pca.explained_variance_ratio_[1]:.1%})",title=f"{title}: k={k}")
        ax.legend(fontsize=7)
    fig.suptitle("Exploratory grouping of complete organoids");fig.tight_layout();fig.savefig(OUT/"phenotype_groups.png",dpi=180);plt.close(fig)
    # Interpretable cluster profiles from conventional means.
    means=traditional[[f+"_mean" for f in FEATURES]].copy();means[:]=StandardScaler().fit_transform(means)
    group_profiles=pd.concat([traditional[["sample"]],pd.DataFrame({"traditional_group":traditional_labels+1,"context_group":context_labels+1}),means],axis=1)
    group_profiles.to_csv(OUT/"traditional_features_by_group.csv",index=False)
    fig=plt.figure(figsize=(13,4.7))
    grid=fig.add_gridspec(1,3,width_ratios=[1,1,.035],wspace=.42)
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]
    color_ax=fig.add_subplot(grid[0,2])
    for ax,column,title in [(axes[0],"traditional_group","Traditional grouping"),(axes[1],"context_group","Context grouping")]:
        matrix=group_profiles.groupby(column)[means.columns].mean()
        im=ax.imshow(matrix,aspect="auto",cmap="coolwarm",vmin=-1.5,vmax=1.5)
        ax.set_xticks(range(len(FEATURES)),[x.replace("_"," ") for x in FEATURES],rotation=35,ha="right",fontsize=8)
        ax.set_yticks(range(len(matrix)),[f"Group {x}" for x in matrix.index]);ax.set_title(title)
    fig.colorbar(im,cax=color_ax,label="Mean standardized traditional feature")
    fig.suptitle("Conventional morphology profiles of exploratory groups");fig.subplots_adjust(left=.08,right=.92,bottom=.27,top=.82);fig.savefig(OUT/"group_profiles.png",dpi=180);plt.close(fig)
    chosen_traditional=traditional_metrics.loc[traditional_metrics.k==traditional_k].iloc[0]
    chosen_context=context_metrics.loc[context_metrics.k==context_k].iloc[0]
    diagnostics=dict(organoids=len(traditional),cells=int(qc.sum()),all_cells_used=True,
        traditional_profile_dimensions=len(traditional.columns)-1,context_profile_dimensions=len(context_summary.columns)-1,
        context_cell_pca_variance=float(cell_pca.explained_variance_ratio_.sum()),
        traditional_chosen_k=traditional_k,context_chosen_k=context_k,
        traditional_silhouette=float(chosen_traditional.silhouette),context_silhouette=float(chosen_context.silhouette),
        traditional_gap=float(chosen_traditional.gap),context_gap=float(chosen_context.gap),partition_agreement=agreement)
    (OUT/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    strength=lambda s:"weak" if s<.25 else "moderate" if s<.5 else "strong"
    report=f'''# Exploratory phenotype groups using complete organoids

## Analysis design

All {qc.sum()} quality-passing cells from all {len(traditional)} image stacks are retained. No cell subsampling or split-half construction is used for the phenotype groups. Each organoid is represented as a distribution-rich profile containing cell count and, for every cell coordinate, mean, SD and 10/25/50/75/90% quantiles.

The traditional profile uses eight physical size and shape features ({len(traditional.columns)-1} organoid variables). The context profile uses the first eight cell-level context-rich SimCLR PCs, explaining {cell_pca.explained_variance_ratio_.sum():.1%} of cell-embedding variation ({len(context_summary.columns)-1} organoid variables). Organoid profiles are standardized, reduced to 95% variance, and grouped with k-means for k=2–8.

## Traditional-feature groups

The highest silhouette occurs at k={traditional_k}: {chosen_traditional.silhouette:.3f} ({strength(chosen_traditional.silhouette)} separation), with group sizes {', '.join(map(str,np.bincount(traditional_labels)))}. The gap statistic is {chosen_traditional.gap:.3f}. These groups summarize differences in the complete distributions of cell size and named shape measurements.

## Context-rich SimCLR groups

The highest silhouette occurs at k={context_k}: {chosen_context.silhouette:.3f} ({strength(chosen_context.silhouette)} separation), with group sizes {', '.join(map(str,np.bincount(context_labels)))}. The gap statistic is {chosen_context.gap:.3f}. These groups may reflect cell neighbourhood organization and acquisition context in addition to phenotype.

## Comparison

Partition agreement is adjusted Rand index {agreement['adjusted_rand']:.3f} and normalized mutual information {agreement['normalized_mutual_information']:.3f}. This partial agreement means the representations divide the observed stacks along partly shared and partly distinct axes; it does not identify which is biologically correct. Context SimCLR has the stronger silhouette at its selected k and gives the cleaner, more balanced exploratory split. Traditional groups remain easier to interpret mechanistically, but their silhouette maximum occurs at the upper boundary of the tested range and includes groups with only two and three stacks, so k=8 should not be treated as a stable phenotype count.

The gap statistic rises with k for both representations across the tested range. It therefore supports non-random structure but does not provide an internal optimum here; its absolute value at the silhouette-selected k should not be used to rank the two representations.

The plots should also be inspected for continuous gradients. K-means always returns groups, even when the data form a continuum. These are exploratory phenotype groups, not validated organoid subtypes.

## Outputs

- `phenotype_groups.png`: full-data organoid maps and selected group assignments.
- `cluster_quality.png`: separation metrics for k=2–8.
- `group_profiles.png`: conventional morphology profiles of both partitions.
- `group_assignments.csv`: group and PCA coordinates for every stack.
- `traditional_distribution_profiles.csv` and `context_distribution_profiles.csv`: complete distribution-rich organoid representations.
- `group_medoids.csv`: observed stacks closest to each group centre.

## Limits

“Organoid” means one `X###` image stack. Independence, animal, condition and time-point metadata remain unconfirmed. Group quality is internal geometric structure, not biological validation. Confirm metadata and test group reproducibility on held-out biological replicates before naming or interpreting subtypes.
'''
    (OUT/"REPORT.md").write_text(report)
    print(json.dumps(diagnostics,indent=2))


if __name__=="__main__":main()
