"""Create a descriptive report of organoid phenotypes in two representation spaces."""
from pathlib import Path
import json, os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/organoid_phenotype_report"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import numpy as np
import pandas as pd
import tifffile
from scipy.cluster.hierarchy import linkage,leaves_list
from scipy.spatial.distance import pdist,squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES=["volume_um3","surface_area_um2","sphericity","aspect_ratio",
          "elongation","flatness","mesh_solidity","surface_to_volume_per_um"]
DISPLAY={"volume_um3":"Volume","surface_area_um2":"Surface area","sphericity":"Sphericity",
         "aspect_ratio":"Aspect ratio","elongation":"Elongation","flatness":"Flatness",
         "mesh_solidity":"Solidity","surface_to_volume_per_um":"Surface / volume"}


def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def permutation_distance_correlation(a,b,n=5000,seed=20260922):
    observed=float(spearmanr(pdist(a),pdist(b,metric="cosine")).statistic)
    rng=np.random.default_rng(seed);null=np.empty(n)
    for i in range(n):null[i]=spearmanr(pdist(a),pdist(b[rng.permutation(len(b))],metric="cosine")).statistic
    return observed,float((1+np.sum(np.abs(null)>=abs(observed)))/(n+1))


def main():
    manifest=pd.read_csv(ROOT/"results/simclr3d/manifest.csv")
    raw_manifest=pd.read_csv(ROOT/"results/simclr3d_raw/manifest.csv")
    cells=pd.read_csv(ROOT/"results/cpsam_v2_all_analysis/all_cells.csv").set_index("cell_id").loc[manifest.cell_id].reset_index()
    qc=manifest.shape_qc_pass.to_numpy();cells=cells.loc[qc].reset_index(drop=True)
    samples=manifest.loc[qc,"sample"].to_numpy();embedding=np.load(ROOT/"results/simclr3d_raw/embeddings.npy")[qc]
    # Conventional organoid summaries contain both central tendency and cell-to-cell spread.
    rows=[]
    for sample in sorted(np.unique(samples)):
        use=samples==sample;row={"sample":sample,"cell_count":int(use.sum())}
        for feature in FEATURES:
            values=cells.loc[use,feature].to_numpy(float)
            row[feature+"_mean"]=float(values.mean());row[feature+"_sd"]=float(values.std())
        rows.append(row)
    summary=pd.DataFrame(rows);sample_order=summary["sample"].to_numpy()
    traditional_columns=[c for c in summary if c!="sample"]
    traditional_scaled=StandardScaler().fit_transform(summary[traditional_columns])
    traditional_pca=PCA().fit(traditional_scaled);traditional_scores=traditional_pca.transform(traditional_scaled)
    # Context-rich organoid centroids and cell-level dispersion.
    centroids=[];context_diversity=[];traditional_diversity=[]
    cell_traditional=StandardScaler().fit_transform(cells[FEATURES])
    for sample in sample_order:
        use=samples==sample
        centroid=normalize(embedding[use].mean(0,keepdims=True))[0];centroids.append(centroid)
        context_diversity.append(float(np.mean(1-embedding[use]@centroid)))
        values=cell_traditional[use];traditional_diversity.append(float(np.sqrt(np.mean(np.sum((values-values.mean(0))**2,axis=1)))))
    centroids=np.stack(centroids);context_pca=PCA().fit(centroids);context_scores=context_pca.transform(centroids)
    summary["traditional_pc1"]=traditional_scores[:,0];summary["traditional_pc2"]=traditional_scores[:,1]
    summary["context_pc1"]=context_scores[:,0];summary["context_pc2"]=context_scores[:,1]
    summary["traditional_cell_diversity"]=traditional_diversity;summary["context_cell_diversity"]=context_diversity
    centroid_frame=pd.DataFrame(centroids,columns=[f"context_centroid_{j:03d}" for j in range(centroids.shape[1])])
    summary=pd.concat([summary,centroid_frame],axis=1)
    summary.to_csv(OUT/"organoid_phenotype_summary.csv",index=False)
    traditional_distance=pdist(traditional_scaled);context_distance=pdist(centroids,metric="cosine")
    correlation,pvalue=permutation_distance_correlation(traditional_scaled,centroids)
    pair_rows=[];k=0
    for i in range(len(summary)):
        for j in range(i+1,len(summary)):
            pair_rows.append(dict(sample_a=sample_order[i],sample_b=sample_order[j],
                traditional_distance=traditional_distance[k],context_distance=context_distance[k]));k+=1
    pd.DataFrame(pair_rows).to_csv(OUT/"pairwise_organoid_distances.csv",index=False)
    diversity_correlation=float(spearmanr(traditional_diversity,context_diversity).statistic)
    # Select nonduplicated extremes for microscopy thumbnails.
    candidates=[]
    for label,values in [("Low traditional PC1",traditional_scores[:,0]),("High traditional PC1",traditional_scores[:,0]),
                         ("Low traditional PC2",traditional_scores[:,1]),("High traditional PC2",traditional_scores[:,1]),
                         ("Low context PC1",context_scores[:,0]),("High context PC1",context_scores[:,0]),
                         ("Low context PC2",context_scores[:,1]),("High context PC2",context_scores[:,1])]:
        order=np.argsort(values) if label.startswith("Low") else np.argsort(values)[::-1]
        index=next(int(i) for i in order if int(i) not in [x[1] for x in candidates]);candidates.append((label,index))
    representatives=[]
    for label,index in candidates:
        row=summary.iloc[index]
        representatives.append(dict(role=label,sample=row["sample"],cell_count=int(row.cell_count),
            mean_volume_um3=row.volume_um3_mean,mean_sphericity=row.sphericity_mean,
            traditional_pc1=row.traditional_pc1,traditional_pc2=row.traditional_pc2,
            context_pc1=row.context_pc1,context_pc2=row.context_pc2))
    pd.DataFrame(representatives).to_csv(OUT/"representative_organoids.csv",index=False)
    # Figure 1: two phenotype maps.
    fig,axes=plt.subplots(1,2,figsize=(14,5.5));volume=summary.volume_um3_mean.to_numpy()
    sizes=25+4*summary.cell_count.to_numpy()
    for ax,scores,pca,title in [(axes[0],traditional_scores,traditional_pca,"Traditional size and shape summaries"),
                                (axes[1],context_scores,context_pca,"Context-rich SimCLR centroids")]:
        sc=ax.scatter(scores[:,0],scores[:,1],c=volume,s=sizes,cmap="viridis",alpha=.85,edgecolor="white",linewidth=.4)
        ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",title=title)
    for label,index in candidates:
        target=axes[0] if "traditional" in label else axes[1];scores=traditional_scores if target is axes[0] else context_scores
        target.annotate(sample_order[index],scores[index,:2],xytext=(3,3),textcoords="offset points",fontsize=7)
    cax=fig.add_axes([.92,.18,.015,.65]);fig.colorbar(sc,cax=cax,label="Mean cell volume (µm³)");fig.suptitle("Observed organoid phenotype spaces (point size = cell count)")
    fig.subplots_adjust(left=.07,right=.89,bottom=.12,top=.87,wspace=.28);fig.savefig(OUT/"phenotype_spaces.png",dpi=180);plt.close(fig)
    # Figure 2: feature heatmap clustered by conventional phenotype.
    mean_columns=[f+"_mean" for f in FEATURES];heat=StandardScaler().fit_transform(summary[mean_columns])
    order=leaves_list(linkage(heat,method="ward"));fig,ax=plt.subplots(figsize=(15,4.6))
    im=ax.imshow(heat[order].T,aspect="auto",cmap="coolwarm",vmin=-2.5,vmax=2.5)
    ax.set_yticks(range(len(FEATURES)),[DISPLAY[x] for x in FEATURES]);ax.set_xticks(range(len(order)),summary["sample"].to_numpy()[order],rotation=90,fontsize=5)
    ax.set(title="Traditional mean cell phenotypes across organoids",xlabel="Image stack / putative organoid")
    fig.colorbar(im,ax=ax,label="Standard deviations from dataset mean",shrink=.8);fig.tight_layout();fig.savefig(OUT/"traditional_feature_heatmap.png",dpi=180);plt.close(fig)
    # Figure 3: cross-representation agreement and within-organoid diversity.
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5))
    hb=axes[0].hexbin(traditional_distance,context_distance,gridsize=35,mincnt=1,cmap="magma")
    fig.colorbar(hb,ax=axes[0],label="Organoid pairs");axes[0].set(xlabel="Traditional phenotype distance",ylabel="Context SimCLR cosine distance",title=f"Distance agreement: Spearman ρ={correlation:.2f}")
    axes[1].scatter(traditional_diversity,context_diversity,c=volume,cmap="viridis",s=sizes,alpha=.8)
    axes[1].set(xlabel="Within-organoid traditional diversity",ylabel="Within-organoid context diversity",title=f"Cell-diversity agreement: ρ={diversity_correlation:.2f}")
    fig.tight_layout();fig.savefig(OUT/"representation_agreement.png",dpi=180);plt.close(fig)
    # Figure 4: raw maximum projections of phenotype-space extremes.
    image_paths=raw_manifest.groupby("sample").raw_image.first().to_dict();fig,axes=plt.subplots(2,4,figsize=(13,8.5))
    for ax,(label,index) in zip(axes.ravel(),candidates):
        image=tifffile.imread(image_paths[sample_order[index]]).astype(float);projection=np.percentile(image,95,axis=0)
        lo,hi=np.percentile(projection,[1,99.5]);ax.imshow(np.clip((projection-lo)/(hi-lo),0,1),cmap="gray")
        ax.set_title(f"{sample_order[index]}\n{label}\n{int(summary.iloc[index].cell_count)} cells",fontsize=9);ax.axis("off")
    fig.suptitle("Representative stacks at phenotype-space extremes\n95th-percentile Z projections; identical percentile display normalization")
    fig.subplots_adjust(left=.03,right=.99,bottom=.04,top=.87,wspace=.18,hspace=.36);fig.savefig(OUT/"representative_organoids.png",dpi=180);plt.close(fig)
    diagnostics=dict(organoids=len(summary),quality_passing_cells=int(qc.sum()),cell_count=dict(min=int(summary.cell_count.min()),median=float(summary.cell_count.median()),max=int(summary.cell_count.max())),
        traditional_pca_variance=traditional_pca.explained_variance_ratio_[:5].tolist(),context_pca_variance=context_pca.explained_variance_ratio_[:5].tolist(),
        traditional_context_distance_spearman=correlation,permutation_pvalue=pvalue,within_diversity_spearman=diversity_correlation,
        traditional_pair_distance=dict(median=float(np.median(traditional_distance)),q10=float(np.quantile(traditional_distance,.1)),q90=float(np.quantile(traditional_distance,.9))),
        context_pair_distance=dict(median=float(np.median(context_distance)),q10=float(np.quantile(context_distance,.1)),q90=float(np.quantile(context_distance,.9))))
    (OUT/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    report=f'''# Diversity of observed organoid phenotypes

## Scope

This report describes 108 image stacks through 985 quality-passing reconstructed cells (median {summary.cell_count.median():.0f}, range {summary.cell_count.min()}–{summary.cell_count.max()} cells per stack). It presents two complementary phenotype views. “Organoid” below means one `X###` image stack; independent organoid identity, animal, experimental condition and time point have not been confirmed.

## Traditional morphology view

Each organoid is summarized by cell count plus the mean and standard deviation of cell volume, surface area, sphericity, aspect ratio, elongation, flatness, mesh solidity and surface-to-volume ratio. After standardization, PC1 and PC2 explain {traditional_pca.explained_variance_ratio_[0]:.1%} and {traditional_pca.explained_variance_ratio_[1]:.1%} of between-stack variation. Pairwise phenotype distances have median {np.median(traditional_distance):.2f} (10th–90th percentile {np.quantile(traditional_distance,.1):.2f}–{np.quantile(traditional_distance,.9):.2f}). This view is interpretable: distances can be traced to physical size and named shape measurements.

## Context-rich SimCLR view

Each organoid is represented by the normalized centroid of its cells’ 128-dimensional context-rich raw-image embeddings. These inputs retain a 40 µm neighbourhood around each cell, so the representation may encode local packing, neighbouring membranes, illumination and acquisition characteristics in addition to phenotype. PC1 and PC2 explain {context_pca.explained_variance_ratio_[0]:.1%} and {context_pca.explained_variance_ratio_[1]:.1%} of centroid variation. Pairwise cosine distances have median {np.median(context_distance):.4f} (10th–90th percentile {np.quantile(context_distance,.1):.4f}–{np.quantile(context_distance,.9):.4f}).

## Agreement and diversity

The two pairwise distance maps have Spearman correlation ρ={correlation:.3f} (label-permutation p={pvalue:.4f}). The association measures whether organoid pairs that differ conventionally also differ in learned context; it does not establish biological validity. Within-organoid cell heterogeneity agrees with ρ={diversity_correlation:.3f} between the traditional and context representations.

The representations should therefore be treated as complementary. Traditional features describe known size and shape axes. Context-rich SimCLR captures a stack-level signature that can include organization and image context, but earlier split-half retrieval was only 4.4% among 108 stacks (0.9% chance), so the identity signal is modest.

## Figures and tables

- `phenotype_spaces.png`: side-by-side PCA maps; point color is mean cell volume and point size is reconstructed cell count.
- `traditional_feature_heatmap.png`: interpretable mean morphology profiles, hierarchically ordered for display.
- `representation_agreement.png`: pairwise distance and within-organoid diversity agreement.
- `representative_organoids.png`: raw projections at extremes of both phenotype spaces. These are examples, not discovered phenotype classes.
- `organoid_phenotype_summary.csv`: organoid-level traditional summaries, PCA coordinates, diversity values and context centroids.
- `pairwise_organoid_distances.csv`: all pairwise distances in both representations.
- `representative_organoids.csv`: identifiers and measurements for displayed examples.

## Interpretation limits

No clustering result in this report should be called an organoid subtype. Cells from the same image are not independent biological replicates, segmentation errors affect both views, and the assumed voxel spacing was taken from dataset literature rather than TIFF metadata. Confirm organoid/animal/condition/time metadata before formal group testing. A strong next analysis would aggregate at the true biological replicate level and evaluate whether predefined conditions or blinded phenotype annotations generalize to held-out organoids.
'''
    (OUT/"REPORT.md").write_text(report)
    print(json.dumps(diagnostics,indent=2))


if __name__=="__main__":main()
