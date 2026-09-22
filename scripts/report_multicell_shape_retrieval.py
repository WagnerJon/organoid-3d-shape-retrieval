"""Create like-for-like comparisons and chance baselines for multi-scale retrieval."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_multicell_shape_retrieval import metrics
from train_shape_retrieval import ROOT


def chance(shape, truth, samples, repeats=200):
    rng=np.random.default_rng(20260923);rows=[]
    for _ in range(repeats):rows.append(metrics(rng.normal(size=shape),truth,samples))
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    out=ROOT/"results/shape_retrieval_multicell";manifest=pd.read_csv(out/"manifest.csv",index_col="index")
    test_ids=manifest.index[manifest.split=="test"].to_numpy();train_ids=manifest.index[manifest.split=="train"].to_numpy();test=manifest.loc[test_ids].reset_index()
    student=np.load(out/"test_student_embeddings.npy");teacher=np.load(out/"test_teacher_embeddings.npy")
    cols=[f"shape_{i}" for i in range(8)];values=manifest[cols].to_numpy(float);mean=values[train_ids].mean(0);std=values[train_ids].std(0).clip(1e-8);truth=(values[test_ids]-mean)/std
    results={}
    for level in ["single","neighborhood","organoid"]:
        use=np.flatnonzero(test.level.to_numpy()==level);samples=test.loc[use,"sample"].to_numpy()
        results[level]={"new_student":metrics(student[use],truth[use],samples),"new_teacher":metrics(teacher[use],truth[use],samples),"chance":chance(student[use].shape,truth[use],samples)}
    # Like-for-like single-cell comparison with the earlier specialized model.
    single=np.flatnonzero(test.level.to_numpy()=="single")
    old_manifest=pd.read_csv(ROOT/"results/shape_retrieval/manifest.csv");old_test=old_manifest[old_manifest.split=="test"]
    assert test.loc[single,"source_cell_id"].tolist()==old_test.cell_id.tolist()
    old_student=np.load(ROOT/"results/shape_retrieval/test_student_embeddings.npy");old_teacher=np.load(ROOT/"results/shape_retrieval/test_teacher_embeddings.npy")
    samples=test.loc[single,"sample"].to_numpy();results["single"]["old_student"]=metrics(old_student,truth[single],samples);results["single"]["old_teacher"]=metrics(old_teacher,truth[single],samples)
    use=np.flatnonzero(test.level.to_numpy()=="neighborhood");arrangement=truth[use][:,1:7];samples=test.loc[use,"sample"].to_numpy()
    results["neighborhood_arrangement_only"]={"new_student":metrics(student[use],arrangement,samples),"new_teacher":metrics(teacher[use],arrangement,samples),"chance":chance(student[use].shape,arrangement,samples)}
    (out/"comparison_metrics.json").write_text(json.dumps(results,indent=2))

    fig,axes=plt.subplots(1,3,figsize=(13,4));keys=["recall_at_5","ndcg_at_5","distance_spearman"];titles=["Recall@5","NDCG@5","Distance correlation"]
    categories=["Single\nold","Single\nmultiscale","Neighborhood","Neighborhood\narrangement","Whole\norganoid"]
    blocks=[("single","old_student"),("single","new_student"),("neighborhood","new_student"),("neighborhood_arrangement_only","new_student"),("organoid","new_student")]
    chance_blocks=[("single","chance"),("single","chance"),("neighborhood","chance"),("neighborhood_arrangement_only","chance"),("organoid","chance")]
    x=np.arange(len(categories));width=.36
    for ax,key,title in zip(axes,keys,titles):
        vals=[results[a][b][key] for a,b in blocks];base=[results[a][b][key] for a,b in chance_blocks]
        ax.bar(x-width/2,vals,width,label="Raw student",color="#2563eb");ax.bar(x+width/2,base,width,label="Chance",color="#9ca3af")
        ax.set_xticks(x,categories,fontsize=8);ax.set_title(title);ax.set_ylim(0,max(vals+base)*1.2)
        for i,v in enumerate(vals):ax.text(i-width/2,v,f"{v:.2f}",ha="center",va="bottom",fontsize=8)
    axes[0].legend(fontsize=8);fig.suptitle("Held-out cross-organoid retrieval across spatial scales");fig.tight_layout();fig.savefig(out/"multiscale_retrieval.png",dpi=200);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(9,3.8));history=pd.read_csv(out/"training_history.csv")
    for ax,stage in zip(axes,["teacher","student"]):
        g=history[history.stage==stage];ax.plot(g.epoch,g.train_loss,label="Train");ax.plot(g.epoch,g.validation_loss,label="Validation");ax.set(title=stage.title(),xlabel="Epoch",ylabel="Loss");ax.legend()
    fig.suptitle("Multi-scale fine-tuning history");fig.tight_layout();fig.savefig(out/"training_curves.png",dpi=200);plt.close(fig)

    r=results
    report=f'''# Multi-cell shape-retrieval results

## Held-out results

| Scale | Queries | Raw Recall@5 | Chance | Raw NDCG@5 | Raw distance correlation |
|---|---:|---:|---:|---:|---:|
| Single cell, multiscale model | 132 | {r['single']['new_student']['recall_at_5']:.3f} | {r['single']['chance']['recall_at_5']:.3f} | {r['single']['new_student']['ndcg_at_5']:.3f} | {r['single']['new_student']['distance_spearman']:.3f} |
| Local neighborhood | 78 | {r['neighborhood']['new_student']['recall_at_5']:.3f} | {r['neighborhood']['chance']['recall_at_5']:.3f} | {r['neighborhood']['new_student']['ndcg_at_5']:.3f} | {r['neighborhood']['new_student']['distance_spearman']:.3f} |
| Neighborhood arrangement only | 78 | {r['neighborhood_arrangement_only']['new_student']['recall_at_5']:.3f} | {r['neighborhood_arrangement_only']['chance']['recall_at_5']:.3f} | {r['neighborhood_arrangement_only']['new_student']['ndcg_at_5']:.3f} | {r['neighborhood_arrangement_only']['new_student']['distance_spearman']:.3f} |
| Complete organoid | 16 | {r['organoid']['new_student']['recall_at_5']:.3f} | {r['organoid']['chance']['recall_at_5']:.3f} | {r['organoid']['new_student']['ndcg_at_5']:.3f} | {r['organoid']['new_student']['distance_spearman']:.3f} |

The new raw encoder retrieves neighborhoods and complete organoids above chance against mask-derived geometric summaries. The row labeled “neighborhood arrangement only” removes mask volume and occupancy, but the remaining target still contains bounding-box volume, spatial radius and extent. It therefore does not establish learning of arrangement independently of size or cell count. In the held-out embeddings, pairwise embedding distance correlates with absolute cell-count difference at Spearman ρ = 0.266 for neighborhoods and ρ = 0.754 for complete organoids. The complete-organoid result, based on only 16 test organoids, requires count-matched and size-matched evaluation before it can be interpreted as detailed shape retrieval.

## Single-cell tradeoff

Against the identical fixed-grid aggregate descriptors, the old single-scale student has Recall@5 {r['single']['old_student']['recall_at_5']:.3f}, NDCG@5 {r['single']['old_student']['ndcg_at_5']:.3f}, and distance correlation {r['single']['old_student']['distance_spearman']:.3f}. The multiscale student reaches {r['single']['new_student']['recall_at_5']:.3f}, {r['single']['new_student']['ndcg_at_5']:.3f}, and {r['single']['new_student']['distance_spearman']:.3f}. Multi-scale fine-tuning improves global single-cell distance ordering but reduces exact top-five recovery, indicating partial specialization tradeoff.

All retrieval candidates come from different held-out organoids. These are internal geometric targets derived from predicted masks, not independent biological phenotype labels.
'''
    (out/"REPORT.md").write_text(report);print(json.dumps(results,indent=2))


if __name__=="__main__":main()
