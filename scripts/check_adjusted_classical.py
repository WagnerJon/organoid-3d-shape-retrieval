"""Post-primary, symmetric nuisance-adjustment sensitivity check; no selection."""
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from prepare_spatial_benchmark import OUT
from pointnet_shape_core import sample_surface,save_json
from spatial_benchmark_core import morphology,CONTROLS,CLASSICAL,mesh_for,geometry_cost
from evaluate_spatial_benchmark import residual_fit,residual_apply
from evaluate_sdf_shape import group_ci


def main():
    assert (OUT/'test/metrics.json').exists()
    train=pd.read_csv(OUT/'dev_manifest.csv');train=train[train.split=='train'].reset_index(drop=True)
    frame=pd.read_csv(OUT/'test/manifest.csv');records=pd.read_csv(OUT/'test/retrieval.csv')
    original=morphology(train,CLASSICAL);mean=original.mean(0);std=original.std(0).clip(1e-6)
    fit=residual_fit(train,(original-mean)/std)
    residual=residual_apply(frame,(morphology(frame,CLASSICAL)-mean)/std,fit)
    np.savez(OUT/'adjusted_classical_fit.npz',**fit,classical_mean=mean,classical_std=std)
    ctr=morphology(train,CONTROLS);controls=(morphology(frame,CONTROLS)-ctr.mean(0))/ctr.std(0).clip(1e-6)
    matching=cdist(controls,controls);distances=cdist(residual,residual);clouds=[]
    for _,row in frame.iterrows():
        v,f=mesh_for(row);clouds.append(sample_surface(v,f,384,np.random.default_rng(20261030+int(row.original_index))))
    rows=[];costs={};groups=frame.similarity_group.to_numpy()
    for record in records[records.method=='pca_residual'].itertuples():
        i=int(record.query);ids=np.flatnonzero((groups!=groups[i])&(matching[i]<=1))
        if len(ids)<5:raise AssertionError('Candidate set changed')
        ids=ids[np.argsort(matching[i,ids])[:20]];j=int(ids[np.argmin(distances[i,ids])]);pair=tuple(sorted([i,j]))
        if pair not in costs:costs[pair]=geometry_cost(clouds[pair[0]],clouds[pair[1]])
        cost=costs[pair]
        rows.append(dict(query=i,cell_id=record.cell_id,similarity_group=int(groups[i]),classical_cell=frame.iloc[j].cell_id,
            adjusted_classical_cost=cost,pca_residual_cost=float(record.geometry_cost),improvement=cost-float(record.geometry_cost)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'adjusted_classical_queries.csv',index=False)
    grouped=result.groupby('similarity_group')[['adjusted_classical_cost','pca_residual_cost']].mean().mean()
    summary=group_ci(result,'improvement')
    save_json(OUT/'adjusted_classical.json',dict(scope='Post-primary sensitivity check after visual review; not a new confirmatory test or basis for model selection.',
        comparison='PCA residual features vs nine classical features, both adjusted using the identical training-only degree-2 Ridge procedure.',
        result=summary,queries=len(result),relative_lower_cost=float(1-grouped.pca_residual_cost/grouped.adjusted_classical_cost),
        model_or_primary_analysis_changed=False))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
