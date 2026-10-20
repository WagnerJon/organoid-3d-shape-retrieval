"""Development-only matched-neighbor benchmark of harmonic and CAJAL methods."""
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pointnet_shape_core import ROOT,sample_surface
from spatial_benchmark_core import mesh_for,geometry_cost
from evaluate_sdf_shape import group_ci
from atlas_shape_space import distances
from check_cajal_validation import candidate_sets

OUT=ROOT/'results/harmonics_cajal'


def main():
    data=json.loads((ROOT/'results/explorer_dataset/dataset.json').read_text());lookup={c['id']:c for c in data['cells']}
    frame,pools,_=candidate_sets();matrices={};names={}
    keys=['morph_controls','morph_size','morph_shape','morph_all','morph_adjusted','spatial_robust','spatial_adjusted',
          'spherical_harmonics','spherical_harmonics_adjusted','cajal_qgw','cajal_profile_adjusted']
    for m in data['methods']:
        if m['id'] in keys:
            matrices[m['id']]=distances([lookup[c]['representations'][m['id']]['vector'] for c in frame.cell_id],m['metric']);names[m['id']]=m['name']
    if not set(keys)<=set(matrices):raise ValueError('Export the new methods before evaluation')
    meta=json.loads((OUT/'validation_gw.json').read_text())
    if frame.cell_id.tolist()!=meta['ids']:raise ValueError('Validation identities changed')
    matrices['cajal_gw_unquantized']=np.load(OUT/'validation_gw.npy');names['cajal_gw_unquantized']='CAJAL · unquantized GW (validation check)'
    clouds=[]
    for row in frame.to_dict('records'):
        v,f=mesh_for(row);clouds.append(sample_surface(v,f,384,np.random.default_rng(20261030+int(row['original_index']))))
    # Reuse exactly the existing proxy costs; source identities and random seeds
    # are identical. New selected pairs get the same proper-rotation ICP proxy.
    costs={};positions={cid:i for i,cid in enumerate(frame.cell_id)}
    old=pd.read_csv(ROOT/'results/atlas_spatial/development_neighbors.csv')
    for row in old.itertuples():
        if row.cell_id in positions and row.neighbor in positions:costs[tuple(sorted([positions[row.cell_id],positions[row.neighbor]]))]=float(row.geometry_cost)
    records=[]
    for key,d in matrices.items():
        for i,ids in pools.items():
            if not np.isfinite(d[i,ids]).all():raise ValueError('Missing candidate comparisons')
            j=int(ids[np.argmin(d[i,ids])]);pair=tuple(sorted([i,j]))
            if pair not in costs:costs[pair]=geometry_cost(clouds[pair[0]],clouds[pair[1]])
            records.append(dict(method=key,query=i,cell_id=frame.iloc[i].cell_id,neighbor=frame.iloc[j].cell_id,
                                similarity_group=int(frame.iloc[i].similarity_group),geometry_cost=costs[pair]))
        print('Evaluated',names[key],flush=True)
    results=pd.DataFrame(records);results.to_csv(OUT/'development_neighbors.csv',index=False)
    baselines={key:results[results.method==key].set_index('query') for key in ['morph_all','morph_adjusted']};summary=[]
    for key in matrices:
        r=results[results.method==key].copy();entry=dict(id=key,name=names[key],queries=len(r),groups=int(r.similarity_group.nunique()),
            geometry_cost=float(r.groupby('similarity_group').geometry_cost.mean().mean()))
        for baseline,table in baselines.items():
            r['improvement']=[table.loc[q].geometry_cost-c for q,c in zip(r['query'],r.geometry_cost)]
            entry['vs_'+baseline]=group_ci(r,'improvement')
        summary.append(entry)
    a=results[results.method=='cajal_qgw'].set_index('query');b=results[results.method=='cajal_gw_unquantized'].set_index('query')
    ids=np.triu_indices(len(frame),1);mask=np.zeros((len(frame),len(frame)),bool)
    for i,pool in pools.items():mask[i,pool]=True;mask[pool,i]=True
    valid=mask[ids] & np.isfinite(matrices['cajal_gw_unquantized'][ids])
    from scipy.stats import spearmanr
    approximation=dict(top1_agreement=float((a.neighbor==b.neighbor).mean()),
        candidate_distance_spearman=float(spearmanr(matrices['cajal_qgw'][ids][valid],matrices['cajal_gw_unquantized'][ids][valid]).statistic),
        note='The unquantized non-convex solver is a reference, not a globally certified optimum. Different local optima may contribute to disagreement.')
    (OUT/'evaluation.json').write_text(json.dumps(dict(scope='Exploratory validation-only; same matched candidate pools and geometry proxy as the existing Atlas comparison. No new held-out claim or biological ground truth.',
        methods=summary,approximation=approximation,queries=len(pools),groups=11,candidate_pairs=int(valid.sum())),indent=2))
    print(json.dumps(approximation),flush=True)


if __name__=='__main__':main()
