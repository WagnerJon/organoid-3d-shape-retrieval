"""Development-only evaluation of Atlas fallback and feature comparisons.

The prior held-out spatial benchmark remains frozen. These are exploratory
comparisons on validation cells, not new evidence from an untouched test set.
"""
import json
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.distance import cdist
from pointnet_shape_core import ROOT, sample_surface
from spatial_benchmark_core import mesh_for, morphology, CONTROLS, geometry_cost, surface_frame
from atlas_shape_space import signature, distances, pack
from evaluate_sdf_shape import group_ci

OUT=ROOT/'results/atlas_spatial'


def main():
    data=json.loads((ROOT/'results/explorer_dataset/dataset.json').read_text());lookup={c['id']:c for c in data['cells']}
    frame=pd.read_csv(ROOT/'results/spatial_benchmark/dev_manifest.csv');train=frame[frame.split=='train'];frame=frame[frame.split=='validation'].reset_index(drop=True)
    controls=morphology(train,CONTROLS);x=(morphology(frame,CONTROLS)-controls.mean(0))/controls.std(0).clip(1e-6)
    matching=cdist(x,x);groups=frame.similarity_group.to_numpy();clouds=[]
    for row in frame.to_dict('records'):
        v,f=mesh_for(row);clouds.append(sample_surface(v,f,384,np.random.default_rng(20261030+int(row['original_index']))))
    methods=[m for m in data['methods'] if m.get('family') is not None]
    records=[];costs={}
    for m in methods:
        vectors=[lookup[c]['representations'][m['id']]['vector'] for c in frame.cell_id];d=distances(vectors,m['metric'])
        for i in range(len(frame)):
            ids=np.flatnonzero((groups!=groups[i])&(matching[i]<=1))
            if len(ids)<5:continue
            ids=ids[np.argsort(matching[i,ids])[:20]];j=int(ids[np.argmin(d[i,ids])]);pair=tuple(sorted([i,j]))
            if pair not in costs:costs[pair]=geometry_cost(clouds[pair[0]],clouds[pair[1]])
            records.append(dict(method=m['id'],query=i,cell_id=frame.iloc[i].cell_id,neighbor=frame.iloc[j].cell_id,
                                similarity_group=int(groups[i]),geometry_cost=costs[pair]))
        print('Evaluated',m['name'],flush=True)
    records=pd.DataFrame(records);records.to_csv(OUT/'development_neighbors.csv',index=False)
    baseline=records[records.method=='morph_all'].set_index('query');summary=[]
    for m in methods:
        subset=records[records.method==m['id']].copy();subset['improvement']=[baseline.loc[q].geometry_cost-c for q,c in zip(subset['query'],subset.geometry_cost)]
        summary.append(dict(method=m['id'],name=m['name'],queries=len(subset),mean_cost=float(subset.groupby('similarity_group').geometry_cost.mean().mean()),
                            vs_all9=group_ci(subset,'improvement')))
    # Reproduce the exact same engineered nearly-symmetric family as a diagnostic,
    # now using only the independent fallback signature. No fitting on these shapes.
    model=np.load(OUT/'signature_encoder.npz');mesh=trimesh.creation.icosphere(subdivisions=4);u=mesh.vertices.copy();vlist=[]
    for amp in [.02,.03,.04,.05,.06,.07,.08,.10]:
        v=u*(1+amp*((u**4).sum(1)-.6))[:,None];v/=np.linalg.norm(v,axis=1).max();vlist.append(v)
    encode=lambda v: (signature(v,mesh.faces)-model['mean'])@model['components'].T*float(model['scale'])
    base=np.array([encode(v) for v in vlist]);rng=np.random.default_rng(20261031);correct=[];errors=[]
    for v_index,v in enumerate(vlist):
        for _ in range(8):
            q,_=np.linalg.qr(rng.normal(size=(3,3)))
            if np.linalg.det(q)<0:q[:,0]*=-1
            moved=encode(v@q);correct.append(int(np.argmin(np.linalg.norm(base-moved,axis=1))==v_index));errors.append(float(np.max(np.abs(moved-base[v_index]))))
    diagnostics=dict(engineered_identity=float(np.mean(correct)),rotations=len(correct),max_code_difference=max(errors),
        note='Same meshes under rigid transforms; not a remeshing test or estimate of natural-cell performance. Signature loses chirality and is not a complete invariant.')
    payload=dict(scope='Exploratory validation-only comparison; no new held-out claims. Candidate pools matched within 1 training SD of log volume, elongation and sphericity, other groups only. Geometry is source-segmentation surface agreement, not biological ground truth.',
        methods=summary,symmetry=diagnostics,ambiguous_validation_cells=int(sum(surface_frame(*mesh_for(r))[3]<.05 for r in frame.to_dict('records'))))
    (OUT/'evaluation.json').write_text(json.dumps(payload,indent=2))
    lines=['# Atlas shape-space comparison','',payload['scope'],'',
           '| Comparison | Mean geometry cost ↓ | Improvement vs all 9, group bootstrap 95% CI |',
           '|---|---:|---|']
    for r in summary:
        ci=r['vs_all9'];lines.append(f"| {r['name']} | {r['mean_cost']:.6f} | {ci} |")
    lines+=['',f"Ambiguous validation cells: {payload['ambiguous_validation_cells']} / 48.",'',
            f"Engineered near-symmetry rotation identity: {diagnostics['engineered_identity']:.1%} over 64 rotations. Maximum code change {diagnostics['max_code_difference']:.3g}.",'',
            'The fallback is a separate deterministic geometric signature, not an improved neural encoder. Full spatial comparison switches to this signature whenever either cell has ambiguous axes. Projection is MDS of the same hybrid distances used for neighbors; the hybrid need not be Euclidean or satisfy the triangle inequality. Positive-inertia coverage is shown, not a PCA variance claim. Traditional and signature spaces use PCA and k-means.','',
            'The adjusted spatial summary is the benchmark’s pooled spatial code, with no ambiguity fallback; its limitation is shown in Atlas. Both adjusted shape and traditional views use identical training-only degree-2 Ridge nuisance adjustment. Organoid plots average included QC cell features; the hybrid is cell-only.','',
            'A surface signature can discard spatial arrangements and cannot distinguish all shapes or chirality. Rigid-rotation invariance assumes the same triangulation. Nearly spherical fallback selection may jump at the fixed axis-gap threshold. New phenotype/condition labels are still needed to determine biological meaning.','',
            'Open the Compare Methods tab. Choose Shape and traditional, After size / shape adjustment, or All methods. Change the reference; reference colors and linked selections let you inspect the same cells in each space. Use Other stacks only for neighbors. The global Split filter can restrict exploration to validation or test, but display PCA/group fits remain descriptive included-cohort fits.','']
    (OUT/'report.md').write_text('\n'.join(lines));print(json.dumps(diagnostics),flush=True)


if __name__=='__main__':main()
