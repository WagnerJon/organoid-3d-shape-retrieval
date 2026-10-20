"""Publish frozen spatial shape comparisons, preserving original Atlas records."""
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from pointnet_shape_core import ROOT, sha256
from spatial_benchmark_core import mesh_for, surface_frame, orbit_grids, summary_features, morphology, CLASSICAL, CONTROLS
from atlas_shape_space import signature, pack, GAP
from evaluate_spatial_benchmark import residual_fit, residual_apply
from organoid_membership import rebuild

OUT = ROOT/'results/atlas_spatial'
ATLAS = ROOT/'results/explorer_dataset'


def encode_one(args):
    row, mean, components = args
    destination=OUT/'cache'/f"{row['cell_id'].replace(':','_')}.npz"
    if destination.exists():
        with np.load(destination) as saved:
            if str(saved['mesh_hash'])==row['source_mesh_sha256'] and str(saved['pca_hash'])==sha256(ROOT/'results/spatial_benchmark/aligned_pca.npz'):
                return row['cell_id']
    v,f=mesh_for(row);frame=surface_frame(v,f)
    z=(orbit_grids(v,f,frame).reshape(4,-1)-mean)@components.T
    s=signature(v,f)
    np.savez_compressed(destination,orbit=z,signature=s,gap=frame[3],mesh_hash=row['source_mesh_sha256'],
                        pca_hash=sha256(ROOT/'results/spatial_benchmark/aligned_pca.npz'))
    return row['cell_id']


def main():
    OUT.mkdir(exist_ok=True);(OUT/'cache').mkdir(exist_ok=True)
    path=ATLAS/'dataset.json';before=sha256(path);data=json.loads(path.read_text())
    manifest=pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv')
    if set(manifest.cell_id)!=set(c['id'] for c in data['cells']):raise ValueError('Cell identities changed')
    # Frozen checkpoint is never refit on the display cohort.
    freeze=json.loads((ROOT/'results/spatial_benchmark/freeze.json').read_text())
    weights=ROOT/'results/spatial_benchmark/aligned_pca.npz'
    if sha256(weights)!=freeze['artifacts']['aligned_pca.npz']:raise ValueError('Frozen PCA checkpoint changed')
    pca=np.load(weights)
    # Validate every source, even when a cached representation exists.
    lookup={c['id']:c for c in data['cells']}
    for row in manifest.to_dict('records'):
        c=lookup[row['cell_id']]
        if sha256(ATLAS/c['sample']/f"{c['label']}.mesh")!=row['source_mesh_sha256']:raise ValueError('Stale original geometry')
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,cid in enumerate(pool.map(encode_one,((r,pca['mean'],pca['components']) for r in manifest.to_dict('records')))):
            if i%25==0:print(f'Encoding shapes {i+1}/{len(manifest)} · {cid}',flush=True)
    loaded=[]
    for cid in manifest.cell_id:
        with np.load(OUT/'cache'/f"{cid.replace(':','_')}.npz") as a:
            loaded.append({k:a[k].copy() for k in ['orbit','signature','gap']})
    z=np.stack([a['orbit'] for a in loaded]);raw=np.stack([a['signature'] for a in loaded]);gaps=np.array([float(a['gap']) for a in loaded])
    dev=pd.read_csv(ROOT/'results/spatial_benchmark/dev_manifest.csv');train_ids=set(dev[dev.split=='train'].cell_id)
    train=np.flatnonzero(manifest.cell_id.isin(train_ids));frame=manifest.iloc[train]
    if len(train)!=96:raise ValueError('Expected original 96 development training cells')
    # This descriptor fit is new and exploratory; test cohort was previously opened.
    sig_pca=PCA(n_components=95,svd_solver='full').fit(raw[train]);sig=sig_pca.transform(raw)
    orbit_pairs=[]
    for i in range(4):
        for j in range(4):orbit_pairs.append(np.square(z[train,i][:,None,:]-z[train,j][None,:,:]).sum(-1))
    d=np.sqrt(np.minimum.reduce(orbit_pairs));upper=np.triu_indices(len(train),1)
    scale=float(np.median(d[upper])/max(np.median(pdist(sig[train])),1e-12));sig*=scale
    np.savez_compressed(OUT/'signature_encoder.npz',mean=sig_pca.mean_,components=sig_pca.components_,scale=scale)
    specs=[]
    def add(key,name,vectors,description,family='shape',metric='euclidean',projection='pca'):
        specs.append(dict(id=key,name=name,metric=metric,dimensions=vectors.shape[1],variance=[0.,0.],stackVariance=[0.,0.],
                          family=family,description=description,projection=projection))
        for cell,vector in zip((lookup[cid] for cid in manifest.cell_id),vectors):
            cell['representations'][key]=dict(vector=np.asarray(vector,float).tolist(),point=[0.,0.],cluster=0)
    add('spatial_robust','Spatial PCA · ambiguity fallback',pack(gaps,z,sig),
        'Frozen 95D spatial PCA in four proper sign frames. If either cell has an axis gap < 0.05, distance uses an independent rotation-invariant surface signature, scaled on training cells. MDS depicts these mixed distances; average-linkage groups. The fallback discards chirality and cannot distinguish all shapes. Cell-level only.',metric='shape_hybrid95',projection='mds')
    add('surface_signature','Surface signature · 95D',sig,
        'Geometric descriptor, not a neural encoder: surface radial and pair-distance quantiles, compressed by PCA fitted on the original 96 training cells. Rotation invariant for rigid transforms of the same mesh; sensitive to remeshing and discards chirality.')
    fit=residual_fit(frame,sig[train]);res=residual_apply(manifest,sig,fit)
    add('surface_signature_adjusted','Surface signature · adjusted',res,
        'Surface signature after degree-2 Ridge adjustment for log volume, elongation and sphericity fitted only on the original 96 training cells. Residualization does not guarantee biological independence.')
    np.savez(OUT/'signature_nuisance.npz',**fit)
    pooled=summary_features(z);fit=residual_fit(frame,pooled[train]);residual=residual_apply(manifest,pooled,fit)
    add('spatial_adjusted','Spatial PCA · adjusted summary',residual,
        'Mean/std over the four spatial PCA frames, adjusted for log volume, elongation and sphericity using training-only degree-2 Ridge. This pooled view has no ambiguity fallback and can remain alignment-sensitive in near-spherical cells.')
    np.savez(OUT/'spatial_nuisance.npz',**fit)
    groups=[('morph_controls','Volume + elongation + sphericity',CONTROLS),
            ('morph_size','Size features',['volume_um3','surface_area_um2','major_axis_um','middle_axis_um','minor_axis_um']),
            ('morph_shape','Shape features',['sphericity','elongation','flatness','mesh_solidity']),
            ('morph_all','All 9 traditional features',CLASSICAL)]
    for key,name,cols in groups:
        values=morphology(manifest,cols);mean=values[train].mean(0);std=values[train].std(0).clip(1e-6);values=(values-mean)/std
        add(key,name,values,'Training-standardized features: '+', '.join(cols)+'. Positive size measures are log transformed.','traditional')
        if key=='morph_all':
            fit=residual_fit(frame,values[train]);adjusted=residual_apply(manifest,values,fit)
            add('morph_adjusted','All 9 traditional · adjusted',adjusted,
                'Identical training-only degree-2 Ridge adjustment for log volume, elongation and sphericity as the shape comparisons. Residual features need not be independent of these measurements.','traditional')
    new_ids={s['id'] for s in specs};data['methods']=[m for m in data['methods'] if m['id'] not in new_ids]+specs
    data['organoids']=data.get('membershipStackTemplates',data['organoids'])
    data['shapeComparison']=dict(version=1,checkpoint_sha256=sha256(weights),training_cells=96,axis_gap_threshold=GAP,
        ambiguous_cells=int((gaps<GAP).sum()),scope='Exploratory Atlas export. Existing held-out benchmark is unchanged; new fallback evaluation is development-only. Organoid spaces average included QC cell descriptors, not whole-organoid geometry.',
        note='Separation and cluster agreement do not establish biological validity. No independent condition labels are available.')
    rebuild(data)
    if sha256(path)!=before:raise ValueError('Atlas changed during export; publication cancelled')
    backup=OUT/f'dataset.before-{before[:12]}.json'
    if not backup.exists():shutil.copy2(path,backup)
    tmp=path.with_suffix('.spatial.tmp');tmp.write_text(json.dumps(data,allow_nan=False,separators=(',',':')));os.replace(tmp,path)
    np.savez_compressed(OUT/'comparison_vectors.npz',ids=manifest.cell_id.to_numpy(dtype=str),orbits=z,signatures=sig,gaps=gaps)
    (OUT/'export.json').write_text(json.dumps(dict(status='complete',before_sha256=before,after_sha256=sha256(path),methods=specs,
        training_ids=sorted(train_ids),checkpoint_sha256=sha256(weights),ambiguous_cells=int((gaps<GAP).sum()),backup=str(backup)),indent=2))
    print(f'Published {len(specs)} comparison spaces for {len(manifest)} cells; {int((gaps<GAP).sum())} ambiguous axes.',flush=True)


if __name__=='__main__':main()
