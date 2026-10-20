"""Attach shell harmonics and official CAJAL qGW to the Atlas comparison view."""
import json,os,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from pointnet_shape_core import ROOT,sha256
from evaluate_spatial_benchmark import residual_fit,residual_apply
from organoid_membership import rebuild

OUT=ROOT/'results/harmonics_cajal';ATLAS=ROOT/'results/explorer_dataset'


def main():
    path=ATLAS/'dataset.json';before=sha256(path);data=json.loads(path.read_text());old={c['id']:c for c in data['cells']}
    source=np.load(OUT/'surfaces.npz');frame=pd.read_csv(OUT/'manifest.csv');ids=source['ids']
    if list(ids)!=frame.cell_id.tolist() or set(ids)!=set(old):raise ValueError('Source identities differ')
    for row in frame.to_dict('records'):
        c=old[row['cell_id']]
        if sha256(ATLAS/c['sample']/f"{c['label']}.mesh")!=row['source_mesh_sha256']:raise ValueError('Stale surface encoding')
    protocol=json.loads((OUT/'cajal_protocol.json').read_text())
    if protocol['source_sha256']!=sha256(OUT/'surfaces.npz'):raise ValueError('Stale GW matrix')
    d=np.load(OUT/'cajal_qgw.npy');n=len(ids)
    if d.shape!=(n,n) or not np.isfinite(d).all() or d.min()<0 or not np.allclose(d,d.T,atol=1e-12) or np.max(np.abs(np.diag(d)))>1e-12:raise ValueError('Invalid CAJAL matrix')
    dev=pd.read_csv(ROOT/'results/spatial_benchmark/dev_manifest.csv');train_ids=set(dev[dev.split=='train'].cell_id);train=np.flatnonzero(frame.cell_id.isin(train_ids))
    if len(train)!=96:raise ValueError('Original training cohort changed')
    specs=[];vectors={}
    def add(key,name,values,description,metric='euclidean',projection='pca',family='shape',record_ids=None):
        vectors[key]=values
        specs.append(dict(id=key,name=name,dimensions=values.shape[1],metric=metric,projection=projection,family=family,description=description,variance=[0.,0.],stackVariance=[0.,0.]))
        for cid,v in zip(ids if record_ids is None else record_ids,values):old[cid]['representations'][key]=dict(vector=v.tolist(),point=[0.,0.],cluster=0)
    sh=source['harmonics']
    add('spherical_harmonics','Spherical harmonics · shell spectrum',sh,
        'Rotation-invariant spherical-shell harmonic powers: 32 radial shells, degrees 0–12, soft occupancy sampled from the original mesh. Exact surface centroid/RMS-radius normalization removes position and scale. Independent implementation of the shell-spectrum construction, not SPHARM-PDM. Finite quadrature produces small rotation errors; phase and chirality are lost.')
    fit=residual_fit(frame.iloc[train],sh[train]);adjusted=residual_apply(frame,sh,fit);np.savez(OUT/'harmonics_adjustment.npz',**fit)
    add('spherical_harmonics_adjusted','Spherical harmonics · adjusted',adjusted,
        'Shell harmonic spectrum after the same training-only degree-2 Ridge adjustment for log volume, elongation and sphericity used by traditional and spatial features. Adjustment is fitted on the original 96 training cells; residuals are not guaranteed independent of controls.')
    add('cajal_qgw','CAJAL · quantized GW',np.c_[np.arange(n),d],
        'Official CAJAL 2.0.3 quantized Gromov–Wasserstein comparison, 128 area-sampled surface points, 32 clusters, Euclidean distances within each cell. Position, scale and orientation (including reflections) removed. Cached pairwise distances drive neighbors and MDS; packed rows are not embedding coordinates. Average-linkage groups; cell-level only. Approximation, sampling and solver local minima affect rankings.',metric='precomputed',projection='mds')
    profile=d[:,train];fit=residual_fit(frame.iloc[train],profile[train]);adjusted=residual_apply(frame,profile,fit);np.savez(OUT/'cajal_profile_adjustment.npz',**fit,landmarks=ids[train])
    add('cajal_profile_adjusted','CAJAL · adjusted landmark profile',adjusted,
        'Distances to the original 96 training cells form a landmark profile, then receive the identical training-only volume/elongation/sphericity adjustment. Euclidean profile distance is an approximation and is not the direct CAJAL GW distance. Organoid plots average included QC cell profiles.')
    validation_meta=json.loads((OUT/'validation_gw.json').read_text());validation_ids=validation_meta['ids']
    validation_d=np.load(OUT/'validation_gw.npy')
    if validation_d.shape!=(48,48) or not np.isfinite(validation_d).all():raise ValueError('Complete unquantized validation matrix required')
    add('cajal_gw_validation','CAJAL · unquantized GW · validation',np.c_[np.arange(48),validation_d],
        'Official unquantized CAJAL solver on all 48 validation cells, with 128 sampled surface points. This is the reference for the faster full-dataset qGW approximation. MDS/average-linkage; numerical non-convex solver, not certified globally optimal. Validation benchmarks use only shared cells across all panels and neighbors. No new held-out claim.',
        metric='precomputed',projection='mds',family='validation',record_ids=validation_ids)
    for cid in ids:
        if cid not in set(validation_ids):old[cid]['representations'].pop('cajal_gw_validation',None)
    keys=set(vectors);data['methods']=[m for m in data['methods'] if m['id'] not in keys]+specs
    data['organoids']=data.get('membershipStackTemplates',data['organoids'])
    data['harmonicsCajal']=dict(version=1,protocol=json.loads((OUT/'protocol.json').read_text()),cajal=protocol,
        source_sha256=sha256(OUT/'surfaces.npz'),distance_sha256=sha256(OUT/'cajal_qgw.npy'),training_ids=sorted(train_ids),
        validation_ids=validation_ids,scope='Published-method comparisons; development-only evaluation. Source microscopy, masks, meshes and physical measurements preserved. No new held-out claims.')
    rebuild(data)
    if sha256(path)!=before:raise ValueError('Atlas changed while exporting; publication cancelled')
    backup=OUT/f'dataset.before-{before[:12]}.json'
    if not backup.exists():shutil.copy2(path,backup)
    tmp=path.with_suffix('.harmonics.tmp');tmp.write_text(json.dumps(data,allow_nan=False,separators=(',',':')));os.replace(tmp,path)
    np.savez_compressed(OUT/'atlas_vectors.npz',ids=ids,**vectors)
    (OUT/'export.json').write_text(json.dumps(dict(status='complete',before_sha256=before,after_sha256=sha256(path),backup=str(backup),methods=specs,
        training_ids=sorted(train_ids),sources={name:sha256(ROOT/'scripts'/name) for name in ['harmonics_cajal_core.py','prepare_harmonics_cajal.py','run_cajal_comparison.py','export_harmonics_cajal.py']}),indent=2))
    print(f'Published {len(specs)} methods, {n} cells, {n*(n-1)//2:,} cached pair distances',flush=True)


if __name__=='__main__':main()
