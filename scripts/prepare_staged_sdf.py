"""Prepare three 2048-point views and cached k=20 geometry graphs. No SDF changes."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pointnet_shape_core import ROOT,sample_surface,sha256,save_json
from prepare_pointnet_shape import read_mesh
from train_sdf_shape import load_pilot

OUT=ROOT/'results/staged_sdf'


def prepare():
    manifest,_=load_pilot();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'data.json').exists():raise FileExistsError('Staged data already exists')
    points=[];graphs=[]
    for index,row in manifest.iterrows():
        path=ROOT/'results/explorer_dataset'/row['sample']/f'{row.label}.mesh'
        if sha256(path)!=row.source_mesh_sha256:raise ValueError('Mesh changed')
        v,f=read_mesh(path);center=np.array([row['normalization_center_'+a+'_um'] for a in 'xyz'])
        v=(v-center)/row.normalization_radius_um;rng=np.random.default_rng(20261021+index)
        views=np.stack([sample_surface(v,f,2048,rng) for _ in range(3)])
        neighbors=np.stack([cKDTree(p).query(p,k=20)[1] for p in views]).astype(np.int32)
        points.append(views);graphs.append(neighbors)
        if (index+1)%24==0:print(f'PREPARED {index+1}/{len(manifest)} larger point views',flush=True)
    np.save(OUT/'points.npy',np.asarray(points,dtype=np.float32));np.save(OUT/'neighbors.npy',np.asarray(graphs))
    manifest.to_csv(OUT/'manifest.csv',index=False)
    save_json(OUT/'data.json',dict(points=2048,neighbors=20,views=3,manifest_sha256=sha256(OUT/'manifest.csv'),
        arrays={name:sha256(OUT/(name+'.npy')) for name in ['points','neighbors']},
        source_manifest_sha256=sha256(ROOT/'results/sdf_shape_pilot/manifest.csv'),
        atlas_sha256=sha256(ROOT/'results/explorer_dataset/dataset.json'),
        split='Same 96 training / 48 validation cells and disjoint similarity groups as prior pilot. No test cells.',
        scope='Reference-style encoder, independently implemented. Point budget is 2048, not the published 8192.'))


def load():
    import json
    meta=json.loads((OUT/'data.json').read_text())
    if sha256(OUT/'manifest.csv')!=meta['manifest_sha256']:raise ValueError('Manifest changed')
    for name,digest in meta['arrays'].items():
        if sha256(OUT/(name+'.npy'))!=digest:raise ValueError('Point view changed')
    manifest,arrays=load_pilot()
    if sha256(ROOT/'results/sdf_shape_pilot/manifest.csv')!=meta['source_manifest_sha256']:raise ValueError('Source manifest changed')
    arrays.update(large_points=np.load(OUT/'points.npy',mmap_mode='r'),neighbors=np.load(OUT/'neighbors.npy',mmap_mode='r'))
    return manifest,arrays


if __name__=='__main__':prepare()
