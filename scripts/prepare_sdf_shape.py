"""Prepare a group-disjoint, geometry-only SDF pilot without modifying Atlas."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import time
import numpy as np
import pandas as pd
import trimesh
from pointnet_shape_core import ROOT, sample_surface, hierarchy, sha256, save_json
from prepare_pointnet_shape import read_mesh

OUT=ROOT/'results/sdf_shape_pilot'
FEATURES=['sphericity','elongation','flatness','mesh_solidity']


def select_balanced(frame,count):
    """Morphology diversity within each group; balanced round-robin across groups."""
    groups=[]
    values=frame[FEATURES]; mean=values.mean(); std=values.std().clip(lower=1e-6)
    for _,group in frame.groupby('similarity_group',sort=True):
        x=((group[FEATURES]-mean)/std).to_numpy(); d=np.full(len(x),np.inf); order=[]
        at=int(np.square(x).sum(1).argmax())
        for i in range(len(x)):
            order.append(group.index[at]); d=np.minimum(d,np.square(x-x[at]).sum(1)); d[[list(group.index).index(j) for j in order]]=-1
            at=int(d.argmax())
        groups.append(order)
    chosen=[]
    for i in range(max(map(len,groups))):
        for group in groups:
            if i<len(group): chosen.append(group[i])
            if len(chosen)==count:return chosen
    return chosen


def prepare_one(record):
    index,row=record
    destination=OUT/'data'/f'{index:03d}.npz'
    mesh_path=ROOT/'results/explorer_dataset'/row['sample']/f"{row['label']}.mesh"
    if sha256(mesh_path)!=row['source_mesh_sha256']:raise ValueError('Source mesh changed')
    vertices,faces=read_mesh(mesh_path)
    center=np.array([row['normalization_center_'+a+'_um'] for a in 'xyz']); radius=row['normalization_radius_um']
    mesh=trimesh.Trimesh((vertices-center)/radius,faces,process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent:raise ValueError('Non-closed mesh '+row['cell_id'])
    rng=np.random.default_rng(20261018+index)
    views=np.stack([sample_surface(mesh.vertices,mesh.faces,512,rng) for _ in range(3)])
    groups={k:np.stack([hierarchy(p,[128,32],16)[k] for p in views]) for k in ['centers1','neighbors1','centers2','neighbors2']}
    # Two near-surface bands, uniform volume, and exact surface constraints.
    near=sample_surface(mesh.vertices,mesh.faces,1536,rng)
    noise=rng.normal(size=near.shape)*np.repeat([.025,.10],768)[:,None]
    volume=rng.uniform(-1.15,1.15,(1024,3))
    surface=sample_surface(mesh.vertices,mesh.faces,512,rng)
    query=np.concatenate([near+noise,volume,surface]).astype(np.float32)
    # trimesh uses positive-inside; this experiment uses negative-inside.
    distances=-trimesh.proximity.signed_distance(mesh,query[:2560])
    target=np.concatenate([distances,np.zeros(512)]).astype(np.float32)
    if not np.isfinite(target).all():raise ValueError('Non-finite SDF')
    np.savez_compressed(destination,points=views,query=query,target=target,**groups)
    return dict(index=index,cell_id=row['cell_id'],split=row['split'],seconds=0,watertight=True,
                file_sha256=sha256(destination),source_mesh_sha256=sha256(mesh_path))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    if (OUT/'manifest.csv').exists():raise FileExistsError('Pilot already prepared')
    frame=pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv')
    chosen=[]
    for split,count in [('train',96),('validation',48)]:
        eligible=frame[(frame.split==split)&frame.training_eligible]
        chosen+=select_balanced(eligible,count)
    manifest=frame.loc[chosen].reset_index(names='original_index');manifest['pilot_index']=np.arange(len(manifest))
    # Fixed twelve distinct training groups, selected independently of learned outputs.
    overfit=select_balanced(manifest[manifest.split=='train'],12)
    manifest['overfit']=manifest.index.isin(overfit)
    manifest.to_csv(OUT/'preparing_manifest.csv',index=False)
    started=time.monotonic();audit=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(prepare_one,manifest.to_dict('index').items()):
            audit.append(result)
            print(f"PREPARED {len(audit)}/{len(manifest)} {result['cell_id']} {time.monotonic()-started:.1f}s",flush=True)
    manifest.to_csv(OUT/'manifest.csv',index=False)
    save_json(OUT/'data_audit.json',dict(records=audit,source_manifest_sha256=sha256(ROOT/'results/pointnet_shape/data/manifest.csv'),
        manifest_sha256=sha256(OUT/'manifest.csv'),atlas_sha256=sha256(ROOT/'results/explorer_dataset/dataset.json'),
        dataset='96 training cells / 48 validation cells, balanced across the existing similarity groups; no test cells selected.',
        target='Signed Euclidean distance to ORIGINAL closed triangle meshes, negative inside; inferred micrometer calibration inherited.',
        scope='Paper-inspired reduced-compute pilot; 512 input points, 128 vector channels, k=12; not a published-model reproduction.',
        atlas_modified=False))


if __name__=='__main__':main()
