"""Geometry-first spatial SDF encoder pilot, independent of Atlas exports.

Dense 32^3 mesh SDF -> spatial CNN -> 128 scalars -> coordinate SDF MLP.
No encoder skip connection, fitted validation code, or rotation-invariance claim.
"""
import argparse
import json
import time
import shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
from torch import nn
import trimesh
import igl
from pointnet_shape_core import ROOT,sha256,save_json,select_device
from prepare_pointnet_shape import read_mesh
from train_sdf_shape import load_pilot,batch
from sdf_shape_core import sdf_loss

OUT=ROOT/'results/spatial_sdf';EXTENT=1.15


class SpatialSDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial=nn.Sequential(nn.Conv3d(1,16,3,2,1),nn.SiLU(),nn.Conv3d(16,32,3,2,1),nn.SiLU(),
            nn.Conv3d(32,64,3,2,1),nn.SiLU(),nn.Flatten(),nn.Linear(4096,256),nn.SiLU(),nn.Linear(256,128))
        self.first=nn.Sequential(nn.Linear(167,256),nn.Softplus(beta=10),nn.Linear(256,256),nn.Softplus(beta=10))
        self.last=nn.Sequential(nn.Linear(423,256),nn.Softplus(beta=10),nn.Linear(256,128),nn.Softplus(beta=10),nn.Linear(128,1))
        self.register_buffer('frequencies',torch.tensor([1.,2.,4.,8.,16.,32.])*np.pi)

    def encode(self,grid):return self.spatial(grid.clamp(-.15,.15)/.15)

    def decode(self,z,q):
        phase=q[...,None]*self.frequencies
        positional=torch.cat([q,phase.sin().flatten(-2),phase.cos().flatten(-2)],-1)
        x=torch.cat([z[:,None].expand(-1,q.shape[1],-1),positional],-1)
        return self.last(torch.cat([self.first(x),x],-1)).squeeze(-1)


def make_grid(item):
    index,row=item;path=ROOT/'results/explorer_dataset'/row['sample']/f"{row['label']}.mesh"
    if sha256(path)!=row['source_mesh_sha256']:raise ValueError('Source mesh changed')
    destination=OUT/'grids'/f'{index:03d}.npy'
    if destination.exists():
        try:
            cached=np.load(destination)
            if cached.shape==(32,32,32) and np.isfinite(cached).all():
                metadata=destination.with_suffix('.json')
                backend=json.loads(metadata.read_text())['backend'] if metadata.exists() else 'trimesh'
                return dict(index=index,sha256=sha256(destination),backend=backend)
        except (ValueError,EOFError):pass
    vertices,faces=read_mesh(path);center=np.array([row['normalization_center_'+a+'_um'] for a in 'xyz'])
    mesh=trimesh.Trimesh((vertices-center)/row['normalization_radius_um'],faces,process=False)
    if not mesh.is_watertight:raise ValueError('Open input mesh')
    axis=np.linspace(-EXTENT,EXTENT,32,dtype=np.float32)
    q=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    sdf=igl.signed_distance(q.astype(np.float64),mesh.vertices.astype(np.float64),mesh.faces.astype(np.int64),
        igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0].clip(-.15,.15).reshape(32,32,32).astype(np.float32)
    temporary=destination.with_suffix('.tmp.npy');np.save(temporary,sdf);temporary.replace(destination)
    save_json(destination.with_suffix('.json'),dict(backend='libigl-2.6.3-pseudonormal'))
    return dict(index=index,sha256=sha256(destination),backend='libigl-2.6.3-pseudonormal')


def prepare():
    manifest,_=load_pilot();OUT.mkdir(exist_ok=True);(OUT/'grids').mkdir(exist_ok=True)
    if (OUT/'data.json').exists():raise FileExistsError('Prepared dataset exists')
    records=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(make_grid,manifest.to_dict('index').items()):
            records.append(result);print(f"GRID {len(records)}/{len(manifest)} {time.monotonic()-started:.1f}s",flush=True)
    save_json(OUT/'data.json',dict(records=records,manifest_sha256=sha256(ROOT/'results/sdf_shape_pilot/manifest.csv'),
        atlas_sha256=sha256(ROOT/'results/explorer_dataset/dataset.json'),resolution=32,extent=EXTENT,
        source='Original mesh signed distances clipped to +/-.15; negative inside; original normalization. Validated libigl pseudonormal plus retained trimesh cache; backend recorded per cell.',
        representation_change='Dense mesh SDF input carries more information than a sampled point cloud. Not an encoder-only ablation.',
        rotation='Laboratory orientation retained; no rotation invariance claimed.'))


def load(overfit_only=False):
    manifest,arrays=load_pilot()
    if overfit_only and not (OUT/'data.json').exists():
        # Allows the training-only capacity check while other cells are prepared.
        # Unneeded entries cannot be used: train and evaluation IDs are both the
        # audited overfit subset in this mode.
        needed=manifest.index[manifest.overfit].to_numpy()
        assert (manifest.iloc[needed].split=='train').all()
        grids=np.zeros((len(manifest),1,32,32,32),dtype=np.float32);records=[]
        for index in needed:
            path=OUT/'grids'/f'{index:03d}.npy';grids[index,0]=np.load(path)
            records.append(dict(index=int(index),sha256=sha256(path)))
        save_json(OUT/'overfit_input_audit.json',dict(records=records,manifest_sha256=sha256(ROOT/'results/sdf_shape_pilot/manifest.csv')))
        arrays['grids']=grids;return manifest,arrays
    meta=json.loads((OUT/'data.json').read_text())
    assert meta['manifest_sha256']==sha256(ROOT/'results/sdf_shape_pilot/manifest.csv')
    grids=[]
    for record in meta['records']:
        path=OUT/'grids'/f"{record['index']:03d}.npy"
        assert sha256(path)==record['sha256'];grids.append(np.load(path))
    arrays['grids']=np.stack(grids)[:,None]
    return manifest,arrays


def train(mode,device_name,epochs):
    torch.set_num_threads(4);torch.manual_seed(20261026);rng=np.random.default_rng(20261026)
    device=select_device(device_name);manifest,arrays=load(overfit_only=mode=='overfit');folder=OUT/mode;folder.mkdir(exist_ok=True)
    if (folder/'best.pt').exists():raise FileExistsError('Checkpoint exists')
    train_ids=manifest.index[manifest.overfit if mode=='overfit' else manifest.split=='train'].to_numpy()
    valid_ids=train_ids if mode=='overfit' else manifest.index[manifest.split=='validation'].to_numpy()
    model=SpatialSDF().to(device)
    if mode=='pilot':
        initial=torch.load(OUT/'overfit/best.pt',map_location='cpu',weights_only=False)
        assert initial['manifest_sha256']==sha256(ROOT/'results/sdf_shape_pilot/manifest.csv')
        model.load_state_dict(initial['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.0001)
    best=float('inf');history=[];started=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();losses=[];order=rng.permutation(train_ids)
        for first in range(0,len(order),8):
            ids=order[first:first+8];grid=torch.tensor(arrays['grids'][ids],device=device)
            _,q,t,_=batch(arrays,ids,device,rng);z=model.encode(grid);prediction=model.decode(z,q)
            loss=sdf_loss(prediction,t)+.00001*z.square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch%10==0 or epoch==1 or epoch==epochs:
            model.eval();errors=[]
            with torch.no_grad():
                for first in range(0,len(valid_ids),8):
                    ids=valid_ids[first:first+8];_,q,t,_=batch(arrays,ids,device)
                    z=model.encode(torch.tensor(arrays['grids'][ids],device=device))
                    errors.append(float(sdf_loss(model.decode(z,q),t).cpu()))
            error=float(np.mean(errors));row=dict(epoch=epoch,train_sdf_l1=float(np.mean(losses)),validation_sdf_l1=error,seconds=time.monotonic()-started)
            history.append(row)
            if error<best:
                best=error;torch.save(dict(model=model.state_dict(),epoch=epoch,manifest_sha256=sha256(ROOT/'results/sdf_shape_pilot/manifest.csv')),folder/'best.pt')
            pd.DataFrame(history).to_csv(folder/'history.csv',index=False);print(json.dumps(dict(mode=mode,**row)),flush=True)
    save_json(folder/'status.json',dict(status='complete',epochs=epochs,training_cells=len(train_ids),evaluation_cells=len(valid_ids),
        device=str(device),seconds=time.monotonic()-started,best_sdf_l1=best,test_used=False,
        per_validation_cell_fitting=False,embedding_dimensions=128,rotation_invariant=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','overfit','pilot'],required=True)
    parser.add_argument('--device',default='auto');parser.add_argument('--epochs',type=int,default=200);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:
        snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
        destination=snapshot/f'spatial_sdf_{args.stage}.py'
        if not destination.exists():shutil.copy2(__file__,destination)
        train(args.stage,args.device,args.epochs)
