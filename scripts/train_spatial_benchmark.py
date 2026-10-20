"""Fit aligned encoders on development training cells only."""
import json
import shutil
import time
import numpy as np
import pandas as pd
import torch
import igl
from sklearn.decomposition import PCA
from pointnet_shape_core import ROOT,sha256,save_json,select_device
from spatial_sdf import SpatialSDF
from prepare_spatial_benchmark import OUT
from spatial_benchmark_core import SIGNS,mesh_for
from train_sdf_shape import load_pilot,batch
from sdf_shape_core import sdf_loss


def train():
    torch.set_num_threads(4);torch.manual_seed(20261028);rng=np.random.default_rng(20261028)
    device=select_device('mps');manifest=pd.read_csv(OUT/'dev_manifest.csv');_,arrays=load_pilot()
    grids=np.load(OUT/'dev_grids.npy',mmap_mode='r');frames=np.load(OUT/'dev_frames.npz')
    train=manifest.index[manifest.split=='train'].to_numpy();valid=manifest.index[manifest.split=='validation'].to_numpy()
    if (OUT/'aligned_cnn.pt').exists():raise FileExistsError('Aligned run exists')
    pca=PCA(n_components=95,svd_solver='randomized',random_state=28).fit(np.array(grids[train]).reshape(-1,32768))
    np.savez_compressed(OUT/'aligned_pca.npz',mean=pca.mean_,components=pca.components_)
    model=SpatialSDF().to(device)
    model.load_state_dict(torch.load(ROOT/'results/spatial_sdf/boundary/best.pt',map_location='cpu',weights_only=False)['model'])
    qb=rng.uniform(-1.15,1.15,(384,3)).astype(np.float32)
    for face in range(6):qb[face*64:(face+1)*64,face//2]=1.15*(1 if face%2 else -1)
    boundary=[]
    for index,row in manifest.iterrows():
        v,f=mesh_for(row);boundary.append([igl.signed_distance((qb@(frames['axes'][index]*s).T+frames['centers'][index]).astype(float),v,f,
            igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0].clip(-.15,.15) for s in SIGNS])
    boundary=np.array(boundary,dtype=np.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,120,eta_min=.00003)
    history=[];best=float('inf');started=time.monotonic()
    for epoch in range(1,121):
        model.train();losses=[];order=rng.permutation(train)
        for first in range(0,len(order),8):
            ids=order[first:first+8];view=int(rng.integers(4));_,q,t,_=batch(arrays,ids,device,rng)
            center=torch.tensor(frames['centers'][ids],device=device,dtype=torch.float32)
            axes=torch.tensor(frames['axes'][ids]*SIGNS[view],device=device,dtype=torch.float32)
            q=(q-center[:,None])@axes;chosen=rng.choice(384,192,replace=False)
            q=torch.cat([q,torch.tensor(qb[chosen],device=device)[None].expand(len(ids),-1,-1)],1)
            t=torch.cat([t,torch.tensor(boundary[ids,view][:,chosen],device=device)],1)
            z=model.encode(torch.tensor(np.array(grids[ids,view,None]),device=device))
            loss=sdf_loss(model.decode(z,q),t)+1e-5*z.square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch%10==0 or epoch==1:
            model.eval();errors=[];outside=[]
            with torch.no_grad():
                for first in range(0,len(valid),8):
                    ids=valid[first:first+8];_,q,t,_=batch(arrays,ids,device)
                    for view in range(4):
                        axes=torch.tensor(frames['axes'][ids]*SIGNS[view],device=device,dtype=torch.float32)
                        center=torch.tensor(frames['centers'][ids],device=device,dtype=torch.float32)
                        z=model.encode(torch.tensor(np.array(grids[ids,view,None]),device=device))
                        errors.append(float(sdf_loss(model.decode(z,(q-center[:,None])@axes),t).cpu()))
                        outside.append(float(sdf_loss(model.decode(z,torch.tensor(qb,device=device)[None].expand(len(ids),-1,-1)),torch.tensor(boundary[ids,view],device=device)).cpu()))
            selection=float(np.mean(errors)+.25*np.mean(outside))
            row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_sdf_l1=float(np.mean(errors)),selection=selection,seconds=time.monotonic()-started)
            history.append(row)
            if selection<best:
                best=selection;torch.save(dict(model=model.state_dict(),epoch=epoch,manifest_sha256=sha256(OUT/'dev_manifest.csv')),OUT/'aligned_cnn.pt')
            pd.DataFrame(history).to_csv(OUT/'training.csv',index=False);print(json.dumps(row),flush=True)
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['spatial_benchmark_core.py','prepare_spatial_benchmark.py','train_spatial_benchmark.py']:
        shutil.copy2(ROOT/'scripts'/name,snapshot/name)
    save_json(OUT/'training.json',dict(status='complete',epochs=120,seed=20261028,training_cells=len(train),validation_cells=len(valid),test_used=False,
        aligned_pca_sha256=sha256(OUT/'aligned_pca.npz'),aligned_cnn_sha256=sha256(OUT/'aligned_cnn.pt'),
        sources={name:sha256(snapshot/name) for name in ['spatial_benchmark_core.py','prepare_spatial_benchmark.py','train_spatial_benchmark.py']}))


if __name__=='__main__':train()
