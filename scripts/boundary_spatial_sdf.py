"""Train-only exterior SDF supervision following the capacity-check box artifacts."""
import json
import time
import shutil
import numpy as np
import pandas as pd
import torch
import igl
from spatial_sdf import OUT,SpatialSDF,load,EXTENT
from pointnet_shape_core import ROOT,select_device,save_json,sha256
from prepare_pointnet_shape import read_mesh
from train_sdf_shape import batch
from sdf_shape_core import sdf_loss


def train(device_name='mps',epochs=60):
    torch.set_num_threads(4);torch.manual_seed(20261027);rng=np.random.default_rng(20261027)
    device=select_device(device_name);manifest,arrays=load();out=OUT/'boundary';out.mkdir(exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError('Existing boundary run')
    # Same spatial queries on the six box faces for every cell; exact per-mesh targets.
    queries=rng.uniform(-EXTENT,EXTENT,(384,3)).astype(np.float32)
    for face in range(6):queries[face*64:(face+1)*64,face//2]=EXTENT*(1 if face%2 else -1)
    targets=[]
    for _,row in manifest.iterrows():
        v,f=read_mesh(ROOT/'results/explorer_dataset'/row['sample']/f'{row.label}.mesh')
        center=np.array([row['normalization_center_'+a+'_um'] for a in 'xyz']);v=(v-center)/row.normalization_radius_um
        targets.append(igl.signed_distance(queries.astype(np.float64),v.astype(np.float64),f.astype(np.int64),igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0].clip(-.15,.15))
    targets=np.asarray(targets,dtype=np.float32);assert targets.min()>0
    np.savez_compressed(out/'boundary_targets.npz',queries=queries,targets=targets)
    ids=manifest.index[manifest.split=='train'].to_numpy();valid=manifest.index[manifest.split=='validation'].to_numpy()
    model=SpatialSDF().to(device);saved=torch.load(OUT/'pilot/best.pt',weights_only=False,map_location='cpu');model.load_state_dict(saved['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.00003)
    best=float('inf');history=[];started=time.monotonic();qbox=torch.tensor(queries,device=device)
    for epoch in range(epochs+1):
        losses=[]
        if epoch:
            model.train();order=rng.permutation(ids)
            for first in range(0,len(order),8):
                selected=order[first:first+8];_,q,t,_=batch(arrays,selected,device,rng)
                chosen=rng.choice(384,192,replace=False)
                q=torch.cat([q,qbox[chosen][None].expand(len(selected),-1,-1)],1)
                t=torch.cat([t,torch.tensor(targets[selected][:,chosen],device=device)],1)
                z=model.encode(torch.tensor(arrays['grids'][selected],device=device));loss=sdf_loss(model.decode(z,q),t)+1e-5*z.square().mean()
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach().cpu()))
            scheduler.step()
        if epoch%10==0 or epoch==epochs:
            model.eval();errors=[];exterior=[]
            with torch.no_grad():
                for first in range(0,len(valid),8):
                    selected=valid[first:first+8];_,q,t,_=batch(arrays,selected,device)
                    z=model.encode(torch.tensor(arrays['grids'][selected],device=device))
                    errors.append(float(sdf_loss(model.decode(z,q),t).cpu()))
                    exterior.append(float(sdf_loss(model.decode(z,qbox[None].expand(len(selected),-1,-1)),torch.tensor(targets[selected],device=device)).cpu()))
            selection=float(np.mean(errors)+.25*np.mean(exterior))
            row=dict(epoch=epoch,validation_sdf_l1=float(np.mean(errors)),validation_boundary_l1=float(np.mean(exterior)),selection_loss=selection,seconds=time.monotonic()-started)
            history.append(row)
            if selection<best:
                best=selection;torch.save(dict(model=model.state_dict(),epoch=epoch,manifest_sha256=sha256(ROOT/'results/sdf_shape_pilot/manifest.csv')),out/'best.pt')
            pd.DataFrame(history).to_csv(out/'history.csv',index=False);print(json.dumps(row),flush=True)
    save_json(out/'status.json',dict(status='complete',epochs=epochs,selection='validation SDF L1 + .25 boundary L1, including epoch zero',
        initial_checkpoint_sha256=sha256(OUT/'pilot/best.pt'),training_cells=len(ids),validation_cells=len(valid),test_used=False,per_cell_fitting=False))
    shutil.copy2(__file__,OUT/'source_snapshot/boundary_spatial_sdf.py')


if __name__=='__main__':train()
