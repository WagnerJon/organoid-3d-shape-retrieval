"""Run isolated SDF overfit or group-disjoint pilot; never exports to Atlas."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from pointnet_shape_core import ROOT, rotations, select_device, save_json, sha256
from prepare_sdf_shape import OUT
from sdf_shape_core import SDFModel,sdf_loss


def load_pilot():
    manifest=pd.read_csv(OUT/'manifest.csv')
    audit=json.loads((OUT/'data_audit.json').read_text())
    if sha256(OUT/'manifest.csv')!=audit['manifest_sha256']:raise ValueError('Pilot manifest changed')
    if (manifest.groupby('similarity_group').split.nunique()>1).any():raise ValueError('Group leakage')
    if set(manifest.split)!={'train','validation'}:raise ValueError('Pilot must not contain test cells')
    objects=[]
    for record in audit['records']:
        path=OUT/'data'/f"{record['index']:03d}.npz"
        if sha256(path)!=record['file_sha256']:raise ValueError('SDF file changed')
        with np.load(path) as d:objects.append({k:d[k] for k in d.files})
    return manifest,{k:np.stack([d[k] for d in objects]) for k in objects[0]}


def batch(arrays,ids,device,rng=None,rotate=False):
    view=int(rng.integers(2)) if rng is not None else 0
    query_ids=(np.concatenate([rng.choice(1536,384,replace=False),rng.choice(1024,256,replace=False)+1536,
                              rng.choice(512,128,replace=False)+2560]) if rng is not None else np.arange(3072))
    p=torch.tensor(arrays['points'][ids,view],device=device)
    q=torch.tensor(arrays['query'][ids][:,query_ids],device=device)
    t=torch.tensor(arrays['target'][ids][:,query_ids],device=device)
    groups={k:torch.tensor(arrays[k][ids,view],device=device).long() for k in ['centers1','neighbors1','centers2','neighbors2']}
    if rotate:
        r=torch.tensor(rotations(len(ids),rng),device=device)
        p=p@r.transpose(1,2);q=q@r.transpose(1,2)
    return p,q,t,groups


def train(kind,mode,device_name,epochs,initial_checkpoint=None):
    torch.set_num_threads(4);torch.manual_seed(20261018);rng=np.random.default_rng(20261018)
    device=select_device(device_name);manifest,arrays=load_pilot()
    out=OUT/f'{mode}_{kind}';out.mkdir(exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError('Choose a new run; do not overwrite pilot results')
    train_ids=manifest.index[manifest.overfit if mode=='overfit' else manifest.split=='train'].to_numpy()
    valid_ids=train_ids if mode=='overfit' else manifest.index[manifest.split=='validation'].to_numpy()
    model=SDFModel(kind).to(device)
    if initial_checkpoint is not None:
        saved=torch.load(initial_checkpoint,map_location='cpu',weights_only=False)
        if saved['kind']!=kind or saved['mode']!='overfit' or saved['manifest_sha256']!=sha256(OUT/'manifest.csv'):
            raise ValueError('Warm start must be the matching training-only sanity-check model')
        if not (manifest.loc[manifest.overfit,'split']=='train').all():raise ValueError('Warm-start data leakage')
        model.load_state_dict(saved['model'])
    save_json(out/'run_config.json',dict(seed=20261018,kind=kind,mode=mode,epochs=epochs,batch_size=8,
        learning_rate=.001,minimum_learning_rate=.0001,optimizer='AdamW',weight_decay=1e-6,
        training_queries=768,query_pool=3072,sdf_clip=.15,input_points=512,
        training_views=2,independent_evaluation_view=3,rotation_augmentation=mode=='pilot',
        manifest_sha256=sha256(OUT/'manifest.csv'),
        initial_checkpoint=str(initial_checkpoint) if initial_checkpoint is not None else None,
        initial_checkpoint_sha256=sha256(initial_checkpoint) if initial_checkpoint is not None else None,
        source_sha256={name:sha256(ROOT/'scripts'/name) for name in ['sdf_shape_core.py','prepare_sdf_shape.py','train_sdf_shape.py']},
        scope='Paper-inspired reduced-compute pilot; not a reproduction of published hyperparameters.'))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.0001)
    history=[];best=float('inf');started=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();losses=[];order=rng.permutation(train_ids)
        for first in range(0,len(order),8):
            p,q,t,g=batch(arrays,order[first:first+8],device,rng,rotate=mode=='pilot')
            _,prediction=model(p,q,g);loss=sdf_loss(prediction,t)
            if not torch.isfinite(loss):raise ValueError('Non-finite SDF loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step();model.eval();validation=[]
        # Fixed validation pool, no training gradients; same-cell pool only for explicit overfit sanity check.
        if epoch%5==0 or epoch==epochs or epoch==1:
            with torch.no_grad():
                for first in range(0,len(valid_ids),4):
                    p,q,t,g=batch(arrays,valid_ids[first:first+4],device)
                    _,prediction=model(p,q,g);validation.append(float(sdf_loss(prediction,t).cpu()))
            value=float(np.mean(validation))
            if value<best:
                best=value
                torch.save(dict(kind=kind,model=model.state_dict(),epoch=epoch,mode=mode,manifest_sha256=sha256(OUT/'manifest.csv')),out/'best.tmp')
                (out/'best.tmp').replace(out/'best.pt')
        else:value=None
        row=dict(epoch=epoch,train_sdf_l1=float(np.mean(losses)),validation_sdf_l1=value,seconds=time.monotonic()-started)
        history.append(row);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        save_json(out/'status.json',dict(status='training',kind=kind,mode=mode,device=str(device),**row))
        if value is not None:print(json.dumps(dict(kind=kind,mode=mode,**row)),flush=True)
    save_json(out/'status.json',dict(status='complete',kind=kind,mode=mode,device=str(device),epochs=epochs,best_sdf_l1=best,
        train_cells=len(train_ids),validation_cells=len(valid_ids),parameters=sum(p.numel() for p in model.parameters()),
        atlas_modified=False,test_cells_used=False,seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['pointnet','vn'],required=True)
    p.add_argument('--mode',choices=['overfit','pilot'],default='overfit');p.add_argument('--device',default='auto');p.add_argument('--epochs',type=int,default=150)
    args=p.parse_args();train(args.kind,args.mode,args.device,args.epochs)
