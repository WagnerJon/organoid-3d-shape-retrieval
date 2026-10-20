"""Offline paired-surface consistency ablation; never writes Atlas data.

Warm starts the same staged encoder in both arms. The shared decoder stays frozen.
No fitted teacher vectors are used as targets. This is a reduced-compute pilot.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from pointnet_shape_core import ROOT,select_device,save_json,sha256,rotations
from prepare_staged_sdf import OUT as SOURCE,load
from staged_sdf_core import ReferenceVectorEncoder,stability
from sdf_shape_core import SDFModel,sdf_loss
from train_sdf_shape import batch
from evaluate_staged_sdf import encode_views

OUT=ROOT/'results/consistent_sdf'


def consistency_loss(a,b):
    """Same-pose vector agreement plus symmetric instance discrimination.

    Both views receive the same rotation. Norm features are invariant; negatives
    are the other training cells in the minibatch. Reconstruction is essential:
    this contrastive term alone does not guarantee escape from exact collapse.
    """
    scale=((a.square().mean()+b.square().mean())/2).detach().clamp_min(1e-4)
    agreement=(a-b).square().mean()/scale
    fa=F.normalize(a.norm(dim=-1),dim=-1);fb=F.normalize(b.norm(dim=-1),dim=-1)
    logits=fa@fb.T/.02;labels=torch.arange(len(a),device=a.device)
    discrimination=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
    return agreement,discrimination


def train(arm,device_name,epochs):
    torch.set_num_threads(4);torch.manual_seed(20261025);rng=np.random.default_rng(20261025)
    device=select_device(device_name);manifest,arrays=load();out=OUT/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError('Completed/checkpointed arm exists')
    saved=torch.load(SOURCE/'decoder/best.pt',map_location='cpu',weights_only=False)
    start=torch.load(SOURCE/'encoder/best.pt',map_location='cpu',weights_only=False)
    assert start['decoder_sha256']==sha256(SOURCE/'decoder/best.pt')
    ids=manifest.index[manifest.split=='train'].to_numpy();validation=manifest.index[manifest.split=='validation'].to_numpy()
    assert not set(manifest.iloc[ids].similarity_group)&set(manifest.iloc[validation].similarity_group)
    model=SDFModel('vn').to(device);model.decoder.load_state_dict(saved['decoder']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    encoder=ReferenceVectorEncoder().to(device);encoder.load_state_dict(start['encoder'])
    optimizer=torch.optim.AdamW(encoder.parameters(),lr=.0003,weight_decay=1e-5)
    best=float('inf');history=[];started=time.monotonic()
    def validate(epoch,values):
        nonlocal best
        encoder.eval()
        with torch.no_grad():
            errors=[]
            for first in range(0,len(validation),4):
                selected=validation[first:first+4]
                p=torch.tensor(np.array(arrays['large_points'][selected,0]),device=device)
                g=torch.tensor(np.array(arrays['neighbors'][selected,0]),device=device).long()
                _,q,t,_=batch(arrays,selected,device)
                errors.append(float(sdf_loss(model.decode(encoder(p,g),q),t).cpu()))
            a=encode_views(encoder,arrays,validation,device);b=encode_views(encoder,arrays,validation,device,view=2)
        metrics=stability(a.norm(dim=-1).cpu().numpy(),b.norm(dim=-1).cpu().numpy(),a.norm(dim=-1).cpu().numpy())
        row=dict(epoch=epoch,validation_sdf_l1=float(np.mean(errors)),seconds=time.monotonic()-started,
                 resampling_self_top1=metrics['resampling_self_top1'],effective_rank=metrics['effective_rank'],**values)
        history.append(row)
        # Same selection rule in both arms, including the unchanged starting model.
        if row['validation_sdf_l1']<best:
            best=row['validation_sdf_l1'];torch.save(dict(encoder=encoder.state_dict(),epoch=epoch,
                decoder_sha256=sha256(SOURCE/'decoder/best.pt')),out/'best.pt')
        torch.save(dict(encoder=encoder.state_dict(),epoch=epoch,decoder_sha256=sha256(SOURCE/'decoder/best.pt')),out/'last.pt')
        pd.DataFrame(history).to_csv(out/'history.csv',index=False);print(json.dumps(dict(arm=arm,**row)),flush=True)
    validate(0,{})
    for epoch in range(1,epochs+1):
        encoder.train();losses=[];order=rng.permutation(ids)
        for first in range(0,len(order),4):
            selected=order[first:first+4];r=torch.tensor(rotations(len(selected),rng),device=device)
            _,q,t,_=batch(arrays,selected,device,rng);q=q@r.transpose(1,2)
            zs=[]
            for view in [0,1]:
                p=torch.tensor(np.array(arrays['large_points'][selected,view]),device=device)@r.transpose(1,2)
                g=torch.tensor(np.array(arrays['neighbors'][selected,view]),device=device).long()
                zs.append(encoder(p,g))
            reconstruction=sum(sdf_loss(model.decode(z,q),t) for z in zs)/2
            agreement,discrimination=consistency_loss(*zs)
            loss=reconstruction+(.01*agreement+.02*discrimination if arm=='paired' else 0)
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(encoder.parameters(),5);optimizer.step()
            losses.append([float(x.detach().cpu()) for x in [reconstruction,agreement,discrimination]])
        for group in optimizer.param_groups:group['lr']=.00003+.00027*(1+math.cos(math.pi*epoch/epochs))/2
        validate(epoch,dict(zip(['train_sdf_l1','train_agreement','train_discrimination'],np.mean(losses,axis=0).tolist())))
    save_json(out/'status.json',dict(status='complete',epochs=epochs,seconds=time.monotonic()-started,
        training_cells=len(ids),validation_cells=len(validation),test_used=False,decoder_frozen=True,
        code_regression=False,selection='Lowest validation SDF L1, including epoch 0',arm=arm))


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'provenance.json').exists():raise FileExistsError('Experiment already prepared')
    snapshot=OUT/'source_snapshot';snapshot.mkdir()
    names=['consistent_sdf.py','staged_sdf_core.py','sdf_shape_core.py','evaluate_staged_sdf.py','prepare_staged_sdf.py','train_sdf_shape.py']
    for name in names:shutil.copy2(ROOT/'scripts'/name,snapshot/name)
    save_json(OUT/'provenance.json',dict(seed=20261025,arms=['control','paired'],epochs=12,
        source_hashes={name:sha256(snapshot/name) for name in names},
        initial_encoder_sha256=sha256(SOURCE/'encoder/best.pt'),decoder_sha256=sha256(SOURCE/'decoder/best.pt'),
        manifest_sha256=sha256(SOURCE/'manifest.csv'),atlas_sha256=sha256(ROOT/'results/explorer_dataset/dataset.json'),
        paired_weights=dict(vector_agreement=.01,instance_discrimination=.02),temperature=.02,
        scope='Both arms warm-start same encoder; two independent precomputed training views, shared SO3 augmentation; third view reserved for validation stability. Frozen decoder. No test cells or Atlas integration.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','control','paired','all'],default='all')
    parser.add_argument('--device',default='auto');parser.add_argument('--epochs',type=int,default=12);args=parser.parse_args()
    if args.stage in ['prepare','all']:prepare()
    for arm in ['control','paired']:
        if args.stage in [arm,'all']:train(arm,args.device,args.epochs)
