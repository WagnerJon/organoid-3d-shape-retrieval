"""Shared training-only SDF codes, then distill into a reference-style VN encoder."""
import argparse
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from pointnet_shape_core import ROOT,select_device,save_json,sha256,rotations
from sdf_shape_core import SDFModel,sdf_loss
from staged_sdf_core import ReferenceVectorEncoder,VectorAdam,code_loss
from prepare_staged_sdf import OUT,load
from train_sdf_shape import batch


def train_decoder(device_name,steps):
    torch.set_num_threads(4);torch.manual_seed(20261021);rng=np.random.default_rng(20261021)
    device=select_device(device_name);manifest,arrays=load();ids=manifest.index[manifest.split=='train'].to_numpy()
    out=OUT/'decoder';out.mkdir(exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError('Do not overwrite a decoder run')
    model=SDFModel('vn').to(device)
    codes=nn.Parameter(torch.randn(len(ids),128,3,device=device)*.3)
    optimizer=torch.optim.AdamW(model.decoder.parameters(),lr=.001,weight_decay=1e-5)
    code_optimizer=VectorAdam(codes,lr=.01)
    best=float('inf');history=[];started=time.monotonic()
    for step in range(1,steps+1):
        chosen=rng.choice(len(ids),8,replace=False);_,q,t,_=batch(arrays,ids[chosen],device,rng)
        z=codes[torch.tensor(chosen,device=device)]
        reconstruction=sdf_loss(model.decode(z,q),t)
        loss=reconstruction+.001*z.square().mean()
        optimizer.zero_grad(set_to_none=True);code_optimizer.zero_grad();loss.backward();optimizer.step();code_optimizer.step()
        factor=.1+.9*(1+math.cos(math.pi*step/steps))/2
        for group in optimizer.param_groups:group['lr']=.001*factor
        code_optimizer.lr=.01*factor
        if step%250==0 or step==1 or step==steps:
            with torch.no_grad():
                values=[]
                for first in range(0,len(ids),8):
                    _,q,t,_=batch(arrays,ids[first:first+8],device)
                    values.append(float(sdf_loss(model.decode(codes[first:first+8],q),t).cpu()))
                error=float(np.mean(values))
            row=dict(step=step,training_sdf_l1=error,seconds=time.monotonic()-started);history.append(row)
            if error<best:
                best=error;torch.save(dict(decoder=model.decoder.state_dict(),codes=codes.detach(),train_ids=ids,
                    step=step,manifest_sha256=sha256(OUT/'manifest.csv')),out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            save_json(out/'status.json',dict(status='training',device=str(device),**row));print(json.dumps(row),flush=True)
    save_json(out/'status.json',dict(status='complete',device=str(device),steps=steps,training_cells=len(ids),
        best_training_sdf_l1=best,seconds=time.monotonic()-started,validation_used=False,test_used=False,
        latent='128 SO(3)-equivariant vectors; norm gives 128 analysis features',code_prior_weight=.001))


def train_encoder(device_name,epochs):
    torch.set_num_threads(4);torch.manual_seed(20261022);rng=np.random.default_rng(20261022)
    device=select_device(device_name);manifest,arrays=load();out=OUT/'encoder';out.mkdir(exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError('Do not overwrite encoder run')
    saved=torch.load(OUT/'decoder/best.pt',map_location='cpu',weights_only=False)
    if saved['manifest_sha256']!=sha256(OUT/'manifest.csv'):raise ValueError('Decoder data mismatch')
    ids=np.asarray(saved['train_ids']);assert (manifest.iloc[ids].split=='train').all()
    validation=manifest.index[manifest.split=='validation'].to_numpy()
    teacher=saved['codes'].to(device);scale=teacher.square().mean((0,2),keepdim=True).sqrt().clamp_min(.03)
    decoder=SDFModel('vn').to(device);decoder.decoder.load_state_dict(saved['decoder']);decoder.eval()
    for parameter in decoder.parameters():parameter.requires_grad_(False)
    encoder=ReferenceVectorEncoder().to(device)
    optimizer=torch.optim.AdamW(encoder.parameters(),lr=.001,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.0001)
    best=float('inf');history=[];started=time.monotonic()
    for epoch in range(1,epochs+1):
        encoder.train();losses=[]
        order=rng.permutation(len(ids))
        for first in range(0,len(order),4):
            positions=order[first:first+4];selected=ids[positions];view=int(rng.integers(2))
            p=torch.tensor(np.array(arrays['large_points'][selected,view]),device=device)
            g=torch.tensor(np.array(arrays['neighbors'][selected,view]),device=device).long()
            _,q,t,_=batch(arrays,selected,device,rng)
            r=torch.tensor(rotations(len(selected),rng),device=device)
            p=p@r.transpose(1,2);q=q@r.transpose(1,2)
            target=teacher[torch.tensor(positions,device=device)]@r.transpose(1,2)
            z=encoder(p,g)
            # Explicit code supervision prevents a good decoder from concealing encoder collapse.
            latent_error=code_loss(z,target,scale)
            sdf_error=sdf_loss(decoder.decode(z,q),t)
            loss=sdf_error+.05*latent_error
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(encoder.parameters(),5);optimizer.step()
            losses.append((float(sdf_error.detach().cpu()),float(latent_error.detach().cpu())))
        scheduler.step();encoder.eval()
        if epoch%5==0 or epoch==1 or epoch==epochs:
            with torch.no_grad():
                values=[]
                for first in range(0,len(validation),4):
                    selected=validation[first:first+4]
                    p=torch.tensor(np.array(arrays['large_points'][selected,0]),device=device)
                    g=torch.tensor(np.array(arrays['neighbors'][selected,0]),device=device).long()
                    _,q,t,_=batch(arrays,selected,device)
                    values.append(float(sdf_loss(decoder.decode(encoder(p,g),q),t).cpu()))
                error=float(np.mean(values))
            row=dict(epoch=epoch,train_sdf_l1=float(np.mean(losses,axis=0)[0]),train_code_mse=float(np.mean(losses,axis=0)[1]),
                     validation_sdf_l1=error,seconds=time.monotonic()-started);history.append(row)
            if error<best:
                best=error;torch.save(dict(encoder=encoder.state_dict(),epoch=epoch,decoder_sha256=sha256(OUT/'decoder/best.pt'),
                    manifest_sha256=sha256(OUT/'manifest.csv')),out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            save_json(out/'status.json',dict(status='training',device=str(device),**row));print(json.dumps(row),flush=True)
    save_json(out/'status.json',dict(status='complete',device=str(device),epochs=epochs,training_cells=len(ids),validation_cells=len(validation),
        best_validation_sdf_l1=best,seconds=time.monotonic()-started,decoder_frozen=True,code_regression_weight=.05,
        test_used=False,scope='Reference-style independently implemented VN encoder; 2048 points; not exact paper reproduction.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['decoder','encoder'],required=True);p.add_argument('--device',default='auto')
    p.add_argument('--steps',type=int,default=12000);p.add_argument('--epochs',type=int,default=80)
    args=p.parse_args();(train_decoder(args.device,args.steps) if args.stage=='decoder' else train_encoder(args.device,args.epochs))
