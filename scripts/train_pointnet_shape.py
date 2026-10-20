"""Train a deterministic 128D PointNet++ autoencoder; test rows are never indexed here."""
import argparse
from pathlib import Path
import json
import time
import random
import numpy as np
import pandas as pd
import torch
from pointnet_shape_core import ROOT, read_config, load_data, tensors, PointCloudAutoencoder, chamfer, rotations, select_device, save_json, sha256


def train(config, smoke=False, device_override=None, resume=False):
    torch.set_num_threads(config['threads']);torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    device=select_device(device_override or config['device'])
    data=ROOT/config['output_dir']/'data';out=ROOT/config['output_dir']/('smoke' if smoke else 'baseline');out.mkdir(parents=True,exist_ok=True)
    manifest,arrays=load_data(data)
    atlas=json.loads((ROOT/config['atlas_dataset']/'dataset.json').read_text())
    current={cell['id']:cell for cell in atlas['cells']}
    if set(current)!=set(manifest.cell_id):raise ValueError('Atlas cell inventory changed; prepare a new dataset')
    for row in manifest.itertuples():
        cell=current[row.cell_id]
        if cell.get('inMainOrganoid')!=bool(row.in_main_organoid) or cell['qc']!=bool(row.shape_qc_pass):
            raise ValueError('Atlas membership/QC changed; prepare a new dataset before training')
    metadata=json.loads((data/'dataset.json').read_text())
    if (metadata['points'],metadata['centers'],metadata['neighbors']) != (config['points'],config['centers'],config['neighbors']):
        raise ValueError('Prepared geometry differs from model configuration')
    train_ids=manifest.index[(manifest.split=='train') & manifest.training_eligible].to_numpy()
    validation_ids=manifest.index[(manifest.split=='validation') & manifest.training_eligible].to_numpy()
    if smoke: train_ids=train_ids[:8];validation_ids=validation_ids[:4]
    model=PointCloudAutoencoder(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    epochs=2 if smoke else config['epochs'];batch_size=min(config['batch_size'],4) if smoke else config['batch_size']
    history=[];best=float('inf');start_epoch=1;bad_epochs=0
    rng=np.random.default_rng(config['seed'])
    signature=sha256(data/'manifest.csv')
    if (out/'last.pt').exists() and not resume:
        raise FileExistsError(f'{out}/last.pt exists; use --resume or a new output_dir')
    if resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if saved['manifest_sha256'] != signature or saved['config'] != config:
            raise ValueError('Cannot resume a different dataset or configuration')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);history=saved['history']
        best=saved['best'];bad_epochs=saved['bad_epochs'];start_epoch=saved['epoch']+1
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
        if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('status')=='trained':
            print('Completed training already exists; reusing best checkpoint',flush=True)
            return out
    else:
        torch.save(dict(model=model.cpu().state_dict(),config=config,manifest_sha256=signature),out/'untrained.pt')
        model.to(device)
    started=time.monotonic()
    for epoch in range(start_epoch,epochs+1):
        model.train();losses=[]
        order=rng.permutation(train_ids)
        for step in range(0,len(order),batch_size):
            points,target,groups=tensors(arrays,order[step:step+batch_size],device)
            if config['rotation_augmentation']:
                rotation=torch.from_numpy(rotations(len(points),rng)).to(device)
                points=points@rotation.transpose(1,2);target=target@rotation.transpose(1,2)
            _,prediction=model(points,groups)
            loss=chamfer(prediction,target,config['chamfer_chunk']).mean()
            if not torch.isfinite(loss): raise ValueError('Non-finite reconstruction loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append((float(loss.detach().cpu()),len(points)))
        model.eval();values=[]
        with torch.no_grad():
            for step in range(0,len(validation_ids),batch_size):
                points,target,groups=tensors(arrays,validation_ids[step:step+batch_size],device)
                _,prediction=model(points,groups)
                values.extend(chamfer(prediction,target,config['chamfer_chunk']).cpu().tolist())
        row=dict(epoch=epoch,train_chamfer=sum(a*b for a,b in losses)/sum(b for _,b in losses),validation_chamfer=float(np.mean(values)),seconds=time.monotonic()-started)
        history.append(row)
        improved=row['validation_chamfer']<best
        if improved: best=row['validation_chamfer'];bad_epochs=0
        else: bad_epochs+=1
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,epoch=epoch,history=history,best=best,
                     bad_epochs=bad_epochs,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),manifest_sha256=signature)
        torch.save(payload,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
        if improved: torch.save(dict(model=model.state_dict(),config=config,epoch=epoch,manifest_sha256=signature),out/'best.pt')
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        save_json(out/'status.json',dict(status='training',device=str(device),smoke=smoke,epochs_requested=epochs,test_rows_loaded=False,**row))
        print(json.dumps(dict(device=str(device),**row)),flush=True)
        if bad_epochs>=config['patience']:break
    save_json(out/'status.json',dict(status='trained',device=str(device),smoke=smoke,completed_epochs=int(history[-1]['epoch']),best_validation_chamfer=best,
                                   train_cells=len(train_ids),validation_cells=len(validation_ids),test_rows_loaded=False))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config/pointnet_shape.json');parser.add_argument('--device');parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();train(read_config(ROOT/args.config),args.smoke,args.device,args.resume)
