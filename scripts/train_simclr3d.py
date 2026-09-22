"""Train a compact shape encoder; reference annotations are never training targets."""
import json, math, time, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from simclr3d_core import ResNet3DSimCLR, augment, nt_xent, pair_metrics, augment_raw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/simclr3d'

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config/simclr3d.json');args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text());OUT=ROOT/cfg.get('output_dir','results/simclr3d')
    seed=cfg['seed']; random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    device=torch.device(cfg['device'])
    if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS GPU unavailable; run outside filesystem sandbox')
    data=np.load(OUT/cfg.get('data_file','masks64.npy'),mmap_mode='r'); meta=pd.read_csv(OUT/'manifest.csv')
    train=meta.index[meta.split=='train'].to_numpy(); val=meta.index[meta.split=='validation'].to_numpy()
    model=ResNet3DSimCLR(cfg['base_channels'],cfg['embedding_dim'],cfg['projection_dim']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed); augrng=torch.Generator().manual_seed(seed+1)
    history=[];best=float('inf');start=1;started=time.time()
    if (OUT/'last.pt').exists():
        ck=torch.load(OUT/'last.pt',map_location='cpu',weights_only=False)
        if ck['config']!=cfg:raise ValueError('Configuration changed: use a separate output directory')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['epoch']+1
        history=ck['history'];best=ck['best_validation_loss'];rng.bit_generator.state=ck['numpy_rng'];augrng.set_state(ck['augmentation_rng'])
    def batch(indices):return torch.from_numpy(np.array(data[indices],copy=True)).unsqueeze(1).float().to(device)
    def paired(x,g):
        if cfg.get('input_kind')=='raw':return augment_raw(x,g,cfg),augment_raw(x,g,cfg)
        return (augment(x,g,cfg['augmentation_scale'],cfg['augmentation_translation']),augment(x,g,cfg['augmentation_scale'],cfg['augmentation_translation']))
    def evaluate():
        model.eval();g=torch.Generator().manual_seed(seed+999);records=[];hs=[]
        with torch.no_grad():
            for k in range(0,len(val),cfg['batch_size']):
                ids=val[k:k+cfg['batch_size']]
                if len(ids)<2:continue
                x=batch(ids);a,b=paired(x,g);h,z=model(torch.cat([a,b]));z1,z2=z.chunk(2)
                records.append(dict(n=len(ids),loss=float(nt_xent(z1,z2,cfg['temperature']).cpu()),**pair_metrics(z1,z2)))
                hs.append(h[:len(ids)].cpu().numpy())
        out={key:float(np.average([r[key] for r in records],weights=[r['n'] for r in records])) for key in records[0] if key!='n'}
        h=np.concatenate(hs);s=np.linalg.svd(h-h.mean(0),compute_uv=False);p=s*s/(s@s+1e-12)
        out['effective_rank']=float(np.exp(-(p*np.log(p+1e-12)).sum()));out['encoder_std']=float(h.std(0).mean())
        return out
    if start==1:
        if cfg.get('input_kind')=='raw':
            model.eval();initial=[]
            with torch.no_grad():
                for k in range(0,len(data),cfg['batch_size']):
                    h,_=model(batch(np.arange(k,min(k+cfg['batch_size'],len(data)))));initial.append(h.cpu().numpy())
            np.save(OUT/'untrained_embeddings.npy',np.concatenate(initial))
        baseline=evaluate();(OUT/'baseline.json').write_text(json.dumps(baseline,indent=2));print('Baseline '+json.dumps(baseline),flush=True)
    for epoch in range(start,cfg['epochs']+1):
        tick=time.time();model.train();order=rng.permutation(train);losses=[]
        factor=epoch/cfg['warmup_epochs'] if epoch<=cfg['warmup_epochs'] else .5*(1+math.cos(math.pi*(epoch-cfg['warmup_epochs'])/(cfg['epochs']-cfg['warmup_epochs'])))
        for group in opt.param_groups:group['lr']=cfg['learning_rate']*max(factor,.01)
        for k in range(0,len(order)-1,cfg['batch_size']):
            ids=order[k:k+cfg['batch_size']]
            if len(ids)<2:continue
            x=batch(ids);a,b=paired(x,augrng);opt.zero_grad(set_to_none=True)
            _,z=model(torch.cat([a,b]));z1,z2=z.chunk(2);loss=nt_xent(z1,z2,cfg['temperature'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();losses.append((float(loss.detach().cpu()),len(ids)))
        row=dict(epoch=epoch,train_loss=float(np.average([x[0] for x in losses],weights=[x[1] for x in losses])),learning_rate=opt.param_groups[0]['lr'])
        improved=False
        if epoch==1 or epoch%cfg['validation_every']==0 or epoch==cfg['epochs']:
            metrics=evaluate();row.update({'validation_'+k:v for k,v in metrics.items()});improved=metrics['loss']<best
            if improved:best=metrics['loss']
        row['seconds']=time.time()-tick;history.append(row)
        ck=dict(epoch=epoch,config=cfg,model=model.state_dict(),optimizer=opt.state_dict(),history=history,best_validation_loss=best,numpy_rng=rng.bit_generator.state,augmentation_rng=augrng.get_state())
        torch.save(ck,OUT/'last.tmp');(OUT/'last.tmp').replace(OUT/'last.pt')
        if improved:torch.save(ck,OUT/'best.pt')
        pd.DataFrame(history).to_csv(OUT/'training_history.csv',index=False)
        (OUT/'status.json').write_text(json.dumps(dict(status='training',epoch=epoch,total_epochs=cfg['epochs'],**{k:v for k,v in row.items() if k!='epoch'}),indent=2))
        print(json.dumps(row),flush=True)
    ck=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.eval();features=[]
    with torch.no_grad():
        for k in range(0,len(data),cfg['batch_size']):
            h,_=model(batch(np.arange(k,min(k+cfg['batch_size'],len(data)))));features.append(h.cpu().numpy())
    h=np.concatenate(features);np.save(OUT/'embeddings_raw.npy',h)
    h=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12);np.save(OUT/'embeddings.npy',h)
    pd.concat([meta[['cell_id','sample','label','split','shape_qc_pass','original_volume_um3']],pd.DataFrame(h,columns=[f'embedding_{i:03d}' for i in range(h.shape[1])])],axis=1).to_csv(OUT/'embeddings.csv',index=False)
    (OUT/'status.json').write_text(json.dumps(dict(status='complete',epochs=cfg['epochs'],best_epoch=ck['epoch'],best_validation_loss=best,objects_embedded=len(h),parameters=sum(p.numel() for p in model.parameters()),run_seconds=time.time()-started),indent=2))
    print('COMPLETE '+(OUT/'status.json').read_text(),flush=True)
if __name__=='__main__':main()
