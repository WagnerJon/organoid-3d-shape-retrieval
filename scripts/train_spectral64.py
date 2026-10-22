"""Matched A/B 64D VAE backbones; development gate controls optional C."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import copy,time
import pandas as pd
import torch
from torch import nn
from scipy.spatial.distance import cdist
from spectral64_core import *
torch.set_num_threads(2)
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(DIM,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
        self.mean=nn.Linear(128,64);self.logv=nn.Linear(128,64);nn.init.constant_(self.logv.bias,-6)
        self.decoder=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,256),nn.SiLU(),nn.Linear(256,DIM))
    def encode(self,x):return self.mean(self.features(x))
    def forward(self,x):
        h=self.features(x);mu,lv=self.mean(h),self.logv(h).clamp(-12,4)
        return self.decoder(mu+torch.randn_like(mu)*torch.exp(.5*lv)),mu,lv


def load():
    entries=json.loads((OUT/'entries.json').read_text());raw=np.load(OUT/'coefficients.npz')['coefficients']
    train=np.array([i for i,e in enumerate(entries) if e['split']=='train'])
    dev=np.array([i for i,e in enumerate(entries) if e['split']=='validation' and e['kind']!='mirror'])
    return entries,raw,train,dev


def inputs(kind,raw,train):
    if kind=='A':
        source=powers(raw);mean=source[train].mean(0);scale=np.sqrt(np.square(source[train]-mean).mean());n=416
        out=np.zeros((len(raw),DIM),np.float32);out[:,:n]=(source-mean)/scale
    else:
        source=weighted(raw);mean=np.zeros(DIM);m=mean.reshape(32,K);m[:,0]=source[train].reshape(-1,32,K)[:,:,0].mean(0)
        scale=np.sqrt(np.square(source[train]-mean).mean());out=((source-mean)/scale).astype(np.float32);n=DIM
    return out,dict(mean=mean,scale=np.array(scale),target_dimensions=np.array(n))


def train(kind):
    entries,raw,train,dev=load();x,scaler=inputs(kind,raw,train)
    folder=OUT/kind;folder.mkdir(exist_ok=True);np.savez_compressed(folder/'scaler.npz',**scaler)
    n=int(scaler['target_dimensions']);t=torch.tensor(x,device=DEVICE)
    degree=np.arange(13)
    if kind=='A':weights=np.tile(np.sqrt(1+.05*degree*(degree+1)),32)
    else:weights=np.tile(np.concatenate([np.full(2*l+1,np.sqrt(1+.05*l*(l+1))) for l in range(13)]),32)
    w=torch.tensor(weights,dtype=torch.float32,device=DEVICE)
    identity=torch.eye(K,device=DEVICE)
    views=None;rotations=None
    if kind=='C':
        a=np.load(OUT/'rotated_training_views.npz');views=torch.tensor(((weighted(a['coefficients'].reshape(-1,32,K))-scaler['mean'])/scaler['scale']).reshape(len(raw),2,DIM).astype(np.float32),device=DEVICE);rotations=torch.tensor(a['actions'],device=DEVICE)
    logs=[]
    for seed in [42,43,44]:
        if (folder/f'seed{seed}.pt').exists():continue
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Model().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=.001)
        best=float('inf');history=[];start=time.time()
        for epoch in range(121):
            if epoch:
                model.train()
                for batch in np.array_split(rng.permutation(train),max(1,len(train)//32)):
                    losses=[];latents=[]
                    for j in [0,1]:
                        inp=t[batch] if views is None else views[batch,j]
                        prediction,mu,lv=model(inp);latents.append(mu)
                        action=identity.expand(len(batch),-1,-1) if rotations is None else rotations[batch,j]
                        posed=torch.bmm(prediction.reshape(-1,32,K),action.transpose(1,2)).reshape(-1,DIM)
                        target=t[batch] if views is None else inp
                        rec=(((posed[:,:n]-target[:,:n])*w)**2).mean()
                        kl=-.5*(1+lv-mu.square()-lv.exp()).mean()
                        losses.append(rec+1e-5*kl)
                    loss=sum(losses)/2
                    if kind=='C':
                        joined=torch.cat(latents,0)
                        variance=(joined-joined.mean(0,keepdim=True)).square().mean()
                        consistency=(latents[0]-latents[1]).square().mean()/(variance+1e-8)
                        loss=loss+.1*consistency
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
            model.eval()
            with torch.no_grad():
                # Canonical validation is available for all arms; no test or mirror
                # challenge contributes to checkpoint selection.
                z=model.encode(t[dev]);pred=model.decoder(z)
                score=float((((pred[:,:n]-t[dev,:n])*w)**2).mean())
            history.append(dict(epoch=epoch,validation_loss=score))
            if score<best:best=score;state=copy.deepcopy(model.state_dict());chosen=epoch
            if epoch%30==0:print(f'{kind} seed{seed} epoch{epoch}: validation={score:.6f} device={DEVICE}',flush=True)
        model.load_state_dict(state)
        torch.save(dict(state={k:v.cpu() for k,v in state.items()},seed=seed,epoch=chosen,latent=64,input_dim=DIM,parameters=sum(p.numel() for p in model.parameters())),folder/f'seed{seed}.pt')
        pd.DataFrame(history).to_csv(folder/f'seed{seed}_history.csv',index=False)
        with torch.no_grad():
            codes=model.encode(t).cpu().numpy();prediction=model.decoder(torch.tensor(codes,device=DEVICE)).cpu().numpy()
        np.savez_compressed(folder/f'seed{seed}_readouts.npz',codes=codes,prediction=prediction)
        logs.append(dict(seed=seed,chosen_epoch=chosen,seconds=time.time()-start,best_validation=best))
    save_json(folder/'training.complete.json',dict(kind=kind,epochs=120,seeds=[42,43,44],logs=logs,train=len(train),validation=len(dev),architecture_parameters=sum(p.numel() for p in Model().parameters())))


def gate():
    entries,raw,train,dev=load();ids=[i for i,e in enumerate(entries) if e['kind']=='mirror' and e['split']=='validation']
    rows=[]
    for kind in ['A','B']:
        scaler=np.load(OUT/kind/'scaler.npz');target=weighted(raw[ids])
        for seed in [42,43,44]:
            z=np.load(OUT/kind/f'seed{seed}_readouts.npz');codes=z['codes'][ids];dist=cdist(codes,codes)
            # Query is itself; compare to its paired mirror with self-query
            # leave-one-out jitter in latent numerical precision avoided below.
            # Separation is tested using independently re-integrated geometry in
            # the final readout; this first gate also requires decoded phase R2.
            pair_distance=np.array([dist[i,i^1] for i in range(len(ids))])
            if kind=='B':
                pred=z['prediction'][ids]*scaler['scale']+scaler['mean']
                delta=target[::2]-target[1::2];pdelta=pred[::2]-pred[1::2]
                r2=1-np.square(pdelta-delta).sum()/np.square(delta).sum()
            else:r2=0.
            # Avoid counting tiny quadrature roundoff as discrimination. Full
            # coefficient distances and powers use the same fixed mesh pair.
            nonmirror=dist.copy()
            for i in range(len(ids)):nonmirror[i,[i,i^1]]=np.inf
            margin=pair_distance/np.median(nonmirror[np.isfinite(nonmirror)])
            separation=float(np.mean(margin>.01))
            rows.append(dict(model=kind,seed=seed,phase_difference_R2=float(r2),relative_mirror_separation=float(np.median(margin)),mirror_separated_fraction=separation))
    b=[r for r in rows if r['model']=='B'];passed=np.mean([r['phase_difference_R2'] for r in b])>=.25 and np.mean([r['mirror_separated_fraction'] for r in b])>=.9
    save_json(OUT/'B_information_gate.json',dict(passed=bool(passed),rows=rows,criterion='Mean phase-difference reconstruction R2 >=.25 and >=90% mirrors separated by >1% of median between-pair latent distance. Uses development pairs only; independent physical-view retrieval is evaluated subsequently.',note='This operationalizes information retention without rewarding sub-floating-point code differences. Model C remains conditional.'))
    print('B information gate',passed,rows,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['A','B','C','gate']);stage=p.parse_args().stage
    if stage=='gate':gate()
    else:train(stage)
