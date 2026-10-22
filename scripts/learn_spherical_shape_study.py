"""Predeclared self-supervised spherical-descriptor VAE and held-out readouts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['VECLIB_MAXIMUM_THREADS']='1'
import sys
import time
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from spherical_shape_study_core import *
from spatial_benchmark_core import mesh_for,geometry_cost
from check_cajal_validation import candidate_sets
from harmonics_cajal_core import harmonics,angular_basis

torch.set_num_threads(2)
RADII=np.linspace(.04,2.6,32)

def radial_input(r,Y):
    c=coefficients(r,Y)
    r=r/np.sqrt(np.mean(r*r))
    occupancy=expit((r[None,:]-RADII[:,None])/.035)
    cs=(4*np.pi/len(Y))*(occupancy@Y.conj())
    p=np.stack([np.linalg.norm(cs[:,l*l:(l+1)**2],axis=1) for l in range(13)],1)
    shell=(p*RADII[:,None]).ravel();shell/=max(np.linalg.norm(shell),1e-12)
    return np.r_[all_features(c),shell]


def synthetic_data():
    cache=OUT/'learning_synthetic.npz'
    if cache.exists():return np.load(cache)
    d,Y=fib_basis(2048,12);dv,Yv=fib_basis(1024,12)
    rng=np.random.default_rng(8791)
    rotations=Rotation.random(24,random_state=rng).as_matrix()
    rotated=[basis(dv@r,12) for r in rotations]
    clean=[];views=[]
    for i in range(576):
        c=random_radial(100000+i,12)
        clean.append(radial_input((Y@c).real,Y))
        x=[]
        for j,noise in enumerate([0,.01,.03]):
            r=(rotated[(i*3+j)%24]@c).real+rng.normal(0,noise,len(dv))
            x.append(radial_input(r,Yv))
        views.append(x)
        if i%64==0:print(f'Synthetic learning inputs {i+1}/576',flush=True)
    np.savez_compressed(cache,clean=clean,views=views)
    return np.load(cache)


class Encoder(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(n,256),nn.SiLU(),nn.Linear(256,192),nn.SiLU())
        self.mean=nn.Linear(192,128);self.logv=nn.Linear(192,128)
        nn.init.constant_(self.logv.bias,-6)
        self.decode=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,n))
    def encode(self,x):return self.mean(self.features(x))
    def forward(self,x):
        h=self.features(x);mu,lv=self.mean(h),self.logv(h).clamp(-12,4)
        return self.decode(mu+torch.randn_like(mu)*torch.exp(.5*lv)),mu,lv


def fit_domain(name,clean,views,train,dev,queries,gallery,query_ids):
    folder=OUT/'learning'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'retrieval.csv').exists() and (folder/'gallery_codes.npz').exists():
        return pd.read_csv(folder/'retrieval.csv').to_dict('records')
    mean=clean[train].mean(0);sd=clean[train].std(0).clip(1e-6)
    x=((clean-mean)/sd).astype('float32')
    v=((views-mean)/sd).astype('float32')
    baselines={
        'radial_power':(gallery[:,:13],queries[...,:13]),
        'radial_power_bispectrum':(gallery[:,:73],queries[...,:73]),
        'shell_power':(gallery[:,73:],queries[...,73:]),
        'all_invariants_raw':(gallery,queries),
        'all_invariants_standardized':((gallery-mean)/sd,(queries-mean)/sd)}
    standardized=(gallery-mean)/sd;qs=(queries-mean)/sd
    pcas={}
    for dim in [32,128]:
        pca=PCA(n_components=min(dim,len(train)-1),svd_solver='full').fit(x[train])
        pcas[dim]=pca;baselines[f'PCA{dim}']=(pca.transform(standardized),pca.transform(qs.reshape(-1,qs.shape[-1])).reshape(*qs.shape[:-1],pca.n_components_))
    result=[];all_codes={}
    for key,(g,q) in baselines.items():
        all_codes[key]=g
        for noise in range(q.shape[1]):
            distances=cdist(q[:,noise],g);rank=np.argsort(distances,axis=1)
            result.append(dict(domain=name,method=key,seed=-1,noise_index=noise,identity=float(np.mean(rank[:,0]==query_ids)),recall5=float(np.mean([i in ids[:5] for i,ids in zip(query_ids,rank)]))))
    tclean=torch.tensor(x);tviews=torch.tensor(v)
    for seed in [42,43,44]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=Encoder(clean.shape[-1]);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        best=float('inf');best_state=None;history=[]
        for epoch in range(121):
            if epoch:
                model.train()
                for batch in np.array_split(rng.permutation(train),max(1,len(train)//32)):
                    view=rng.integers(0,v.shape[1],len(batch));a=tviews[batch,view]
                    noisy=a+torch.randn_like(a)*.03
                    prediction,mu,lv=model(noisy)
                    rec=(prediction-tclean[batch]).square().mean()
                    kl=-.5*(1+lv-mu.square()-lv.exp()).mean()
                    consistency=(mu-model.encode(a)).square().mean()
                    loss=rec+1e-5*kl+.1*consistency
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
            model.eval()
            with torch.no_grad():
                z=model.encode(tclean[dev]);score=float((model.decode(z)-tclean[dev]).square().mean())
            history.append(dict(epoch=epoch,development_reconstruction_mse=score))
            if score<best:
                best=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
            if epoch%30==0:print(f'{name} VAE seed {seed} epoch {epoch}: dev MSE {score:.5f}',flush=True)
        model.load_state_dict(best_state);torch.save(dict(state=best_state,input_dimension=clean.shape[-1],latent=128,best_epoch=best_epoch),folder/f'seed{seed}.pt')
        pd.DataFrame(history).to_csv(folder/f'seed{seed}_training.csv',index=False)
        with torch.no_grad():
            g=model.encode(torch.tensor(standardized,dtype=torch.float32)).numpy()
            q=model.encode(torch.tensor(qs.reshape(-1,qs.shape[-1]),dtype=torch.float32)).numpy().reshape(*qs.shape[:-1],128)
        all_codes[f'VAE_seed{seed}']=g
        for noise in range(q.shape[1]):
            distance=cdist(q[:,noise],g);rank=np.argsort(distance,axis=1)
            singular=np.linalg.svd(g-g.mean(0),compute_uv=False);prob=singular/singular.sum();effective_rank=float(np.exp(-(prob*np.log(prob.clip(1e-15))).sum()))
            result.append(dict(domain=name,method='VAE128',seed=seed,noise_index=noise,identity=float(np.mean(rank[:,0]==query_ids)),recall5=float(np.mean([i in ids[:5] for i,ids in zip(query_ids,rank)])),best_epoch=best_epoch,development_mse=best,effective_rank=effective_rank))
    np.savez_compressed(folder/'gallery_codes.npz',**all_codes)
    np.savez_compressed(folder/'scaler.npz',mean=mean,sd=sd)
    pd.DataFrame(result).to_csv(folder/'retrieval.csv',index=False)
    return result


def real_data():
    frame=pd.read_csv(OUT/'real_manifest.csv');a=np.load(OUT/'real_features.npz')
    shell=np.load(ROOT/'results/harmonics_cajal/surfaces.npz');lookup={cid:i for i,cid in enumerate(shell['ids'])}
    clean=np.c_[a['features'],shell['harmonics'][[lookup[c] for c in frame.cell_id]]]
    train=np.flatnonzero(frame.split.to_numpy()=='train');validation=np.flatnonzero(frame.split.to_numpy()=='validation')
    # Separate development reconstruction selection from the 48-cell geometric
    # readout. Development cells come from the original training partition.
    rng=np.random.default_rng(420);groups=frame.iloc[train].similarity_group.unique();rng.shuffle(groups)
    dev_groups=set(groups[:max(1,len(groups)//6)])
    dev=np.array([i for i in train if frame.iloc[i].similarity_group in dev_groups]);fit=np.setdiff1d(train,dev)
    if len(dev)<10:raise ValueError('Insufficient group-separated development cells')
    view=np.stack([clean,clean],axis=1)
    view[:,1]+=rng.normal(size=clean.shape)*clean[fit].std(0)*.03
    # New mesh probes on 24 validation objects: smooth boundary displacement
    # (0.5% and 2% of RMS radius), plus a proper random rotation, no label changes.
    qi=validation[np.linspace(0,len(validation)-1,24).round().astype(int)]
    cache=OUT/'real_noise_queries.npz'
    if cache.exists():queries=np.load(cache)['queries']
    else:
        queries=[]
        for k,i in enumerate(qi):
            row=frame.iloc[i].to_dict();v,f=mesh_for(row)
            center,_,_,_=__import__('spatial_benchmark_core').surface_frame(v,f);vn=v-center
            vn/=np.sqrt(np.mean(np.square(vn).sum(1)))
            normal=trimesh.Trimesh(vn,f,process=False).vertex_normals
            phase=rng.normal(size=(3,3))*5;angles=rng.uniform(0,6.28,3)
            perturb=np.sin(vn@phase.T+angles).mean(1)
            perturb/=max(np.std(perturb),1e-12)
            R=Rotation.random(random_state=rng).as_matrix();codes=[]
            for noise in [0,.005,.02]:
                vv=(vn+noise*perturb[:,None]*normal)@R
                c,qc,_,_=radial_coefficients(vv,f,2048,12)
                codes.append(np.r_[all_features(c),harmonics(vv,f)])
            queries.append(codes)
            print(f'Physical real-surface noise probes {k+1}/24',flush=True)
        queries=np.array(queries);np.savez_compressed(cache,queries=queries,query_indices=qi)
    qid=np.searchsorted(validation,qi)
    np.savez_compressed(OUT/'learning_real_data.npz',clean=clean,train=fit,dev=dev,validation=validation,query_indices=qi)
    return frame,clean,view,fit,dev,queries,clean[validation],qid


def matched_evaluation(frame):
    valid,pools,_=candidate_sets();real_ids={cid:i for i,cid in enumerate(frame[frame.split=='validation'].cell_id)}
    indices=np.array([real_ids[cid] for cid in valid.cell_id])
    z=np.load(OUT/'learning/real/gallery_codes.npz');codes={k:z[k][indices] for k in z.files}
    # General-map descriptors are evaluated on the same valid subset for all
    # methods, not against full-cohort scores from other methods.
    general=np.load(OUT/'general_features.npz');assert general['ids'].tolist()==valid.cell_id.tolist()
    codes['general_XYZ_power']=general['features']
    balanced=OUT/'general_balanced_features.npz'
    if balanced.exists():
        b=np.load(balanced);assert b['ids'].tolist()==valid.cell_id.tolist()
        codes['general_XYZ_power_balanced']=b['features']
    data=json.loads((ROOT/'results/explorer_dataset/dataset.json').read_text());lookup={c['id']:c for c in data['cells']}
    codes['frozen_shell_adjusted']=np.array([lookup[cid]['representations']['spherical_harmonics_adjusted']['vector'] for cid in valid.cell_id])
    codes['traditional_nine']=np.array([lookup[cid]['representations']['morph_all']['vector'] for cid in valid.cell_id])
    geometry={}
    for p in [ROOT/'results/atlas_spatial/development_neighbors.csv',ROOT/'results/harmonics_cajal/development_neighbors.csv',ROOT/'results/shapeembed3d/retrieval.csv']:
        old=pd.read_csv(p)
        if not {'cell_id','neighbor','geometry_cost'}<=set(old.columns):continue
        for row in old.itertuples():geometry[tuple(sorted([row.cell_id,row.neighbor]))]=float(row.geometry_cost)
    clouds={}
    def cost(i,j):
        a,b=valid.iloc[i].cell_id,valid.iloc[j].cell_id;key=tuple(sorted([a,b]))
        if key not in geometry:
            for index in [i,j]:
                cid=valid.iloc[index].cell_id
                if cid not in clouds:
                    v,f=mesh_for(valid.iloc[index].to_dict())
                    from pointnet_shape_core import sample_surface
                    clouds[cid]=sample_surface(v,f,384,np.random.default_rng(20261030+int(valid.iloc[index].original_index)))
            geometry[key]=float(geometry_cost(clouds[a],clouds[b]))
        return geometry[key]
    records=[]
    eligible=np.isfinite(codes['general_XYZ_power']).all(1)
    if 'general_XYZ_power_balanced' in codes:eligible &= np.isfinite(codes['general_XYZ_power_balanced']).all(1)
    for scope in ['full','general_valid_only']:
        for method,x in codes.items():
            if scope=='full' and method.startswith('general_XYZ_power'):continue
            distances=cdist(x,x)
            for i,pool in pools.items():
                ids=pool if scope=='full' else pool[eligible[pool]]
                if scope!='full' and (not eligible[i] or len(ids)<3):continue
                j=int(ids[np.argmin(distances[i,ids])])
                records.append(dict(scope=scope,method=method,query=valid.iloc[i].cell_id,neighbor=valid.iloc[j].cell_id,group=int(valid.iloc[i].similarity_group),geometry_cost=cost(i,j)))
    pd.DataFrame(records).to_csv(OUT/'matched_retrieval.csv',index=False)
    print('Matched geometric-neighbor evaluation complete',flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    p=json.loads((OUT/'protocol.json').read_text())
    p['learning']['input']='489D: 13 radial powers + 60 L8 bispectrum components + 416 shell powers; training std floor 1e-6'
    p['learning']['development']='Synthetic 64 shape IDs; real original training groups held out for development reconstruction, all 140 original validation cells excluded from training/selection'
    p['learning']['real_noise']='24 original validation cells, proper random rotation plus smooth normal displacement with standard deviation 0, .005, .02 RMS radius'
    save_json(OUT/'protocol.json',p)
    a=synthetic_data();clean=a['clean'];views=a['views']
    fit_domain('synthetic',clean,views,np.arange(384),np.arange(384,448),views[448:],clean[448:],np.arange(128))
    frame,clean,views,train,dev,queries,gallery,qid=real_data()
    fit_domain('real',clean,views,train,dev,queries,gallery,qid)
    matched_evaluation(frame)
    save_json(OUT/'learning.complete.json',dict(status='complete',seeds=[42,43,44],synthetic_test=128,real_noise_objects=24))
if __name__=='__main__':main()
