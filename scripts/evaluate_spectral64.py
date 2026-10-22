"""Frozen-model test readouts. Test metrics never select a model or seed."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import time
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from skimage.measure import marching_cubes
from spectral64_core import *
from train_spectral64 import Model,inputs
from spatial_benchmark_core import morphology,CONTROLS,mesh_for,geometry_cost
from evaluate_spatial_benchmark import residual_fit,residual_apply
from pointnet_shape_core import sample_surface

torch.set_num_threads(2)

def load_model(kind,seed):
    checkpoint=torch.load(OUT/kind/f'seed{seed}.pt',map_location='cpu',weights_only=True);model=Model();model.load_state_dict(checkpoint['state']);model.eval();return model

def encode(kind,source,seed):
    scale=np.load(OUT/kind/'scaler.npz');x=np.zeros((len(source),DIM),np.float32)
    if kind=='A':x[:,:416]=(powers(source)-scale['mean'])/scale['scale']
    else:x=(weighted(source)-scale['mean'])/scale['scale']
    with torch.no_grad():return load_model(kind,seed).encode(torch.tensor(x,dtype=torch.float32)).numpy()

def group_ci(values,groups):
    f=pd.DataFrame(dict(x=values,g=groups));x=f.groupby('g').x.mean().to_numpy();rng=np.random.default_rng(42)
    return dict(mean=float(x.mean()),ci95=np.percentile(rng.choice(x,(2000,len(x)),replace=True).mean(1),[2.5,97.5]).tolist(),groups=len(x))

def prepare_grid(res=40):
    axis=np.linspace(-2.6,2.6,res);grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    r=np.linalg.norm(grid,axis=1);d=grid/np.maximum(r[:,None],1e-12);Y=basis(d,12)
    high=np.searchsorted(RADII,r).clip(1,31);low=high-1;weight=((r-RADII[low])/(RADII[high]-RADII[low])).clip(0,1)
    return axis,grid,Y,low,high,weight,r>RADII[-1]

def grid_field(c,config):
    axis,grid,Y,lo,hi,w,outside=config;c=unpack(c)
    v=np.sum(Y*(c[lo]*(1-w[:,None])+c[hi]*w[:,None]),axis=1).real;v[outside]=0
    return v.reshape((len(axis),)*3)

def surface_metrics(values,v,f,config):
    if values.min()>=.5 or values.max()<=.5:return dict(surface_valid=False,chamfer_squared=None,p95=None)
    axis=config[0];rv,rf,_,_=marching_cubes(values,.5,spacing=(axis[1]-axis[0],)*3);rv+=axis[0]
    target,_=trimesh.sample.sample_surface(trimesh.Trimesh(v,f,process=False),2048,seed=42)
    pred,_=trimesh.sample.sample_surface(trimesh.Trimesh(rv,rf,process=False),2048,seed=42)
    a=igl.point_mesh_squared_distance(target,rv,rf)[0];b=igl.point_mesh_squared_distance(pred,v,f)[0]
    return dict(surface_valid=True,chamfer_squared=float(a.mean()+b.mean()),p95=float(np.percentile(np.sqrt(np.r_[a,b]),95)))

def evaluate():
    entries=json.loads((OUT/'entries.json').read_text());raw=np.load(OUT/'coefficients.npz')['coefficients'];frame=pd.read_csv(OUT/'manifest.csv');nreal=len(frame)
    kinds=['A','B'];gate=json.loads((OUT/'B_information_gate.json').read_text())
    if gate['passed']:
        assert (OUT/'C/training.complete.json').exists(),'C must be frozen before opening final test metrics';kinds.append('C')
    frozen={f'{k}_seed{s}':sha(OUT/k/f'seed{s}.pt') for k in kinds for s in [42,43,44]}
    save_json(OUT/'test_freeze.json',dict(checkpoint_sha256=frozen,test_opening='After all planned checkpoints frozen; no subsequent selection on test',C_gate=gate['passed']))
    train_idx=np.flatnonzero(frame.split.to_numpy()=='train');test_idx=np.flatnonzero(frame.split.to_numpy()=='test');test=frame.iloc[test_idx].reset_index(drop=True)
    noise=np.load(OUT/'test_noise.npz');assert noise['ids'].tolist()==test.cell_id.tolist()
    codes={'shell_power':powers(raw[test_idx])};noise_codes={'shell_power':powers(noise['coefficients'])};rotation_codes={'shell_power':powers(noise['rotated_coefficients'])}
    # Both direct and nuisance-adjusted fixed historical reference are recomputed
    # on precisely the new test cells. Their old validation score is not reused.
    atlas=json.loads((ROOT/'results/explorer_dataset/dataset.json').read_text());lookup={c['id']:c for c in atlas['cells']}
    for key,name in [('morph_all','traditional_nine_frozen'),('spherical_harmonics_adjusted','shell_adjusted_frozen')]:codes[name]=np.array([lookup[cid]['representations'][key]['vector'] for cid in test.cell_id])
    fit=dict(np.load(ROOT/'results/harmonics_cajal/harmonics_adjustment.npz'))
    noise_codes['shell_adjusted_frozen']=residual_apply(test,powers(noise['coefficients']),fit)
    rotation_codes['shell_adjusted_frozen']=residual_apply(test,powers(noise['rotated_coefficients']),fit)
    # Query nuisance controls remain the original values: isolates representation
    # drift under surface noise, not a full remeasurement/correction workflow.
    train_all=np.array([i for i,e in enumerate(entries) if e['split']=='train']);X=weighted(raw);pca=PCA(n_components=64,svd_solver='randomized',random_state=42).fit(X[train_all]);codes['spectral_PCA64']=pca.transform(X[test_idx]);noise_codes['spectral_PCA64']=pca.transform(weighted(noise['coefficients']));rotation_codes['spectral_PCA64']=pca.transform(weighted(noise['rotated_coefficients']))
    np.savez_compressed(OUT/'PCA64.npz',mean=pca.mean_,components=pca.components_)
    for kind in kinds:
        for seed in [42,43,44]:
            key=f'{kind}_seed{seed}';z=np.load(OUT/kind/f'seed{seed}_readouts.npz');codes[key]=z['codes'][test_idx];noise_codes[key]=encode(kind,noise['coefficients'],seed);rotation_codes[key]=encode(kind,noise['rotated_coefficients'],seed)
    raw_rotated=None;raw_rotated_second=None
    if gate['passed']:
        rotation_data=np.load(OUT/'rotated_training_views.npz')
        raw_rotated=rotation_data['coefficients'][test_idx,0]
        raw_rotated_second=rotation_data['coefficients'][test_idx,1]
        raw_noisy=np.load(OUT/'test_raw_noise.npz')['coefficients']
        for seed in [42,43,44]:
            key=f'C_unaligned_seed{seed}'
            codes[key]=encode('C',raw_rotated,seed)
            noise_codes[key]=encode('C',raw_noisy,seed)
            rotation_codes[key]=encode('C',raw_rotated_second,seed)
    # Predeclared secondary nuisance-adjusted variants, fit only on original
    # training biological objects (no synthetic controls, validation or test).
    shell_fit=residual_fit(frame.iloc[train_idx],powers(raw[train_idx]))
    codes['shell_adjusted_train469']=residual_apply(test,powers(raw[test_idx]),shell_fit)
    noise_codes['shell_adjusted_train469']=residual_apply(test,powers(noise['coefficients']),shell_fit)
    rotation_codes['shell_adjusted_train469']=residual_apply(test,powers(noise['rotated_coefficients']),shell_fit)
    fitted_nuisance={}
    for kind in kinds:
        for seed in [42,43,44]:
            key=f'{kind}_seed{seed}';all_codes=np.load(OUT/kind/f'seed{seed}_readouts.npz')['codes']
            nf=residual_fit(frame.iloc[train_idx],all_codes[train_idx]);fitted_nuisance[key]=nf
            codes[key+'_adjusted']=residual_apply(test,codes[key],nf)
            noise_codes[key+'_adjusted']=residual_apply(test,noise_codes[key],nf)
            rotation_codes[key+'_adjusted']=residual_apply(test,rotation_codes[key],nf)
    # Same matched pools for every method, no baseline-specific omissions.
    controls=morphology(frame.iloc[train_idx],CONTROLS);c=(morphology(test,CONTROLS)-controls.mean(0))/controls.std(0).clip(1e-6);d=cdist(c,c);groups=test.similarity_group.to_numpy();pools={}
    for i in range(len(test)):
        allowed=np.flatnonzero((groups!=groups[i])&(d[i]<=1))
        if len(allowed)>=5:pools[i]=allowed[np.argsort(d[i,allowed])[:20]]
    selected=[];geometry={};clouds={}
    for method,z in codes.items():
        distance=cdist(z,z)
        for i,allowed in pools.items():
            j=int(allowed[np.argmin(distance[i,allowed])]);pair=tuple(sorted((i,j)))
            if pair not in geometry:
                for ix in pair:
                    if ix not in clouds:clouds[ix]=sample_surface(*mesh_for(test.iloc[ix].to_dict()),384,np.random.default_rng(20261030+int(test.iloc[ix].original_index)))
                geometry[pair]=float(geometry_cost(clouds[pair[0]],clouds[pair[1]]))
            selected.append(dict(method=method,query=test.iloc[i].cell_id,neighbor=test.iloc[j].cell_id,group=int(groups[i]),geometry_cost=geometry[pair]))
    selected=pd.DataFrame(selected);selected.to_csv(OUT/'test_neighbors.csv',index=False)
    unaligned={}
    if raw_rotated is not None:
        unaligned['shell_power']=powers(raw_rotated)
        unaligned['shell_adjusted_frozen']=residual_apply(test,powers(raw_rotated),fit)
        unaligned['spectral_PCA64']=pca.transform(weighted(raw_rotated))
        for kind in kinds:
            for seed in [42,43,44]:unaligned[f'{kind}_seed{seed}']=encode(kind,raw_rotated,seed)
    for key,nf in fitted_nuisance.items():
        kind,seed=key.split('_seed')
        unaligned[key+'_adjusted']=residual_apply(test,encode(kind,raw_rotated,int(seed)),nf) if raw_rotated is not None else None
    if raw_rotated is not None:unaligned['shell_adjusted_train469']=residual_apply(test,powers(raw_rotated),shell_fit)
    raw_noise=np.load(OUT/'test_raw_noise.npz')['coefficients']
    raw_noise_codes={'shell_power':powers(raw_noise),'shell_adjusted_frozen':residual_apply(test,powers(raw_noise),fit),'spectral_PCA64':pca.transform(weighted(raw_noise))}
    for kind in kinds:
        for seed in [42,43,44]:raw_noise_codes[f'{kind}_seed{seed}']=encode(kind,raw_noise,seed)
    raw_noise_codes['shell_adjusted_train469']=residual_apply(test,powers(raw_noise),shell_fit)
    for key,nf in fitted_nuisance.items():
        kind,seed=key.split('_seed');raw_noise_codes[key+'_adjusted']=residual_apply(test,encode(kind,raw_noise,int(seed)),nf)
    for key in list(codes):
        if key.startswith('C_unaligned_'):
            unaligned[key]=rotation_codes[key];raw_noise_codes[key]=noise_codes[key]
    stability=[]
    for method,z in noise_codes.items():
        if method not in codes:continue
        pred=cdist(z,codes[method]).argmin(1);rot=cdist(rotation_codes[method],codes[method]).argmin(1)
        una=cdist(unaligned[method],codes[method]).argmin(1) if method in unaligned else None
        combined=cdist(raw_noise_codes[method],codes[method]).argmin(1)
        for i in range(len(test)):stability.append(dict(method=method,cell_id=test.iloc[i].cell_id,group=int(groups[i]),noise_correct=int(pred[i]==i),rotation_correct=int(rot[i]==i),unaligned_rotation_correct=int(una[i]==i) if una is not None else None,unaligned_noise_correct=int(combined[i]==i)))
    pd.DataFrame(stability).to_csv(OUT/'test_stability.csv',index=False)
    np.savez_compressed(OUT/'test_codes.npz',ids=test.cell_id.to_numpy(dtype=str),**codes)
    # Keep both old radial-equal-power challenge and stricter mirror challenge.
    challenges=[]
    for scope,ids in [('synthetic',[i for i,e in enumerate(entries) if e['kind']=='synthetic' and e['split']=='test']),('mirror',[i for i,e in enumerate(entries) if e['kind']=='mirror' and e['split']=='test'])]:
        for method in ['shell_power','spectral_PCA64']+[f'{k}_seed{s}' for k in kinds for s in [42,43,44]]:
            if method=='shell_power':z=powers(raw[ids])
            elif method=='spectral_PCA64':z=pca.transform(X[ids])
            else:
                k,s=method.split('_seed');z=np.load(OUT/k/f'seed{s}_readouts.npz')['codes'][ids]
            if scope=='mirror':
                distance=cdist(z,z);pair=np.array([distance[i,i^1] for i in range(len(ids))]);off=distance.copy()
                for i in range(len(ids)):off[i,[i,i^1]]=np.inf
                median=np.median(off[np.isfinite(off)]);fraction=float(np.mean(pair>.01*median))
                target=X[ids];recovery=None
                if method=='spectral_PCA64':
                    pred=pca.inverse_transform(z);delta=target[::2]-target[1::2];pdiff=pred[::2]-pred[1::2];recovery=float(1-np.square(delta-pdiff).sum()/np.square(delta).sum())
                elif method.startswith(('B_','C_')):
                    k,s=method.split('_seed');sc=np.load(OUT/k/'scaler.npz');pred=np.load(OUT/k/f'seed{s}_readouts.npz')['prediction'][ids]*sc['scale']+sc['mean'];delta=target[::2]-target[1::2];pdiff=pred[::2]-pred[1::2];recovery=float(1-np.square(delta-pdiff).sum()/np.square(delta).sum())
                challenges.append(dict(scope=scope,method=method,pairs=len(ids)//2,mirror_separated_fraction=fraction,phase_difference_R2=recovery))
    physical=np.load(OUT/'physical_challenges.npz')
    physical_rows=[]
    for scope in ['legacy','mirror']:
        mask=physical['kind']==scope;g=physical['gallery'][mask];q=physical['query'][mask];pair=physical['pair'][mask]
        alternatives={'shell_power':(powers(g),powers(q)),'spectral_PCA64':(pca.transform(weighted(g)),pca.transform(weighted(q)))}
        for kind in kinds:
            for seed in [42,43,44]:alternatives[f'{kind}_seed{seed}']=(encode(kind,g,seed),encode(kind,q,seed))
        for method,(a,b) in alternatives.items():
            distances=cdist(b,a);prediction=distances.argmin(1)
            for i in range(len(g)):
                same=distances[i,i];mirror=distances[i,i^1]
                # Exact equal-distance ties receive half-credit rather than an
                # arbitrary identity from row ordering or floating-point noise.
                pair_accuracy=.5 if abs(same-mirror)<1e-8*max(1,abs(same),abs(mirror)) else float(same<mirror)
                physical_rows.append(dict(scope=scope,method=method,query=i,pair=int(pair[i]),identity_correct=int(prediction[i]==i),within_pair_correct=pair_accuracy))
    pd.DataFrame(physical_rows).to_csv(OUT/'physical_challenge_retrieval.csv',index=False)
    save_json(OUT/'test_challenges.json',challenges)
    # Full-field reconstruction is available only for full-coefficient models.
    decoded_test={}
    for kind in kinds:
        if kind=='A':continue
        scale=np.load(OUT/kind/'scaler.npz')
        for seed in [42,43,44]:decoded_test[f'{kind}_seed{seed}']=unweighted(np.load(OUT/kind/f'seed{seed}_readouts.npz')['prediction'][test_idx]*scale['scale']+scale['mean'])
    if gate['passed']:
        sc=np.load(OUT/'C/scaler.npz')
        for seed in [42,43,44]:
            model=load_model('C',seed)
            with torch.no_grad():prediction=model.decoder(torch.tensor(codes[f'C_unaligned_seed{seed}'],dtype=torch.float32)).numpy()
            decoded_test[f'C_unaligned_seed{seed}']=unweighted(prediction*sc['scale']+sc['mean'])
    config=prepare_grid();reconstruction=[];(OUT/'reconstructions').mkdir(exist_ok=True)
    hf=np.concatenate([np.arange(l*l,(l+1)**2) for l in range(9,13)])
    for q,i in enumerate(test_idx):
        path=OUT/'reconstructions'/f'{q:03d}.json'
        if path.exists():reconstruction.extend(json.loads(path.read_text()));continue
        v,f=mesh_for(frame.iloc[i].to_dict());v,_=canonical(v,f);mesh=trimesh.Trimesh(v,f,process=False);closed=bool(mesh.is_watertight and mesh.is_winding_consistent)
        sdf=igl.signed_distance(config[1],v,f,igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0].reshape((40,)*3)
        gt=sdf<=0;boundary=abs(sdf)<.15
        actual=raw[i];actual_hf=np.zeros_like(actual);actual_hf[:,hf]=actual[:,hf];target_boundary_hf=grid_field(actual_hf,config)[boundary]
        methods={'coefficient_ceiling':actual,'spectral_PCA64':unweighted(pca.inverse_transform(pca.transform(X[i:i+1])))[0]}
        methods.update({key:value[q] for key,value in decoded_test.items()})
        results=[]
        for method,coef in methods.items():
            values=grid_field(coef,config);mask=values>=.5;union=(gt|mask).sum();iou=float((gt&mask).sum()/max(union,1)) if closed else None
            metric=surface_metrics(values,v,f,config)
            detail=float(1-np.square((coef-actual)[:,hf]).sum()/max(np.square(actual[:,hf]).sum(),1e-15))
            local_hf=np.zeros_like(coef);local_hf[:,hf]=coef[:,hf];pred_boundary_hf=grid_field(local_hf,config)[boundary]
            boundary_detail=float(1-np.square(pred_boundary_hf-target_boundary_hf).sum()/max(np.square(target_boundary_hf).sum(),1e-15))
            result=dict(method=method,cell_id=frame.iloc[i].cell_id,group=int(frame.iloc[i].similarity_group),volume_eligible=closed,volumetric_IoU=iou,high_degree9_12_R2=detail,boundary_detail9_12_R2=boundary_detail,**metric);results.append(result)
            if q in [0,5,10]:np.savez_compressed(OUT/'reconstructions'/f'{q:03d}_{method}.npz',field=values,vertices=v,faces=f)
        save_json(path,results);reconstruction.extend(results)
        if q%15==0:print(f'Held-out volumetric/surface/detail reconstruction {q+1}/{len(test_idx)}',flush=True)
    pd.DataFrame(reconstruction).to_csv(OUT/'test_reconstruction.csv',index=False)
    save_json(OUT/'evaluation.complete.json',dict(test_cells=len(test),organoids=test['sample'].nunique(),groups=test.similarity_group.nunique(),matched_queries=len(pools),models=kinds,weights_frozen=frozen))
    print('Held-out evaluation complete',flush=True)
if __name__=='__main__':evaluate()
