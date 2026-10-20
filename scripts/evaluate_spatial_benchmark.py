"""Development-led pose and geometry benchmark with a guarded held-out stage."""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from pointnet_shape_core import ROOT,select_device,save_json,sha256,sample_surface
from prepare_spatial_benchmark import OUT
from spatial_benchmark_core import mesh_for,surface_frame,sampled_frame,sdf_grid,orbit_grids,orbit_distance,summary_features,morphology,CONTROLS,CLASSICAL
from spatial_sdf import SpatialSDF
from evaluate_spatial_sdf import GridDecoder
from sdf_shape_core import extract_surface
from evaluate_sdf_shape import surface_metrics,group_ci


def probe_one(item):
    index,row,folder=item;path=OUT/folder/f'{index:04d}.npz'
    if path.exists():return index,str(path)
    v,f=mesh_for(row);raw=[];aligned=[];stress=[];rng=np.random.default_rng(20261029+index)
    for trial in range(8):
        r=np.linalg.qr(rng.normal(size=(3,3)))[0]
        if np.linalg.det(r)<0:r[:,-1]*=-1
        rotated=v@r.T;frame=surface_frame(rotated,f)
        raw.append(sdf_grid(rotated,f));aligned.append(orbit_grids(rotated,f,frame))
        if trial==0:r0=r;center0=frame[0];axes0=frame[1]
    for trial in range(2):
        p=sample_surface(v,f,2048,rng);stress.append(orbit_grids(v,f,sampled_frame(p)))
    frame=surface_frame(v,f)
    np.savez(path,raw_base=sdf_grid(v,f),aligned_base=orbit_grids(v,f,frame),raw_rotation=np.asarray(raw),
        aligned_rotation=np.asarray(aligned),stress=np.asarray(stress),r0=r0,rot0_center=center0,rot0_axes=axes0,
        center=frame[0],axes=frame[1],axis_gap=frame[3])
    return index,str(path)


class Encoder:
    def __init__(self,kind,aligned,device):
        self.kind=kind;self.aligned=aligned;self.device=device
        if kind=='pca':
            self.path=OUT/'aligned_pca.npz' if aligned else ROOT/'results/spatial_sdf/pca95_encoder.npz'
            saved=np.load(self.path);self.mean=torch.tensor(saved['mean'],device=device);self.components=torch.tensor(saved['components'],device=device)
        else:
            self.path=OUT/'aligned_cnn.pt' if aligned else ROOT/'results/spatial_sdf/boundary/best.pt'
            self.model=SpatialSDF().to(device);self.model.load_state_dict(torch.load(self.path,map_location='cpu',weights_only=False)['model']);self.model.eval()

    def encode(self,grids):
        values=[]
        with torch.no_grad():
            for first in range(0,len(grids),16):
                x=torch.tensor(np.array(grids[first:first+16]),device=self.device)
                z=(x.flatten(1)-self.mean)@self.components.T if self.kind=='pca' else self.model.encode(x[:,None])
                values.append(z.cpu().numpy())
        return np.concatenate(values)

    def surface(self,code):
        if self.kind=='pca':
            with torch.no_grad():grid=(torch.tensor(code,device=self.device)@self.components+self.mean).cpu().reshape(1,32,32,32)
            return extract_surface(GridDecoder(),grid)
        return extract_surface(self.model,torch.tensor(code[None],device=self.device))


def accuracy(a,b,groups):
    distances=orbit_distance(a,b);identity=float((distances.argmin(1)==np.arange(len(a))).mean())
    base=orbit_distance(b,b);jaccard=[]
    for i in range(len(a)):
        permitted=np.flatnonzero(groups!=groups[i]);k=min(5,len(permitted))
        before=set(permitted[np.argsort(base[i,permitted])[:k]]);after=set(permitted[np.argsort(distances[i,permitted])[:k]])
        jaccard.append(len(before&after)/max(len(before|after),1))
    return identity,float(np.mean(jaccard))


def residual_fit(train_frame,train_features):
    x=morphology(train_frame,CONTROLS);mean=x.mean(0);std=x.std(0).clip(1e-6)
    poly=PolynomialFeatures(2,include_bias=False);features=poly.fit_transform((x-mean)/std)
    ridge=Ridge(alpha=1).fit(features,train_features)
    return dict(mean=mean,std=std,powers=poly.powers_,coef=ridge.coef_,intercept=ridge.intercept_)


def residual_apply(frame,features,fit):
    x=(morphology(frame,CONTROLS)-fit['mean'])/fit['std']
    polynomial=np.prod(x[:,None,:]**fit['powers'][None],axis=-1)
    return features-(polynomial@fit['coef'].T+fit['intercept'])


def retrieval(frame,train_frame,codes,train_codes,out,fits=None):
    x=morphology(frame,CONTROLS);xt=morphology(train_frame,CONTROLS);control=(x-xt.mean(0))/xt.std(0).clip(1e-6)
    classical=morphology(frame,CLASSICAL);ct=morphology(train_frame,CLASSICAL);classical=(classical-ct.mean(0))/ct.std(0).clip(1e-6)
    distances={'three_measures':cdist(control,control),'nine_measures':cdist(classical,classical)};fits={} if fits is None else fits
    summaries={};residuals={}
    for kind,z in codes.items():
        distances[kind]=orbit_distance(z,z)
        if kind not in fits:fits[kind]=residual_fit(train_frame,summary_features(train_codes[kind]))
        summary=summary_features(z);residual=residual_apply(frame,summary,fits[kind]);residuals[kind]=residual
        distances[kind+'_residual']=cdist(residual,residual)
        np.savez(out/(kind+'_nuisance_fit.npz'),**fits[kind])
    pairs=set();candidates={};groups=frame.similarity_group.to_numpy()
    for i in range(len(frame)):
        ids=np.flatnonzero((groups!=groups[i])&(distances['three_measures'][i]<=1))
        if len(ids)<5:continue
        ids=ids[np.argsort(distances['three_measures'][i,ids])[:20]];candidates[i]=ids
        for j in ids:pairs.add(tuple(sorted([i,int(j)])))
    from spatial_benchmark_core import geometry_cost
    clouds=[]
    for index,row in frame.iterrows():
        v,f=mesh_for(row);clouds.append(sample_surface(v,f,384,np.random.default_rng(20261030+int(row.original_index))))
    costs={}
    for n,(i,j) in enumerate(sorted(pairs)):
        costs[(i,j)]=geometry_cost(clouds[i],clouds[j])
    rows=[]
    for i,ids in candidates.items():
        for name,distance in distances.items():
            chosen=int(ids[np.argmin(distance[i,ids])]);cost=costs[tuple(sorted([i,chosen]))]
            rows.append(dict(method=name,query=i,cell_id=frame.iloc[i].cell_id,similarity_group=int(groups[i]),
                neighbor=chosen,neighbor_cell=frame.iloc[chosen].cell_id,candidates=len(ids),geometry_cost=cost,
                random_candidate_cost=float(np.mean([costs[tuple(sorted([i,int(j)]))] for j in ids]))))
    records=pd.DataFrame(rows);records.to_csv(out/'retrieval.csv',index=False)
    if len(records):
        wide=records.pivot(index=['query','similarity_group'],columns='method',values='geometry_cost').reset_index()
        for method in distances:
            if method in ['three_measures','nine_measures']:continue
            summaries[method]={}
            for baseline in ['three_measures','nine_measures']:
                wide['improvement']=wide[baseline]-wide[method]
                summaries[method][baseline]=group_ci(wide,'improvement')
    diagnostics={}
    for kind,residual in residuals.items():
        active=residual.std(0)>1e-7
        correlations=np.corrcoef(residual[:,active].T,control.T)[:-3,-3:] if active.any() else np.zeros((1,3))
        diagnostics[kind]=dict(max_residual_control_correlation=float(np.nanmax(np.abs(correlations))),
            mean_residual_control_correlation=float(np.nanmean(np.abs(correlations))))
    return dict(queries=len(candidates),groups=len(set(groups[list(candidates)])) if candidates else 0,
        methods=summaries,residual_diagnostics=diagnostics,geometry_pairs=len(pairs)),fits


def evaluate(stage):
    if stage=='test' and not (OUT/'freeze.json').exists():raise RuntimeError('Test is sealed until development selection is frozen')
    torch.set_num_threads(4);device=select_device('mps')
    lock=json.loads((OUT/'freeze.json').read_text()) if stage=='test' else None
    if lock:
        if sha256(ROOT/'results/pointnet_shape/data/manifest.csv')!=lock['test_manifest_sha256']:raise ValueError('Frozen test manifest changed')
        for name,digest in lock['sources'].items():
            if sha256(ROOT/'scripts'/name)!=digest:raise ValueError('Frozen benchmark source changed')
        for name,digest in lock['artifacts'].items():
            if sha256(OUT/name)!=digest:raise ValueError('Frozen encoder/nuisance fit changed')
    dev=pd.read_csv(OUT/'dev_manifest.csv');train_frame=dev[dev.split=='train'].reset_index(drop=True)
    if stage=='development':frame=dev[dev.split=='validation'].reset_index(drop=True);kinds=['pca','cnn']
    else:
        original=pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv').reset_index(names='original_index')
        frame=original[original.split.eq('test')&original.training_eligible].reset_index(drop=True);kinds=[lock['selected']]
        if frame.cell_id.tolist()!=lock['test_cell_ids']:raise ValueError('Frozen test cohort changed')
    out=OUT/stage;out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Completed evaluation exists; do not reopen test for tuning')
    # Record test opening before reading its mesh files; do not permit silent restarts.
    if stage=='test':
        if (out/'opened.json').exists():raise FileExistsError('Test already opened; recover this run explicitly, never silently restart')
        save_json(out/'opened.json',dict(freeze_sha256=sha256(OUT/'freeze.json'),cells=len(frame)))
    folder=stage+'_probes';(OUT/folder).mkdir(exist_ok=True);paths=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        items=[(int(row.original_index),row.to_dict(),folder) for _,row in frame.iterrows()]
        for index,path in pool.map(probe_one,items):paths.append(path);print(f'PROBED {len(paths)}/{len(frame)}',flush=True)
    bundles={(kind,aligned):Encoder(kind,aligned,device) for kind in kinds for aligned in [False,True]}
    codes={};train_codes={};features={};pose={};rows=[];gaps=[]
    for kind in kinds:
        train_grids=np.load(OUT/'dev_grids.npy',mmap_mode='r')[dev.split=='train']
        train_codes[kind]=bundles[kind,True].encode(train_grids.reshape(-1,32,32,32)).reshape(len(train_frame),4,-1)
        for aligned in [False,True]:
            bundle=bundles[kind,aligned];base=[];rotations=[];stress=[]
            for path in paths:
                with np.load(path) as p:
                    fields=p['aligned_base'] if aligned else p['raw_base'][None]
                    base.append(bundle.encode(fields));rotations.append(bundle.encode(p['aligned_rotation'].reshape(-1,32,32,32) if aligned else p['raw_rotation']).reshape(8,4 if aligned else 1,-1))
                    if aligned:stress.append(bundle.encode(p['stress'].reshape(-1,32,32,32)).reshape(2,4,-1))
            base=np.asarray(base);rotations=np.asarray(rotations);name=kind+('_aligned' if aligned else '_raw')
            features[name]=dict(base=base,rotated=rotations)
            if aligned:features[name]['stress']=np.asarray(stress);codes[kind]=base
            groups=frame.similarity_group.to_numpy();gap=np.array([float(np.load(path)['axis_gap']) for path in paths]);gaps=gap
            trials=[accuracy(rotations[:,trial],base,groups) for trial in range(8)]
            pose[name]=dict(rotation_identity=float(np.mean(trials,axis=0)[0]),rotation_neighbor_jaccard=float(np.mean(trials,axis=0)[1]),
                rotation_identity_per_trial=[r[0] for r in trials])
            if aligned:
                trials=[accuracy(np.asarray(stress)[:,trial],base,groups) for trial in range(2)]
                pose[name].update(sampled_frame_identity=float(np.mean(trials,axis=0)[0]),sampled_frame_neighbor_jaccard=float(np.mean(trials,axis=0)[1]),
                    single_frame_rotation_identity=float(np.mean([accuracy(rotations[:,trial,:1],base[:,:1],groups)[0] for trial in range(8)])))
                for label,mask in [('near_degenerate',gap<.05),('other',gap>=.05),('nearly_spherical',frame.sphericity.to_numpy()>=.9)]:
                    if mask.any():
                        # Full gallery, subgroup queries only.
                        rate=[]
                        for trial in range(2):
                            d=orbit_distance(np.asarray(stress)[:,trial],base)
                            rate.append(float((d.argmin(1)[mask]==np.arange(len(base))[mask]).mean()))
                        pose[name][label]=dict(cells=int(mask.sum()),sampled_frame_identity=float(np.mean(rate)))
    frame['axis_gap']=gaps
    # Reconstruction before and after one predetermined rotation, using frame 0.
    for position,row in frame.iterrows():
        v,f=mesh_for(row);reference=sample_surface(v,f,2048,np.random.default_rng(20261023+int(row.original_index)))
        with np.load(paths[position]) as p:
            for kind in kinds:
                for aligned in [False,True]:
                    name=kind+('_aligned' if aligned else '_raw');bundle=bundles[kind,aligned]
                    for rotated in [False,True]:
                        code=features[name]['rotated'][position,0,0] if rotated else features[name]['base'][position,0]
                        surface,field=bundle.surface(code)
                        record=dict(cell_id=row.cell_id,similarity_group=int(row.similarity_group),method=name,rotated=rotated)
                        if surface is None:rows.append(dict(**record,failed_surface=True));continue
                        vertices,faces=surface
                        if aligned:vertices=vertices@(p['rot0_axes'] if rotated else p['axes']).T+(p['rot0_center'] if rotated else p['center'])
                        if rotated:vertices=vertices@p['r0']
                        prediction=sample_surface(vertices,faces,2048,np.random.default_rng(20262023+int(row.original_index)))
                        mesh=__import__('trimesh').Trimesh(vertices,faces,process=False)
                        true=__import__('trimesh').Trimesh(v,f,process=False)
                        rows.append(dict(**record,failed_surface=False,**surface_metrics(reference,prediction,row.normalization_radius_um),watertight=bool(mesh.is_watertight),volume_error=float(abs(abs(mesh.volume)-abs(true.volume))/abs(true.volume))))
        print(f'GEOMETRY {position+1}/{len(frame)}',flush=True)
    results=pd.DataFrame(rows);results.to_csv(out/'reconstruction.csv',index=False)
    reconstruction={}
    for (method,rotated),sub in results.groupby(['method','rotated']):
        reconstruction[method+('_rotated' if rotated else '')]=dict(cells=len(sub),failed_surfaces=int(sub.failed_surface.sum()),nonwatertight=int((sub.watertight==False).sum()),
            **{c:float(sub[c].mean()) for c in ['squared_chamfer','f1_at_005_radius','volume_error']})
    fits={kind:dict(np.load(OUT/lock['nuisance_files'][kind])) for kind in kinds} if lock else None
    retrieval_metrics,fits=retrieval(frame,train_frame,codes,train_codes,out,fits)
    frame.to_csv(out/'manifest.csv',index=False)
    np.savez_compressed(out/'codes.npz',**{name+'_'+key:values for name,entries in features.items() for key,values in entries.items()})
    save_json(out/'metrics.json',dict(pose=pose,reconstruction=reconstruction,retrieval=retrieval_metrics,
        cells=len(frame),groups=int(frame.similarity_group.nunique()),test_used=stage=='test',
        atlas_unchanged=sha256(ROOT/'results/explorer_dataset/dataset.json')==json.loads((OUT/'dev.json').read_text())['atlas_sha256']))
    print(json.dumps(dict(pose=pose,reconstruction=reconstruction,retrieval=retrieval_metrics)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['development','test'],required=True);evaluate(parser.parse_args().stage)
