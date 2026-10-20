"""Standalone released-model and partial-adaptation comparison; no Atlas writes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch
import trimesh
from allen_reference_core import OUT
from train_allen_reference import CachedAllen
from pointnet_shape_core import ROOT,sha256,sample_surface
from spatial_benchmark_core import mesh_for,surface_frame,geometry_cost
from sdf_shape_core import extract_surface
from evaluate_sdf_shape import surface_metrics,group_ci
from evaluate_spatial_benchmark import residual_fit,residual_apply,Encoder
from check_cajal_validation import candidate_sets
from atlas_shape_space import distances

ARMS=['pretrained','decoder_only','head_decoder']


def load_arm(name,device='cpu'):
    m=CachedAllen()
    if name!='pretrained':
        ckpt=torch.load(OUT/name/'best.pt',map_location='cpu',weights_only=False)
        if ckpt['manifest_sha256']!=sha256(OUT/'manifest.csv'):raise ValueError('Checkpoint/data mismatch')
        m.load_state_dict(ckpt['model'],strict=True)
    return m.to(device).eval()


def load_features(frame,view=0,indices=None):
    indices=range(len(frame)) if indices is None else indices
    return np.stack([np.load(OUT/'features'/f'{i:04d}_{view}.npz')['features'] for i in indices])


def embeddings():
    frame=pd.read_csv(OUT/'manifest.csv');features=load_features(frame)
    dev,pools,_=candidate_sets();positions={cid:i for i,cid in enumerate(frame.cell_id)};ids=np.array([positions[c] for c in dev.cell_id])
    rotation_ids=[i for i in ids if (OUT/'features'/f'{i:04d}_2.npz').exists()]
    secondary=load_features(frame,view=1,indices=ids);rotated=load_features(frame,view=2,indices=rotation_ids)
    all_codes={};stability={};fits={};readouts={}
    original96=set(pd.read_csv(ROOT/'results/spatial_benchmark/dev_manifest.csv').query("split=='train'").cell_id)
    train96=np.flatnonzero(frame.cell_id.isin(original96));train_frame=frame.iloc[train96];groups=dev.similarity_group.to_numpy()
    for name in ARMS:
        m=load_arm(name)
        with torch.no_grad():
            z=m.encode(torch.tensor(features));r=m.encode(torch.tensor(secondary));q=m.encode(torch.tensor(rotated))
        norms=z.norm(dim=-1).numpy().astype(float);rs=r.norm(dim=-1).numpy().astype(float);rr=q.norm(dim=-1).numpy().astype(float)
        all_codes[name]=z.numpy();all_codes[name+'_norms']=norms
        center=norms[train96].mean(0);scale=norms[train96].std(0).clip(1e-6);standard=(norms-center)/scale
        nuisance=residual_fit(train_frame,standard[train96]);adjusted=residual_apply(frame,standard,nuisance)
        fits[name]=dict(center=center,scale=scale,**nuisance);np.savez_compressed(OUT/'evaluation'/(name+'_readout_fit.npz'),**fits[name])
        readouts[name+'_raw']=norms[ids];readouts[name+'_standardized']=standard[ids];readouts[name+'_adjusted']=adjusted[ids]
        gallery=norms[ids];resamp=cdist(rs,gallery);rotation=cdist(rr,gallery)
        rotation_positions=np.array([np.flatnonzero(ids==i)[0] for i in rotation_ids]);base=cdist(gallery,gallery);jaccard=[]
        for i in range(len(ids)):
            allowed=np.flatnonzero(groups!=groups[i]);a=set(allowed[np.argsort(base[i,allowed])[:5]]);b=set(allowed[np.argsort(resamp[i,allowed])[:5]])
            jaccard.append(len(a&b)/max(len(a|b),1))
        energy=np.linalg.svd(norms[frame.split=='train']-norms[frame.split=='train'].mean(0),compute_uv=False)**2;p=energy/max(energy.sum(),1e-15)
        rotation_rng=np.random.default_rng(20261118)
        from allen_reference_core import random_rotation
        algebraic=np.stack([code@random_rotation(rotation_rng) for code in z.numpy()])
        algebraic_error=np.linalg.norm(np.linalg.norm(algebraic,axis=-1)-norms,axis=1)/np.maximum(np.linalg.norm(norms,axis=1),1e-12)
        stability[name]=dict(resampling_cells=len(ids),resampling_self_top1=float((resamp.argmin(1)==np.arange(len(ids))).mean()),
            resampling_neighbor_jaccard=float(np.mean(jaccard)),fresh_rotation_cells=len(rotation_ids),fresh_rotation_self_top1=float((rotation.argmin(1)==rotation_positions).mean()),
            fresh_rotation_relative_code_error=float(np.mean(np.linalg.norm(rr-gallery[rotation_positions],axis=1)/np.linalg.norm(gallery[rotation_positions],axis=1))),
            effective_rank=float(np.exp(-(p*np.log(p.clip(1e-15))).sum())) if energy.sum()>1e-15 else 0.,algebraic_norm_max_relative_error=float(algebraic_error.max()),
            scope='Full8192-point resampling for48 cells; one fresh full-backbone rotation for12 fixed cells, compared against48-cell gallery. Normalization transform is held fixed during rotation.')
        print('Readouts',name,json.dumps(stability[name]),flush=True)
    np.savez_compressed(OUT/'evaluation/codes.npz',ids=frame.cell_id.to_numpy(dtype=str),**all_codes)
    np.savez_compressed(OUT/'evaluation/readouts.npz',ids=dev.cell_id.to_numpy(dtype=str),**readouts)
    (OUT/'evaluation/stability.json').write_text(json.dumps(stability,indent=2))
    return frame,ids,all_codes,readouts


def retrieval(readouts,codes,ids):
    frame,pools,_=candidate_sets();snapshot=json.loads((OUT/'atlas_snapshot.json').read_text());lookup={c['id']:c for c in snapshot['cells']}
    selected=['morph_all','morph_adjusted','spatial_robust','spatial_adjusted','spherical_harmonics','spherical_harmonics_adjusted','pointnet_shape128','cajal_qgw','cajal_profile_adjusted','cajal_gw_validation']
    matrices={};names={}
    for method in snapshot['methods']:
        if method['id'] in selected:
            x=[lookup[c]['representations'][method['id']]['vector'] for c in frame.cell_id]
            matrices[method['id']]=cdist(x,x,metric='cosine') if method['metric']=='cosine' else distances(x,method['metric']);names[method['id']]=method['name']
    for name,x in readouts.items():matrices['allen_'+name]=cdist(x,x);names['allen_'+name]='Allen '+name.replace('_',' ')
    from allen_reference_core import latent_rotation_distances
    for arm in ['pretrained','head_decoder']:
        key='allen_'+arm+'_vector_alignment';matrices[key]=latent_rotation_distances(codes[arm][ids]);names[key]='Allen '+arm.replace('_',' ')+' · pose-aligned full vector code (secondary)'
    clouds=[]
    for row in frame.to_dict('records'):
        v,f=mesh_for(row);clouds.append(sample_surface(v,f,384,np.random.default_rng(20261030+int(row['original_index']))))
    positions={c:i for i,c in enumerate(frame.cell_id)};costs={}
    old=pd.read_csv(ROOT/'results/harmonics_cajal/development_neighbors.csv')
    for row in old.itertuples():
        if row.cell_id in positions and row.neighbor in positions:costs[tuple(sorted([positions[row.cell_id],positions[row.neighbor]]))]=float(row.geometry_cost)
    records=[]
    for method,d in matrices.items():
        for i,candidates in pools.items():
            j=int(candidates[np.argmin(d[i,candidates])]);pair=tuple(sorted([i,j]))
            if pair not in costs:costs[pair]=geometry_cost(clouds[pair[0]],clouds[pair[1]])
            records.append(dict(method=method,query=i,cell_id=frame.iloc[i].cell_id,neighbor=frame.iloc[j].cell_id,
                similarity_group=int(frame.iloc[i].similarity_group),geometry_cost=costs[pair]))
    records=pd.DataFrame(records);records.to_csv(OUT/'evaluation/retrieval.csv',index=False)
    summary=[]
    for method in matrices:
        subset=records[records.method==method].copy();entry=dict(id=method,name=names[method],queries=len(subset),geometry_cost=float(subset.groupby('similarity_group').geometry_cost.mean().mean()))
        for baseline in ['morph_all','morph_adjusted']:
            b=records[records.method==baseline].set_index('query');subset['improvement']=[b.loc[q].geometry_cost-c for q,c in zip(subset['query'],subset.geometry_cost)]
            entry['vs_'+baseline]=group_ci(subset,'improvement')
        summary.append(entry)
    primary=next(r for r in summary if r['id']=='allen_head_decoder_adjusted')['vs_morph_adjusted']
    (OUT/'evaluation/retrieval.json').write_text(json.dumps(dict(methods=summary,primary_comparison=primary,scope='Exploratory matched-validation geometry comparison; validation also selects adaptation checkpoints. Not independent biological evidence.'),indent=2))
    print('Primary geometry-proxy comparison',json.dumps(primary),flush=True)


def geometry(frame,ids,codes,device):
    rows=[];models={name:load_arm(name,device) for name in ARMS};pca=Encoder('pca',True,'cpu');clouds={};refs={};selected_ids=frame.iloc[ids].cell_id.tolist()
    orbit=np.load(ROOT/'results/spatial_benchmark/development/codes.npz')['pca_aligned_base'];order=pd.read_csv(ROOT/'results/spatial_benchmark/development/manifest.csv').cell_id.tolist()
    pca_lookup={cid:z[0] for cid,z in zip(order,orbit)}
    for index in ids:
        row=frame.iloc[index];v,f=mesh_for(row);mesh=trimesh.Trimesh(v,f,process=False);refs[row.cell_id]=sample_surface(v,f,2048,np.random.default_rng(20261121+int(row.original_index)))
        with np.load(OUT/'data'/f'{index:04d}.npz') as data:center=data['center'];scale=float(data['scale'])
        for name in ARMS+['spatial_pca']:
            dest=OUT/'evaluation/meshes'/f"{row.cell_id.replace(':','_')}_{name}.npz"
            if dest.exists():
                with np.load(dest) as a:vertices=a['vertices'];faces=a['faces'];field=a['field']
                surface=(vertices,faces) if len(vertices) else None
            else:
                if name=='spatial_pca':
                    surface,field=pca.surface(pca_lookup[row.cell_id]);pc,axes,_,_=surface_frame(v,f)
                    if surface is not None:surface=(surface[0]@axes.T+pc,surface[1])
                else:
                    z=torch.tensor(codes[name][index:index+1],device=device);surface,field=extract_surface(models[name],z,resolution=56,extent=.6)
                    if surface is not None:surface=(surface[0]*scale+center,surface[1])
                if surface is None:vertices=np.empty((0,3));faces=np.empty((0,3),int)
                else:vertices,faces=surface
                np.savez_compressed(dest,vertices=vertices,faces=faces,field=field)
            entry=dict(cell_id=row.cell_id,similarity_group=int(row.similarity_group),method=name,failed_surface=surface is None)
            if surface is not None:
                reconstructed=trimesh.Trimesh(vertices,faces,process=False);sample=sample_surface(vertices,faces,2048,np.random.default_rng(20261122+int(row.original_index)))
                clouds[row.cell_id,name]=sample;boundary=np.concatenate([field[0].ravel(),field[-1].ravel(),field[:,0].ravel(),field[:,-1].ravel(),field[:,:,0].ravel(),field[:,:,-1].ravel()])
                entry.update(**surface_metrics(refs[row.cell_id],sample,float(row.normalization_radius_um)),
                    volume_relative_error=float(abs(abs(reconstructed.volume)-abs(mesh.volume))/abs(mesh.volume)),
                    watertight=bool(reconstructed.is_watertight),boundary_crossing=bool((boundary<0).any()))
            rows.append(entry)
        print(f'Geometry {row.cell_id} · {selected_ids.index(row.cell_id)+1}/{len(ids)}',flush=True)
    results=pd.DataFrame(rows);results.to_csv(OUT/'evaluation/reconstruction.csv',index=False);summary={}
    for name,sub in results.groupby('method'):
        summary[name]=dict(cells=len(sub),failed_surfaces=int(sub.failed_surface.sum()),successful=int((~sub.failed_surface).sum()),
            **{c:float(sub[c].mean()) for c in ['squared_chamfer','f1_at_005_radius','p95_surface_error_um','volume_relative_error']},
            nonwatertight=int((sub.watertight==False).sum()),boundary_crossings=int((sub.boundary_crossing==True).sum()))
    (OUT/'evaluation/geometry.json').write_text(json.dumps(summary,indent=2))
    visual_ids=[r['query'] for r in json.loads((ROOT/'results/spatial_benchmark/development/visual_queries.json').read_text())]
    np.savez_compressed(OUT/'evaluation/visuals.npz',ids=np.asarray(visual_ids),references=np.stack([refs[c] for c in visual_ids]),
        **{name:np.stack([clouds.get((c,name),np.full((2048,3),np.nan)) for c in visual_ids]) for name in ARMS+['spatial_pca']})
    print('Geometry summary',json.dumps(summary),flush=True)


def main():
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.55)
    (OUT/'evaluation/meshes').mkdir(parents=True,exist_ok=True)
    signature=dict(manifest_sha256=sha256(OUT/'manifest.csv'),official_sha256=sha256(OUT/'checkpoints/Rotation_invariant_pointcloud_SDF.ckpt'),
        decoder_only_sha256=sha256(OUT/'decoder_only/best.pt'),head_decoder_sha256=sha256(OUT/'head_decoder/best.pt'),analysis_protocol_sha256=sha256(OUT/'analysis_protocol.json'))
    path=OUT/'evaluation/provenance.json'
    if path.exists() and json.loads(path.read_text())!=signature:raise ValueError('Cached evaluation belongs to different weights or protocol')
    path.write_text(json.dumps(signature,indent=2))
    frame,ids,codes,readouts=embeddings();retrieval(readouts,codes,ids);geometry(frame,ids,codes,'mps')
    protocol=json.loads((OUT/'protocol.json').read_text())
    (OUT/'evaluation/completed.json').write_text(json.dumps(dict(status='complete',atlas_unchanged=protocol['atlas_sha256']==sha256(ROOT/'results/explorer_dataset/dataset.json'),test_used=False),indent=2))


if __name__=='__main__':main()
