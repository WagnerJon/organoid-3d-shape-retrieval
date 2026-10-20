"""Compare isolated SDF pilot with the existing AE on exactly the same cells."""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
import trimesh
from pointnet_shape_core import ROOT, sample_surface, rotations, select_device, save_json, sha256
from prepare_pointnet_shape import read_mesh
from train_sdf_shape import OUT,load_pilot,batch
from sdf_shape_core import SDFModel,extract_surface
from evaluate_pointnet_shape import normalized,conditional_retrieval


def surface_metrics(reference,prediction,radius):
    a=cKDTree(prediction).query(reference)[0];b=cKDTree(reference).query(prediction)[0]
    precision=float((b<.05).mean());recall=float((a<.05).mean())
    return dict(squared_chamfer=float(np.square(a).mean()+np.square(b).mean()),
        p95_surface_error_um=float(np.quantile(np.concatenate([a,b]),.95)*radius),
        f1_at_005_radius=float(2*precision*recall/max(precision+recall,1e-12)))


def group_ci(frame,column):
    values=frame.groupby('similarity_group')[column].mean().to_numpy()
    rng=np.random.default_rng(20261018)
    return dict(mean=float(values.mean()),ci95=np.quantile(rng.choice(values,(2000,len(values)),replace=True).mean(1),[.025,.975]).tolist(),groups=len(values))


def evaluate(mode,device_name):
    torch.set_num_threads(4);device=select_device(device_name);manifest,arrays=load_pilot()
    ids=manifest.index[manifest.overfit if mode=='overfit' else manifest.split=='validation'].to_numpy()
    out=OUT/(mode+'_evaluation');out.mkdir(exist_ok=True)
    original=pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv')
    previous=np.load(ROOT/'results/pointnet_shape/baseline/validation_reconstructions.npz')
    old_recon=dict(zip(previous['cell_ids'],previous['reconstructions']))
    models={};checkpoints={}
    for kind in ['pointnet','vn']:
        saved=torch.load(OUT/f'{mode}_{kind}'/'best.pt',weights_only=False,map_location='cpu')
        if saved['manifest_sha256']!=sha256(OUT/'manifest.csv'):raise ValueError('Checkpoint manifest mismatch')
        model=SDFModel(kind).to(device);model.load_state_dict(saved['model']);model.eval();models[kind]=model
        checkpoints[kind]=dict(epoch=saved['epoch'],sha256=sha256(OUT/f'{mode}_{kind}'/'best.pt'))
    rows=[];clouds=[];references=[];latent={k:[] for k in models};resampled={k:[] for k in models};rotated={k:[] for k in models}
    equivariance={k:[] for k in models};rng=np.random.default_rng(20261019);started=time.monotonic()
    for n,index in enumerate(ids):
        row=manifest.iloc[index];v,f=read_mesh(ROOT/'results/explorer_dataset'/row['sample']/f'{row.label}.mesh')
        center=np.array([row['normalization_center_'+a+'_um'] for a in 'xyz']);radius=float(row.normalization_radius_um)
        mesh=trimesh.Trimesh((v-center)/radius,f,process=False)
        reference=sample_surface(mesh.vertices,mesh.faces,2048,rng);references.append(reference)
        displayed={};base=dict(cell_id=row.cell_id,similarity_group=int(row.similarity_group))
        if row.cell_id in old_recon:
            pred=old_recon[row.cell_id];displayed['baseline']=pred
            rows.append(dict(**base,method='baseline',**surface_metrics(reference,pred,radius)))
        p,q,t,g=batch(arrays,[index],device)
        rot=torch.tensor(rotations(1,rng),device=device)
        with torch.no_grad():
            for kind,model in models.items():
                z=model.encode(p,g);latent[kind].append(model.embedding(z).cpu().numpy()[0])
                zr=model.encode(p@rot.transpose(1,2),g);rotated[kind].append(model.embedding(zr).cpu().numpy()[0])
                second=torch.tensor(arrays['points'][index,2],device=device)[None]
                g2={k:torch.tensor(arrays[k][index,2],device=device)[None].long() for k in g}
                zs=model.encode(second,g2);resampled[kind].append(model.embedding(zs).cpu().numpy()[0])
                sr=model.decode(zr,q[:,:64]@rot.transpose(1,2));s=model.decode(z,q[:,:64])
                equivariance[kind].append(float((sr-s).abs().max().cpu()))
                surface,field=extract_surface(model,z)
                if surface is None:
                    rows.append(dict(**base,method=kind,failed_surface=True));continue
                vertices,faces=surface
                reconstructed=trimesh.Trimesh(vertices,faces,process=False)
                pred=sample_surface(vertices,faces,2048,rng);displayed[kind]=pred
                reconstructed.export(out/f"{row.cell_id.replace(':','_')}_{kind}.ply")
                boundary=np.concatenate([field[0].ravel(),field[-1].ravel(),field[:,0].ravel(),field[:,-1].ravel(),field[:,:,0].ravel(),field[:,:,-1].ravel()])
                rows.append(dict(**base,method=kind,**surface_metrics(reference,pred,radius),
                    volume_relative_error=float(abs(abs(reconstructed.volume)-abs(mesh.volume))/abs(mesh.volume)),
                    watertight=bool(reconstructed.is_watertight),boundary_negative=bool((boundary<0).any()),failed_surface=False))
        clouds.append(displayed)
        print(f'EVALUATED {n+1}/{len(ids)} {row.cell_id} {time.monotonic()-started:.1f}s',flush=True)
    results=pd.DataFrame(rows);results.to_csv(out/'cell_metrics.csv',index=False)
    summary={}
    for method,frame in results.groupby('method'):
        summary[method]=dict(cells=len(frame),failed_surfaces=int(frame.get('failed_surface',pd.Series(False,index=frame.index)).fillna(False).sum()),
            **{c:float(frame[c].mean()) for c in ['squared_chamfer','p95_surface_error_um','f1_at_005_radius']})
        if method!='baseline':
            summary[method].update(volume_relative_error=float(frame.volume_relative_error.mean()),
                nonwatertight=int((frame.watertight==False).sum()),boundary_crossings=int((frame.boundary_negative==True).sum()))
    for kind in models:
        a=np.asarray(latent[kind]);b=np.asarray(resampled[kind]);c=np.asarray(rotated[kind]);unit=normalized(a)
        values=np.linalg.svd(a-a.mean(0),compute_uv=False)**2;p=values/max(values.sum(),1e-15)
        summary[kind].update(resampling_self_top1=float((np.argmax(normalized(b)@unit.T,axis=1)==np.arange(len(ids))).mean()),
            rotation_self_top1=float((np.argmax(normalized(c)@unit.T,axis=1)==np.arange(len(ids))).mean()),
            maximum_joint_rotation_sdf_error=float(max(equivariance[kind])),effective_rank=float(np.exp(-(p*np.log(p.clip(1e-15))).sum())))
        if mode=='pilot':
            original_ids=manifest.iloc[ids].original_index.to_numpy()
            training_ids=manifest[manifest.split=='train'].original_index.to_numpy()
            original_arrays={'targets':np.load(ROOT/'results/pointnet_shape/data/targets.npy',mmap_mode='r')}
            retrieval,matched=conditional_retrieval(original,original_arrays,original_ids,a,training_ids,20261018)
            summary[kind]['conditional_retrieval']=retrieval
            pd.DataFrame(matched).to_csv(out/(kind+'_matched_retrieval.csv'),index=False)
        np.savez_compressed(out/(kind+'_embeddings.npz'),cell_ids=manifest.iloc[ids].cell_id.to_numpy(str),latent=a,resampled=b,rotated=c)
    comparisons={}
    if mode=='pilot':
        pivot=results.pivot(index=['cell_id','similarity_group'],columns='method',values='squared_chamfer').reset_index()
        for a,b in [('baseline','pointnet'),('baseline','vn'),('pointnet','vn')]:
            pivot['improvement']=pivot[a]-pivot[b];comparisons[a+'_minus_'+b]=group_ci(pivot,'improvement')
    save_json(out/'metrics.json',dict(mode=mode,methods=summary,paired_group_comparisons=comparisons,checkpoints=checkpoints,
        evaluation='2048 independent equal-area samples per surface, common normalization; nearest-point metrics include sampling error. Threshold F1 = 5% of original normalization radius. SDF meshes extracted at 56^3.',
        atlas_unchanged=sha256(ROOT/'results/explorer_dataset/dataset.json')==json.loads((OUT/'data_audit.json').read_text())['atlas_sha256'],
        limitation='Reduced-compute pilot, not a published-model reproduction; validation used for model selection, no final independent test or biological labels. Original model used more training cells and more input points.'))
    # Predefined low/mid/high sphericity examples, independent of model error.
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ordered=np.argsort(manifest.iloc[ids].sphericity.to_numpy());chosen=ordered[np.linspace(0,len(ordered)-1,3).astype(int)]
    columns=['reference','baseline','pointnet','vn'] if mode=='pilot' else ['reference','pointnet','vn']
    names=dict(reference='Original surface samples',baseline='Existing PointNet++ AE',pointnet='PointNet++ + SDF',vn='VN-DGCNN + SDF')
    fig=plt.figure(figsize=(4*len(columns),10));fig.suptitle(f'{mode.capitalize()} reconstruction comparison · low / middle / high sphericity',fontsize=15)
    for ri,at in enumerate(chosen):
        for ci,key in enumerate(columns):
            ax=fig.add_subplot(3,len(columns),ri*len(columns)+ci+1,projection='3d')
            cloud=references[at] if key=='reference' else clouds[at].get(key)
            if cloud is not None:ax.scatter(*cloud.T,s=1.8,alpha=.75,color='#ec8a35' if key!='reference' else '#247e91',rasterized=True)
            ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1),box_aspect=(1,1,1));ax.view_init(20,-65);ax.set_axis_off()
            label=manifest.iloc[ids[at]].cell_id
            ax.set_title(names[key]+('\n'+label if key=='reference' else ''),fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'reconstruction_comparison.png',dpi=170);plt.close(fig)
    # Orthogonal midplane contours check indentations hidden in the 3D rendering.
    fig,axes=plt.subplots(3,len(columns),figsize=(4*len(columns),10),squeeze=False)
    for ri,at in enumerate(chosen):
        for ci,key in enumerate(columns):
            ax=axes[ri,ci];cloud=references[at] if key=='reference' else clouds[at].get(key)
            if cloud is not None:
                section=cloud[np.abs(cloud[:,2])<.075];ax.scatter(section[:,0],section[:,1],s=5)
            ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),aspect='equal');ax.set_title(names[key]);ax.grid(alpha=.2)
    fig.suptitle('XY midplane surface bands · |z| < 0.075 normalized radius');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'midplane_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['overfit','pilot'],required=True);p.add_argument('--device',default='auto');args=p.parse_args()
    evaluate(args.mode,args.device)
