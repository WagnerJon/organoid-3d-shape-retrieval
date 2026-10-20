"""Evaluate encoder and optional SDF-code fitting without publishing to Atlas.

Fitted codes see signed-distance observations from the known segmentation.
They are explicitly distinct from encoder-only inference, and shared decoder
weights never learn from validation cells. All surface scores use fresh samples.
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import trimesh
from pointnet_shape_core import ROOT,select_device,rotations,save_json,sha256,sample_surface
from sdf_shape_core import SDFModel,sdf_loss,extract_surface
from staged_sdf_core import ReferenceVectorEncoder,VectorAdam,stability
from prepare_staged_sdf import OUT,load
from prepare_pointnet_shape import read_mesh
from evaluate_sdf_shape import surface_metrics,group_ci
from evaluate_pointnet_shape import conditional_retrieval


def observation_indices(view):
    # Disjoint, identically balanced SDF subsets: 768 near, 512 volume, 256 surface.
    # Interleave both near-surface bands instead of putting a different band in each view.
    return np.concatenate([np.arange(view,1536,2),np.arange(1536+view,2560,2),np.arange(2560+view,3072,2)])


def fit_codes(model,initial,query,target,steps,seed=20261023):
    """Frozen decoder, vector-wise adaptive code fitting; no cell labels used."""
    for p in model.parameters():p.requires_grad_(False)
    code=nn.Parameter(initial.detach().clone());optimizer=VectorAdam(code,lr=.015)
    rng=np.random.default_rng(seed);best=code.detach().clone()
    with torch.no_grad():
        best_error=(model.decode(code,query)-target.clamp(-.15,.15)).abs().mean(1)+.001*code.square().mean((1,2))
    trajectory=[]
    for step in range(1,steps+1):
        columns=rng.choice(query.shape[1],min(512,query.shape[1]),replace=False)
        q=query[:,columns];t=target[:,columns]
        loss=sdf_loss(model.decode(code,q),t)+.001*code.square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        optimizer.lr=.015*(.2+.8*(1+np.cos(np.pi*step/steps))/2)
        if step%50==0 or step==steps:
            with torch.no_grad():
                error=(model.decode(code,query)-target.clamp(-.15,.15)).abs().mean(1)+.001*code.square().mean((1,2))
                improved=error<best_error;best[improved]=code.detach()[improved];best_error=torch.minimum(best_error,error)
            trajectory.append(dict(step=step,mean_fitting_objective=float(best_error.mean().cpu())))
    return best.detach(),trajectory


def encode_views(encoder,arrays,ids,device,view=0,rotation=None):
    result=[]
    with torch.no_grad():
        for first in range(0,len(ids),4):
            selected=ids[first:first+4]
            p=torch.tensor(np.array(arrays['large_points'][selected,view]),device=device)
            g=torch.tensor(np.array(arrays['neighbors'][selected,view]),device=device).long()
            if rotation is not None:p=p@rotation[first:first+4].transpose(1,2)
            result.append(encoder(p,g))
    return torch.cat(result)


def refine_all(model,initial,arrays,ids,device,steps,observation_view,rotation=None):
    indices=observation_indices(observation_view);codes=[];history=[]
    for first in range(0,len(ids),8):
        started=time.monotonic()
        selected=ids[first:first+8]
        q=torch.tensor(arrays['query'][selected][:,indices],device=device)
        t=torch.tensor(arrays['target'][selected][:,indices],device=device)
        if rotation is not None:q=q@rotation[first:first+8].transpose(1,2)
        z,trajectory=fit_codes(model,initial[first:first+8],q,t,steps)
        codes.append(z);history.append(dict(cell_ids=selected.tolist(),trajectory=trajectory,seconds=time.monotonic()-started))
        print(f'FITTED {min(first+8,len(ids))}/{len(ids)} observation view {observation_view}',flush=True)
    return torch.cat(codes),history


def evaluate(device_name,steps,decoder_only=False):
    torch.set_num_threads(4);device=select_device(device_name);manifest,arrays=load()
    ids=manifest.index[manifest.split=='validation'].to_numpy();train_ids=manifest.index[manifest.split=='train'].to_numpy()
    out=OUT/('decoder_validation' if decoder_only else 'evaluation');out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Completed evaluation exists')
    saved=torch.load(OUT/'decoder/best.pt',map_location='cpu',weights_only=False)
    if saved['manifest_sha256']!=sha256(OUT/'manifest.csv'):raise ValueError('Decoder manifest changed')
    model=SDFModel('vn').to(device);model.decoder.load_state_dict(saved['decoder']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    rng=np.random.default_rng(20261023);rotation=torch.tensor(rotations(len(ids),rng),device=device)
    codes={};fresh={};rotated={};histories={}
    if not decoder_only:
        enc_saved=torch.load(OUT/'encoder/best.pt',map_location='cpu',weights_only=False)
        if enc_saved['decoder_sha256']!=sha256(OUT/'decoder/best.pt'):raise ValueError('Encoder decoder mismatch')
        encoder=ReferenceVectorEncoder().to(device);encoder.load_state_dict(enc_saved['encoder']);encoder.eval()
        codes['encoder']=encode_views(encoder,arrays,ids,device)
        fresh['encoder']=encode_views(encoder,arrays,ids,device,view=2)
        rotated['encoder']=encode_views(encoder,arrays,ids,device,rotation=rotation)
        for key,initial,view,r in [('refined',codes['encoder'],0,None),('fresh',fresh['encoder'],1,None),
                                   ('rotated',rotated['encoder'],0,rotation)]:
            fitted,history=refine_all(model,initial,arrays,ids,device,steps,view,rotation=r)
            if key=='refined':codes['refined']=fitted
            elif key=='fresh':fresh['refined']=fitted
            else:rotated['refined']=fitted
            histories[key]=history
    # A separate direct-code control starts from zero; it uses no encoder.
    cached=OUT/'decoder_validation'
    if not decoder_only and (cached/'metrics.json').exists():
        meta=json.loads((cached/'metrics.json').read_text())
        if meta['shared_decoder_sha256']!=sha256(OUT/'decoder/best.pt') or meta['refinement_steps']!=steps:
            raise ValueError('Cached direct-code control uses different weights or fitting budget')
        with np.load(cached/'codes.npz') as data:
            if not np.array_equal(data['cell_ids'],manifest.iloc[ids].cell_id.to_numpy(str)):raise ValueError('Cached cell IDs differ')
            codes['fitted_zero']=torch.tensor(data['fitted_zero'],device=device)
            fresh['fitted_zero']=torch.tensor(data['fitted_zero_fresh'],device=device)
            rotated['fitted_zero']=torch.tensor(data['fitted_zero_rotated'],device=device)
        histories['fitted_zero']='Reused matching completed decoder-only control; no refitting.'
    else:
        zero=torch.zeros(len(ids),128,3,device=device)
        codes['fitted_zero'],histories['fitted_zero']=refine_all(model,zero,arrays,ids,device,steps,0)
        fresh['fitted_zero'],_=refine_all(model,zero,arrays,ids,device,steps,1)
        rotated['fitted_zero'],_=refine_all(model,zero,arrays,ids,device,steps,0,rotation=rotation)
    save_json(out/'fitting_histories.json',histories)
    np.savez_compressed(out/'codes.npz',cell_ids=manifest.iloc[ids].cell_id.to_numpy(str),
        **{k:z.cpu().numpy() for k,z in codes.items()},**{k+'_fresh':z.cpu().numpy() for k,z in fresh.items()},
        **{k+'_rotated':z.cpu().numpy() for k,z in rotated.items()})
    old=np.load(ROOT/'results/pointnet_shape/baseline/validation_reconstructions.npz');baseline=dict(zip(old['cell_ids'],old['reconstructions']))
    rows=[];references=[];clouds=[];started=time.monotonic()
    for position,index in enumerate(ids):
        row=manifest.iloc[index];v,f=read_mesh(ROOT/'results/explorer_dataset'/row['sample']/f'{row.label}.mesh')
        center=np.array([row['normalization_center_'+a+'_um'] for a in 'xyz']);radius=float(row.normalization_radius_um)
        mesh=trimesh.Trimesh((v-center)/radius,f,process=False);reference=sample_surface(mesh.vertices,mesh.faces,2048,rng)
        references.append(reference);displayed={'baseline':baseline[row.cell_id]}
        common=dict(cell_id=row.cell_id,similarity_group=int(row.similarity_group))
        rows.append(dict(**common,method='baseline',**surface_metrics(reference,baseline[row.cell_id],radius)))
        for method,values in codes.items():
            z=values[position:position+1];surface,field=extract_surface(model,z)
            if surface is None:
                rows.append(dict(**common,method=method,failed_surface=True,squared_chamfer=None,p95_surface_error_um=None,f1_at_005_radius=None,
                                 volume_relative_error=None,watertight=False,boundary_negative=True));continue
            vertices,faces=surface;reconstructed=trimesh.Trimesh(vertices,faces,process=False)
            pred=sample_surface(vertices,faces,2048,rng);displayed[method]=pred
            reconstructed.export(out/f"{row.cell_id.replace(':','_')}_{method}.ply")
            boundary=np.concatenate([field[0].ravel(),field[-1].ravel(),field[:,0].ravel(),field[:,-1].ravel(),field[:,:,0].ravel(),field[:,:,-1].ravel()])
            unseen=observation_indices(1)
            with torch.no_grad():
                q=torch.tensor(arrays['query'][index:index+1,unseen],device=device)
                t=torch.tensor(arrays['target'][index:index+1,unseen],device=device)
                heldout=float(sdf_loss(model.decode(z,q),t).cpu())
            rows.append(dict(**common,method=method,**surface_metrics(reference,pred,radius),
                volume_relative_error=float(abs(abs(reconstructed.volume)-abs(mesh.volume))/abs(mesh.volume)),
                watertight=bool(reconstructed.is_watertight),boundary_negative=bool((boundary<0).any()),failed_surface=False,
                heldout_query_sdf_l1=heldout))
        clouds.append(displayed);print(f'EVALUATED {position+1}/{len(ids)} {time.monotonic()-started:.1f}s',flush=True)
    results=pd.DataFrame(rows);results.to_csv(out/'cell_metrics.csv',index=False);summary={}
    for method,frame in results.groupby('method'):
        def mean(name):
            value=float(frame[name].mean());return value if np.isfinite(value) else None
        summary[method]=dict(cells=len(frame),failed_surfaces=int(frame.failed_surface.fillna(False).sum()),
            **{name:mean(name) for name in ['squared_chamfer','p95_surface_error_um','f1_at_005_radius']})
        if method!='baseline':
            summary[method].update(volume_relative_error=mean('volume_relative_error'),heldout_query_sdf_l1=mean('heldout_query_sdf_l1'),
                nonwatertight=int((frame.watertight==False).sum()),boundary_crossings=int((frame.boundary_negative==True).sum()),
                **stability(codes[method].norm(dim=-1).cpu().numpy(),fresh[method].norm(dim=-1).cpu().numpy(),rotated[method].norm(dim=-1).cpu().numpy()))
            # Query observations rotate together; inference uses a vector-wise optimizer.
            summary[method]['relative_rotation_equivariance_error']=float((rotated[method]-codes[method]@rotation.transpose(1,2)).norm().cpu()/codes[method].norm().clamp_min(1e-8).cpu())
            original_manifest=pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv')
            retrieval,matched=conditional_retrieval(original_manifest,{'targets':np.load(ROOT/'results/pointnet_shape/data/targets.npy',mmap_mode='r')},
                manifest.iloc[ids].original_index.to_numpy(),codes[method].norm(dim=-1).cpu().numpy(),
                manifest.iloc[train_ids].original_index.to_numpy(),20261023)
            summary[method]['conditional_retrieval']=retrieval;pd.DataFrame(matched).to_csv(out/(method+'_matched_retrieval.csv'),index=False)
    paired={};wide=results.pivot(index=['cell_id','similarity_group'],columns='method',values='squared_chamfer').reset_index()
    for method in codes:
        valid=wide.dropna(subset=['baseline',method]).copy();valid['improvement']=valid.baseline-valid[method]
        if len(valid):paired[method]=group_ci(valid,'improvement')
    meta=json.loads((OUT/'data.json').read_text())
    save_json(out/'metrics.json',dict(methods=summary,paired_group_improvements=paired,refinement_steps=steps,
        shared_decoder_sha256=sha256(OUT/'decoder/best.pt'),
        scope='Validation-cell codes fit to known segmentation SDF observations with frozen decoder; encoder-only reported separately. No test cells.',
        fitting_observations='1536 points: 768 near surface / 512 uniform / 256 exact surface. Fresh view uses disjoint SDF observations and independent input sample.',
        evaluation='Surface metrics on 2048 independent samples, extracted at 56^3; held-out query error on unused half of SDF pool.',
        atlas_unchanged=meta['atlas_sha256']==sha256(ROOT/'results/explorer_dataset/dataset.json')))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns=['reference','baseline']+list(codes)
    labels=dict(reference='Original surface',baseline='Existing PointNet++ AE',encoder='Staged encoder',refined='Encoder + code fitting',fitted_zero='SDF fitting, no encoder')
    order=np.argsort(manifest.iloc[ids].sphericity.to_numpy());chosen=order[np.linspace(0,len(order)-1,3).astype(int)]
    fig=plt.figure(figsize=(3.5*len(columns),10))
    for ri,at in enumerate(chosen):
        for ci,key in enumerate(columns):
            ax=fig.add_subplot(3,len(columns),ri*len(columns)+ci+1,projection='3d');cloud=references[at] if key=='reference' else clouds[at].get(key)
            if cloud is not None:ax.scatter(*cloud.T,s=1.8,alpha=.8,color='#247e91' if key=='reference' else '#ec8a35')
            ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1),box_aspect=(1,1,1));ax.view_init(20,-65);ax.set_axis_off()
            ax.set_title(labels[key]+('\n'+manifest.iloc[ids[at]].cell_id if key=='reference' else ''),fontsize=10)
    fig.suptitle('Staged SDF validation · same cells · fitting uses known segmentation distances');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='auto');p.add_argument('--steps',type=int,default=500)
    p.add_argument('--decoder-only',action='store_true');args=p.parse_args();evaluate(args.device,args.steps,args.decoder_only)
