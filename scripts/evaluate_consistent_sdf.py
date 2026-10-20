"""Matched geometry and stability evaluation for the offline consistency ablation."""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import trimesh
from consistent_sdf import OUT,SOURCE
from prepare_staged_sdf import load
from pointnet_shape_core import ROOT,select_device,sha256,save_json,rotations,sample_surface
from staged_sdf_core import ReferenceVectorEncoder,stability
from sdf_shape_core import SDFModel,extract_surface
from evaluate_staged_sdf import encode_views
from evaluate_sdf_shape import surface_metrics,group_ci
from prepare_pointnet_shape import read_mesh
from evaluate_pointnet_shape import conditional_retrieval


def evaluate(device_name):
    torch.set_num_threads(4);device=select_device(device_name);manifest,arrays=load()
    out=OUT/'evaluation';out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Completed evaluation exists')
    provenance=json.loads((OUT/'provenance.json').read_text())
    assert provenance['decoder_sha256']==sha256(SOURCE/'decoder/best.pt')
    assert provenance['manifest_sha256']==sha256(SOURCE/'manifest.csv')
    ids=manifest.index[manifest.split=='validation'].to_numpy();train=manifest.index[manifest.split=='train'].to_numpy()
    model=SDFModel('vn').to(device);saved=torch.load(SOURCE/'decoder/best.pt',weights_only=False,map_location='cpu')
    model.decoder.load_state_dict(saved['decoder']);model.eval()
    rng=np.random.default_rng(20261023);rotation=torch.tensor(rotations(len(ids),rng),device=device)
    codes={};summaries={};features={}
    for arm in ['control','paired']:
        for selection in ['best','last']:
            key=arm+'_'+selection;checkpoint=OUT/arm/(selection+'.pt')
            saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
            encoder=ReferenceVectorEncoder().to(device);encoder.load_state_dict(saved['encoder']);encoder.eval()
            a=encode_views(encoder,arrays,ids,device);b=encode_views(encoder,arrays,ids,device,view=2)
            c=encode_views(encoder,arrays,ids,device,rotation=rotation);codes[key]=a
            norms=[z.norm(dim=-1).cpu().numpy() for z in [a,b,c]];features[key]=norms
            summaries[key]=dict(epoch=saved['epoch'],checkpoint_sha256=sha256(checkpoint),**stability(*norms),
                relative_rotation_error=float(((c-a@rotation.transpose(1,2)).norm()/a.norm().clamp_min(1e-8)).cpu()))
    rows=[];references=[];clouds=[]
    # Identical reference samples for every method and independent deterministic
    # samples for each reconstruction. No validation surface enters training.
    for position,index in enumerate(ids):
        row=manifest.iloc[index];v,f=read_mesh(ROOT/'results/explorer_dataset'/row['sample']/f'{row.label}.mesh')
        center=np.array([row['normalization_center_'+axis+'_um'] for axis in 'xyz']);radius=float(row.normalization_radius_um)
        mesh=trimesh.Trimesh((v-center)/radius,f,process=False)
        reference=sample_surface(mesh.vertices,mesh.faces,2048,np.random.default_rng(20261023+index));references.append(reference)
        displayed={}
        for key,z in codes.items():
            surface,field=extract_surface(model,z[position:position+1])
            base=dict(cell_id=row.cell_id,similarity_group=int(row.similarity_group),method=key)
            if surface is None:rows.append(dict(**base,failed_surface=True));continue
            vertices,faces=surface;reconstructed=trimesh.Trimesh(vertices,faces,process=False)
            prediction=sample_surface(vertices,faces,2048,np.random.default_rng(20262023+index));displayed[key]=prediction
            boundary=np.concatenate([field[0].ravel(),field[-1].ravel(),field[:,0].ravel(),field[:,-1].ravel(),field[:,:,0].ravel(),field[:,:,-1].ravel()])
            rows.append(dict(**base,failed_surface=False,**surface_metrics(reference,prediction,radius),
                volume_relative_error=float(abs(abs(reconstructed.volume)-abs(mesh.volume))/abs(mesh.volume)),
                watertight=bool(reconstructed.is_watertight),boundary_crossing=bool((boundary<0).any())))
        clouds.append(displayed)
        print(f'EVALUATED {position+1}/{len(ids)}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'cell_metrics.csv',index=False)
    original=pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv')
    targets={'targets':np.load(ROOT/'results/pointnet_shape/data/targets.npy',mmap_mode='r')}
    for key,summary in summaries.items():
        values=frame[frame.method==key]
        summary.update({column:float(values[column].mean()) for column in ['squared_chamfer','p95_surface_error_um','f1_at_005_radius','volume_relative_error']})
        summary.update(failed_surfaces=int(values.failed_surface.sum()),nonwatertight=int((values.watertight==False).sum()),boundary_crossings=int(values.boundary_crossing.sum()))
        retrieval,matched=conditional_retrieval(original,targets,manifest.iloc[ids].original_index.to_numpy(),features[key][0],manifest.iloc[train].original_index.to_numpy(),20261023)
        summary['conditional_retrieval']=retrieval;pd.DataFrame(matched).to_csv(out/(key+'_matched_retrieval.csv'),index=False)
    paired={};wide=frame.pivot(index=['cell_id','similarity_group'],columns='method',values='squared_chamfer').reset_index()
    for selection in ['best','last']:
        wide['improvement']=wide['control_'+selection]-wide['paired_'+selection]
        paired[selection]=group_ci(wide,'improvement')
    np.savez_compressed(out/'features.npz',cell_ids=manifest.iloc[ids].cell_id.to_numpy(str),**{key:values[0] for key,values in features.items()},**{key+'_fresh':values[1] for key,values in features.items()})
    save_json(out/'metrics.json',dict(methods=summaries,paired_group_geometry_improvement=paired,
        validation_cells=len(ids),test_used=False,per_cell_fitting=False,
        atlas_unchanged=provenance['atlas_sha256']==sha256(ROOT/'results/explorer_dataset/dataset.json'),
        limitations='One seed, 12 fine-tuning epochs, 96 training cells, batch-four negatives; development validation also selects best checkpoints. Last epoch separately reported. Reduced-compute paper-inspired encoder, not a reproduction.'))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=['control_last','paired_last'];order=np.argsort(manifest.iloc[ids].sphericity.to_numpy());chosen=order[np.linspace(0,len(order)-1,3).astype(int)]
    fig=plt.figure(figsize=(10.5,10))
    for ri,at in enumerate(chosen):
        for ci,key in enumerate(['reference']+keys):
            ax=fig.add_subplot(3,3,ri*3+ci+1,projection='3d');cloud=references[at] if key=='reference' else clouds[at].get(key)
            if cloud is not None:ax.scatter(*cloud.T,s=2,color='#247e91' if key=='reference' else '#ec8a35',alpha=.8)
            ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1),box_aspect=(1,1,1));ax.view_init(20,-65);ax.set_axis_off()
            ax.set_title({'reference':'Original surface','control_last':'SDF-only fine-tuning','paired_last':'SDF + sample consistency'}[key]+('\n'+manifest.iloc[ids[at]].cell_id if key=='reference' else ''),fontsize=10)
    fig.suptitle('Encoder-only inference after 12 matched fine-tuning epochs');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for arm in ['control','paired']:
        history=pd.read_csv(OUT/arm/'history.csv')
        for ax,column in zip(axes,['validation_sdf_l1','resampling_self_top1','effective_rank']):
            ax.plot(history.epoch,history[column],label=arm);ax.set(xlabel='Fine-tuning epoch',title=column.replace('_',' '));ax.legend()
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=170);plt.close(fig)
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='auto');evaluate(parser.parse_args().device)
