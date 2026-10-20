"""Mesh-grounded offline evaluation of spatial geometry encoders."""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA
from spatial_sdf import OUT,SpatialSDF,load,EXTENT
from sdf_shape_core import extract_surface
from pointnet_shape_core import ROOT,select_device,save_json,sha256,sample_surface
from prepare_pointnet_shape import read_mesh
from evaluate_sdf_shape import surface_metrics,group_ci


class GridDecoder:
    def decode(self,grid,query):
        # Input arrays are x,y,z; scipy interpolation preserves that order.
        axis=np.linspace(-EXTENT,EXTENT,32)
        values=RegularGridInterpolator((axis,axis,axis),grid[0].cpu().numpy(),bounds_error=True)(query[0].cpu().numpy())
        return torch.tensor(values,dtype=torch.float32)[None]


def evaluate(mode,device_name):
    torch.set_num_threads(4);device=select_device(device_name);manifest,arrays=load()
    ids=manifest.index[manifest.overfit if mode=='overfit' else manifest.split=='validation'].to_numpy()
    out=OUT/(mode+'_evaluation');out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Completed evaluation exists')
    saved=torch.load(OUT/mode/'best.pt',map_location='cpu',weights_only=False)
    assert saved['manifest_sha256']==sha256(ROOT/'results/sdf_shape_pilot/manifest.csv')
    model=SpatialSDF().to(device);model.load_state_dict(saved['model']);model.eval()
    with torch.no_grad():codes=model.encode(torch.tensor(arrays['grids'][ids],device=device))
    # Linear spatial encoder: train-only PCA, maximum rank 95 in this small pilot.
    train=manifest.index[manifest.split=='train'].to_numpy();flat=arrays['grids'].reshape(len(manifest),-1)
    pca=PCA(n_components=95,svd_solver='randomized',random_state=26);pca.fit(flat[train])
    pca_codes=pca.transform(flat[ids]);pca_grids=pca.inverse_transform(pca_codes).reshape(-1,32,32,32)
    np.savez_compressed(out/'codes.npz',cell_ids=manifest.iloc[ids].cell_id.to_numpy(str),spatial128=codes.cpu().numpy(),pca95=pca_codes)
    rows=[];refs=[];clouds=[];griddecoder=GridDecoder()
    for at,index in enumerate(ids):
        row=manifest.iloc[index];v,f=read_mesh(ROOT/'results/explorer_dataset'/row['sample']/f'{row.label}.mesh')
        center=np.array([row['normalization_center_'+a+'_um'] for a in 'xyz']);radius=float(row.normalization_radius_um)
        mesh=trimesh.Trimesh((v-center)/radius,f,process=False)
        reference=sample_surface(mesh.vertices,mesh.faces,2048,np.random.default_rng(20261023+index));refs.append(reference);displayed={}
        methods=dict(spatial128=(model,codes[at:at+1]),input_grid=(griddecoder,torch.tensor(arrays['grids'][index])),
            pca95=(griddecoder,torch.tensor(pca_grids[at:at+1])))
        for name,(decoder,z) in methods.items():
            surface,field=extract_surface(decoder,z)
            base=dict(method=name,cell_id=row.cell_id,similarity_group=int(row.similarity_group))
            if surface is None:rows.append(dict(**base,failed_surface=True));continue
            vertices,faces=surface;reconstructed=trimesh.Trimesh(vertices,faces,process=False)
            cloud=sample_surface(vertices,faces,2048,np.random.default_rng(20262023+index));displayed[name]=cloud
            reconstructed.export(out/f"{row.cell_id.replace(':','_')}_{name}.ply")
            boundary=np.concatenate([field[0].ravel(),field[-1].ravel(),field[:,0].ravel(),field[:,-1].ravel(),field[:,:,0].ravel(),field[:,:,-1].ravel()])
            rows.append(dict(**base,failed_surface=False,**surface_metrics(reference,cloud,radius),
                volume_relative_error=float(abs(abs(reconstructed.volume)-abs(mesh.volume))/abs(mesh.volume)),
                watertight=bool(reconstructed.is_watertight),boundary_crossing=bool((boundary<0).any())))
        clouds.append(displayed);print(f'EVALUATED {at+1}/{len(ids)}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'cell_metrics.csv',index=False);summary={}
    for name,values in frame.groupby('method'):
        summary[name]=dict(cells=len(values),failed_surfaces=int(values.failed_surface.sum()),
            **{c:float(values[c].mean()) for c in ['squared_chamfer','f1_at_005_radius','p95_surface_error_um','volume_relative_error']},
            nonwatertight=int((values.watertight==False).sum()),boundary_crossings=int(values.boundary_crossing.sum()))
    comparisons={}
    if mode!='overfit':
        previous=pd.read_csv(ROOT/'results/consistent_sdf/evaluation/cell_metrics.csv')
        before=previous[previous.method=='control_last'][['cell_id','squared_chamfer']].rename(columns={'squared_chamfer':'before'})
        paired=frame[frame.method=='spatial128'].merge(before,on='cell_id',validate='one_to_one')
        paired['improvement']=paired.before-paired.squared_chamfer
        comparisons['vs_previous_encoder']=group_ci(paired,'improvement')
    meta=json.loads((OUT/'data.json').read_text())
    save_json(out/'metrics.json',dict(methods=summary,comparisons=comparisons,mode=mode,checkpoint_epoch=saved['epoch'],
        checkpoint_sha256=sha256(OUT/mode/'best.pt'),atlas_unchanged=meta['atlas_sha256']==sha256(ROOT/'results/explorer_dataset/dataset.json'),
        test_used=False,per_cell_fitting=False,rotation_invariant=False,pca_components=95,
        limits='Orientation-dependent mesh SDF inputs, more information than prior point clouds. Development validation used for selection. No biological superiority claim.'))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=np.argsort(manifest.iloc[ids].sphericity.to_numpy());chosen=order[np.linspace(0,len(order)-1,3).astype(int)]
    fig=plt.figure(figsize=(14,10));names=['reference','input_grid','pca95','spatial128']
    for ri,at in enumerate(chosen):
        for ci,name in enumerate(names):
            ax=fig.add_subplot(3,4,ri*4+ci+1,projection='3d');cloud=refs[at] if name=='reference' else clouds[at].get(name)
            if cloud is not None:ax.scatter(*cloud.T,s=2,alpha=.8,color='#247e91' if name=='reference' else '#ec8a35')
            ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1),box_aspect=(1,1,1));ax.view_init(20,-65);ax.set_axis_off()
            ax.set_title(dict(reference='Original mesh',input_grid='Uncompressed 32³ SDF',pca95='Train-only PCA · 95D',spatial128='Spatial CNN · 128D')[name]+('\n'+manifest.iloc[ids[at]].cell_id if name=='reference' else ''),fontsize=10)
    fig.suptitle('Geometry-first spatial encoder · '+mode+' · no per-cell fitting');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['overfit','pilot','boundary'],required=True);parser.add_argument('--device',default='auto')
    args=parser.parse_args();evaluate(args.mode,args.device)
