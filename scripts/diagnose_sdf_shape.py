"""Training-only decoder capacity diagnostic with per-cell learnable codes.

These are NOT encoders or transferable embeddings. No validation/test data are
used. If the same decoder can fit these cells with free codes but not an encoder,
the observed average-shape failure is upstream/optimization, not SDF capacity.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
from torch import nn
import trimesh
from pointnet_shape_core import ROOT,select_device,save_json,sample_surface
from train_sdf_shape import OUT,load_pilot,batch
from sdf_shape_core import SDFModel,sdf_loss,extract_surface
from prepare_pointnet_shape import read_mesh
from evaluate_sdf_shape import surface_metrics


def run(device_name,steps):
    torch.set_num_threads(4);device=select_device(device_name)
    manifest,arrays=load_pilot();ids=manifest.index[manifest.overfit].to_numpy()
    out=OUT/'decoder_diagnostic';out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Completed diagnostic exists')
    summaries={};examples={};history=[]
    for kind in ['pointnet','vn']:
        torch.manual_seed(20261020);rng=np.random.default_rng(20261020)
        model=SDFModel(kind).to(device)
        # Generous, independent codes: this measures decoder capacity, not learned shape features.
        shape=(len(ids),128,3) if kind=='vn' else (len(ids),128)
        codes=nn.Parameter(torch.randn(shape,device=device)*(.3 if kind=='vn' else .5))
        optimizer=torch.optim.Adam([dict(params=model.decoder.parameters(),lr=.001),dict(params=[codes],lr=.01)])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.0001)
        best=float('inf');started=time.monotonic()
        for step in range(1,steps+1):
            chosen=rng.choice(len(ids),8,replace=False);_,q,t,_=batch(arrays,ids[chosen],device,rng)
            z=codes[torch.tensor(chosen,device=device)]
            prediction=model.decode(z,q);loss=sdf_loss(prediction,t)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();scheduler.step()
            if step%50==0 or step==1:
                with torch.no_grad():
                    errors=[]
                    for first in range(0,len(ids),4):
                        _,q,t,_=batch(arrays,ids[first:first+4],device)
                        errors.append(float(sdf_loss(model.decode(codes[first:first+4],q),t).cpu()))
                error=float(np.mean(errors));history.append(dict(kind=kind,step=step,sdf_l1=error))
                if error<best:
                    best=error;torch.save(dict(kind=kind,decoder=model.decoder.state_dict(),codes=codes.detach(),step=step),out/(kind+'.pt'))
                print(json.dumps(dict(kind=kind,step=step,training_sdf_l1=error,seconds=time.monotonic()-started)),flush=True)
        saved=torch.load(out/(kind+'.pt'),weights_only=False,map_location=device);model.decoder.load_state_dict(saved['decoder']);model.eval()
        results=[];clouds=[]
        with torch.no_grad():
            for i,index in enumerate(ids):
                row=manifest.iloc[index];vertices,faces=read_mesh(ROOT/'results/explorer_dataset'/row['sample']/f'{row.label}.mesh')
                center=np.array([row['normalization_center_'+a+'_um'] for a in 'xyz']);radius=row.normalization_radius_um
                reference=sample_surface((vertices-center)/radius,faces,2048,rng)
                surface,_=extract_surface(model,saved['codes'][i:i+1])
                if surface is None:raise ValueError('Decoder diagnostic produced no surface')
                v,f=surface;pred=sample_surface(v,f,2048,rng)
                results.append(dict(cell_id=row.cell_id,**surface_metrics(reference,pred,radius)))
                clouds.append((reference,pred));trimesh.Trimesh(v,f,process=False).export(out/f'{i:02d}_{kind}.ply')
        summaries[kind]=dict(training_cells=len(ids),steps=steps,best_step=saved['step'],best_training_sdf_l1=best,
            **{key:float(np.mean([r[key] for r in results])) for key in ['squared_chamfer','p95_surface_error_um','f1_at_005_radius']},
            cell_metrics=results)
        examples[kind]=clouds
    save_json(out/'metrics.json',dict(scope='Decoder-only capacity check on the SAME twelve training cells; free per-cell codes, no encoder, no generalization evidence.',methods=summaries))
    save_json(out/'history.json',history)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,10));ordered=np.argsort(manifest.iloc[ids].sphericity.to_numpy());chosen=ordered[[0,len(ids)//2,-1]]
    for ri,at in enumerate(chosen):
        for ci,key in enumerate(['reference','pointnet','vn']):
            cloud=examples['pointnet'][at][0] if key=='reference' else examples[key][at][1]
            ax=fig.add_subplot(3,3,ri*3+ci+1,projection='3d');ax.scatter(*cloud.T,s=2,color='#247e91' if key=='reference' else '#ec8a35')
            ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1),box_aspect=(1,1,1));ax.set_axis_off();ax.view_init(20,-65)
            ax.set_title(('Original '+manifest.iloc[ids[at]].cell_id) if key=='reference' else key+' decoder + free code')
    fig.suptitle('Decoder capacity diagnostic · training cells only · no learned encoder');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'decoder_capacity.png',dpi=160);plt.close(fig)
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='auto');p.add_argument('--steps',type=int,default=1500);args=p.parse_args();run(args.device,args.steps)
