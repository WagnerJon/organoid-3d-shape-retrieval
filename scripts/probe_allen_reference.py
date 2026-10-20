"""Measure full released model on Apple GPU before fixing local training budget."""
import json,time
import numpy as np
import pandas as pd
import torch
from allen_reference_core import AllenModel,OUT
from pointnet_shape_core import sample_surface
from spatial_benchmark_core import mesh_for


def main():
    torch.set_num_threads(4);device='mps';torch.manual_seed(42)
    frame=pd.read_csv('results/pointnet_shape/data/manifest.csv');row=frame[(frame.split=='train')&frame.training_eligible].iloc[0]
    v,f=mesh_for(row);center=(v.min(0)+v.max(0))/2;scale=np.ptp(v,axis=0).max();v=(v-center)/scale
    p=sample_surface(v,f,8192,np.random.default_rng(42)).astype(np.float32)
    model=AllenModel().to(device);opt=torch.optim.Adam(model.parameters(),lr=.0002)
    points=torch.tensor(p[None],device=device);q=torch.rand(1,20000,3,device=device)-.5;t=torch.linalg.vector_norm(q,dim=-1)-.35
    torch.mps.synchronize();started=time.monotonic()
    with torch.no_grad():z=model.encode(points);prediction=model.decode(z,q[:,:1024])
    torch.mps.synchronize();inference=time.monotonic()-started
    started=time.monotonic();opt.zero_grad(set_to_none=True);z=model.encode(points);leaf=z.detach().requires_grad_(True);losses=[]
    for first in range(0,20000,2048):
        pred=model.decode(leaf,q[:,first:first+2048]);loss=torch.abs(pred-t[:,first:first+2048]).sum()/20000;loss.backward();losses.append(float(loss.detach().cpu()))
    z.backward(leaf.grad);opt.step();torch.mps.synchronize();elapsed=time.monotonic()-started
    result=dict(device=device,parameters=sum(p.numel() for p in model.parameters()),input_points=8192,query_points=20000,
        inference_seconds=inference,training_cell_seconds=elapsed,estimated_epoch_seconds=elapsed*469,
        allocated_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory(),recommended_bytes=torch.mps.recommended_max_memory(),
        finite=bool(torch.isfinite(prediction).all().cpu()),latent_shape=list(z.shape))
    (OUT/'runtime_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
