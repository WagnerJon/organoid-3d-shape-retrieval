"""Evaluate the existing point AE's stability on the SAME pilot validation pool."""
import argparse
import numpy as np
import pandas as pd
import torch
from pointnet_shape_core import (ROOT,PointCloudAutoencoder,tensors,batch_hierarchy,rotations,
                                select_device,load_data,read_config,save_json,sha256)
from evaluate_pointnet_shape import normalized
from prepare_sdf_shape import OUT


def main(device_name):
    torch.set_num_threads(4);device=select_device(device_name)
    selected=pd.read_csv(OUT/'manifest.csv');ids=selected[selected.split=='validation'].original_index.to_numpy()
    config=read_config(ROOT/'config/pointnet_shape.json');manifest,arrays=load_data(ROOT/config['output_dir']/'data')
    saved=torch.load(ROOT/config['output_dir']/'baseline/best.pt',weights_only=False,map_location='cpu')
    if saved['manifest_sha256']!=sha256(ROOT/config['output_dir']/'data/manifest.csv'):raise ValueError('Baseline data changed')
    model=PointCloudAutoencoder(config).to(device);model.load_state_dict(saved['model']);model.eval()
    z=[];fresh=[];rotated=[];rng=np.random.default_rng(20261019)
    with torch.no_grad():
        for first in range(0,len(ids),8):
            p,t,g=tensors(arrays,ids[first:first+8],device)
            z.extend(model.encoder(p,g).cpu().numpy())
            fresh.extend(model.encoder(t,batch_hierarchy(t,config['centers'],config['neighbors'],device)).cpu().numpy())
            r=torch.tensor(rotations(len(p),rng),device=device)
            rotated.extend(model.encoder(p@r.transpose(1,2),g).cpu().numpy())
    z=np.asarray(z);unit=normalized(z);energy=np.linalg.svd(z-z.mean(0),compute_uv=False)**2
    probability=energy/max(energy.sum(),1e-15)
    result=dict(cells=len(ids),resampling_self_top1=float((np.argmax(normalized(np.asarray(fresh))@unit.T,axis=1)==np.arange(len(ids))).mean()),
                rotation_self_top1=float((np.argmax(normalized(np.asarray(rotated))@unit.T,axis=1)==np.arange(len(ids))).mean()),
                effective_rank=float(np.exp(-(probability*np.log(probability.clip(1e-15))).sum())),
                scope='Same 48-cell candidate pool as SDF pilot; original model has 2048 input points and more training cells. Independent resampling protocol differs: original cached independent target sample versus fresh third SDF input sample.')
    save_json(OUT/'baseline_stability.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='auto');args=p.parse_args();main(args.device)
