"""Controlled surface identities under SO(3) rotations/resampling; no generated shapes enter training."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import trimesh
from pointnet_shape_core import ROOT,read_config,PointCloudAutoencoder,sample_surface,normalize_cloud,hierarchy,rotations,select_device,save_json
from evaluate_pointnet_shape import normalized


def synthetic_shapes():
    for family in range(6):
        for strength in [.12,.25,.4,.6]:
            mesh=trimesh.creation.icosphere(subdivisions=3)
            directions=np.asarray(mesh.vertices).copy()
            if family==0:
                mesh.vertices*=np.array([1+strength,1,1-strength/2])
            else:
                centers=[np.array([0.,0.,1.])]
                if family==2:centers.append(np.array([0.,0.,-1.]))
                if family==3:centers.append(np.array([1.,0.,0.]))
                if family==4:centers=[p/np.linalg.norm(p) for p in np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])]
                if family==5:centers=[np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])]
                radial=1+strength*sum(np.exp((directions@center-1)/.045) for center in centers)
                mesh.vertices=directions*radial[:,None]
            mesh.vertices*= (1/max(abs(mesh.volume),1e-12))**(1/3) # equal physical volume
            covariance=np.linalg.eigvalsh(np.cov(np.asarray(mesh.vertices).T)).clip(1e-12)
            features=[mesh.area,(covariance[-1]/covariance[0])**.5,(covariance[-2]/covariance[0])**.5]
            yield f'family{family}_strength{strength}',mesh,np.array(features)


def benchmark(config,device_override=None):
    device=select_device(device_override or config['device']);torch.set_num_threads(config['threads'])
    run=ROOT/config['output_dir']/'baseline';saved=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    model=PointCloudAutoencoder(config).to(device);model.load_state_dict(saved['model']);model.eval()
    rng=np.random.default_rng(config['seed']+313);clouds=[];groups=[];features=[];names=[]
    for name,mesh,metrics in synthetic_shapes():
        names.append(name);features.append(metrics)
        for _ in range(2):
            raw=sample_surface(mesh.vertices,mesh.faces,config['points'],rng)
            points,_,_,_=normalize_cloud(raw,raw)
            points=points@rotations(1,rng)[0].T
            clouds.append(points);groups.append(hierarchy(points,config['centers'],config['neighbors']))
    vectors=[]
    with torch.no_grad():
        for start in range(0,len(clouds),config['batch_size']):
            ids=range(start,min(len(clouds),start+config['batch_size']))
            points=torch.from_numpy(np.stack([clouds[i] for i in ids])).to(device)
            h={key:torch.from_numpy(np.stack([groups[i][key] for i in ids])).long().to(device) for key in groups[0]}
            vectors.extend(model.encoder(points,h).cpu().numpy())
    vectors=normalized(np.asarray(vectors));query=vectors[1::2];candidate=vectors[::2]
    distances=1-query@candidate.T
    ranks=np.argsort(np.argsort(distances,axis=1),axis=1)[np.arange(len(names)),np.arange(len(names))]+1
    feature=np.asarray(features);std=feature.std(0).clip(1e-12);feature/=std
    classical=np.linalg.norm(feature[:,None]-feature[None],axis=-1).argmin(1)
    result=dict(shapes=len(names),families=6,equal_volume=True,rotated_resampled_identity_top1=float(np.mean(ranks==1)),
        rotated_resampled_mean_reciprocal_rank=float(np.mean(1/ranks)),classical_identity_top1=float(np.mean(classical==np.arange(len(names)))),
        interpretation='Controlled synthetic identity stress test, not cell-type labels. Descriptor baseline uses exact source mesh area and covariance axis ratios. Strong classical performance here is expected and must not be called a learned advantage.')
    save_json(run/'synthetic_benchmark.json',result)
    pd.DataFrame(dict(shape=names,rotated_resampled_rank=ranks)).to_csv(run/'synthetic_ranks.csv',index=False)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config/pointnet_shape.json');parser.add_argument('--device')
    args=parser.parse_args();benchmark(read_config(ROOT/args.config),args.device)
