"""Validation-led reconstruction/retrieval diagnostics; test results never choose the next architecture."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from pointnet_shape_core import ROOT,read_config,load_data,tensors,PointCloudAutoencoder,chamfer,rotations,batch_hierarchy,select_device,save_json,sha256

FEATURES=['volume_um3','surface_area_um2','sphericity','elongation','flatness','mesh_solidity','major_axis_um','middle_axis_um','minor_axis_um']


def normalized(values):
    return values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)


def canonical(points):
    points=points-points.mean(0)
    _,vectors=np.linalg.eigh(points.T@points)
    vectors=vectors[:,::-1].copy()
    if np.linalg.det(vectors)<0:vectors[:,-1]*=-1
    return points@vectors


def geometric_cost(a,b):
    # A geometry proxy, not biological ground truth. Repeated eigenvalues make PCA frames ambiguous.
    best=float('inf');ta=cKDTree(a)
    for signs in [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]:
        moved=b*np.array(signs);tb=cKDTree(moved)
        value=float(np.square(ta.query(moved)[0]).mean()+np.square(tb.query(a)[0]).mean())
        best=min(best,value)
    return best


def conditional_retrieval(manifest,arrays,ids,embeddings,train_ids,seed):
    frame=manifest.iloc[ids].reset_index(drop=True);train=manifest.iloc[train_ids]
    values=frame[FEATURES].to_numpy(float);training=train[FEATURES].to_numpy(float)
    mean,std=training.mean(0),training.std(0).clip(1e-6)
    classical=(values-mean)/std
    controls=['volume_um3','sphericity','elongation']
    control=(frame[controls].to_numpy(float)-train[controls].mean().to_numpy())/train[controls].std(ddof=0).to_numpy().clip(1e-6)
    cloud=[canonical(np.asarray(arrays['targets'][i])[::8]) for i in ids]
    rows=[];vectors=normalized(embeddings)
    for query in range(len(ids)):
        distances=np.linalg.norm(control-control[query],axis=1)
        candidates=np.flatnonzero((frame.similarity_group.to_numpy()!=frame.iloc[query].similarity_group)&(distances<=1.0))
        if len(candidates)<5:continue
        candidates=candidates[np.argsort(distances[candidates])[:20]]
        learned=candidates[np.argmin(1-vectors[candidates]@vectors[query])]
        traditional=candidates[np.argmin(np.linalg.norm(classical[candidates]-classical[query],axis=1))]
        costs={int(i):geometric_cost(cloud[query],cloud[i]) for i in set([learned,traditional])}
        rows.append(dict(cell_id=frame.iloc[query].cell_id,similarity_group=int(frame.iloc[query].similarity_group),candidates=len(candidates),
                         learned_cell=frame.iloc[learned].cell_id,classical_cell=frame.iloc[traditional].cell_id,
                         learned_cost=costs[int(learned)],classical_cost=costs[int(traditional)],
                         improvement=costs[int(traditional)]-costs[int(learned)]))
    summary=dict(queries=len(rows),interpretation='Cross-similarity-group retrieval among >=5 candidates matched within unit standardized volume/sphericity/elongation distance. Cost: normalized point Chamfer after PCA frame and four proper sign flips; an imperfect predicted-surface geometry proxy, not biological ground truth.')
    if rows:
        result=pd.DataFrame(rows);grouped=result.groupby('similarity_group').improvement.mean().to_numpy()
        rng=np.random.default_rng(seed)
        bootstrap=rng.choice(grouped,size=(2000,len(grouped)),replace=True).mean(1)
        summary.update(groups=len(grouped),mean_group_improvement=float(grouped.mean()),group_bootstrap_95_ci=np.quantile(bootstrap,[.025,.975]).tolist())
    else:summary.update(groups=0,mean_group_improvement=None,group_bootstrap_95_ci=None)
    return summary,rows


def gate(metrics,status,config):
    thresholds=config['stage2_gate'];retrieval=metrics['conditional_retrieval']
    ci=retrieval.get('group_bootstrap_95_ci')
    checks=dict(full_training_not_smoke=not status['smoke'],
        enough_training=status['completed_epochs']>=thresholds['minimum_completed_epochs'],
        enough_validation_cells=metrics['cells']>=thresholds['minimum_validation_cells'],
        enough_validation_groups=metrics['groups']>=thresholds['minimum_validation_groups'],
        beats_untrained=metrics['untrained_improvement']>=thresholds['minimum_untrained_improvement'],
        beats_training_template=metrics['template_improvement']>=thresholds['minimum_template_improvement'],
        stable_resampling=metrics['resampling_self_top1']>=thresholds['minimum_resampling_top1'],
        noncollapsed_embedding=metrics['effective_rank']>=thresholds['minimum_effective_rank'],
        complementary_geometry=bool(ci and retrieval['groups']>=3 and ci[0]>0))
    return dict(eligible=all(checks.values()),checks=checks,decision_basis='Validation only; fixed thresholds before training. No test metric is used.',
        next_stage='VN-DGCNN-style SO(3)-equivariant vectors -> invariant 128D features; pose supplied separately to a coordinate-conditioned SDF decoder (negative inside).',
        status='eligible_for_rotation_invariant_sdf_experiment' if all(checks.values()) else 'keep_pointnet_baseline; do_not_advance_yet',
        limitation='Single-dataset exploratory evidence. Positive geometry-proxy scores do not establish biological utility or fine-protrusion fidelity. Inspect reconstructions and repeat with independent specimens before biological claims.')


def evaluate(config,smoke=False,device_override=None,split='validation'):
    torch.set_num_threads(config['threads']);device=select_device(device_override or config['device'])
    data=ROOT/config['output_dir']/'data';out=ROOT/config['output_dir']/('smoke' if smoke else 'baseline')
    manifest,arrays=load_data(data);status=json.loads((out/'status.json').read_text())
    if status['status']!='trained':raise ValueError('Finish training before evaluating the architecture gate')
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    if saved['manifest_sha256']!=sha256(data/'manifest.csv'):raise ValueError('Checkpoint belongs to different surfaces')
    ids=manifest.index[(manifest.split==split)&manifest.training_eligible].to_numpy()
    train_ids=manifest.index[(manifest.split=='train')&manifest.training_eligible].to_numpy()
    if smoke:ids=ids[:4]
    model=PointCloudAutoencoder(config).to(device);model.load_state_dict(saved['model']);model.eval()
    initial=PointCloudAutoencoder(config).to(device)
    initial.load_state_dict(torch.load(out/'untrained.pt',map_location='cpu',weights_only=False)['model']);initial.eval()
    # A real training surface nearest the median radial profile, not an invalid mean of unordered points.
    profiles=np.quantile(np.linalg.norm(arrays['points'][train_ids],axis=-1),np.linspace(0,1,16),axis=1).T
    medoid=train_ids[np.argmin(np.square(profiles-np.median(profiles,axis=0)).sum(1))]
    template=torch.from_numpy(np.array(arrays['targets'][medoid],copy=True)).to(device)
    scores=[];latent=[];recon=[];resampled=[];rotated=[]
    rng=np.random.default_rng(config['seed']+101)
    with torch.no_grad():
        for start in range(0,len(ids),config['batch_size']):
            batch=ids[start:start+config['batch_size']];points,target,groups=tensors(arrays,batch,device)
            z,pred=model(points,groups);_,untrained=initial(points,groups)
            template_batch=template[None].expand(len(points),-1,-1)
            loss=chamfer(pred,target,config['chamfer_chunk']);base=chamfer(untrained,target,config['chamfer_chunk']);templ=chamfer(template_batch,target,config['chamfer_chunk'])
            scores.extend(zip(loss.cpu().tolist(),base.cpu().tolist(),templ.cpu().tolist()))
            latent.extend(z.cpu().numpy());recon.extend(pred.cpu().numpy())
            resampling_groups=batch_hierarchy(target,config['centers'],config['neighbors'],device)
            resampled.extend(model.encoder(target,resampling_groups).cpu().numpy())
            r=torch.from_numpy(rotations(len(points),rng)).to(device)
            rotated.extend(model.encoder(points@r.transpose(1,2),groups).cpu().numpy())
    vectors=normalized(np.asarray(latent));resampled=normalized(np.asarray(resampled));rotated=normalized(np.asarray(rotated));scores=np.asarray(scores)
    singular=np.linalg.svd(np.asarray(latent)-np.mean(latent,axis=0),compute_uv=False);energy=singular**2
    probability=energy/max(energy.sum(),1e-12);effective_rank=float(np.exp(-(probability*np.log(probability.clip(1e-12))).sum())) if energy.sum()>1e-12 else 0.
    retrieval,rows=conditional_retrieval(manifest,arrays,ids,np.asarray(latent),train_ids,config['seed'])
    metrics=dict(split=split,cells=len(ids),groups=int(manifest.iloc[ids].similarity_group.nunique()),smoke=smoke,
        mean_squared_chamfer=float(scores[:,0].mean()),untrained_chamfer=float(scores[:,1].mean()),training_template_chamfer=float(scores[:,2].mean()),
        untrained_improvement=float(1-scores[:,0].mean()/max(scores[:,1].mean(),1e-12)),
        template_improvement=float(1-scores[:,0].mean()/max(scores[:,2].mean(),1e-12)),
        resampling_self_top1=float(np.mean(np.argmax(resampled@vectors.T,axis=1)==np.arange(len(ids)))),
        rotation_self_top1=float(np.mean(np.argmax(rotated@vectors.T,axis=1)==np.arange(len(ids)))),
        mean_rotation_cosine_distance=float(np.mean(1-(vectors*rotated).sum(1))),effective_rank=effective_rank,
        mean_reconstruction_rms_um=float(np.mean(np.sqrt(scores[:,0]/2)*manifest.iloc[ids].normalization_radius_um.to_numpy())),
        conditional_retrieval=retrieval,checkpoint_epoch=int(saved['epoch']))
    save_json(out/(split+'_metrics.json'),metrics)
    pd.DataFrame(rows).to_csv(out/(split+'_matched_retrieval.csv'),index=False)
    pd.DataFrame(dict(cell_id=manifest.iloc[ids].cell_id.to_numpy(),squared_chamfer=scores[:,0],untrained_chamfer=scores[:,1],template_chamfer=scores[:,2])).to_csv(out/(split+'_cell_errors.csv'),index=False)
    np.savez_compressed(out/(split+'_reconstructions.npz'),cell_ids=manifest.iloc[ids].cell_id.to_numpy(str),inputs=np.asarray(arrays['points'][ids]),targets=np.asarray(arrays['targets'][ids]),reconstructions=np.asarray(recon))
    if split=='validation':save_json(out/'stage2_decision.json',gate(metrics,status,config))
    print(json.dumps(metrics),flush=True)
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config/pointnet_shape.json');parser.add_argument('--device');parser.add_argument('--smoke',action='store_true');parser.add_argument('--split',choices=['validation','test'],default='validation')
    args=parser.parse_args();evaluate(read_config(ROOT/args.config),args.smoke,args.device,args.split)
