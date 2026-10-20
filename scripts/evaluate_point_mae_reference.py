"""Matched geometry, resampling/rotation and maskedpatch diagnostics, offline only."""
import json
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from point_mae_reference import OUT,ROOT,model,encode,masked_prediction
from run_point_mae_reference import draw
from spatial_benchmark_core import SIGNS,frame_from_cov,mesh_for,geometry_cost
from allen_reference_core import random_rotation
from pointnet_shape_core import sample_surface,sha256
from check_cajal_validation import candidate_sets
from evaluate_spatial_benchmark import residual_fit,residual_apply
from evaluate_sdf_shape import group_ci


def load_arm(arm):
    m=model()
    if arm=='adapted':
        c=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False)
        if c['protocol_sha256']!=sha256(OUT/'protocol.json'):raise ValueError('Checkpoint protocol mismatch')
        m.load_state_dict(c['model'],strict=True)
    return m.to('mps').eval()


def features(m,points,axes):
    with torch.no_grad():
        native=encode(m,torch.tensor(points,device='mps')).cpu().numpy()
        orbit=[]
        for sign in SIGNS:
            p=(points@axes*sign).astype(np.float32)
            orbit.append(encode(m,torch.tensor(p,device='mps')).cpu().numpy())
    return native,np.mean(orbit,axis=0)


def encode_arm(arm,frame,pools,axes):
    path=OUT/(arm+'_features.npz')
    if path.exists():
        with np.load(path) as a:return {k:a[k] for k in a.files}
    m=load_arm(arm);native=[];orbit=[]
    for first in range(0,len(frame),8):
        ids=np.arange(first,min(first+8,len(frame)));points=np.stack([draw(pools[i],i) for i in ids])
        a,b=features(m,points,axes[ids]);native.extend(a);orbit.extend(b)
        if first%80==0:
            progress=dict(status='extracting',arm=arm,completed=min(first+8,len(frame)),total=len(frame))
            (OUT/'status.json').write_text(json.dumps(progress));print(arm,'features',min(first+8,len(frame)),'/',len(frame),flush=True)
    answer=dict(native=np.array(native),pose_average=np.array(orbit));np.savez_compressed(path,**answer)
    return answer


def stability(m,arm,features0,frame,pools,axes,covariance,ids):
    repeats={k:[] for k in features0};rotation={k:[] for k in features0};rot_ids=[]
    for first in range(0,len(ids),8):
        ii=ids[first:first+8];p=np.stack([draw(pools[i],i,1) for i in ii]);a,b=features(m,p,axes[ii])
        repeats['native'].extend(a);repeats['pose_average'].extend(b)
    for i in ids:
        if not (ROOT/'results/allen_reference/features'/f'{i:04d}_2.npz').exists():continue
        r=random_rotation(np.random.default_rng(20261021+int(i)))
        p=(draw(pools[i],i)@r)[None];_,basis,_,_=frame_from_cov(np.zeros(3),r.T@covariance[i]@r)
        a,b=features(m,p,basis[None]);rotation['native'].extend(a);rotation['pose_average'].extend(b);rot_ids.append(i)
    result={};groups=frame.iloc[ids].similarity_group.to_numpy();positions=np.array([np.flatnonzero(ids==i)[0] for i in rot_ids])
    for kind,z in features0.items():
        gallery=z[ids];res=np.asarray(repeats[kind]);rot=np.asarray(rotation[kind]);d=cdist(gallery,gallery);dr=cdist(res,gallery);dt=cdist(rot,gallery);jaccard=[]
        for i in range(len(ids)):
            allowed=np.flatnonzero(groups!=groups[i]);a=set(allowed[np.argsort(d[i,allowed])[:5]]);b=set(allowed[np.argsort(dr[i,allowed])[:5]]);jaccard.append(len(a&b)/len(a|b))
        train=z[frame.split=='train'];energy=np.linalg.svd(train-train.mean(0),compute_uv=False)**2;p=energy/max(energy.sum(),1e-15)
        result[kind]=dict(resampling_cells=len(ids),resampling_self_top1=float((dr.argmin(1)==np.arange(len(ids))).mean()),neighbor_jaccard=float(np.mean(jaccard)),rotation_cells=len(rot_ids),rotation_self_top1=float((dt.argmin(1)==positions).mean()),rotation_relative_code_error=float(np.mean(np.linalg.norm(rot-gallery[positions],axis=1)/np.linalg.norm(gallery[positions],axis=1))),effective_rank=float(np.exp(-(p*np.log(p.clip(1e-15))).sum())))
    np.savez_compressed(OUT/(arm+'_stability_features.npz'),resample_native=repeats['native'],resample_pose=repeats['pose_average'],rotation_native=rotation['native'],rotation_pose=rotation['pose_average'],rotation_indices=rot_ids)
    return result


def retrieval(readouts):
    frame,pools,_=candidate_sets();old=pd.read_csv(ROOT/'results/allen_reference/evaluation/retrieval.csv');old_summary=json.loads((ROOT/'results/allen_reference/evaluation/retrieval.json').read_text())
    costs={};lookup={c:i for i,c in enumerate(frame.cell_id)}
    for row in old.itertuples():costs[tuple(sorted([lookup[row.cell_id],lookup[row.neighbor]]))]=row.geometry_cost
    clouds=[]
    for row in frame.to_dict('records'):
        v,f=mesh_for(row);clouds.append(sample_surface(v,f,384,np.random.default_rng(20261030+int(row['original_index']))))
    new=[]
    for method,x in readouts.items():
        d=cdist(x,x)
        for i,candidates in pools.items():
            j=int(candidates[np.argmin(d[i,candidates])]);pair=tuple(sorted([i,j]))
            if pair not in costs:costs[pair]=geometry_cost(clouds[pair[0]],clouds[pair[1]])
            new.append(dict(method=method,query=i,cell_id=frame.iloc[i].cell_id,neighbor=frame.iloc[j].cell_id,similarity_group=int(frame.iloc[i].similarity_group),geometry_cost=costs[pair]))
    rows=pd.concat([old,pd.DataFrame(new)],ignore_index=True);rows.to_csv(OUT/'retrieval.csv',index=False)
    names={r['id']:r['name'] for r in old_summary['methods']};names.update({k:k.replace('point_mae_','Point-MAE ').replace('_',' ') for k in readouts});summary=[]
    for method,sub in rows.groupby('method',sort=False):
        sub=sub.copy();entry=dict(id=method,name=names[method],queries=len(sub),geometry_cost=float(sub.groupby('similarity_group').geometry_cost.mean().mean()))
        for base in ['morph_all','morph_adjusted','spatial_robust','spherical_harmonics_adjusted','allen_head_decoder_adjusted']:
            b=rows[rows.method==base].set_index('query');sub['improvement']=[b.loc[q].geometry_cost-v for q,v in zip(sub['query'],sub.geometry_cost)];entry['vs_'+base]=group_ci(sub,'improvement')
        summary.append(entry)
    result=dict(methods=summary,primary=next(r for r in summary if r['id']=='point_mae_adapted_pose_average_adjusted'),scope='Exploratory validation geometry; positive paired differences favor the candidate. Multiple readouts, not confirmatory biology.')
    (OUT/'retrieval.json').write_text(json.dumps(result,indent=2));print('Primary retrieval',json.dumps(result['primary']),flush=True)


def patch_diagnostic(m,arm,frame,pools,ids):
    state=np.random.get_state();np.random.seed(1219);rows=[];visual={};visual_ids=[r['query'] for r in json.loads((ROOT/'results/spatial_benchmark/development/visual_queries.json').read_text())]
    for i in ids:
        p=draw(pools[i],i)
        with torch.no_grad():
            pred,target,center,mask,groups=masked_prediction(m,torch.tensor(p[None],device='mps'))
            loss=float(m.loss_func(pred,target).cpu());baseline=float(m.loss_func(torch.zeros_like(pred),target).cpu())
        rows.append(dict(arm=arm,cell_id=frame.iloc[i].cell_id,similarity_group=int(frame.iloc[i].similarity_group),patch_cd=loss,center_only_cd=baseline))
        cid=frame.iloc[i].cell_id
        if cid in visual_ids:
            c=center[mask].cpu().numpy();visible=(groups[~mask]+center[~mask,None]).reshape(-1,3).cpu().numpy();rebuild=(pred.cpu().numpy()+c[:,None]).reshape(-1,3);truth=(target.cpu().numpy()+c[:,None]).reshape(-1,3)
            visual[cid+'_visible']=visible;visual[cid+'_predicted']=rebuild;visual[cid+'_target']=truth
    np.random.set_state(state);np.savez_compressed(OUT/(arm+'_patch_visuals.npz'),**visual)
    return rows


def main():
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.55)
    (OUT/'status.json').write_text(json.dumps(dict(status='evaluating')))
    signature=dict(protocol_sha256=sha256(OUT/'protocol.json'),checkpoint_sha256=sha256(OUT/'checkpoints/pretrain.pth'),adapted_sha256=sha256(OUT/'best.pt'),manifest_sha256=sha256(OUT/'manifest.csv'),baseline_sha256=sha256(ROOT/'results/allen_reference/evaluation/retrieval.csv'))
    path=OUT/'evaluation_provenance.json'
    if path.exists() and json.loads(path.read_text())!=signature:raise ValueError('Stale evaluation cache')
    path.write_text(json.dumps(signature,indent=2))
    frame=pd.read_csv(OUT/'manifest.csv');pools=np.load(OUT/'pools.npy',mmap_mode='r');f=np.load(OUT/'frames.npz');axes=f['axes'];cov=f['covariance'];dev,_,_=candidate_sets();positions={c:i for i,c in enumerate(frame.cell_id)};ids=np.array([positions[c] for c in dev.cell_id])
    original96=set(pd.read_csv(ROOT/'results/spatial_benchmark/dev_manifest.csv').query("split=='train'").cell_id);train=np.flatnonzero(frame.cell_id.isin(original96));readouts={};checks={};patches=[]
    for arm in ['pretrained','adapted']:
        all_features=encode_arm(arm,frame,pools,axes);m=load_arm(arm)
        checks[arm]=stability(m,arm,all_features,frame,pools,axes,cov,ids);print('Stability',arm,json.dumps(checks[arm]),flush=True)
        patches.extend(patch_diagnostic(m,arm,frame,pools,ids))
        for kind,x in all_features.items():
            center=x[train].mean(0);scale=x[train].std(0).clip(1e-6);standard=(x-center)/scale;fit=residual_fit(frame.iloc[train],standard[train]);adjusted=residual_apply(frame,standard,fit)
            np.savez_compressed(OUT/f'{arm}_{kind}_fit.npz',center=center,scale=scale,**fit)
            for name,array in [('raw',x),('standardized',standard),('adjusted',adjusted)]:readouts[f'point_mae_{arm}_{kind}_{name}']=array[ids]
        del m;torch.mps.empty_cache()
    (OUT/'stability.json').write_text(json.dumps(checks,indent=2));pd.DataFrame(patches).to_csv(OUT/'patch_reconstruction.csv',index=False);np.savez_compressed(OUT/'readouts.npz',ids=dev.cell_id.to_numpy(dtype=str),**readouts)
    retrieval(readouts)
    protocol=json.loads((OUT/'protocol.json').read_text());unchanged=sha256(ROOT/'results/explorer_dataset/dataset.json')==protocol['atlas_sha256']
    assert unchanged
    (OUT/'evaluation.completed.json').write_text(json.dumps(dict(status='complete',atlas_unchanged=unchanged,test_used=False,embedding_cells=len(frame),evaluated_cells=len(ids)),indent=2));print('Point-MAE evaluation complete',flush=True)


if __name__=='__main__':main()
