"""Offline evaluation of the Lite-based 3D pilot and its literal CNN reference."""
import itertools,json
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from shapeembed3d_core import OUT,ROOT,ShapeEmbed3D,edm,normalize_edm,gw_edm_loss
from pointnet_shape_core import sha256,sample_surface
from spatial_benchmark_core import mesh_for,geometry_cost
from allen_reference_core import random_rotation
from check_cajal_validation import candidate_sets
from evaluate_spatial_benchmark import residual_fit,residual_apply
from evaluate_sdf_shape import group_ci,surface_metrics


def load_model(kind):
    state=torch.load(OUT/kind/'best.pt',map_location='cpu',weights_only=False)
    if state['protocol_sha256']!=sha256(OUT/'protocol.json'):raise ValueError('Stale model/protocol')
    m=ShapeEmbed3D(kind).to('mps');m.load_state_dict(state['model'],strict=True)
    return m.eval()


def encode(m,matrices):
    values=[];variances=[]
    with torch.no_grad():
        for first in range(0,len(matrices),16):
            mean,log_var,_,_=m.encode(torch.tensor(np.array(matrices[first:first+16]),device='mps'))
            values.extend(mean.cpu().numpy());variances.extend(log_var.cpu().numpy())
    return np.array(values),np.array(variances)


def stability(m,codes,frame,ids,points,matrices):
    resample,_=encode(m,matrices[ids,1]);rng=np.random.default_rng(20261023)
    perms=np.stack([rng.permutation(128) for _ in ids]);d=matrices[ids,0];permuted=d[np.arange(len(ids))[:,None,None],perms[:,:,None],perms[:,None,:]];permutation,_=encode(m,permuted)
    rot_ids=[i for i in ids if (ROOT/'results/allen_reference/features'/f'{i:04d}_2.npz').exists()];probe={k:[] for k in ['rotation','reflection','translate_scale']}
    with torch.no_grad():
        for i in rot_ids:
            p=np.array(points[i,0]);r=random_rotation(np.random.default_rng(20261024+int(i)))
            for k,q in [('rotation',p@r),('reflection',p*np.array([-1,1,1],np.float32)),('translate_scale',p*3.7+np.array([2,-5,3],np.float32))]:
                probe[k].append(edm(torch.tensor(q[None],device='mps')).cpu().numpy()[0])
    gallery=codes[ids];positions=np.array([np.flatnonzero(ids==i)[0] for i in rot_ids]);groups=frame.iloc[ids].similarity_group.to_numpy();base=cdist(gallery,gallery);jaccard=[]
    for i in range(len(ids)):
        allowed=np.flatnonzero(groups!=groups[i]);a=set(allowed[np.argsort(base[i,allowed])[:5]]);b=set(allowed[np.argsort(cdist(resample[i:i+1],gallery)[0,allowed])[:5]]);jaccard.append(len(a&b)/len(a|b))
    result=dict(resampling_cells=len(ids),resampling_self_top1=float((cdist(resample,gallery).argmin(1)==np.arange(len(ids))).mean()),neighbor_jaccard=float(np.mean(jaccard)),permutation_cells=len(ids),permutation_self_top1=float((cdist(permutation,gallery).argmin(1)==np.arange(len(ids))).mean()),permutation_relative_error=float(np.mean(np.linalg.norm(permutation-gallery,axis=1)/np.linalg.norm(gallery,axis=1))))
    for k,d in probe.items():
        z,_=encode(m,np.array(d));result[k+'_cells']=len(rot_ids);result[k+'_self_top1']=float((cdist(z,gallery).argmin(1)==positions).mean());result[k+'_relative_error']=float(np.mean(np.linalg.norm(z-gallery[positions],axis=1)/np.linalg.norm(gallery[positions],axis=1)))
    train=codes[frame.split=='train'];energy=np.linalg.svd(train-train.mean(0),compute_uv=False)**2;p=energy/max(energy.sum(),1e-15);result['effective_rank']=float(np.exp(-(p*np.log(p.clip(1e-15))).sum())) if energy.sum()>1e-15 else 0.;return result


def retrieve(readouts,input_matrices):
    frame,pools,pairs=candidate_sets();old=pd.read_csv(ROOT/'results/point_mae/retrieval.csv');old_names={r['id']:r['name'] for r in json.loads((ROOT/'results/point_mae/retrieval.json').read_text())['methods']};positions={c:i for i,c in enumerate(frame.cell_id)};costs={};clouds=[]
    for r in old.itertuples():costs[tuple(sorted([positions[r.cell_id],positions[r.neighbor]]))]=r.geometry_cost
    for row in frame.to_dict('records'):
        v,f=mesh_for(row);clouds.append(sample_surface(v,f,384,np.random.default_rng(20261030+int(row['original_index']))))
    matrices={k:cdist(x,x) for k,x in readouts.items()};direct=np.full((len(frame),len(frame)),np.inf);np.fill_diagonal(direct,0.)
    with torch.no_grad():
        for first in range(0,len(pairs),16):
            pair=pairs[first:first+16];a=torch.tensor(np.stack([input_matrices[i] for i,j in pair]),device='mps');b=torch.tensor(np.stack([input_matrices[j] for i,j in pair]),device='mps');a,_=normalize_edm(a);b,_=normalize_edm(b);ab,_=gw_edm_loss(a,b);ba,_=gw_edm_loss(b,a)
            for (i,j),v in zip(pair,((ab+ba)/2).cpu().numpy()):direct[i,j]=direct[j,i]=v
    np.save(OUT/'direct_edm_gw.npy',direct);matrices['shapeembed3d_direct_edm_gw']=direct
    new=[]
    for method,d in matrices.items():
        for i,candidates in pools.items():
            j=int(candidates[np.argmin(d[i,candidates])]);pair=tuple(sorted([i,j]))
            if pair not in costs:costs[pair]=geometry_cost(clouds[pair[0]],clouds[pair[1]])
            new.append(dict(method=method,query=i,cell_id=frame.iloc[i].cell_id,neighbor=frame.iloc[j].cell_id,similarity_group=int(frame.iloc[i].similarity_group),geometry_cost=costs[pair]))
    rows=pd.concat([old,pd.DataFrame(new)],ignore_index=True);rows.to_csv(OUT/'retrieval.csv',index=False);summary=[]
    for method,sub in rows.groupby('method',sort=False):
        sub=sub.copy();r=dict(id=method,name=old_names.get(method,method.replace('shapeembed3d_','Lite3D ').replace('_',' ')),queries=len(sub),geometry_cost=float(sub.groupby('similarity_group').geometry_cost.mean().mean()))
        for baseline in ['morph_all','morph_adjusted','spatial_robust','spherical_harmonics_adjusted','point_mae_adapted_pose_average_raw','shapeembed3d_direct_edm_gw']:
            b=rows[rows.method==baseline].set_index('query');sub['delta']=[b.loc[q].geometry_cost-v for q,v in zip(sub['query'],sub.geometry_cost)];r['vs_'+baseline]=group_ci(sub,'delta')
        summary.append(r)
    result=dict(methods=summary,primary=next(r for r in summary if r['id']=='shapeembed3d_invariant_adjusted'),scope='Exploratory matched-validation geometry, not biological discrimination.')
    (OUT/'retrieval.json').write_text(json.dumps(result,indent=2));print('Primary retrieval',json.dumps(result['primary']),flush=True)


def align_cloud(reference,cloud,allow_reflection=False):
    """Finite multi-start ICP; O3 and proper-only variants, never scale fitting."""
    ac=reference.mean(0);bc=cloud.mean(0);a=reference-ac;b=cloud-bc;_,aa=np.linalg.eigh(a.T@a);_,bb=np.linalg.eigh(b.T@b);a=a@aa;b=b@bb;ta=cKDTree(a);best=float('inf');answer=None;parity=None
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product([-1,1],repeat=3):
            r=np.eye(3)[:,permutation]*signs;det=float(np.linalg.det(aa@r@bb.T))
            if not allow_reflection and det<0:continue
            moved=b@r.T
            for _ in range(8):
                _,idx=ta.query(moved);u,_,vt=np.linalg.svd(moved.T@a[idx]);step=u@vt
                if np.linalg.det(step)<0:u[:,-1]*=-1;step=u@vt
                moved=moved@step
            cost=float(np.square(ta.query(moved)[0]).mean()+np.square(cKDTree(moved).query(a)[0]).mean())
            if cost<best:best=cost;answer=moved@aa.T+ac;parity=det
    return answer,parity


def reconstruction(m,kind,frame,ids,points,matrices):
    records=[];visual={};queries=[r['query'] for r in json.loads((ROOT/'results/spatial_benchmark/development/visual_queries.json').read_text())];self_cost=[];repeat_cost=[];marginal=[]
    with torch.no_grad():
        for i in ids:
            row=frame.iloc[i];output=m(torch.tensor(np.array(matrices[i:i+1,0]),device='mps'),deterministic=True);pred,target,_,_,cloud,s=output
            gw,t=gw_edm_loss(pred,target);self_gw,_=gw_edm_loss(target,target);other,_=normalize_edm(torch.tensor(np.array(matrices[i:i+1,1]),device='mps'));repeat_gw,_=gw_edm_loss(target,other)
            self_cost.append(float(self_gw.cpu()));repeat_cost.append(float(repeat_gw.cpu()));error=max(float((t.sum(-1)-1/128).abs().max().cpu()),float((t.sum(-2)-1/128).abs().max().cpu()))*128
            if error>1e-3:raise ValueError('Invalid uniform transport marginals')
            marginal.append(error)
            with np.load(ROOT/'results/allen_reference/data'/f'{i:04d}.npz') as a:scale=float(a['scale']);center=a['center']
            p=cloud.cpu().numpy()[0];p=(p-p.mean(0))*float(s.cpu().reshape(-1)[0])*scale
            v,f=mesh_for(row);reference=sample_surface(v,f,2048,np.random.default_rng(20261025+int(row.original_index)));sampled=np.array(points[i,0])*scale+center
            for label,raw,reflect in [('source_sampling_128',sampled,False),(kind+'_proper',p,False),(kind+'_o3',p,True)]:
                aligned,det=(raw,1.) if label=='source_sampling_128' else align_cloud(reference,raw,reflect)
                rec=dict(cell_id=row.cell_id,similarity_group=int(row.similarity_group),method=label,gw_edm=float(gw.cpu()) if label!='source_sampling_128' else None,alignment_reflected=det<0,**surface_metrics(reference,aligned,float(row.normalization_radius_um)));records.append(rec)
                if row.cell_id in queries:visual[row.cell_id+'_'+label]=aligned
            if row.cell_id in queries:visual[row.cell_id+'_reference']=reference
    np.savez_compressed(OUT/(kind+'_reconstructions.npz'),**visual)
    return records,dict(self_gw_mean=float(np.mean(self_cost)),resampling_gw_mean=float(np.mean(repeat_cost)),max_relative_marginal_error=float(max(marginal)))


def main():
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.55)
    prepared=json.loads((OUT/'prepared.json').read_text())
    assert sha256(OUT/'points.npy')==prepared['points_sha256'] and sha256(OUT/'edms.npy')==prepared['edms_sha256']
    signature=dict(protocol_sha256=sha256(OUT/'protocol.json'),input_baseline_protocol_sha256=sha256(OUT/'input_baseline_protocol.json'),models={k:sha256(OUT/k/'best.pt') for k in ['invariant','cnn']},points_sha256=sha256(OUT/'points.npy'),matrices_sha256=sha256(OUT/'edms.npy'),baseline_sha256=sha256(ROOT/'results/point_mae/retrieval.csv'))
    p=OUT/'evaluation_provenance.json'
    if p.exists() and json.loads(p.read_text())!=signature:raise ValueError('Cached evaluation provenance mismatch')
    p.write_text(json.dumps(signature,indent=2));frame=pd.read_csv(OUT/'manifest.csv');points=np.load(OUT/'points.npy',mmap_mode='r');matrices=np.load(OUT/'edms.npy',mmap_mode='r');dev,_,_=candidate_sets();positions={c:i for i,c in enumerate(frame.cell_id)};ids=np.array([positions[c] for c in dev.cell_id]);train96=set(pd.read_csv(ROOT/'results/spatial_benchmark/dev_manifest.csv').query("split=='train'").cell_id);train=np.flatnonzero(frame.cell_id.isin(train96));assert len(train)==96
    readouts={};checks={};records=[];calibration={}
    for kind in ['invariant','cnn']:
        m=load_model(kind);z,logvar=encode(m,matrices[:,0]);np.savez_compressed(OUT/(kind+'_codes.npz'),ids=frame.cell_id.to_numpy(dtype=str),mean=z,logvar=logvar)
        checks[kind]=stability(m,z,frame,ids,points,matrices);print('Stability',kind,json.dumps(checks[kind]),flush=True)
        center=z[train].mean(0);scale=z[train].std(0).clip(1e-6);standard=(z-center)/scale;fit=residual_fit(frame.iloc[train],standard[train]);adjusted=residual_apply(frame,standard,fit);np.savez_compressed(OUT/(kind+'_fit.npz'),center=center,scale=scale,**fit)
        for name,x in [('raw',z),('standardized',standard),('adjusted',adjusted)]:readouts[f'shapeembed3d_{kind}_{name}']=x[ids]
        rec,cal=reconstruction(m,kind,frame,ids,points,matrices);records.extend(rec);calibration[kind]=cal;del m;torch.mps.empty_cache()
    (OUT/'stability.json').write_text(json.dumps(checks,indent=2));(OUT/'matching_audit.json').write_text(json.dumps(calibration,indent=2));pd.DataFrame(records).drop_duplicates(['cell_id','method']).to_csv(OUT/'reconstruction.csv',index=False);np.savez_compressed(OUT/'readouts.npz',ids=dev.cell_id.to_numpy(dtype=str),**readouts);retrieve(readouts,np.array(matrices[ids,0]))
    unchanged=sha256(ROOT/'results/explorer_dataset/dataset.json')==json.loads((OUT/'protocol.json').read_text())['atlas_sha256'];assert unchanged
    (OUT/'evaluation.completed.json').write_text(json.dumps(dict(status='complete',atlas_unchanged=unchanged,embedding_cells=len(frame),evaluated_cells=len(ids),test_used=False),indent=2));print('Lite3D evaluation complete',flush=True)


if __name__=='__main__':main()
