"""Data preparation and controlled training of the ShapeEmbedLite-based 3D pilot."""
import json,time,subprocess,sys,os
import numpy as np
import pandas as pd
import torch
from pointnet_shape_core import ROOT,sha256
from shapeembed3d_core import OUT,SOURCE,ShapeEmbed3D,edm,vae_loss,fps_indices


def protocol():
    p=OUT/'protocol.json'
    if p.exists():return json.loads(p.read_text())
    d=dict(name='Experimental ShapeEmbedLite3D, not an official published 3D implementation',
        official_source='https://github.com/uhlmanngroup/ShapeEmbedLite',source_commit='df178ec4623f59e3a1eb2f37ed46e84e4ff0e22c',source_sha256=sha256(SOURCE),
        atlas_sha256=sha256(ROOT/'results/explorer_dataset/dataset.json'),manifest_sha256=sha256(ROOT/'results/allen_reference/manifest.csv'),
        surface_pool_sha256=sha256(ROOT/'results/point_mae/pools.npy'),
        retained='Normalized Euclidean distance matrix input (Frobenius norm/N),128-dimensional variational latent, original Lite DecoderMLP class/default three-hidden-layer recipe with space_dim3, coordinate-derived valid EDM reconstruction, EDM-only reconstruction objective, beta1e-10 KL; deterministic latent means for analysis.',
        adaptations='Ordered2D contour becomes128 area-based FPS surface points. New scalar permutation-equivariant distance-message encoder; reference original unpretrained ResNet18 without contour circular convolutions. Cyclic/reversal EDM matching generalized to GW-initialized hard one-to-one reindexing. No raw intensities, Chamfer training or coordinate/PCA supervision.',
        encoder_conditioning='Version2 scalar distance RBF32centers0..4,width.15 with LayerNorm; initial z_log_var bias-6 for both encoder arms.',encoder_arms=['invariant','cnn'],points=128,latent=128,train_cells=469,validation_cells=140,test_used=False,
        training=dict(epochs=100,batch=16,optimizer='Adam',lr=.001,seed=42,beta=1e-10,views='Four fixed independent area-pool/FPS128 views per training cell, random view and arbitrary simultaneous EDM row/column permutation each step. No pose augmentation required.',selection='Minimum deterministic-mean validation hard-reindexed EDM MSE at epoch0 and every5 epochs. No neighbor-based model/epoch selection.'),
        loss='One hard-reindexed EDM MSE plus beta times mean per-cell KL. Soft GW proposes a Hungarian one-to-one assignment; also evaluate a Hungarian sorted-distance-profile assignment and retain lower actual EDM MSE. GW uses40Sinkhornsteps,epsilon.03/.01/.003,3updates each, feasible marginal rounding. No gradients through correspondence; not a certified global permutation optimum. Fix duplicated reconstruction term in upstream training script.',
        invariances='Invariant arm: translation, positive uniform scale, O(3) rotations/reflections and arbitrary input permutations, up to numerical precision for fixed sampled point sets. Reflection/chirality information intentionally absent as with EDMs. Sampling stability tested separately; no complete graph-invariant or topology reconstruction claim.',
        primary='Invariant encoder128D, training-standardized and degree2Ridge nuisance-adjusted, versus identically adjusted9 traditional measures in existing46-query matched48-cell validation cohort. Fit nuisance/scaling on original96training cells.',
        secondary='CNN-reference and invariant descriptors raw/standardized/adjusted, geometry reconstruction and invariance/resampling. Frozen PCA/harmonics/Allen/Point-MAE/CAJAL references, descriptive11group bootstrap2000. Different data/pretraining budgets; validation reused, no independent biological labels.',
        reconstruction='Global128-point output from latent alone. EDM permits reflection: show O3-aligned geometry and proper-rotation-only geometry separately; no fitted scale. Dense surface errors limited by128-point output; include128-point source sampling reference. No reconstructed surface mesh claim.',
        scope='Offline experiment only. No Atlas updates; no excluded/QC-ineligible/test cells.')
    p.write_text(json.dumps(d,indent=2));return d


def prepare():
    p=protocol();frame=pd.read_csv(ROOT/'results/allen_reference/manifest.csv')
    expected={'epochs':100,'batch':16,'optimizer':'Adam','lr':.001,'seed':42,'beta':1e-10}
    if any(p['training'][k]!=v for k,v in expected.items()):raise ValueError('Protocol differs from this fixed pilot configuration')
    assert sha256(SOURCE)==p['source_sha256'] and sha256(ROOT/'results/allen_reference/manifest.csv')==p['manifest_sha256']
    if (OUT/'prepared.json').exists():return frame
    assert sha256(ROOT/'results/point_mae/pools.npy')==p['surface_pool_sha256']
    frame.to_csv(OUT/'manifest.csv',index=False);pools=np.load(ROOT/'results/point_mae/pools.npy',mmap_mode='r');jobs=[(i,v) for i in range(len(frame)) for v in range(4)]
    points=np.lib.format.open_memmap(OUT/'points.npy',mode='w+',dtype=np.float32,shape=(len(frame),4,128,3))
    matrices=np.lib.format.open_memmap(OUT/'edms.npy',mode='w+',dtype=np.float32,shape=(len(frame),4,128,128))
    for first in range(0,len(jobs),32):
        batch=jobs[first:first+32];candidates=np.stack([pools[i][np.random.default_rng(20261022+4*i+v).choice(32768,2048,replace=False)] for i,v in batch])
        ids=fps_indices(candidates,128);chosen=candidates[np.arange(len(batch))[:,None],ids]
        d=np.sqrt(np.square(chosen[:,:,None]-chosen[:,None]).sum(-1))
        for (i,v),xyz,dist in zip(batch,chosen,d):points[i,v]=xyz;matrices[i,v]=dist
        if first%320==0:print('Prepared surface views',min(first+32,len(jobs)),'/',len(jobs),flush=True)
    points.flush();matrices.flush();(OUT/'prepared.json').write_text(json.dumps(dict(cells=len(frame),views=4,points=128,points_sha256=sha256(OUT/'points.npy'),edms_sha256=sha256(OUT/'edms.npy')),indent=2))
    return frame


def validate(m,data,ids):
    m.eval();values=[]
    with torch.no_grad():
        for first in range(0,len(ids),16):
            ii=ids[first:first+16];output=m(torch.tensor(np.array(data[ii,0]),device='mps'),deterministic=True)
            _,rec,_=vae_loss(output);values.append((float(rec.cpu()),len(ii)))
    return sum(v*n for v,n in values)/sum(n for v,n in values)


def train_arm(kind,frame):
    path=OUT/kind;path.mkdir(exist_ok=True)
    if (path/'completed.json').exists():return
    torch.manual_seed(42);rng=np.random.default_rng(42);m=ShapeEmbed3D(kind).to('mps');opt=torch.optim.Adam(m.parameters(),lr=.001)
    data=np.load(OUT/'edms.npy',mmap_mode='r');tr=frame.index[frame.split=='train'].to_numpy();va=frame.index[frame.split=='validation'].to_numpy();best=float('inf');history=[];start=0;offset=0.;began=time.monotonic()
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location='cpu',weights_only=False);assert state['protocol_sha256']==sha256(OUT/'protocol.json')
        m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng']);torch.mps.set_rng_state(state['mps_rng']);start=state['epoch']+1;best=state['best'];offset=state['seconds'];history=pd.read_csv(path/'history.csv').to_dict('records')
    for epoch in range(start,101):
        training=None
        if epoch:
            m.train();losses=[]
            order=rng.permutation(tr)
            for first in range(0,len(order),16):
                ii=order[first:first+16];views=rng.integers(4,size=len(ii));d=np.array(data[ii,views]);perms=np.stack([rng.permutation(128) for _ in ii]);d=d[np.arange(len(ii))[:,None,None],perms[:,:,None],perms[:,None,:]]
                output=m(torch.tensor(d,device='mps'));loss,rec,kl=vae_loss(output);opt.zero_grad(set_to_none=True);loss.backward();opt.step();value=float(rec.detach().cpu())
                if not np.isfinite(value):raise ValueError('Nonfinite training objective')
                losses.append(value)
            training=float(np.mean(losses))
        if epoch%5==0:
            value=validate(m,data,va)
            if not np.isfinite(value):raise ValueError('Nonfinite validation objective')
            record=dict(epoch=epoch,train_gw=training,validation_gw=value,seconds=offset+time.monotonic()-began);history.append(record)
            if value<best:
                best=value;torch.save(dict(kind=kind,model=m.state_dict(),epoch=epoch,validation_gw=value,protocol_sha256=sha256(OUT/'protocol.json')),path/'best.pt')
            pd.DataFrame(history).to_csv(path/'history.csv',index=False)
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),seconds=offset+time.monotonic()-began,protocol_sha256=sha256(OUT/'protocol.json')),path/'last.pt')
            (OUT/'status.json').write_text(json.dumps(dict(status='training',arm=kind,**record)));print(kind,json.dumps(record),flush=True)
    (path/'completed.json').write_text(json.dumps(dict(status='complete',epochs=100,parameters=sum(p.numel() for p in m.parameters()),best_validation_gw=best),indent=2))


def main():
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.55);frame=prepare()
    for kind in ['invariant','cnn']:train_arm(kind,frame);torch.mps.empty_cache()
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
    for name in ['evaluate_shapeembed3d.py','report_shapeembed3d.py']:subprocess.run([sys.executable,str(ROOT/'scripts'/name)],check=True,env=env)


if __name__=='__main__':main()
