"""Cache full-8192-point official backbone features for a standalone benchmark."""
import json,time,os
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from collections import deque
import numpy as np
import pandas as pd
import torch
from allen_reference_core import AllenModel,OUT,chunked_encode,source_hashes,random_rotation
from pointnet_shape_core import ROOT,sha256

_MODEL=None
_DEVICE=None


def initialize(device):
    global _MODEL,_DEVICE
    torch.set_num_threads(2);_DEVICE=device
    if device=='mps':torch.mps.set_per_process_memory_fraction(.45)
    _MODEL=AllenModel().to(device).eval()


def cache_one(job):
    i,row,view=job;path=OUT/'features'/f'{i:04d}_{view}.npz'
    if path.exists():
        with np.load(path) as a:
            if str(a['source_hash'])!=row['source_mesh_sha256']:raise ValueError('Stale feature geometry')
            if not np.isfinite(a['features']).all():raise ValueError('Invalid cached feature')
        return i,view,_DEVICE
    with np.load(OUT/'data'/f'{i:04d}.npz') as data:
        points=data['points'];rng=np.random.default_rng(20261112+i*17+(view if view==1 else 0))
        p=points[rng.integers(len(points),size=8192)].copy()
        if row['split']=='train':p+=rng.normal(0,.005,p.shape).astype(np.float32)
        rotation=np.eye(3,dtype=np.float32)
        if view==2:rotation=random_rotation(np.random.default_rng(20261113+i));p=p@rotation
    with torch.no_grad():
        features=chunked_encode(_MODEL.encoder,torch.tensor(p[None],device=_DEVICE),chunk=512 if _DEVICE=='cpu' else 256,return_features=True)
        latent=_MODEL.encoder.final_conv[-1](features)
        features=features.cpu().numpy()[0];latent=latent.cpu().numpy()[0]
    if not np.isfinite(features).all() or not np.isfinite(latent).all():raise ValueError('Nonfinite backbone output')
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as out:np.savez_compressed(out,features=features,latent=latent,rotation=rotation,source_hash=row['source_mesh_sha256'],device=_DEVICE)
    os.replace(temp,path)
    return i,view,_DEVICE


def main():
    frame=pd.read_csv(OUT/'manifest.csv');(OUT/'features').mkdir(exist_ok=True)
    dev=pd.read_csv(ROOT/'results/spatial_benchmark/dev_manifest.csv');validation=set(dev[dev.split=='validation'].cell_id)
    # First record per validation group, plus one more, fixed before results.
    fixed=frame[frame.cell_id.isin(validation)].groupby('similarity_group',sort=True).head(1).index.tolist()
    fixed+=(frame[frame.cell_id.isin(validation)&~frame.index.isin(fixed)].index.tolist()[:1])
    jobs=[(i,row,0) for i,row in enumerate(frame.to_dict('records'))]
    jobs += [(i,frame.iloc[i].to_dict(),1) for i in frame.index[frame.cell_id.isin(validation)]]
    jobs += [(i,frame.iloc[i].to_dict(),2) for i in fixed]
    protocol=dict(method='Official released full-capacity NPM1 VN-DGCNN / DecoderInner',encoder_features=512,parameters=14306163,
        input_points=8192,neighbors=20,sdf_training_pool=32768,seed=42,training_cells=469,validation_selection_cells=140,matched_validation_cells=48,
        published_weights='NPM1 Rotation_invariant_pointcloud_SDF.ckpt, epoch270, global_step578314',
        checkpoint_sha256=sha256(OUT/'checkpoints/Rotation_invariant_pointcloud_SDF.ckpt'),official_sources=source_hashes(),
        cache_jobs=len(jobs),resampling_cells=48,fresh_rotation_cells=len(fixed),
        source_precision='FP32; no channel or point reduction; exact chunked pointwise operations and all-point kNN, tested against released forward/gradient.',
        local_training=dict(arms=['decoder_only','encoder_head_and_decoder'],epochs=30,batch_size=8,query_points_per_step=2048,learning_rate=.0002,
            optimizer='Adam',checkpoint_selection='Mean SDF L1 over fixed 20000 queries for all 140 validation cells; include epoch0, then every5epochs.',
            frozen='All backbone layers except final 2736→512 VNLinear. Cached full-resolution features; one fixed noised training input view. Proper rotations applied jointly to cached vectors and SDF coordinates.',
            limitation='Partial adaptation, not full end-to-end 400–2000 epoch paper training. Query count2048 differs from author20000; exact mesh SDF and FP32 storage differ from scanning/float16 preprocessing.'),
        test_used=False,atlas_sha256=sha256(ROOT/'results/explorer_dataset/dataset.json'))
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));started=time.monotonic();done=0
    with ProcessPoolExecutor(max_workers=2,initializer=initialize,initargs=('cpu',)) as cpu, ProcessPoolExecutor(max_workers=1,initializer=initialize,initargs=('mps',)) as gpu:
        pending=deque(jobs);futures={}
        for executor in [cpu,cpu,gpu]:
            if pending:futures[executor.submit(cache_one,pending.popleft())]=executor
        while futures:
            ready,_=wait(futures,return_when=FIRST_COMPLETED)
            for future in ready:
                executor=futures.pop(future);i,view,device=future.result();done+=1
                if done%5==0 or done==len(jobs):
                    state=dict(status='caching',completed=done,total=len(jobs),seconds=time.monotonic()-started,last_cell=frame.iloc[i].cell_id,last_view=view,last_device=device)
                    (OUT/'status.json').write_text(json.dumps(state));print(json.dumps(state),flush=True)
                if pending:futures[executor.submit(cache_one,pending.popleft())]=executor
    (OUT/'cache.completed.json').write_text(json.dumps(dict(status='complete',jobs=len(jobs),seconds=time.monotonic()-started)))
    print('Official backbone cache complete',flush=True)


if __name__=='__main__':main()
