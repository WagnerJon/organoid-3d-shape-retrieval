"""Group-disjoint original-mesh inputs for official pretrained/fine-tuned Allen SDF."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import trimesh
import igl
from allen_reference_core import OUT
from pointnet_shape_core import ROOT,sample_surface,sha256
from spatial_benchmark_core import mesh_for


def work(args):
    i,row=args;path=OUT/'data'/f'{i:04d}.npz'
    if path.exists():
        with np.load(path) as a:
            if str(a['mesh_sha256'])!=row['source_mesh_sha256']:raise ValueError('Stale source geometry')
        return row['cell_id']
    v,f=mesh_for(row);mesh=trimesh.Trimesh(v,f,process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent:raise ValueError('Non-closed source '+row['cell_id'])
    center=(v.min(0)+v.max(0))/2;scale=np.ptp(v,axis=0).max();v=(v-center)/scale
    rng=np.random.default_rng(20261110+i)
    points=sample_surface(v,f,32768,rng).astype(np.float32);query=rng.uniform(-.5,.5,(32768,3)).astype(np.float32)
    target=igl.signed_distance(query.astype(float),v,f,igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0].astype(np.float32)
    if not np.isfinite(target).all():raise ValueError('Nonfinite SDF')
    np.savez_compressed(path,points=points,query=query,target=target,center=center,scale=scale,mesh_sha256=row['source_mesh_sha256'])
    return row['cell_id']


def main():
    (OUT/'data').mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv');frame=frame[frame.training_eligible & frame.split.isin(['train','validation'])].copy().reset_index(names='original_index')
    assert (frame.split=='train').sum()==469 and (frame.split=='validation').sum()==140
    if set(frame[frame.split=='train'].similarity_group)&set(frame[frame.split=='validation'].similarity_group):raise ValueError('Group leakage')
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,cid in enumerate(pool.map(work,enumerate(frame.to_dict('records')))):
            if i%25==0:print(f'Allen surface/SDF preparation {i+1}/{len(frame)} · {cid}',flush=True)
    frame.to_csv(OUT/'manifest.csv',index=False)
    (OUT/'data.json').write_text(json.dumps(dict(training_cells=469,validation_cells=140,training_groups=21,validation_groups=11,
        source_manifest_sha256=sha256(ROOT/'results/pointnet_shape/data/manifest.csv'),manifest_sha256=sha256(OUT/'manifest.csv'),
        atlas_sha256=sha256(ROOT/'results/explorer_dataset/dataset.json'),
        preprocessing='Authors bounding-box center and largest side normalization, 32768 area-sampled surface points, 32768 uniform cube SDF observations. Exact libigl pseudonormal distances replace mesh_to_sdf scanning; float32 replaces float16 storage. Training uses one cached noised 8192-point draw per cell and random proper rotations. Local partial adaptation draws 2048 SDF observations per cell-step; validation uses fixed 20000-query observations. The authors full recipe uses 20000 training queries. Shape-inference geometry only.',
        invariance_scope='Encoder rotation probes rotate already normalized clouds and queries jointly. Axis-aligned bounding-box normalization is not itself invariant when recomputed after arbitrary rotation.',
        test_used=False),indent=2))
    print('Allen reference data prepared',flush=True)


if __name__=='__main__':main()
