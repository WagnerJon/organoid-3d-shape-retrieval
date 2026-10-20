"""Prepare original-mesh spherical harmonics and surface samples, with caching."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from pointnet_shape_core import ROOT,sha256
from spatial_benchmark_core import mesh_for
from harmonics_cajal_core import harmonics,surface_points,angular_basis

OUT=ROOT/'results/harmonics_cajal'
_BASIS=None

def work(args):
    global _BASIS
    i,row,source_hash=args;path=OUT/'cells'/f'{i:04d}.npz'
    if path.exists():
        with np.load(path) as a:
            if str(a['mesh_hash'])==row['source_mesh_sha256'] and str(a['algorithm_hash'])==source_hash:return row['cell_id']
    v,f=mesh_for(row)
    if _BASIS is None:_BASIS=angular_basis()
    vector=harmonics(v,f,_BASIS);points=surface_points(v,f,128,20261101+i)
    np.savez_compressed(path,harmonics=vector,points=points,mesh_hash=row['source_mesh_sha256'],algorithm_hash=source_hash)
    return row['cell_id']


def main():
    OUT.mkdir(exist_ok=True);(OUT/'cells').mkdir(exist_ok=True)
    frame=pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv');algorithm=sha256(ROOT/'scripts/harmonics_cajal_core.py')
    protocol=dict(version=1,lmax=12,radial_shells=32,angular_quadrature=[32,64],soft_boundary_width=.035,surface_points=128,
        normalization='Exact surface-area centroid and RMS surface radius; scale and orientation removed.',
        cajal='Official CAJAL 2.0.3 GW solver, Euclidean within-surface point distances, uniform area-sampled measure; 128 points.',
        development_only=True,test_used_for_selection=False,source_manifest_sha256=sha256(ROOT/'results/pointnet_shape/data/manifest.csv'),algorithm_sha256=algorithm)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,cid in enumerate(pool.map(work,[(i,row,algorithm) for i,row in enumerate(frame.to_dict('records'))])):
            if i%25==0:print(f'Shape descriptors {i+1}/{len(frame)} · {cid}',flush=True)
    values=[];clouds=[]
    for i in range(len(frame)):
        with np.load(OUT/'cells'/f'{i:04d}.npz') as a:values.append(a['harmonics']);clouds.append(a['points'])
    np.savez_compressed(OUT/'surfaces.npz',ids=frame.cell_id.to_numpy(dtype=str),harmonics=np.asarray(values),points=np.asarray(clouds))
    frame.to_csv(OUT/'manifest.csv',index=False)
    print('Surface descriptors ready',flush=True)


if __name__=='__main__':main()
