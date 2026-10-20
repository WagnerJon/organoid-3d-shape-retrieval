"""Official CAJAL quantized GW on all surfaces; resumable rows, no labels fitted."""
import os
from pathlib import Path
import json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial.distance import cdist
from pointnet_shape_core import ROOT,sha256

OUT=ROOT/'results/harmonics_cajal'
_CELLS=None


def initialize():
    global _CELLS
    from cajal.qgw import quantized_icdm
    clouds=np.load(OUT/'surfaces.npz')['points'];u=np.full(128,1/128)
    _CELLS=[quantized_icdm(cdist(p,p),u,32) for p in clouds]


def row_job(i):
    from cajal.qgw import quantized_gw
    n=len(_CELLS);path=OUT/'qgw_rows'/f'{i:04d}.npy'
    if path.exists():
        saved=np.load(path)
        if saved.shape==(n-i-1,) and np.isfinite(saved).all():return i,float(saved.sum())
        raise ValueError('Invalid cached CAJAL row')
    values=[]
    for j in range(i+1,n):
        _,value=quantized_gw(_CELLS[i],_CELLS[j])
        if not np.isfinite(value) or value<0:raise ValueError('Invalid CAJAL distance')
        values.append(value)
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as output:np.save(output,np.asarray(values))
    os.replace(temp,path)
    return i,float(np.sum(values))


def main():
    import importlib.metadata
    OUT.mkdir(exist_ok=True);(OUT/'qgw_rows').mkdir(exist_ok=True)
    spec=dict(method='Official CAJAL quantized GW',version=importlib.metadata.version('cajal'),surface_points=128,clusters=32,
        measure='Uniform over stratified area samples',within_cell_metric='Euclidean',source_sha256=sha256(OUT/'surfaces.npz'),
        interpretation='qGW approximation, not an exact solution of the non-convex GW problem. Local transport approximations can affect rankings.')
    protocol=OUT/'cajal_protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=spec:raise ValueError('CAJAL inputs changed; choose a new cache')
    protocol.write_text(json.dumps(spec,indent=2));n=len(np.load(OUT/'surfaces.npz')['ids']);total=n*(n-1)//2;started=time.monotonic();done=0
    with ProcessPoolExecutor(max_workers=4,initializer=initialize) as pool:
        jobs=[pool.submit(row_job,i) for i in range(n)]
        for future in as_completed(jobs):
            i,_=future.result();done+=n-i-1
            if i%10==0 or done==total:
                print(f'CAJAL surface pairs {done:,}/{total:,} · {100*done/total:.1f}% · {time.monotonic()-started:.0f}s',flush=True)
                (OUT/'progress.json').write_text(json.dumps(dict(status='running',completed_pairs=done,total_pairs=total,seconds=time.monotonic()-started)))
    matrix=np.zeros((n,n))
    for i in range(n):matrix[i,i+1:]=np.load(OUT/'qgw_rows'/f'{i:04d}.npy')
    matrix+=matrix.T;np.save(OUT/'cajal_qgw.npy',matrix)
    (OUT/'progress.json').write_text(json.dumps(dict(status='complete',completed_pairs=total,total_pairs=total,seconds=time.monotonic()-started)))
    print('CAJAL complete',flush=True)


if __name__=='__main__':main()
