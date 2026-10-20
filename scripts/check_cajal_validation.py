"""Unquantized CAJAL solver on predeclared validation candidate pairs only."""
import json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pointnet_shape_core import ROOT,sha256
from spatial_benchmark_core import morphology,CONTROLS
OUT=ROOT/'results/harmonics_cajal'
_CELLS=None


def init():
    global _CELLS
    frame=pd.read_csv(ROOT/'results/spatial_benchmark/dev_manifest.csv');frame=frame[frame.split=='validation']
    a=np.load(OUT/'surfaces.npz');lookup={cid:i for i,cid in enumerate(a['ids'])};clouds=a['points'][[lookup[cid] for cid in frame.cell_id]]
    _CELLS=[cdist(p,p) for p in clouds]


def solve(pair):
    from cajal.gw_cython import gw
    i,j=pair;path=OUT/'validation_gw_pairs'/f'{i:03d}_{j:03d}.npy'
    if path.exists():return i,j,float(np.load(path))
    u=np.full(128,1/128);result=gw(_CELLS[i],u,_CELLS[j],u)
    if result is None:raise RuntimeError('CAJAL descent did not converge')
    _,value=result
    if not np.isfinite(value):raise ValueError('Nonfinite GW')
    np.save(path,value);return i,j,float(value)


def candidate_sets():
    dev=pd.read_csv(ROOT/'results/spatial_benchmark/dev_manifest.csv');train=dev[dev.split=='train'];frame=dev[dev.split=='validation'].reset_index(drop=True)
    ctr=morphology(train,CONTROLS);x=(morphology(frame,CONTROLS)-ctr.mean(0))/ctr.std(0).clip(1e-6);matching=cdist(x,x)
    groups=frame.similarity_group.to_numpy();pools={};pairs=set()
    for i in range(len(frame)):
        ids=np.flatnonzero((groups!=groups[i])&(matching[i]<=1))
        if len(ids)<5:continue
        ids=ids[np.argsort(matching[i,ids])[:20]];pools[i]=ids
        pairs.update(tuple(sorted([i,int(j)])) for j in ids)
    return frame,pools,sorted(pairs)


def main():
    (OUT/'validation_gw_pairs').mkdir(exist_ok=True);frame,pools,pairs=candidate_sets();started=time.monotonic()
    matched_pairs=len(pairs)
    pairs=[(i,j) for i in range(len(frame)) for j in range(i+1,len(frame))]
    matrix=np.full((len(frame),len(frame)),np.nan);np.fill_diagonal(matrix,0)
    with ProcessPoolExecutor(max_workers=3,initializer=init) as pool:
        for k,(i,j,value) in enumerate(pool.map(solve,pairs)):
            matrix[i,j]=matrix[j,i]=value
            if k%50==0:print(f'Unquantized validation GW {k+1}/{len(pairs)}',flush=True)
    np.save(OUT/'validation_gw.npy',matrix)
    (OUT/'validation_gw.json').write_text(json.dumps(dict(ids=frame.cell_id.tolist(),pairs=len(pairs),matched_candidate_pairs=matched_pairs,eligible_queries=len(pools),seconds=time.monotonic()-started,
        surface_sha256=sha256(OUT/'surfaces.npz'),scope='Default official CAJAL GW solver, 128 surface points; non-convex optimization is not certified globally optimal. Complete validation cohort; no training or test cells included.'),indent=2))
    print('Unquantized validation check complete',flush=True)


if __name__=='__main__':main()
