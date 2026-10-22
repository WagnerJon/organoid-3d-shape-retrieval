"""Geometry-derived held-out perturbations. Does not evaluate any model."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from scipy.spatial.transform import Rotation
from spectral64_core import *
from spatial_benchmark_core import mesh_for


def worker(arg):
    i,row=arg;path=OUT/'noise_cells'/f'{i:03d}.npz'
    if path.exists():return i
    v,f=mesh_for(row);vc,pose=canonical(v,f);normal=trimesh.Trimesh(vc,f,process=False).vertex_normals
    rng=np.random.default_rng(33442+i);phase=rng.normal(size=(3,3))*5;angles=rng.uniform(0,6.28,3)
    displacement=np.sin(vc@phase.T+angles).mean(1);displacement-=displacement.mean();displacement/=max(displacement.std(),1e-12)
    noisy=vc+.02*displacement[:,None]*normal;noisy,qpose=canonical(noisy,f)
    coef=field_coefficients(noisy,f)
    # Independent rigid rotation test, explicitly rotate the same clean mesh.
    R=Rotation.random(random_state=rng).as_matrix();rotated=vc@R.T
    aligned,rotpose=canonical(rotated,f);rotcoef=field_coefficients(aligned,f)
    np.savez_compressed(path,coefficients=coef,rotated_coefficients=rotcoef,id=row['cell_id'])
    return i


def main():
    (OUT/'noise_cells').mkdir(exist_ok=True);frame=pd.read_csv(OUT/'manifest.csv');test=frame[frame.split=='test'].reset_index(drop=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for k,_ in enumerate(pool.map(worker,enumerate(test.to_dict('records')))):
            if k%20==0:print(f'Physical 2% noise + rigid rotation probes {k+1}/{len(test)}',flush=True)
    a=[np.load(OUT/'noise_cells'/f'{i:03d}.npz') for i in range(len(test))]
    np.savez_compressed(OUT/'test_noise.npz',ids=test.cell_id.to_numpy(dtype=str),coefficients=np.stack([z['coefficients'] for z in a]),rotated_coefficients=np.stack([z['rotated_coefficients'] for z in a]))
    print('Test geometry probes ready; no model metrics inspected',flush=True)
if __name__=='__main__':main()


def raw_worker(arg):
    i,row=arg;path=OUT/'noise_cells'/f'{i:03d}_raw.npz'
    if path.exists():return i
    v,f=mesh_for(row);vc,pose=canonical(v,f);normal=trimesh.Trimesh(vc,f,process=False).vertex_normals
    rng=np.random.default_rng(33442+i);phase=rng.normal(size=(3,3))*5;angles=rng.uniform(0,6.28,3)
    displacement=np.sin(vc@phase.T+angles).mean(1);displacement-=displacement.mean();displacement/=max(displacement.std(),1e-12)
    noisy,qpose=canonical(vc+.02*displacement[:,None]*normal,f)
    R=Rotation.random(random_state=rng).as_matrix();coefficient=field_coefficients(noisy@R.T,f)
    np.savez_compressed(path,coefficients=coefficient);return i


def raw_noise():
    frame=pd.read_csv(OUT/'manifest.csv');test=frame[frame.split=='test'].reset_index(drop=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for k,_ in enumerate(pool.map(raw_worker,enumerate(test.to_dict('records')))):
            if k%25==0:print(f'Unaligned physical noisy views {k+1}/135',flush=True)
    np.savez_compressed(OUT/'test_raw_noise.npz',coefficients=np.stack([np.load(OUT/'noise_cells'/f'{i:03d}_raw.npz')['coefficients'] for i in range(len(test))]))
