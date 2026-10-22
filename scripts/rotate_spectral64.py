"""Conditional C data: independently rotate actual meshes before SDF sampling."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from scipy.spatial.transform import Rotation
from spectral64_core import *
from spatial_benchmark_core import mesh_for

def real_basis(d):
    Y=basis(d,12);H=pack(Y)
    for l in range(13):
        for m in range(1,l+1):H[:,l*l+l-m]*=-1
    return H

@lru_cache(maxsize=1)
def action_quadrature():
    # Products through L12 have degree <=24. This smaller exact rule avoids
    # rebuilding the oversampled geometry quadrature for each rotation.
    d,T,_=angular_basis(16,32,12)
    return d,T[:,0].real*np.sqrt(4*np.pi),real_basis(d)

def action(R):
    d,weights,H=action_quadrature();Hr=real_basis(d@R);D=np.zeros((K,K))
    for l in range(13):
        s=slice(l*l,(l+1)**2);block=H[:,s].T@(weights[:,None]*Hr[:,s])
        # Round numerical quadrature to the orthogonal representation.
        u,_,vt=np.linalg.svd(block);D[s,s]=u@vt
    return D.astype(np.float32)

def worker(arg):
    i,e=arg;path=OUT/'rotation_cells'/f'{i:04d}.npz'
    if path.exists():return i
    if e['kind']=='real':v,f=mesh_for(e)
    else:v,f=synthetic_mesh(e['seed'],e.get('mirror',False))
    v,_=canonical(v,f);rng=np.random.default_rng(59132+i);views=[];actions=[]
    for _ in range(2):
        R=Rotation.random(random_state=rng).as_matrix();views.append(field_coefficients(v@R.T,f));actions.append(action(R))
    np.savez_compressed(path,coefficients=views,actions=actions)
    return i

def main():
    gate=json.loads((OUT/'B_information_gate.json').read_text());assert gate['passed'],'B gate did not pass; C remains deferred'
    entries=json.loads((OUT/'entries.json').read_text());(OUT/'rotation_cells').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for k,_ in enumerate(pool.map(worker,enumerate(entries))):
            if k%50==0:print(f'Independent physical rotation pairs {k+1}/{len(entries)}',flush=True)
    views=[];actions=[]
    for i in range(len(entries)):
        a=np.load(OUT/'rotation_cells'/f'{i:04d}.npz');views.append(a['coefficients']);actions.append(a['actions'])
    np.savez_compressed(OUT/'rotated_training_views.npz',coefficients=views,actions=actions)
    save_json(OUT/'rotation_preparation.complete.json',dict(objects=len(entries),actual_geometry_rotations=2*len(entries)))
if __name__=='__main__':main()
