"""Independent physical query meshes for the original and stricter challenges."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor
from spectral64_core import *

def worker(arg):
    i,entry=arg;kind=entry['kind'];path=OUT/'challenge_cells'/f'{i:03d}.npz'
    if path.exists():return i
    if kind=='legacy':
        sphere=trimesh.creation.icosphere(subdivisions=3);c=np.array(entry['real'])+1j*np.array(entry['imag']);v=sphere.vertices*(basis(sphere.vertices,8)@c).real[:,None];f=sphere.faces
    else:v,f=synthetic_mesh(entry['seed'],entry.get('mirror',False))
    v,_=canonical(v,f);gallery=field_coefficients(v,f)
    normal=trimesh.Trimesh(v,f,process=False).vertex_normals;rng=np.random.default_rng(87100+i)
    phase=rng.normal(size=(3,3))*5;angles=rng.uniform(0,6.28,3);disp=np.sin(v@phase.T+angles).mean(1);disp-=disp.mean();disp/=max(disp.std(),1e-12)
    vq,qpose=canonical(v+.005*disp[:,None]*normal,f);query=field_coefficients(vq,f)
    np.savez_compressed(path,gallery=gallery,query=query,kind=kind,pair=entry['pair']);return i

def main():
    OUT.joinpath('challenge_cells').mkdir(exist_ok=True);entries=[]
    pairs=np.load(ROOT/'results/spherical_shape_study/equal_power_pairs.npz')['coefficients']
    for i,pair in enumerate(pairs):
        for c in pair:entries.append(dict(kind='legacy',real=c.real.tolist(),imag=c.imag.tolist(),pair=i))
    for i in range(32):
        for mirror in [False,True]:entries.append(dict(kind='mirror',seed=80000+i,mirror=mirror,pair=i))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for k,_ in enumerate(pool.map(worker,enumerate(entries))):
            if k%20==0:print(f'Independent challenge geometry {k+1}/{len(entries)}',flush=True)
    a=[np.load(OUT/'challenge_cells'/f'{i:03d}.npz') for i in range(len(entries))]
    np.savez_compressed(OUT/'physical_challenges.npz',gallery=np.stack([x['gallery'] for x in a]),query=np.stack([x['query'] for x in a]),kind=[str(x['kind']) for x in a],pair=[int(x['pair']) for x in a])
    print('Independent challenge geometry ready',flush=True)
if __name__=='__main__':main()
