"""Supplementary diagnostics; no retraining or primary-test model selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from spherical_shape_study_core import *
from learn_spherical_shape_study import radial_input,Encoder


def area_distortion(v,f,mapped):
    tri=mapped[f];triple=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))
    area=abs(2*np.arctan2(triple,1+np.einsum('ij,ij->i',tri[:,0],tri[:,1])+np.einsum('ij,ij->i',tri[:,1],tri[:,2])+np.einsum('ij,ij->i',tri[:,2],tri[:,0])))
    original=trimesh.triangles.area(v[f]);ratio=(area/area.sum())/(original/original.sum()).clip(1e-20)
    return dict(area_ratio_p05=float(np.percentile(ratio,5)),area_ratio_p95=float(np.percentile(ratio,95)),median_absolute_log_area_distortion=float(np.median(abs(np.log(ratio.clip(1e-20))))))


def mapping_diagnostics():
    records=[]
    for path in sorted((OUT/'mapping_cells').glob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['mapping']['valid']:
            z=np.load(path);records.append(dict(cell_id=meta['cell_id'],**area_distortion(z['vertices'],z['faces'],z['mapped'])))
    save_json(OUT/'mapping_distortion.json',records)
    d,Y=fib_basis(8192,32);sphere=trimesh.creation.icosphere(subdivisions=4);Ye=basis(sphere.vertices,32)
    results=[]
    for name in ['curved_tube','bent_branched_tubes']:
        a=np.load(OUT/'arbitrary_controls'/f'{name}.npz');v,f=a['vertices'],a['faces'];mapped,info=map_sphere(v,f)
        _,values,qc=ray_samples(mapped,f,d,attribute=v);c=coefficients(values,Y)
        for L in [8,12,24,32]:
            reconstruct=(Ye[:,:((L+1)**2)]@c[:((L+1)**2)]).real
            results.append(dict(shape=name,L=L,N=8192,coverage=qc['coverage'],**reconstruction_metrics(reconstruct,v,f),**area_distortion(v,f,mapped)))
    save_json(OUT/'supplementary_general_degree.json',results)
    print('Mapping distortion and supplementary degree diagnostic complete',flush=True)


def phase_challenge():
    pairs=np.load(OUT/'equal_power_pairs.npz')['coefficients']
    coeff=[]
    # Extend each pair through L12 to match the learning generator's bandwidth.
    for i in range(len(pairs)):
        a=random_radial(2000+i,12);b=random_radial(3000+i,12)
        for l in range(1,13):
            sl=slice(l*l,(l+1)**2);b[sl]*=np.linalg.norm(a[sl])/max(np.linalg.norm(b[sl]),1e-15)
        coeff.extend([a,b])
    coeff=np.asarray(coeff)
    d,Y=fib_basis(4096,12);dq,Yq=fib_basis(2048,12)
    gallery=[];queries=[];rng=np.random.default_rng(98134)
    for i,c in enumerate(coeff):
        pad=np.pad(c,(0,169-len(c)))
        gallery.append(radial_input((Y@pad).real,Y))
        R=Rotation.random(random_state=rng).as_matrix();r=(basis(dq@R,12)@c).real
        queries.append(radial_input(r+rng.normal(0,.005,len(r)),Yq))
    g,q=np.asarray(gallery),np.asarray(queries);result=[]
    for domain in ['synthetic','real']:
        p=OUT/'learning'/domain
        if not (p/'gallery_codes.npz').exists():continue
        scale=np.load(p/'scaler.npz');mean,sd=scale['mean'],scale['sd']
        methods={'radial_power':(g[:,:13],q[:,:13]),'radial_bispectrum':(g[:,13:73],q[:,13:73]),'shell_power':(g[:,73:],q[:,73:]),'all_invariants_raw':(g,q),'all_invariants_standardized':((g-mean)/sd,(q-mean)/sd)}
        for seed in [42,43,44]:
            checkpoint=torch.load(p/f'seed{seed}.pt',weights_only=True);model=Encoder(g.shape[1]);model.load_state_dict(checkpoint['state']);model.eval()
            with torch.no_grad():methods[f'VAE_seed{seed}']=(model.encode(torch.tensor((g-mean)/sd,dtype=torch.float32)).numpy(),model.encode(torch.tensor((q-mean)/sd,dtype=torch.float32)).numpy())
        for method,(x,y) in methods.items():
            distances=cdist(y,x);pred=distances.argmin(1)
            pair_decisions=np.array([distances[i,i]<distances[i,i^1] for i in range(len(g))])
            result.append(dict(training_domain=domain,method=method,identity=float(np.mean(pred==np.arange(len(g)))),within_equal_power_pair_accuracy=float(pair_decisions.mean()),queries=len(g)))
    pd.DataFrame(result).to_csv(OUT/'supplementary_phase_challenge.csv',index=False)
    print('Supplementary unseen equal-power phase challenge complete',flush=True)
if __name__=='__main__':mapping_diagnostics();phase_challenge()
