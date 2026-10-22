"""Immutable cohort split, coefficient extraction and development/test gates."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from spectral64_core import *
from spatial_benchmark_core import mesh_for


def protocol():
    OUT.mkdir(exist_ok=True);(OUT/'cells').mkdir(exist_ok=True)
    src=ROOT/'results/pointnet_shape/data/manifest.csv';frame=pd.read_csv(src);frame['original_index']=np.arange(len(frame));frame=frame[frame.training_eligible.astype(bool)].copy()
    assert frame.groupby('sample').split.nunique().max()==1
    assert frame.groupby('similarity_group').split.nunique().max()==1
    frame.to_csv(OUT/'manifest.csv',index=False)
    p=dict(date='2026-10-21',latent=64,seeds=[42,43,44],epochs=120,batch=32,optimizer='Adam',lr=.001,KL=1e-5,
        models={'A':'416 invariant shell band-norm powers -> same powers, VAE64; padded to 5408 for matched parameter count','B':'5408 real degrees of freedom preserving all full complex coefficients -> full coefficients, same VAE64 backbone','C':'Conditional on B information gate: paired coefficients from independently rotated actual source geometry; same B backbone, posed decoder, plus latent consistency'},
        coefficient_field=dict(shells=32,radii=RADII.tolist(),degree=12,quadrature=[32,64],soft_sdf_width=WIDTH,packing='Exact orthonormal real representation of conjugate-symmetric complex coefficients; no information discarded',normalization='Exact surface centroid and RMS radius; proper PCA frame with skewness signs; orientation kept separately for decoding and never appended to latent',weighting='Coefficients multiplied by radial coordinate, proportional to volumetric L2 measure. Shared scalar training residual RMS; no per-coefficient whitening'),
        splits={k:dict(cells=len(t),organoids=t['sample'].nunique(),groups=t.similarity_group.nunique()) for k,t in frame.groupby('split')},
        synthetic=dict(train=384,development=64,test=128,source='Actual icosphere meshes displaced by degree12 radial functions; independent seeds',development_mirror_pairs=16,test_mirror_pairs=32,legacy_radial_equal_power_pairs=32),
        selection='Only validation coefficient/power reconstruction. Freeze all three seed checkpoints before test metrics. No selecting a seed on test outcomes.',
        B_information_gate='Development mirrored-pair same-pose within-pair retrieval >=90% and phase-dependent coefficient-difference recovery R2 >=0.25, averaged over three seeds. Mirrored surfaces have equal multi-shell powers (unlike the legacy equal-radial-power challenge). C is not trained if this fails.',
        final_gates=dict(matched_geometry_cost_below=.0225,noise2percent_retrieval_at_least=.95,latent=64,note='Proposed targets, not observed results. Historical .023836 and95.8% came from a different reused validation cohort and are not directly comparable to the new test split.'),
        evaluation='Held-out 135 segmented test cells from17 organoids/11 similarity groups; not biologically verified cell-state labels. Recompute traditional and frozen shell references on same test pools. Group bootstrap uncertainty and 2% smooth normal surface perturbations; IoU, bidirectional surface Chamfer, and high-degree coefficient/detail recovery separate. Synthetic tests independent. Biological-state evaluation cannot be claimed without annotations and biological replicates.',
        source_hashes={'manifest':sha(src),'atlas':sha(ROOT/'results/explorer_dataset/dataset.json')},scope='Offline only; no Atlas changes, no preprocessing or model fitting on test objects')
    save_json(OUT/'protocol.json',p)


def worker(arg):
    index,entry=arg;kind=entry['kind'];path=OUT/'cells'/f'{index:04d}.npz'
    if path.exists():return index
    if kind=='real':v,f=mesh_for(entry)
    else:v,f=synthetic_mesh(entry['seed'],entry.get('mirror',False))
    v,pose=canonical(v,f);coef=field_coefficients(v,f)
    np.savez_compressed(path,coefficients=coef,id=entry['id'])
    save_json(path.with_suffix('.json'),pose)
    return index


def prepare():
    frame=pd.read_csv(OUT/'manifest.csv');entries=[]
    for row in frame.to_dict('records'):entries.append(dict(**row,kind='real',id=row['cell_id']))
    for i in range(576):entries.append(dict(kind='synthetic',id=f'synthetic_{i}',seed=100000+i,split='train' if i<384 else 'validation' if i<448 else 'test'))
    for split,count,start in [('validation',16,70000),('test',32,80000)]:
        for pair in range(count):
            for mirrored in [False,True]:entries.append(dict(kind='mirror',id=f'{split}_mirror_{pair}_{int(mirrored)}',seed=start+pair,mirror=mirrored,split=split,pair=pair))
    save_json(OUT/'entries.json',entries)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for k,_ in enumerate(pool.map(worker,enumerate(entries))):
            if k%40==0:print(f'Full complex field extraction {k+1}/{len(entries)}',flush=True)
    x=np.stack([np.load(OUT/'cells'/f'{i:04d}.npz')['coefficients'] for i in range(len(entries))])
    np.savez_compressed(OUT/'coefficients.npz',coefficients=x,ids=[e['id'] for e in entries])
    save_json(OUT/'preparation.complete.json',dict(objects=len(entries),real=744,synthetic=576,mirror_challenge=96))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['protocol','prepare']);stage=p.parse_args().stage
    if stage=='protocol':protocol()
    else:prepare()
