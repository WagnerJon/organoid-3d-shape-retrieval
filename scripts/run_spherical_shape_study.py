"""Five offline experiments; CLI stages are restartable and do not update Atlas."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
import argparse
import csv
import time
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
from skimage.measure import marching_cubes
from spherical_shape_study_core import *
from spatial_benchmark_core import mesh_for, geometry_cost
from check_cajal_validation import candidate_sets


def protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    p=dict(name='Spherical ShapeEmbed 3D: five offline experiments',date='2026-10-21',seed=42,
        scope='Standalone research benchmark. No Atlas edits; original meshes and measurements are immutable. Numerical invariance, topology eligibility and sampling bias are measured separately.',
        rotation=dict(L=[4,8,12,16],N=[256,1024,4096],objects=8,rotations_per_object=16,rotation_prefixes=[4,16],segmentation_resolution=[32,64,96],segmentation_objects=4,segmentation_rotations=4),
        sensitivity=dict(families=['elongation','protrusion','indentation','three_lobed_branching_proxy','roughness'],width=[.15,.3,.6],seeds=[0,1,2],amplitudes=[0,.1,.2,.35,.5,.7],N=4096,L=12),
        geometry=dict(cohort='Existing 48-cell validation morphology cohort, reused/descriptive',radial='Outermost positive ray intersection, report miss/multi-crossing fractions; missing rays nearest-angle filled only for descriptor availability',general='Independent cMCF-inspired finite-element spherical map, frozen cotangent stiffness and updated area mass, 45 steps dt.04; only closed connected orientable genus-zero meshes, no topology repair',mapping_validity='No flipped or degenerate spherical faces; total signed spherical area degree one. Not a certified global self-intersection test.',arbitrary_reference='Existing 32-shell occupancy power spectrum; full shell coefficients are used only for reconstruction (power alone cannot reconstruct)',L=12,N=2048),
        bispectrum=dict(L=8,formula='Clebsch-Gordan contraction of a_l1 a_l2 conjugate(a_l3), normalized by c00 cubed; sorted triples, retain even-real and odd-imaginary components',same_power_pairs=32),
        learning=dict(primary='Self-supervised denoising VAE on invariant harmonic descriptors, 128D bottleneck, not an official ShapeEmbed3D reproduction',synthetic_train=384,synthetic_dev=64,synthetic_test=128,epochs=120,seeds=[42,43,44],input='L12 radial power + L8 bispectrum; scale by training standard deviation with floor .001; reconstruct denoised standardized descriptors, KL1e-5, consistency.1',selection='Minimum clean development reconstruction MSE, no test selection',baselines=['raw power','raw power+bispectrum','standardized power+bispectrum','PCA128','PCA32'],real='Train only original 469 training cells; 140 validation cells held out from fitting but already used in previous research. 48-cell matched-neighbor analysis is descriptive. Do not call it a new independent biological test.'),
        source_hashes={p:sha(ROOT/p) for p in ['results/allen_reference/manifest.csv','results/explorer_dataset/dataset.json','results/harmonics_cajal/surfaces.npz']},
        references=['https://github.com/uhlmanngroup/ShapeEmbedLite','https://www.cs.jhu.edu/~misha/MyPapers/SGP03.pdf','https://arxiv.org/abs/1203.6819','https://github.com/mkazhdan/MoebiusRegistration','https://arxiv.org/abs/2004.13371','https://pmc.ncbi.nlm.nih.gov/articles/PMC10676763/'])
    save_json(OUT/'protocol.json',p)
    return p


def foundation():
    rows=[];rng=np.random.default_rng(42)
    source=[random_radial(100+i,L=12) for i in range(8)]
    rotations=Rotation.random(16,random_state=rng).as_matrix()
    for n in [256,1024,4096]:
        d,Y=fib_basis(n,16)
        base=np.stack([coefficients((Y[:,:169]@c).real,Y) for c in source])
        rotated=[]
        for r in rotations:
            Yr=basis(d@r,12)
            rotated.append(np.stack([coefficients((Yr@c).real,Y) for c in source]))
        rotated=np.stack(rotated,axis=1)
        for L in [4,8,12,16]:
            clean=np.stack([power(c,L) for c in base]);between=cdist(clean,clean);np.fill_diagonal(between,np.inf)
            median=float(np.median(between[np.isfinite(between)]));nearest=between.min(1)
            for count in [4,16]:
                rotated_codes=np.array([[power(c,L) for c in row[:count]] for row in rotated])
                change=np.linalg.norm(rotated_codes-clean[:,None],axis=-1)
                identity=np.array([[cdist(x[None],clean).argmin()==i for x in row] for i,row in enumerate(rotated_codes)])
                rows.append(dict(N=n,L=L,rotations=count,median_within=float(np.median(change)),p95_within=float(np.percentile(change,95)),between_median=median,within_between_ratio=float(np.median(change)/median),p95_nearest_ratio=float(np.percentile(change/nearest[:,None],95)),identity=float(identity.mean())))
        print(f'Experiment 1 continuous rotations: N={n} complete',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'rotation.csv',index=False)
    resolution=[]
    for n in [32,64,96]:
        axis=np.linspace(-2,2,n);g=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1);rad=np.linalg.norm(g,axis=-1)
        dirs=g/np.maximum(rad[...,None],1e-12)
        clean=[];rot=[]
        for i,family in enumerate(['elongation','protrusion','indentation','branching']):
            codes=[]
            for k,r in enumerate(np.r_[np.eye(3)[None],rotations[:4]]):
                rv=synthetic_radius((dirs.reshape(-1,3)@r),family,.65,.3,i).reshape(rad.shape)
                mask=rad<=rv
                v,f,_,_=marching_cubes(mask.astype(float),.5,spacing=(axis[1]-axis[0],)*3);v+=axis[0]
                c,qc,_,_=radial_coefficients(v,f,1024,8);codes.append(power(c,8))
            clean.append(codes[0]);rot.append(codes[1:])
        clean=np.array(clean);rot=np.array(rot);between=cdist(clean,clean);np.fill_diagonal(between,np.inf)
        drift=np.linalg.norm(rot-clean[:,None],axis=-1)
        resolution.append(dict(resolution=n,N=1024,L=8,rotations=4,median_within=float(np.median(drift)),median_between=float(np.median(between[np.isfinite(between)])),ratio=float(np.median(drift)/np.median(between[np.isfinite(between)])),identity=float(np.mean([[cdist(c[None],clean).argmin()==i for c in row] for i,row in enumerate(rot)]))))
        print(f'Experiment 1 voxel-rotation probe: {n} cubed complete',flush=True)
    save_json(OUT/'segmentation_resolution.json',resolution)


def synthetic():
    d,Y=fib_basis(4096,12);rows=[];coeff=[];features=[]
    for family in ['elongation','protrusion','indentation','branching','roughness']:
        for width in [.15,.3,.6]:
            for seed in [0,1,2]:
                values=[]
                for amount in [0,.1,.2,.35,.5,.7]:
                    r=synthetic_radius(d,family,amount,width,seed);c=coefficients(r,Y);feat=all_features(c)
                    rows.append(dict(family=family,width=width,seed=seed,amount=amount,radial_deformation_rms=float(np.sqrt(np.mean((r/r.mean()-1)**2)))));coeff.append(c);features.append(feat);values.append(feat)
                values=np.array(values)
                for method,slice_ in [('power',slice(0,13)),('power_bispectrum',slice(None))]:
                    dist=np.linalg.norm(values[:,slice_]-values[0,slice_],axis=1)
                    for i,row in enumerate(rows[-6:]):row[method+'_distance']=float(dist[i])
    pd.DataFrame(rows).to_csv(OUT/'sensitivity.csv',index=False)
    np.savez_compressed(OUT/'sensitivity_features.npz',coefficients=coeff,features=features)
    # Deliberately equal band energies with independently changed phase/relative
    # band orientation. Exact coefficient readout bypasses finite-sampling aliasing.
    pairs=[];stored=[]
    d2,Y2=fib_basis(4096,8)
    for i in range(32):
        a=random_radial(2000+i);b=random_radial(3000+i)
        for l in range(1,9):
            sl=slice(l*l,(l+1)**2);b[sl]*=np.linalg.norm(a[sl])/max(np.linalg.norm(b[sl]),1e-15)
        pa,pb=power(a,8),power(b,8);ba,bb=bispectrum(a),bispectrum(b)
        row=dict(pair=i,power_distance=float(np.linalg.norm(pa-pb)),bispectrum_distance=float(np.linalg.norm(ba-bb)))
        if i<8:
            # An alignment-aware geometry check separates phase changes from pose.
            p=d2*(Y2@a).real[:,None];q=d2*(Y2@b).real[:,None]
            row['proper_rotation_geometry_cost']=float(geometry_cost(p[::8],q[::8]))
        pairs.append(row);stored.append([a,b])
    np.savez_compressed(OUT/'equal_power_pairs.npz',coefficients=stored)
    save_json(OUT/'equal_power_pairs.json',pairs)
    print('Experiments 2 and 4: 270 controlled shapes and 32 equal-power pairs complete',flush=True)


def real_worker(args):
    i,row=args;cache=OUT/'real_cells'/f'{i:04d}.npz'
    if cache.exists():return i
    v,f=mesh_for(row);c,qc,vn,scale=radial_coefficients(v,f,2048,12)
    np.savez_compressed(cache,coefficients=c,features=all_features(c),coverage=qc['coverage'],multiple_crossing_fraction=qc['multiple_crossing_fraction'],id=row['cell_id'],mesh_sha=row['source_mesh_sha256'])
    return i


def real_prepare():
    frame=pd.read_csv(ROOT/'results/allen_reference/manifest.csv')
    (OUT/'real_cells').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for k,i in enumerate(pool.map(real_worker,enumerate(frame.to_dict('records')))):
            if k%25==0:print(f'Real-cell radial/bispectrum preparation {k+1}/{len(frame)}',flush=True)
    codes=[];coefs=[];qc=[]
    for i,row in frame.iterrows():
        with np.load(OUT/'real_cells'/f'{i:04d}.npz') as z:
            assert str(z['id'])==row.cell_id
            codes.append(z['features']);coefs.append(z['coefficients']);qc.append(dict(cell_id=row.cell_id,split=row.split,coverage=float(z['coverage']),multiple_crossing_fraction=float(z['multiple_crossing_fraction'])))
    np.savez_compressed(OUT/'real_features.npz',ids=frame.cell_id.to_numpy(dtype=str),features=codes,coefficients=coefs)
    frame.to_csv(OUT/'real_manifest.csv',index=False);pd.DataFrame(qc).to_csv(OUT/'real_radial_qc.csv',index=False)
    print('All real-cell descriptors prepared',flush=True)


def mapping():
    frame,_,_=candidate_sets();d,Y=fib_basis(2048,12)
    sphere=trimesh.creation.icosphere(subdivisions=4);evald=sphere.vertices;Ye=basis(evald,12)
    records=[];features=[];(OUT/'mapping_cells').mkdir(exist_ok=True)
    for i,row in enumerate(frame.to_dict('records')):
        path=OUT/'mapping_cells'/f'{i:03d}.npz';meta=path.with_suffix('.json')
        if meta.exists():records.append(json.loads(meta.read_text()));features.append(np.load(path)['feature']);continue
        v,f=mesh_for(row);c,qc,v,scale=radial_coefficients(v,f,2048,12)
        radial=evald*np.maximum((Ye@c).real,.001)[:,None]
        result=dict(cell_id=row['cell_id'],radial_qc=qc,radial=reconstruction_metrics(radial,v,f))
        mapped,info=map_sphere(v,f);result['mapping']=info
        feature=np.full(13,np.nan);general=np.zeros((0,3))
        if info['valid']:
            _,values,sample_qc=ray_samples(mapped,f,d,attribute=v)
            coeff=coefficients(values,Y);general=(Ye@coeff).real
            result['general']=reconstruction_metrics(general,v,f);result['general_coverage']=sample_qc['coverage']
            feature=np.array([np.linalg.norm(coeff[l*l:(l+1)**2]) for l in range(13)])
            feature/=max(np.linalg.norm(feature),1e-12)
        np.savez_compressed(path,feature=feature,vertices=v,faces=f,radial=radial,general=general,mapped=mapped if mapped is not None else np.zeros((0,3)))
        save_json(meta,result);records.append(result);features.append(feature)
        print(f'Experiment 3 spherical mapping {i+1}/48: {info["reason"]}',flush=True)
    np.savez_compressed(OUT/'general_features.npz',ids=frame.cell_id.to_numpy(dtype=str),features=features)
    save_json(OUT/'mapping.json',records)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['protocol','foundation','synthetic','real','mapping'])
    stage=p.parse_args().stage
    if stage=='protocol':protocol();return
    if not (OUT/'protocol.json').exists():protocol()
    start=time.time()
    {'foundation':foundation,'synthetic':synthetic,'real':real_prepare,'mapping':mapping}[stage]()
    save_json(OUT/f'{stage}.complete.json',dict(stage=stage,seconds=time.time()-start))
if __name__=='__main__':main()
