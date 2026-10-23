"""Resumable, bounded-memory fractional Zernike experiment on five organoids.

Research outputs only: never modifies loaded Atlas datasets, source geometry,
existing representations, method recommendations, or the bundled demo.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse, hashlib, json, shutil, time, resource
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
from zernike_fractional_core import mesh_descriptors
import five_organoid_retrieval_core as prior
from spectral64_core import save_json, sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fractional_zernike'
OLD=prior.OUT
ALPHAS=(1.,.38,.14,.20)
BASE=[f'{field}_a{alpha:g}' for field in ('volume','surface') for alpha in ALPHAS]
KEYS=BASE+[key+'_minmax' for key in BASE]
CONFIG=dict(order=20,alphas=list(ALPHAS),sampling=[96,32,64],resolution=64,normalization='mass')


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():
        p=json.loads((OUT/'protocol.json').read_text())
        if p['config']!=CONFIG or p['encoder_sha256']!=sha(ROOT/'scripts/zernike_fractional_core.py'):
            raise RuntimeError('Encoder/config changed; use a fresh experiment folder')
        if p['source_manifest_sha256']!=sha(OLD/'manifest.csv') or p['source_plan_sha256']!=sha(OLD/'plan.json'):
            raise RuntimeError('Benchmark identities or query plan changed')
        return
    frame=pd.read_csv(OLD/'manifest.csv')
    save_json(OUT/'protocol.json',dict(config=CONFIG,encoder_sha256=sha(ROOT/'scripts/zernike_fractional_core.py'),
        source_manifest_sha256=sha(OLD/'manifest.csv'),source_plan_sha256=sha(OLD/'plan.json'),
        paper='https://doi.org/10.3390/appliedmath6100174',primary='volume_a0.38 vs volume_a1, L2 Euclidean',
        secondary='Surface-band input; alpha0.14/0.20; mass-normalized descriptors with leave-query-organoid-out MinMax scaling. All are exploratory, not independently validated.',
        geometry='Reuse100 fixed queries,747 included galleries, five organoids, same ICP proxy and16 fixed cross-organoid reference pairs per query. No labels used to tune alpha or feature weights.',
        robustness='Same25 probes: clean re-encoding, actual mesh rotation, smooth2% surface perturbation; identity retrieval in747 included cells. One probe per condition/object; descriptive uncertainty.',
        normalization='Exact closed-mesh volume centroid, area-centroid fallback for open meshes; maximum vertex radius maps to0.9. L2 normalized for primary comparisons. Secondary MinMax fit on included cells from the other four organoids only, without clipping queries or using geometry labels.',
        paper_adaptation='Independent shifted-Jacobi radial basis satisfying Eq4 and Section4 recurrence; spherical product quadrature rather than Cartesian voxel sums. Paper-like binary surface band with half-width sqrt(3)/64; filled-volume soft SDF occupancy matches prior input convention. Both use the same volume centroid, not the paper surface-voxel centroid. Not a reproduction of reported Princeton benchmark accuracy.',
        exclusions='Encode all961 objects, retrieval uses existing747 QC/main-organoid inclusions; no source data/Atlas files modified.',
        inference_only=True,started=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    frame.to_csv(OUT/'manifest.csv',index=False)
    shutil.copy2(OLD/'plan.json',OUT/'plan.json')
    shutil.copy2(OLD/'geometry.sqlite',OUT/'geometry.sqlite')
    shutil.copy2(OLD/'surface_clouds.npy',OUT/'surface_clouds.npy')
    (OUT/'cells').mkdir(exist_ok=True);(OUT/'probes').mkdir(exist_ok=True)


def encode_one(item):
    i,row=item;path=OUT/'cells'/f'{i:04d}.npz'
    if path.exists():
        with np.load(path) as a:
            if str(a['mesh_sha256'])!=row['source_mesh_sha256']:raise ValueError('Changed mesh')
        return i
    t=time.monotonic();v,f=prior.mesh(row)
    codes,meta=mesh_descriptors(v,f,**CONFIG)
    save_codes(path,**codes,mesh_sha256=row['source_mesh_sha256'],seconds=time.monotonic()-t,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,closed=meta['closed'])
    return i


def save_codes(path,**values):
    """An interrupted worker never leaves an apparently completed cache file."""
    temporary=path.with_suffix('.pending')
    with temporary.open('wb') as handle:np.savez_compressed(handle,**values)
    temporary.replace(path)


def encode(workers=2):
    freeze();frame=pd.read_csv(OUT/'manifest.csv');start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for number,_ in enumerate(pool.map(encode_one,enumerate(frame.to_dict('records')),chunksize=1),1):
            if number%25==0 or number==len(frame):print(f'Fractional/classical moments: {number}/{len(frame)} cells; {time.monotonic()-start:.0f}s',flush=True)
    arrays={key:np.stack([np.load(OUT/'cells'/f'{i:04d}.npz')[key] for i in range(len(frame))]) for key in BASE}
    np.savez_compressed(OUT/'codes.npz',ids=frame.cell_id.to_numpy(dtype=str),**arrays)
    save_json(OUT/'encoding.json',dict(cells=len(frame),dimensions=121,fields=BASE,wall_seconds=time.monotonic()-start,
        median_seconds=float(np.median([float(np.load(OUT/'cells'/f'{i:04d}.npz')['seconds']) for i in range(len(frame))])),
        max_worker_rss_bytes=int(max(np.load(OUT/'cells'/f'{i:04d}.npz')['peak_rss_bytes'] for i in range(len(frame))))))


def scaled(x,key,frame,organoid):
    if key.endswith('_minmax'):
        fit=x[frame.included.to_numpy(bool)&(frame['sample'].to_numpy()!=organoid)]
        low=fit.min(0);span=fit.max(0)-low;valid=span>1e-8
        return (x-low)/np.where(valid,span,1)*valid
    return x/np.linalg.norm(x,axis=1,keepdims=True)


def matrix(key,arrays,frame):
    x=arrays[key.removesuffix('_minmax')];d=np.zeros((len(frame),len(frame)))
    # MinMax uses a distinct training fold for each query's source organoid.
    for sample in frame['sample'].unique():
        ids=np.flatnonzero(frame['sample'].to_numpy()==sample)
        z=scaled(x,key,frame,sample);d[ids]=cdist(z[ids],z)
    return d


def evaluate():
    import run_five_organoid_retrieval as geometry
    geometry.OUT=OUT  # Independent cache; source geometry labels remain unchanged.
    frame=pd.read_csv(OUT/'manifest.csv');plan=json.loads((OUT/'plan.json').read_text())
    arrays=np.load(OUT/'codes.npz');distances={key:matrix(key,arrays,frame) for key in KEYS}
    chosen={};requested=set()
    for i in plan['queries']:
        gallery=frame.index[frame.included&(frame['sample']!=frame.iloc[i]['sample'])].to_numpy()
        for key,d in distances.items():
            j=int(gallery[np.argmin(d[i,gallery])]);chosen[key,i]=j;requested.add(tuple(sorted((i,j))))
        requested.update(tuple(sorted((i,int(j)))) for j in plan['random_reference_pairs'][str(i)])
    costs=geometry.cache_geometry(requested);rows=[];separation=[]
    for i in plan['queries']:
        refs=np.array(plan['random_reference_pairs'][str(i)]);g=np.array([costs[tuple(sorted((i,int(j))))] for j in refs])
        order=np.argsort(g);near,far=order[:4],order[-4:]
        for key,d in distances.items():
            j=chosen[key,i];dd=d[i,refs]
            rows.append(dict(method=key,query=i,cell_id=frame.iloc[i].cell_id,organoid=frame.iloc[i]['sample'],neighbor=j,
                neighbor_cell=frame.iloc[j].cell_id,neighbor_organoid=frame.iloc[j]['sample'],geometry_cost=costs[tuple(sorted((i,j)))]))
            separation.append(dict(method=key,query=i,organoid=frame.iloc[i]['sample'],
                near_far_ordering=float(np.mean((dd[near,None]<dd[None,far])+.5*(dd[near,None]==dd[None,far]))),spearman=float(spearmanr(g,dd).statistic)))
    pd.DataFrame(rows).to_csv(OUT/'retrieval.csv',index=False);pd.DataFrame(separation).to_csv(OUT/'separation.csv',index=False)
    np.savez_compressed(OUT/'distances.npz',**distances)
    # Export full-space cross-organoid neighbors for the two primary descriptors.
    neighbors=[]
    for key in ('volume_a1','volume_a0.38','surface_a1','surface_a0.38'):
        for i in frame.index[frame.included]:
            pool=frame.index[frame.included&(frame['sample']!=frame.iloc[i]['sample'])].to_numpy()
            ranking=pool[np.argsort(distances[key][i,pool],kind='stable')]
            for rank,j in enumerate(ranking[:10],1):neighbors.append(dict(method=key,query=frame.iloc[i].cell_id,rank=rank,neighbor=frame.iloc[j].cell_id,distance=distances[key][i,j]))
    pd.DataFrame(neighbors).to_csv(OUT/'neighbors.csv',index=False)


def probe_one(item):
    i,row=item;path=OUT/'probes'/f'{i:04d}.npz'
    if path.exists():return i
    v,f=prior.mesh(row);r=Rotation.random(random_state=np.random.default_rng(89000+i)).as_matrix()
    variants=dict(clean=v,rotation=v@r.T,noise=prior.perturb(v,f,87000+i));values={}
    for scope,vertices in variants.items():
        codes,_=mesh_descriptors(vertices,f,**CONFIG)
        values.update({f'{scope}__{key}':code for key,code in codes.items()})
    save_codes(path,**values);return i


def probes(workers=2):
    frame=pd.read_csv(OUT/'manifest.csv');plan=json.loads((OUT/'plan.json').read_text());arrays=np.load(OUT/'codes.npz')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for number,_ in enumerate(pool.map(probe_one,[(i,frame.iloc[i].to_dict()) for i in plan['probes']]),1):
            if number%5==0:print(f'Physical perturbations: {number}/25 objects',flush=True)
    rows=[];ids=frame.index[frame.included].to_numpy()
    for i in plan['probes']:
        a=np.load(OUT/'probes'/f'{i:04d}.npz');organoid=frame.iloc[i]['sample']
        for key in KEYS:
            base=key.removesuffix('_minmax');x=arrays[base];z=scaled(x,key,frame,organoid)
            clean_neighbors=None
            for scope in ('clean','rotation','noise'):
                q=a[f'{scope}__{base}']
                if key.endswith('_minmax'):
                    fit=x[frame.included.to_numpy(bool)&(frame['sample'].to_numpy()!=organoid)]
                    low=fit.min(0);span=np.ptp(fit,axis=0);valid=span>1e-8;q=(q-low)/np.where(valid,span,1)*valid
                else:q=q/np.linalg.norm(q)
                dd=cdist(q[None],z[ids])[0];ranking=ids[np.argsort(dd,kind='stable')];neighbors=set(ranking[ranking!=i][:5])
                if scope=='clean':clean_neighbors=neighbors
                rows.append(dict(method=key,query=i,organoid=organoid,scope=scope,correct=int(ranking[0]==i),top5=int(i in ranking[:5]),
                    relative_drift=float(np.linalg.norm(q-z[i])/max(np.min(cdist(z[i:i+1],z[ids[ids!=i]])),1e-12)),
                    neighbor_jaccard=len(neighbors&clean_neighbors)/max(len(neighbors|clean_neighbors),1)))
    pd.DataFrame(rows).to_csv(OUT/'robustness.csv',index=False)


def audit():
    """Independent numerical checks, not another alpha/descriptor selection pass."""
    frame=pd.read_csv(OUT/'manifest.csv');plan=json.loads((OUT/'plan.json').read_text());arrays=np.load(OUT/'codes.npz')
    ids=frame.index[frame.included].to_numpy();rows=[]
    for i in plan['probes'][::5]:
        v,f=prior.mesh(frame.iloc[i].to_dict())
        config=dict(CONFIG,sampling=[160,48,96]);fine,_=mesh_descriptors(v,f,**config)
        rotation=Rotation.random(random_state=99800+i).as_matrix();rot,_=mesh_descriptors(v@rotation.T,f,**config)
        for key in BASE:
            x=arrays[key];x=x/np.linalg.norm(x,axis=1,keepdims=True)
            b=fine[key]/np.linalg.norm(fine[key]);c=rot[key]/np.linalg.norm(rot[key]);d=cdist(b[None],x[ids])[0]
            margin=np.min(cdist(x[i:i+1],x[ids[ids!=i]]))
            rows.append(dict(query=i,method=key,coarse_fine_l2=float(np.linalg.norm(x[i]-b)),fine_rotation_l2=float(np.linalg.norm(b-c)),
                             nearest_distinct=float(margin),fine_recall=int(ids[np.argmin(d)]==i)))
        print(f'Finer quadrature check: {i}',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'convergence.csv',index=False)
    old=np.load(OLD/'zernike20.npz')['vectors'];new=arrays['volume_a1'];new=new/np.linalg.norm(new,axis=1,keepdims=True)
    comparison=np.linalg.norm(old-new,axis=1)
    save_json(OUT/'classical_crosscheck.json',dict(reference='Existing independent ZMPY3D64-cubed integration',
        cells=len(frame),median_l2=float(np.median(comparison)),p95_l2=float(np.percentile(comparison,95)),max_l2=float(comparison.max()),
        interpretation='Same polynomial family and normalization, different numerical integration. Report paired alpha changes within the new implementation separately from old-vs-new performance.'))


def demo(workers=2):
    from prepare_pointnet_shape import read_mesh
    dataset=ROOT/'results/demo-shapes-v1';data=json.loads((dataset/'dataset.json').read_text());rows=[];codes={key:[] for key in BASE}
    for i,c in enumerate(data['cells']):
        path=OUT/f'demo-{i:03d}.npz'
        if not path.exists():
            v,f=read_mesh(dataset/c['sample']/f"{c['label']}.mesh")
            z,_=mesh_descriptors(v,f,**CONFIG);save_codes(path,**z)
        a=np.load(path)
        for key in BASE:codes[key].append(a[key])
        rows.append(dict(cell_id=c['id'],sample=c['sample'],family=int(c['metrics']['demo_shape_class'])))
    frame=pd.DataFrame(rows);result=[]
    for key,values in codes.items():
        x=np.array(values);x/=np.linalg.norm(x,axis=1,keepdims=True);d=cdist(x,x)
        for i in range(len(frame)):
            pool=np.flatnonzero(frame['sample'].to_numpy()!=frame.iloc[i]['sample']);j=pool[np.argmin(d[i,pool])]
            result.append(dict(method=key,query=frame.iloc[i].cell_id,family=int(frame.iloc[i].family),correct=int(frame.iloc[j].family==frame.iloc[i].family)))
    pd.DataFrame(result).to_csv(OUT/'demo.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['encode','evaluate','probes','demo','audit']);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.workers not in (1,2):p.error('Use one or two workers on16GB Macs')
    freeze()
    if a.stage in ('encode','probes','demo'):globals()[a.stage](a.workers)
    else:globals()[a.stage]()
