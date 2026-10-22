"""Apply ZMPY3D and select a practical frozen retrieval descriptor on five organoids."""
from five_organoid_retrieval_core import *
import argparse, subprocess, time, sqlite3
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
from spatial_benchmark_core import geometry_cost


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'protocol.json').exists(): return
    data = source(); rows = []
    old = pd.read_csv(JOINT / 'representations/manifest.csv')
    for i, c in enumerate(data['cells']):
        assert old.iloc[i].cell_id == c['id']
        rows.append(dict(index=i, cell_id=c['id'], sample=c['sample'], label=c['label'],
                         included=bool(c['qc'] and c.get('inMainOrganoid', True)),
                         source_mesh_sha256=old.iloc[i].source_mesh_sha256))
    frame = pd.DataFrame(rows); frame.to_csv(OUT / 'manifest.csv', index=False)
    rng = np.random.default_rng(20261021); queries = []; probes = []; references = {}
    for sample, subset in frame[frame.included].groupby('sample'):
        ids = rng.permutation(subset.index.to_numpy()); queries.extend(ids[:20].tolist()); probes.extend(ids[:5].tolist())
    for i in queries:
        pool = frame.index[frame.included & (frame['sample'] != frame.iloc[i]['sample'])].to_numpy()
        references[str(i)] = rng.choice(pool, 16, replace=False).tolist()
    save_json(OUT / 'plan.json', dict(queries=queries, probes=probes, random_reference_pairs=references))
    save_json(OUT / 'protocol.json', dict(date='2026-10-21', purpose='Find similar-looking complete3D cell shapes, rather than reconstruct or remove ordinary morphometry.',
        cells=len(frame), included=int(frame.included.sum()), organoids=frame['sample'].nunique(),
        source_dataset_sha256=sha(ATLAS / 'dataset.json'), source_manifest_sha256=sha(JOINT / 'representations/manifest.csv'),
        methods=METHODS, practical_shortlist=PRIMARY,
        zernike=dict(implementation='Published ZMPY3D NumPy routines, no reimplementation of polynomial transform',
                     repository='https://github.com/tawssie/ZMPY3D', commit=subprocess.check_output(['git','-C',str(ZMPY),'rev-parse','HEAD'], text=True).strip(),
                     order=20, dimensions=121, grid=64, center='Exact volume centroid for watertight consistently oriented meshes; otherwise exact area centroid',
                     scale='Maximum vertex radius maps to0.9 inside the unit ball; physical scale excluded',
                     field='Soft occupancy from original mesh SDF, half-voxel logistic width, clipped to unit ball; no source segmentation changed',
                     normalization='L2-normalize all121 invariant norms. Structural zeros are retained; no per-coordinate whitening, no nuisance regression.',
                     comparison='Euclidean in full121D; norm invariants lose phase/chirality and are not complete invariants'),
        evaluation='100 fixed QC-passing queries,20 per organoid. Each retrieves from QC-passing cells of the other four organoids. Frozen encoders only; no target-dataset representation training. Display PCA/UMAP is never used for search.',
        geometry='Proper-rotation24-start8-iteration surface ICP on768 deterministic area samples. Exact surface-area centroid/RMS-radius normalization removes pose/scale. No scale/reflection optimization. Costs compare methods on these same query/gallery cells, not earlier differently normalized benchmarks.',
        negatives='For each query,16 fixed random other-organoid references. Report within-query Spearman descriptor-vs-geometry correlation and pairwise ordering accuracy between lower and upper geometric quartiles. Geometric proxy, no human shape labels.',
        perturbation='25 fixed queries,5 per organoid: one smooth2%-RMS-radius normal surface perturbation and one independent actual geometry rotation. Fresh clean re-encoding checks consistency with existing gallery codes. Noise recall@1 and top5 overlap use all747 included cells. No segmentation split/merge error claim.',
        selection='Practical shortlist fixed from prior evidence: raw shells, new Zernike20, adapted pose-averaged Point-MAE, spatial PCA with fallback. Choose lowest mean cross-organoid geometric retrieval cost among those with clean recall100%, rotation recall>=96% and noise recall no more than4 percentage points (one probe) below raw shells. If none improves raw, keep raw. Other frozen methods are contextual geometry-only comparisons, not dismissed as universally inferior.',
        uncertainty='Report individual query values and organoid means; descriptive4000 organoid-bootstrap intervals over only five organoids. Method selection reuses this target benchmark: no independent validation claim or biological state claim.',
        outputs='All961 Zernike codes, full-space nearest/farthest indices excluding QC-failed cells by default, comparison report, fixed surface examples, and an additive backed-up Atlas export. Existing methods/metrics/QC/source meshes preserved.'))
    print('Frozen comparison plan:961 represented objects,747 included;100 queries,25 probes', flush=True)


def encode_one(item):
    i, row = item; folder = OUT / 'zernike_cells'; path = folder / f'{i:04d}.npz'
    if path.exists():
        a = np.load(path); assert str(a['mesh_sha256']) == row['source_mesh_sha256']; return i
    v, f = mesh(row); start = time.monotonic(); vector = zernike(v, f)
    np.savez_compressed(path, vector=vector, mesh_sha256=row['source_mesh_sha256'], seconds=time.monotonic()-start)
    return i


def encode():
    frame = pd.read_csv(OUT / 'manifest.csv'); (OUT / 'zernike_cells').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for n, _ in enumerate(pool.map(encode_one, enumerate(frame.to_dict('records')), chunksize=1)):
            if n % 40 == 0: print(f'ZMPY3D order20: {n+1}/{len(frame)} original cell surfaces', flush=True)
    codes = np.stack([np.load(OUT / 'zernike_cells' / f'{i:04d}.npz')['vector'] for i in range(len(frame))])
    np.savez_compressed(OUT / 'zernike20.npz', ids=frame.cell_id.to_numpy(dtype=str), vectors=codes)
    np.save(OUT / 'surface_clouds.npy', np.stack([sample_surface(normalize(*mesh(row)), mesh(row)[1], 768,
            np.random.default_rng(45500+i)) for i, row in enumerate(frame.to_dict('records'))]))
    save_json(OUT / 'encoding.complete.json', dict(objects=len(frame), dimensions=121, implementation='ZMPY3D CPU'))
    print('All961 Zernike descriptors and deterministic geometry clouds ready', flush=True)


def geometry_init():
    global CLOUDS
    CLOUDS = np.load(OUT / 'surface_clouds.npy', mmap_mode='r')


def geometry_one(pair):
    return pair[0], pair[1], geometry_cost(CLOUDS[pair[0]], CLOUDS[pair[1]])


def cache_geometry(requested):
    db = sqlite3.connect(OUT / 'geometry.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS cost (i INTEGER,j INTEGER,value REAL,PRIMARY KEY(i,j))')
    result = {(int(i), int(j)):float(v) for i,j,v in db.execute('SELECT i,j,value FROM cost')}
    missing = sorted(set(tuple(sorted((int(i), int(j)))) for i,j in requested)-result.keys())
    with ProcessPoolExecutor(max_workers=2, initializer=geometry_init) as pool:
        for n, (i,j,value) in enumerate(pool.map(geometry_one, missing, chunksize=4)):
            result[(i,j)] = value; db.execute('INSERT OR REPLACE INTO cost VALUES(?,?,?)',(i,j,value))
            if n % 100 == 0: db.commit(); print(f'Cross-organoid geometry labels {n+1}/{len(missing)}', flush=True)
    db.commit(); db.close(); return result


def evaluate():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sha(ATLAS / 'dataset.json') == protocol['source_dataset_sha256']
    frame = pd.read_csv(OUT / 'manifest.csv'); plan = json.loads((OUT / 'plan.json').read_text())
    d = matrices(source(), np.load(OUT / 'zernike20.npz')['vectors'])
    np.savez_compressed(OUT / 'all_distances.npz', **d)
    requests = set(); chosen = {}
    for i in plan['queries']:
        gallery = frame.index[frame.included & (frame['sample'] != frame.iloc[i]['sample'])].to_numpy()
        for key in METHODS:
            j = int(gallery[np.argmin(d[key][i,gallery])]); chosen[(key,i)] = j
            requests.add(tuple(sorted((i,j))))
        requests.update(tuple(sorted((i,int(j)))) for j in plan['random_reference_pairs'][str(i)])
    costs = cache_geometry(requests); rows = []; separations = []; random_rows = []
    for i in plan['queries']:
        refs = np.array(plan['random_reference_pairs'][str(i)]); g = np.array([costs[tuple(sorted((i,int(j))))] for j in refs])
        order = np.argsort(g); near = order[:4]; far = order[-4:]
        for key in METHODS:
            j = chosen[(key,i)]
            rows.append(dict(method=key,query=i,cell_id=frame.iloc[i].cell_id,organoid=frame.iloc[i]['sample'],
                             neighbor=j,neighbor_cell=frame.iloc[j].cell_id,neighbor_organoid=frame.iloc[j]['sample'],
                             geometry_cost=costs[tuple(sorted((i,j)))]))
            distances = d[key][i,refs]
            ordering = np.mean((distances[near,None]<distances[None,far]) + .5*(distances[near,None]==distances[None,far]))
            correlation = spearmanr(g,distances).statistic
            separations.append(dict(method=key,query=i,organoid=frame.iloc[i]['sample'],
                                    near_far_ordering=float(ordering),spearman=float(correlation) if np.isfinite(correlation) else 0.))
            for r, gg, dd in zip(refs,g,distances):random_rows.append(dict(method=key,query=i,reference=int(r),geometry_cost=float(gg),descriptor_distance=float(dd)))
    pd.DataFrame(rows).to_csv(OUT / 'retrieval_comparison.csv',index=False)
    pd.DataFrame(separations).to_csv(OUT / 'separation_comparison.csv',index=False)
    pd.DataFrame(random_rows).to_csv(OUT / 'fixed_pair_distances.csv',index=False)
    save_json(OUT / 'geometry.complete.json', dict(queries=len(plan['queries']),organoids=5,methods=len(METHODS),
                                                no_target_encoder_training=True))
    print('Frozen-descriptor cross-organoid comparison complete',flush=True)


def spatial_code(v, f, row):
    from atlas_shape_space import signature, pack
    from pointnet_shape_core import normalize_cloud
    from spatial_benchmark_core import orbit_grids
    cfg = json.loads((ROOT/'config/pointnet_shape.json').read_text())
    seed = int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{row['cell_id']}".encode()).digest()[:8], 'little')
    points = sample_surface(v,f,cfg['points'],np.random.default_rng(seed))
    _,_,center,radius = normalize_cloud(points,points); v = (v-center)/radius
    frame = surface_frame(v,f); pca = np.load(ROOT/'results/spatial_benchmark/aligned_pca.npz')
    encoder = np.load(ROOT/'results/atlas_spatial/signature_encoder.npz')
    orbit = (orbit_grids(v,f,frame).reshape(4,-1)-pca['mean'])@pca['components'].T
    sig = (signature(v,f)-encoder['mean'])@encoder['components'].T*encoder['scale']
    return pack([frame[3]],orbit[None],sig[None])[0]


def query_distance(key, query, gallery):
    if key != 'spatial_robust':return cdist(query[None],gallery)[0]
    from atlas_shape_space import GAP
    orbit = query[1:381].reshape(4,95); g = gallery[:,1:381].reshape(-1,4,95)
    d = np.min(np.stack([cdist(orbit[a:a+1],g[:,b])[0] for a in range(4) for b in range(4)]),axis=0)
    fallback = (query[0]>.5)|(gallery[:,0]>.5)
    d[fallback] = cdist(query[None,381:],gallery[fallback,381:])[0]
    return d


def probes():
    import torch
    from harmonics_cajal_core import harmonics,angular_basis
    from point_mae_reference import model,encode as encode_mae
    from run_point_mae_reference import draw
    from spatial_benchmark_core import SIGNS
    frame=pd.read_csv(OUT/'manifest.csv'); plan=json.loads((OUT/'plan.json').read_text()); data=source()
    gallery_ids=frame.index[frame.included].to_numpy(); records=data['cells']
    gallery={key:np.array([records[i]['representations'][key]['vector'] for i in gallery_ids]) for key in PRIMARY if key!='zernike20'}
    have_gallery=(OUT/'zernike20.npz').exists()
    if have_gallery:gallery['zernike20']=np.load(OUT/'zernike20.npz')['vectors'][gallery_ids]
    torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu'
    if device=='mps':torch.mps.set_per_process_memory_fraction(.35)
    m=model(); state=torch.load(ROOT/'results/point_mae/best.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(state['model'],strict=True);m=m.to(device).eval()
    basis=angular_basis(); rows=[]; (OUT/'probes').mkdir(exist_ok=True)
    for number,i in enumerate(plan['probes']):
        path=OUT/'probes'/f'{i:04d}.npz';row=frame.iloc[i].to_dict()
        if not path.exists():
            v,f=mesh(row);rng=np.random.default_rng(89000+i);rotation=Rotation.random(random_state=rng).as_matrix()
            variants={'clean':v,'noise':perturb(v,f,87000+i),'rotation':v@rotation.T}; codes={}
            for scope,vertices in variants.items():
                codes[f'{scope}__spherical_harmonics']=harmonics(vertices,f,basis)
                codes[f'{scope}__zernike20']=zernike(vertices,f)
                codes[f'{scope}__spatial_robust']=spatial_code(vertices,f,row)
                center=(vertices.min(0)+vertices.max(0))/2;scale=np.ptp(vertices,axis=0).max()
                index=200000+i;pool=sample_surface((vertices-center)/scale,f,32768,np.random.default_rng(20261110+index))
                points=draw(pool,index);axes=surface_frame(vertices,f)[1]; poses=[]
                with torch.no_grad():
                    for sign in SIGNS:
                        p=(points@axes*sign).astype(np.float32)
                        poses.append(encode_mae(m,torch.tensor(p[None],device=device)).cpu().numpy()[0])
                codes[f'{scope}__point_mae_adapted_pose_average_raw']=np.mean(poses,axis=0)
            np.savez_compressed(path,**codes)
        if not have_gallery:
            print(f'Prepared clean /noise /rotation geometry probes {number+1}/{len(plan["probes"])}; full gallery still encoding',flush=True)
            continue
        a=np.load(path)
        for key in PRIMARY:
            clean_distance=query_distance(key,a[f'clean__{key}'],gallery[key]);clean_ranking=np.argsort(clean_distance)
            for scope in ['clean','noise','rotation']:
                d=query_distance(key,a[f'{scope}__{key}'],gallery[key]);rank=np.argsort(d)
                clean_neighbors=set(gallery_ids[clean_ranking][gallery_ids[clean_ranking]!=i][:5])
                neighbors=set(gallery_ids[rank][gallery_ids[rank]!=i][:5])
                rows.append(dict(method=key,scope=scope,query=i,cell_id=row['cell_id'],organoid=row['sample'],
                                 correct=int(gallery_ids[rank[0]]==i),top5=int(i in gallery_ids[rank[:5]]),
                                 neighbor_jaccard=len(clean_neighbors&neighbors)/max(len(clean_neighbors|neighbors),1)))
        print(f'Fresh clean /2% noise /rotation probe {number+1}/{len(plan["probes"])} on {device}',flush=True)
    if not have_gallery:
        print('All probe descriptors cached; rerun this stage after full gallery encoding to score them',flush=True)
        return
    pd.DataFrame(rows).to_csv(OUT/'robustness.csv',index=False)
    save_json(OUT/'probes.complete.json',dict(cells=25,methods=PRIMARY,device=device))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','encode','evaluate','probes'])
    globals()[parser.parse_args().stage]()
