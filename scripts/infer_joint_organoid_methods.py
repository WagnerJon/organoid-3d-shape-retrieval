"""Apply all 36 transferable Atlas methods to a new multi-organoid dataset.

Inference only. Saved training encoders, scalers, and nuisance fits are immutable.
All output and caches live under the supplied joint analysis folder.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_runtime_root = Path(__file__).resolve().parents[1]
for _key, _value in {'MPLCONFIGDIR': str(_runtime_root/'.cache/matplotlib'),
                      'XDG_CACHE_HOME': str(_runtime_root/'.cache'),
                      'NUMBA_CACHE_DIR': str(_runtime_root/'.cache/numba'),
                      'PYTORCH_ENABLE_MPS_FALLBACK': '1'}.items():
    os.environ.setdefault(_key, _value)

import numpy as np
import pandas as pd

from pointnet_shape_core import ROOT, sha256, sample_surface, normalize_cloud
from prepare_pointnet_shape import read_mesh
from spatial_benchmark_core import CLASSICAL, surface_frame
from encode_standalone_atlas import residual

DEVICE = 'mps'


def memory_limit(fraction):
    if DEVICE == 'mps':
        import torch
        torch.mps.set_per_process_memory_fraction(fraction)


def save(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix('.tmp')
    compact = path.name == 'dataset.json'
    tmp.write_text(json.dumps(value, allow_nan=False, indent=None if compact else 2,
                              separators=(',', ':') if compact else None))
    tmp.replace(path)


def npz(path, **values):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **values)
    tmp.replace(path)


def configure(folder):
    global JOINT, ATLAS, OUT
    JOINT = Path(folder)
    ATLAS = JOINT/'atlas_dataset'
    OUT = JOINT/'representations'
    OUT.mkdir(exist_ok=True)


def progress(stage, done, total, **extra):
    record = dict(status='running', stage=stage, completed=done, total=total, updated=time.time(), **extra)
    save(OUT/'status.json', record)
    print(json.dumps(record), flush=True)


def dataset():
    return json.loads((ATLAS/'dataset.json').read_text())


def original_methods():
    """Read the source method catalog without parsing its large cell arrays."""
    path = ROOT/'results/explorer_dataset/dataset.json'
    with path.open() as handle:
        header = handle.read(256*1024)
    start = header.find('"methods":')
    if start >= 0:
        try:
            methods, _ = json.JSONDecoder().raw_decode(header[start+len('"methods":'):].lstrip())
            if isinstance(methods, list):
                return methods
        except json.JSONDecodeError:
            pass
    return json.loads(path.read_text())['methods']


def prepare():
    data = dataset()
    original = pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv')
    medians = original[CLASSICAL].median()
    rows = []
    for index, c in enumerate(data['cells']):
        path = ATLAS/c['sample']/f"{c['label']}.mesh"
        row = dict(cell_id=c['id'], sample=c['sample'], label=c['label'], original_index=index,
                   inference_index=200000+index, source_mesh_sha256=sha256(path), **c['metrics'])
        for k in CLASSICAL:
            if k not in row or not np.isfinite(row[k]):
                row[k] = float(medians[k])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'manifest.csv', index=False)
    (OUT/'geometry').mkdir(exist_ok=True)
    (OUT/'allen').mkdir(exist_ok=True)
    hashes = {}
    model_paths = ['pointnet_shape/baseline/best.pt', 'point_mae/checkpoints/pretrain.pth',
                   'point_mae/best.pt', 'allen_reference/checkpoints/Rotation_invariant_pointcloud_SDF.ckpt',
                   'allen_reference/head_decoder/best.pt', 'shapeembed3d/invariant/best.pt', 'shapeembed3d/cnn/best.pt']
    model_paths += [f'{key}/best.pt' for key in ['simclr3d', 'simclr3d_raw', 'simclr3d_raw_isolated', 'icone3d_raw_isolated', 'icone3d_raw_context']]
    paths = [ROOT/'results'/p for p in model_paths]
    for sub in ['atlas_spatial', 'allen_reference/evaluation', 'point_mae', 'shapeembed3d', 'harmonics_cajal']:
        paths += list((ROOT/'results'/sub).glob('*fit.npz'))
    paths += [ROOT/'results/harmonics_cajal/harmonics_adjustment.npz', ROOT/'results/harmonics_cajal/cajal_profile_adjustment.npz']
    paths += [ROOT/'scripts'/name for name in ['spatial_benchmark_core.py', 'atlas_shape_space.py',
        'harmonics_cajal_core.py', 'simclr3d_core.py', 'pointnet_shape_core.py',
        'allen_reference_core.py', 'point_mae_reference.py', 'shapeembed3d_core.py']]
    from allen_reference_core import SOURCES
    from point_mae_reference import SOURCE
    paths += list(SOURCES.values())+[SOURCE]
    for p in paths:
        hashes[str(p.relative_to(ROOT))] = sha256(p)
    frozen = OUT/'frozen_sources.json'
    if frozen.exists() and json.loads(frozen.read_text()) != hashes:
        raise ValueError('A frozen inference source changed')
    save(frozen, hashes)
    for index, row in enumerate(rows):
        path = OUT/'geometry'/f'{index:04d}.npz'
        if path.exists():
            with np.load(path) as cached:
                assert str(cached['source_hash']) == row['source_mesh_sha256']
            continue
        v, f = read_mesh(ATLAS/row['sample']/f"{row['label']}.mesh")
        v = v.astype(float)
        center = (v.min(0)+v.max(0))/2
        scale = np.ptp(v, axis=0).max()
        if not scale > 0:
            raise ValueError('Degenerate mesh '+row['cell_id'])
        points = sample_surface((v-center)/scale, f, 32768, np.random.default_rng(20261110+row['inference_index']))
        _, axes, _, gap = surface_frame(v, f)
        npz(path, points=points, axes=axes, gap=gap, center=center, scale=scale, source_hash=row['source_mesh_sha256'])
        if index % 25 == 0:
            progress('geometry', index+1, len(rows))
    return frame


def learned(frame, stage):
    import torch
    import extend_atlas_shapes as extension
    torch.set_num_threads(2)
    memory_limit(.45)
    extension.OUT = OUT
    if stage == 'lite':
        extension.lite(frame, DEVICE)
    elif stage == 'point-mae':
        extension.point_mae(frame, DEVICE)
    else:
        # A single bounded worker keeps the full 8,192-point Allen encoder from
        # competing with another model for the machine's shared 16 GB memory.
        target = OUT/'allen_codes.npz'
        if target.exists():
            return
        extension.initialize_allen(DEVICE)
        for index, row in enumerate(frame.to_dict('records')):
            extension.allen_one(row)
            if (index+1) % 5 == 0 or index+1 == len(frame):
                progress('Allen 8192-point encoder', index+1, len(frame))
        values = {key: [] for key in ['pretrained', 'head_decoder']}
        for index in range(len(frame)):
            with np.load(OUT/'allen'/f'{index:04d}.npz') as a:
                for key in values:
                    values[key].append(a[key])
        npz(target, ids=frame.cell_id.to_numpy(dtype=str), **{k: np.asarray(v) for k, v in values.items()})


def pointnet(frame):
    import torch
    from pointnet_shape_core import PointCloudAutoencoder
    config = json.loads((ROOT/'config/pointnet_shape.json').read_text())
    destination = OUT/'pointnet_codes.npz'
    if destination.exists():
        return
    torch.set_num_threads(2)
    memory_limit(.4)
    state = torch.load(ROOT/'results/pointnet_shape/baseline/best.pt', map_location='cpu', weights_only=False)
    model = PointCloudAutoencoder(config).to(DEVICE).eval()
    model.load_state_dict(state['model'], strict=True)
    codes, reconstructions = [], []
    for first in range(0, len(frame), 4):
        batch = frame.iloc[first:first+4]
        points, centers, scales = [], [], []
        for row in batch.to_dict('records'):
            v, f = read_mesh(ATLAS/row['sample']/f"{row['label']}.mesh")
            seed = int.from_bytes(hashlib.sha256(f"{config['seed']}:{row['cell_id']}".encode()).digest()[:8], 'little')
            cloud = sample_surface(v, f, config['points'], np.random.default_rng(seed))
            normalized, _, center, scale = normalize_cloud(cloud, cloud)
            points.append(normalized); centers.append(center); scales.append(scale)
        with torch.no_grad():
            latent, prediction = model(torch.tensor(np.stack(points), device=DEVICE))
        z = latent.cpu().numpy()
        codes.extend(z/np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12))
        for row, pred, center, scale in zip(batch.to_dict('records'), prediction.cpu().numpy(), centers, scales):
            relative = Path('pointnet_shape128/frozen')/row['sample']/f"{row['label']}.points"
            path = ATLAS/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            physical = (pred*scale+center).astype('<f4')
            path.write_bytes(struct.pack('<I', len(physical))+physical.tobytes())
            reconstructions.append(str(relative))
        if first % 40 == 0:
            progress('PointNet++', min(first+4, len(frame)), len(frame))
    npz(destination, ids=frame.cell_id.to_numpy(dtype=str), codes=np.asarray(codes), paths=np.asarray(reconstructions))


def image_inputs(frame):
    """Bounded raw windows with physical coordinates; masks stay disk-backed."""
    from scipy import ndimage
    from skimage.transform import resize
    import tifffile
    from prepare_simclr3d import standardize
    from prepare_raw_simclr3d import raw_crop
    from prepare_isolated_raw_simclr3d import isolate
    target = OUT/'image_inputs'
    target.mkdir(exist_ok=True)
    if (target/'complete.json').exists():
        return
    n = len(frame)
    arrays = {name: np.lib.format.open_memmap(target/f'{name}.npy', mode='w+', dtype='u1' if name=='mask' else 'f4', shape=(n, 64, 64, 64)) for name in ['mask', 'context', 'isolated']}
    notes = []
    for sample, group in frame.groupby('sample', sort=False):
        number = int(sample.rsplit('MV', 1)[1])
        run = JOINT/'organoids'/f'MV{number}'
        config = json.loads((run/'run.json').read_text())
        native_spacing = np.array(config['spacingZYXum'])
        spacing = np.array(json.loads((run/'processing.json').read_text())['effectiveSpacingZYXum'])
        labels = np.load(run/'labels_processed.npy', mmap_mode='r')
        raw_path = JOINT/'raw'/Path(config['source']).name
        raw = tifffile.memmap(raw_path, mode='r')
        # Exact native-stack percentiles, with a temporary disk-backed float32
        # work array to avoid a multi-gigabyte RAM copy during normalization.
        low, high = np.percentile(raw, [1, 99.8])
        bounds = ndimage.find_objects(labels)
        for index, row in group.iterrows():
            label = int(row.label)
            bbox = bounds[label-1]
            mask = np.asarray(labels[bbox]) == label
            fallback = []
            try:
                binary, _ = standardize(mask, spacing, size=64, padding=1.15)
            except ValueError as exc:
                if str(exc) != 'Cell vanished during resampling':
                    raise
                # A few already-flagged, disconnected tiny fragments fall
                # between output samples. Conservatively map their REAL
                # occupied voxels to the same normalized cube, rather than
                # encode an empty mask or silently omit those objects.
                iso = ndimage.zoom(mask.astype(np.uint8), spacing/spacing.min(), order=0, prefilter=False)>0
                side = int(np.ceil(np.linalg.norm(iso.shape)*1.15))
                offsets = np.array([(side-s)//2 for s in iso.shape])
                coords = np.argwhere(iso)+offsets
                mapped = np.minimum(63, ((coords+.5)/side*64).astype(int))
                binary = np.zeros((64, 64, 64), dtype=np.uint8)
                binary[tuple(mapped.T)] = 1
                fallback.append('mask: conservative occupied-voxel transfer; standard thresholded resampling vanished')
            arrays['mask'][index] = binary
            # Pixel-center physical transform identical to the original raw
            # pipeline. Read only a cell's bounding box plus interpolation and
            # anti-aliasing halo, then sample onto the processed grid.
            processed_center = np.array([row.centroid_z_um, row.centroid_y_um, row.centroid_x_um])/spacing
            native_center = (processed_center+.5)*spacing/native_spacing-.5
            half = 20/native_spacing
            starts = np.maximum(0, np.floor(native_center-half-8).astype(int))
            stops = np.minimum(raw.shape, np.ceil(native_center+half+9).astype(int))
            slices = tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))
            normalized = np.clip((np.asarray(raw[slices], np.float32)-low)/max(float(high-low), 1e-6), 0, 1)
            context, outside_local = raw_crop(normalized, native_center-starts, native_spacing, fov=40, size=64)
            arrays['context'][index] = context
            # Resample just the original cell bounding box. The anti-aliasing
            # sigma follows skimage.resize on the whole native stack; a halo
            # preserves the local Gaussian support without loading that stack.
            axes = [(np.arange(s.start, s.stop)+.5)*spacing[k]/native_spacing[k]-.5 for k, s in enumerate(bbox)]
            lo = np.maximum(0, np.floor([a.min() for a in axes]).astype(int)-12)
            hi = np.minimum(raw.shape, np.ceil([a.max() for a in axes]).astype(int)+13)
            slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            patch = np.clip((np.asarray(raw[slices], np.float32)-low)/max(float(high-low), 1e-6), 0, 1)
            factors = np.asarray(raw.shape)/np.asarray(labels.shape)
            sigma = np.maximum((factors-1)/2, 0)
            # skimage's default "reflect" maps to scipy's "mirror" boundary
            # convention; match it at the native stack edges as well.
            patch = ndimage.gaussian_filter(patch, sigma=sigma, mode='mirror')
            coords = np.meshgrid(*[axis-lo[k] for k, axis in enumerate(axes)], indexing='ij')
            cell_raw = ndimage.map_coordinates(patch, coords, order=1, mode='nearest', prefilter=False)
            try:
                isolated, mask64, side = isolate(cell_raw, mask.astype(np.uint8), 1, size=64, padding=1.15)
            except ValueError as exc:
                if 'Invalid isolated cell' not in str(exc):
                    raise
                side = int(np.ceil(np.linalg.norm(mask.shape)*1.15))
                pads = [((side-s)//2, side-s-(side-s)//2) for s in mask.shape]
                cube = np.pad(cell_raw*mask, pads)
                isolated = resize(cube, (64,)*3, order=1, preserve_range=True, anti_aliasing=side>64).astype(np.float32)
                coordinates = np.argwhere(mask)+np.array([p[0] for p in pads])
                mapped = np.minimum(63, ((coordinates+.5)/side*64).astype(int))
                mask64 = np.zeros((64,)*3, bool)
                mask64[tuple(mapped.T)] = True
                isolated *= mask64
                # Retain observed intensities when interpolation misses a
                # subvoxel fragment; no new intensity values are synthesized.
                np.maximum.at(isolated, tuple(mapped.T), cell_raw[mask])
                fallback.append('isolated: conservative occupied-voxel/intensity transfer; nearest-neighbor gate vanished')
            arrays['isolated'][index] = isolated
            notes.append(dict(cell_id=row.cell_id, normalization_low=float(low), normalization_high=float(high),
                              context_outside_fraction=float(outside_local), isolated_cube_side=int(side), isolated_mask_voxels=int(mask64.sum()),
                              subvoxel_fragment_fallback=fallback))
        for a in arrays.values():
            a.flush()
        progress('raw/mask input preparation', int(group.index.max())+1, n, sample=sample)
        del raw
    save(target/'complete.json', dict(objects=n, notes=notes, preprocessing='Original per-stack percentile normalization; physical 40 µm context FOV; original masked size-normalized crop. Bounded interpolation windows replace full-stack float32 copies.'))


def image_codes(frame):
    import torch
    from simclr3d_core import ResNet3DSimCLR
    torch.set_num_threads(2)
    memory_limit(.4)
    inputs = OUT/'image_inputs'
    variants = [('simclr3d', 'mask'), ('simclr3d_raw', 'context'), ('simclr3d_raw_isolated', 'isolated'),
                ('icone3d_raw_isolated', 'isolated'), ('icone3d_raw_context', 'context')]
    for key, source in variants:
        path = OUT/f'{key}_codes.npz'
        if path.exists():
            continue
        checkpoint = torch.load(ROOT/'results'/key/'best.pt', map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        model = ResNet3DSimCLR(cfg['base_channels'], cfg['embedding_dim'], cfg.get('projection_dim', 64))
        if key.startswith('icone'):
            model = model.encoder
            model.load_state_dict(checkpoint['backbone'], strict=True)
        else:
            model.load_state_dict(checkpoint['model'], strict=True)
            model = model.encoder
        model = model.to(DEVICE).eval()
        values = []
        x = np.load(inputs/f'{source}.npy', mmap_mode='r')
        with torch.no_grad():
            for first in range(0, len(frame), 8):
                batch = torch.tensor(np.array(x[first:first+8], copy=True), dtype=torch.float32, device=DEVICE)[:, None]
                h = model(batch).cpu().numpy()
                values.extend(h/np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12))
                if first % 80 == 0:
                    progress(key, min(first+8, len(frame)), len(frame))
        npz(path, ids=frame.cell_id.to_numpy(dtype=str), codes=np.asarray(values))
        del model
        if DEVICE == 'mps':
            torch.mps.empty_cache()


def harmonic_worker(task):
    row, atlas, target = task
    from harmonics_cajal_core import harmonics, surface_points
    path = Path(target)/f"{row['original_index']:04d}.npz"
    if path.exists():
        with np.load(path) as a:
            assert str(a['mesh_hash']) == row['source_mesh_sha256']
        return row['cell_id']
    v, f = read_mesh(Path(atlas)/row['sample']/f"{row['label']}.mesh")
    v, f = v.astype(float), f.astype(np.int64)
    h = harmonics(v, f)
    p = surface_points(v, f, 128, 20261101+row['inference_index'])
    npz(path, harmonics=h, points=p, mesh_hash=row['source_mesh_sha256'])
    return row['cell_id']


def harmonic_codes(frame):
    target = OUT/'harmonics'
    target.mkdir(exist_ok=True)
    if (OUT/'surfaces.npz').exists():
        return
    with ProcessPoolExecutor(max_workers=2) as pool:
        for done, _ in enumerate(pool.map(harmonic_worker, [(r, str(ATLAS), str(target)) for r in frame.to_dict('records')]), 1):
            if done % 25 == 0:
                progress('spherical harmonics', done, len(frame))
    values, clouds = [], []
    for i in range(len(frame)):
        with np.load(target/f'{i:04d}.npz') as a:
            values.append(a['harmonics']); clouds.append(a['points'])
    npz(OUT/'surfaces.npz', ids=frame.cell_id.to_numpy(dtype=str), harmonics=np.asarray(values), points=np.asarray(clouds))


_CAJAL_CELLS = None
_CAJAL_LANDMARKS = None
_CAJAL_PATH = None


def cajal_initialize(folder):
    global _CAJAL_CELLS, _CAJAL_LANDMARKS, _CAJAL_PATH
    from cajal.qgw import quantized_icdm
    from scipy.spatial.distance import cdist
    _CAJAL_PATH = Path(folder)
    u = np.full(128, 1/128)
    points = np.load(_CAJAL_PATH/'surfaces.npz')['points']
    fit = np.load(ROOT/'results/harmonics_cajal/cajal_profile_adjustment.npz')
    old = np.load(ROOT/'results/harmonics_cajal/surfaces.npz')
    indices = {cid: i for i, cid in enumerate(old['ids'])}
    _CAJAL_CELLS = [quantized_icdm(cdist(p, p), u, 32) for p in points]
    _CAJAL_LANDMARKS = [quantized_icdm(cdist(old['points'][indices[cid]], old['points'][indices[cid]]), u, 32) for cid in fit['landmarks']]


def cajal_row(i):
    from cajal.qgw import quantized_gw
    path = _CAJAL_PATH/'qgw_rows'/f'{i:04d}.npz'
    if not path.exists():
        values = [quantized_gw(_CAJAL_CELLS[i], _CAJAL_CELLS[j])[1] for j in range(i+1, len(_CAJAL_CELLS))]
        profile = [quantized_gw(_CAJAL_CELLS[i], p)[1] for p in _CAJAL_LANDMARKS]
        npz(path, distances=np.asarray(values), profile=np.asarray(profile))
    return i


def cajal_codes(frame):
    if (OUT/'cajal_codes.npz').exists():
        return
    (OUT/'qgw_rows').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=3, initializer=cajal_initialize, initargs=(str(OUT),)) as pool:
        jobs = [pool.submit(cajal_row, i) for i in range(len(frame))]
        for done, job in enumerate(as_completed(jobs), 1):
            job.result()
            if done % 20 == 0:
                progress('CAJAL rows and original landmarks', done, len(frame))
    d = np.zeros((len(frame), len(frame)))
    profiles = []
    for i in range(len(frame)):
        with np.load(OUT/'qgw_rows'/f'{i:04d}.npz') as a:
            d[i, i+1:] = a['distances']; profiles.append(a['profile'])
    d += d.T
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Invalid CAJAL matrix')
    npz(OUT/'cajal_codes.npz', ids=frame.cell_id.to_numpy(dtype=str), distances=d, profile=np.asarray(profiles))


def publish(frame):
    from organoid_membership import project
    path = ATLAS/'dataset.json'
    before = sha256(path)
    data = dataset()
    templates = {m['id']: m for m in original_methods()}
    new = {}
    def load(name):
        a = np.load(OUT/name)
        assert a['ids'].tolist() == frame.cell_id.tolist()
        return a
    for key in ['simclr3d', 'simclr3d_raw', 'simclr3d_raw_isolated', 'icone3d_raw_isolated', 'icone3d_raw_context']:
        new[key] = load(f'{key}_codes.npz')['codes']
    pn = load('pointnet_codes.npz')
    new['pointnet_shape128'] = pn['codes']
    learned_raw = {}
    allen = load('allen_codes.npz')
    for arm in ['pretrained', 'head_decoder']:
        learned_raw['allen_'+arm] = (allen[arm], ROOT/'results/allen_reference/evaluation'/f'{arm}_readout_fit.npz')
    for arm in ['pretrained', 'adapted']:
        a = load(f'point_mae_{arm}.npz')
        for kind in ['native', 'pose_average']:
            learned_raw[f'point_mae_{arm}_{kind}'] = (a[kind], ROOT/'results/point_mae'/f'{arm}_{kind}_fit.npz')
    for kind in ['invariant', 'cnn']:
        learned_raw[f'shapeembed3d_{kind}'] = (load(f'lite_{kind}.npz')['mean'], ROOT/'results/shapeembed3d'/f'{kind}_fit.npz')
    for base, (raw, fit_path) in learned_raw.items():
        fit = dict(np.load(fit_path))
        new[base+'_raw'] = raw
        new[base+'_adjusted'] = residual(frame, (raw-fit['center'])/fit['scale'], fit)
    sh = load('surfaces.npz')['harmonics']
    new['spherical_harmonics'] = sh
    new['spherical_harmonics_adjusted'] = residual(frame, sh, np.load(ROOT/'results/harmonics_cajal/harmonics_adjustment.npz'))
    cajal = load('cajal_codes.npz')
    new['cajal_qgw'] = np.c_[np.arange(len(frame)), cajal['distances']]
    new['cajal_profile_adjusted'] = residual(frame, cajal['profile'], np.load(ROOT/'results/harmonics_cajal/cajal_profile_adjustment.npz'))
    for index, c in enumerate(data['cells']):
        assert c['id'] == frame.iloc[index].cell_id
        assert sha256(ATLAS/c['sample']/f"{c['label']}.mesh") == frame.iloc[index].source_mesh_sha256
        c.setdefault('pointReconstructions', {})['pointnet_shape128'] = str(pn['paths'][index])
        for key, values in new.items():
            if not np.isfinite(values[index]).all():
                raise ValueError('Nonfinite code '+key)
            c['representations'][key] = dict(vector=np.asarray(values[index], float).tolist(), point=[0., 0.], cluster=0)
    specs = []
    for key, values in new.items():
        spec = dict(templates[key])
        spec.update(dimensions=values.shape[1], coverage='all_cells', variance=project(data['cells'], key, spec['metric']), stackVariance=[0., 0.])
        spec['description'] = spec.get('description', spec['name'])+' Applied to all cells of five new organoids using frozen original-dataset inference; no retraining. Original benchmark results do not establish performance on this dataset.'
        if spec['metric'] == 'precomputed' or spec.get('scope') == 'cells':
            for s in data['organoids']:
                s['representations'].pop(key, None)
        else:
            for stack in data['organoids']:
                cells = [c for c in data['cells'] if c['sample'] == stack['sample'] and c['qc']]
                if cells:
                    v = np.mean([c['representations'][key]['vector'] for c in cells], axis=0)
                    if spec['metric'] == 'cosine':
                        v /= max(np.linalg.norm(v), 1e-12)
                    stack['representations'][key] = dict(vector=v.tolist(), point=[0., 0.], cluster=0)
            spec['stackVariance'] = project([s for s in data['organoids'] if key in s['representations']], key)
        specs.append(spec)
    existing = [m for m in data['methods'] if m['id'] not in new]
    # Keep the best geometry method as Atlas's default initial shape choice.
    data['methods'] = [m for m in existing if m['id'] != 'spatial_robust']+specs+[m for m in existing if m['id'] == 'spatial_robust']
    # The standalone spatial exporter starts with one stack and zero stack
    # coordinates. Refit those descriptive displays for this five-organoid
    # cohort as well; cell-only and pairwise methods remain cell-only.
    for spec in data['methods']:
        if spec.get('scope') != 'cells' and spec['metric'] not in ['precomputed', 'shape_hybrid95']:
            spec['stackVariance'] = project([s for s in data['organoids'] if spec['id'] in s['representations']], spec['id'])
    expected = set(templates)-{'cajal_gw_validation'}
    assert {m['id'] for m in data['methods']} == expected
    assert all(set(c['representations']) == expected for c in data['cells'])
    npz(OUT/'vectors.npz', ids=frame.cell_id.to_numpy(dtype=str),
        **{key: np.array([c['representations'][key]['vector'] for c in data['cells']], dtype=float) for key in sorted(expected)})
    data['frozenTransfer'] = dict(methods=len(expected), objects=len(frame), organoids=5, sources=json.loads((OUT/'frozen_sources.json').read_text()),
        omitted='The old cajal_gw_validation is a 48-original-cell numerical validation reference, not an encoder transferable to new cells.',
        rawPreprocessing=json.loads((OUT/'image_inputs/complete.json').read_text()),
        caveat='New intensity distribution, shapes and physical scales may be outside the original training distribution. Descriptive projections and groups do not prove biological classes.')
    backup = OUT/f'dataset.before-full-{before[:12]}.json'
    if not backup.exists():
        shutil.copy2(path, backup)
    assert sha256(path) == before
    for rel, digest in data['frozenTransfer']['sources'].items():
        assert sha256(ROOT/rel) == digest
    save(path, data)
    save(OUT/'complete.json', dict(status='complete', methods=len(expected), objects=len(frame), organoids=5, dataset_sha256=sha256(path), retrained=False))
    progress('complete', len(frame), len(frame), methods=len(expected))


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--stage', choices=['prepare', 'spatial', 'image-inputs', 'image-codes', 'pointnet', 'lite', 'point-mae', 'allen', 'harmonics', 'cajal', 'publish'], required=True)
    args = parser.parse_args()
    DEVICE = args.device
    configure(args.folder)
    frame = prepare()
    stages = {'prepare': lambda: None,
              'spatial': lambda: subprocess.run([sys.executable, str(ROOT/'scripts/encode_standalone_atlas.py'), str(ATLAS)], check=True),
              'image-inputs': lambda: image_inputs(frame), 'image-codes': lambda: image_codes(frame),
              'pointnet': lambda: pointnet(frame), 'lite': lambda: learned(frame, 'lite'),
              'point-mae': lambda: learned(frame, 'point-mae'), 'allen': lambda: learned(frame, 'allen'),
              'harmonics': lambda: harmonic_codes(frame), 'cajal': lambda: cajal_codes(frame), 'publish': lambda: publish(frame)}
    stages[args.stage]()


if __name__ == '__main__':
    main()
