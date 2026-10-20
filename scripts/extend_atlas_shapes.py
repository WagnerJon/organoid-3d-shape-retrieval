"""Frozen inference for cells missing Allen / Point-MAE / Lite3D codes.

Writes a separate exploration cache; never changes benchmark artifacts or trains.
The final exporter merges these rows with the original, unchanged 609-cell codes.
"""
import argparse
import copy
import json
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd
import torch

from pointnet_shape_core import ROOT, sha256, sample_surface
from prepare_pointnet_shape import read_mesh
from spatial_benchmark_core import surface_frame, SIGNS

OUT = ROOT / 'results/atlas_full_shape'
ATLAS = ROOT / 'results/explorer_dataset'
_MODEL = None
_HEADS = None
_DEVICE = None


def save(path, **values):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def status(stage, completed, total, started, **extra):
    value = dict(stage=stage, completed=completed, total=total, seconds=time.monotonic()-started, **extra)
    (OUT / 'status.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def frozen_files():
    files = [ROOT / 'results/pointnet_shape/data/manifest.csv']
    for folder in ('allen_reference', 'point_mae', 'shapeembed3d'):
        base = ROOT / 'results' / folder
        files += list(base.glob('*fit.npz')) + list(base.glob('*codes.npz')) + list(base.glob('*features.npz'))
        files += [p for p in [base/'protocol.json', base/'manifest.csv', base/'readouts.npz', base/'report.md', base/'retrieval.json'] if p.exists()]
    base = ROOT / 'results/allen_reference'
    files += list((base/'evaluation').glob('*readout_fit.npz'))
    files += [base/'evaluation/codes.npz', base/'evaluation/readouts.npz', base/'checkpoints/Rotation_invariant_pointcloud_SDF.ckpt', base/'head_decoder/best.pt']
    base = ROOT / 'results/point_mae'
    files += [base/'checkpoints/pretrain.pth', base/'best.pt']
    files += [ROOT/'results/shapeembed3d'/kind/'best.pt' for kind in ('invariant', 'cnn')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))}


def verify_frozen():
    expected = json.loads((OUT/'frozen_inputs.json').read_text())
    for name, digest in expected.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Frozen source changed: ' + name)


def prepare():
    OUT.mkdir(exist_ok=True, parents=True)
    for name in ('geometry', 'allen'):
        (OUT/name).mkdir(exist_ok=True)
    snapshot = OUT/'frozen_inputs.json'
    if snapshot.exists():
        verify_frozen()
    else:
        snapshot.write_text(json.dumps(frozen_files(), indent=2))
    frame = pd.read_csv(ROOT/'results/pointnet_shape/data/manifest.csv')
    original = pd.read_csv(ROOT/'results/allen_reference/manifest.csv')
    known = set(original.cell_id)
    frame['original_index'] = np.arange(len(frame))
    frame['inference_index'] = frame.original_index + 100000
    old_order = dict(zip(original.cell_id, range(len(original))))
    frame['benchmark_index'] = [old_order.get(cid, -1) for cid in frame.cell_id]
    path = OUT/'manifest.csv'
    if path.exists() and pd.read_csv(path).cell_id.tolist() != frame.cell_id.tolist():
        raise ValueError('Full inference manifest changed')
    frame.to_csv(path, index=False)
    data = json.loads((ATLAS/'dataset.json').read_text()); lookup = {c['id']: c for c in data['cells']}
    if set(lookup) != set(frame.cell_id):
        raise ValueError('Atlas / inference cell inventories differ')
    missing = frame[~frame.cell_id.isin(known)]
    started = time.monotonic()
    for done, row in enumerate(missing.to_dict('records'), 1):
        cell = lookup[row['cell_id']]; mesh = ATLAS/cell['sample']/f"{cell['label']}.mesh"
        if sha256(mesh) != row['source_mesh_sha256']:
            raise ValueError('Stale source mesh: ' + row['cell_id'])
        path = OUT/'geometry'/f"{row['original_index']:04d}.npz"
        if path.exists():
            with np.load(path) as cached:
                if str(cached['source_hash']) != row['source_mesh_sha256']:
                    raise ValueError('Stale surface pool')
            continue
        v, f = read_mesh(mesh); v = v.astype(float)
        center = (v.min(0)+v.max(0))/2; scale = np.ptp(v, axis=0).max()
        if not scale > 0:
            raise ValueError('Degenerate mesh: ' + row['cell_id'])
        points = sample_surface((v-center)/scale, f, 32768,
                                np.random.default_rng(20261110+row['inference_index'])).astype(np.float32)
        _, axes, _, gap = surface_frame(v, f)
        save(path, points=points, axes=axes, gap=gap, center=center, scale=scale, source_hash=row['source_mesh_sha256'])
        if done % 50 == 0 or done == len(missing):
            status('surface preparation', done, len(missing), started)
    (OUT/'protocol.json').write_text(json.dumps(dict(
        scope='Exploratory full-data frozen inference. No retraining, model selection or benchmark modification.',
        cells=len(frame), existing=len(original), additional=len(missing),
        benchmark_manifest_sha256=sha256(ROOT/'results/allen_reference/manifest.csv'),
        full_manifest_sha256=sha256(OUT/'manifest.csv'),
        source='Original physical triangle surfaces. Bounding-box-center / maximum-side normalization; 32768 area-sampled pool.',
        seeds='New-cell inference index = original full-manifest index + 100000; pool20261110+index, Allen draw20261112+index*17, Point-MAE draw20261019+index*3, Lite FPS20261022+4*index.',
        retained='All original609 vectors remain byte-for-byte identical, including their original fixed training-input views.',
        augmentation='No added jitter or random rotation for any new-cell inference. Point-MAE pose averaging retains the original four proper PCA sign frames.',
        adjustment='Original fixed96-training-cell descriptor scaling and nuisance adjustment; no refitting.',
        exclusions='Encode all cells, including excluded/QC-failed cells for review. Membership and QC still control analyses.',
        exact_cajal='Unquantized48-cell reference remains scoped; quantized CAJAL already covers all1088 cells.'), indent=2))
    return missing


def geometry(row):
    with np.load(OUT/'geometry'/f"{int(row['original_index']):04d}.npz") as data:
        return data['points'], data['axes']


def lite(missing, device):
    from shapeembed3d_core import ShapeEmbed3D, fps_indices, edm
    started = time.monotonic(); rows = missing.to_dict('records')
    for kind in ('invariant', 'cnn'):
        path = OUT/(f'lite_{kind}.npz')
        if path.exists():
            continue
        state = torch.load(ROOT/'results/shapeembed3d'/kind/'best.pt', map_location='cpu', weights_only=False)
        if state['protocol_sha256'] != sha256(ROOT/'results/shapeembed3d/protocol.json'):
            raise ValueError('Lite checkpoint protocol mismatch')
        model = ShapeEmbed3D(kind).to(device).eval(); model.load_state_dict(state['model'], strict=True)
        values = []
        with torch.no_grad():
            for first in range(0, len(rows), 16):
                batch = rows[first:first+16]; candidates = []
                for row in batch:
                    pool, _ = geometry(row)
                    candidates.append(pool[np.random.default_rng(20261022+4*row['inference_index']).choice(32768, 2048, replace=False)])
                candidates = np.stack(candidates); indices = fps_indices(candidates, 128)
                points = candidates[np.arange(len(batch))[:, None], indices]
                matrices = edm(torch.tensor(points, device=device))
                mean, _, _, _ = model.encode(matrices)
                values.extend(mean.cpu().numpy())
                if first % 80 == 0 or first+16 >= len(rows):
                    status('Lite3D '+kind, min(first+16, len(rows)), len(rows), started)
        save(path, ids=missing.cell_id.to_numpy(dtype=str), mean=np.asarray(values))
        del model
        if device == 'mps': torch.mps.empty_cache()


def point_mae(missing, device):
    from point_mae_reference import model, encode
    from run_point_mae_reference import draw
    rows = missing.to_dict('records'); started = time.monotonic()
    for arm in ('pretrained', 'adapted'):
        path = OUT/(f'point_mae_{arm}.npz')
        if path.exists(): continue
        m = model()
        if arm == 'adapted':
            state = torch.load(ROOT/'results/point_mae/best.pt', map_location='cpu', weights_only=False)
            if state['protocol_sha256'] != sha256(ROOT/'results/point_mae/protocol.json'):
                raise ValueError('Point-MAE checkpoint protocol mismatch')
            m.load_state_dict(state['model'], strict=True)
        m = m.to(device).eval(); native = []; averaged = []
        with torch.no_grad():
            for first in range(0, len(rows), 8):
                batch = rows[first:first+8]; clouds = []; axes = []
                for row in batch:
                    pool, basis = geometry(row); clouds.append(draw(pool, row['inference_index'])); axes.append(basis)
                clouds = np.stack(clouds); axes = np.stack(axes)
                native.extend(encode(m, torch.tensor(clouds, device=device)).cpu().numpy())
                poses = []
                for sign in SIGNS:
                    pose = (clouds@axes*sign).astype(np.float32)
                    poses.append(encode(m, torch.tensor(pose, device=device)).cpu().numpy())
                averaged.extend(np.mean(poses, axis=0))
                if first % 40 == 0 or first+8 >= len(rows):
                    status('Point-MAE '+arm, min(first+8, len(rows)), len(rows), started)
        save(path, ids=missing.cell_id.to_numpy(dtype=str), native=np.asarray(native), pose_average=np.asarray(averaged))
        del m
        if device == 'mps': torch.mps.empty_cache()


def initialize_allen(device):
    global _MODEL, _HEADS, _DEVICE
    from allen_reference_core import AllenModel
    torch.set_num_threads(2); _DEVICE = device
    if device == 'mps': torch.mps.set_per_process_memory_fraction(.45)
    _MODEL = AllenModel().to(device).eval()
    released = copy.deepcopy(_MODEL.encoder.final_conv[-1]).cpu().eval()
    adapted = copy.deepcopy(released)
    state = torch.load(ROOT/'results/allen_reference/head_decoder/best.pt', map_location='cpu', weights_only=False)
    if state['manifest_sha256'] != sha256(ROOT/'results/allen_reference/manifest.csv'):
        raise ValueError('Allen checkpoint cohort mismatch')
    adapted.load_state_dict({k.removeprefix('head.'): v for k, v in state['model'].items() if k.startswith('head.')}, strict=True)
    _HEADS = dict(pretrained=released, head_decoder=adapted.eval())


def allen_one(row):
    from allen_reference_core import chunked_encode
    path = OUT/'allen'/f"{row['original_index']:04d}.npz"
    if path.exists():
        with np.load(path) as data:
            if str(data['source_hash']) != row['source_mesh_sha256'] or not all(np.isfinite(data[k]).all() for k in _HEADS):
                raise ValueError('Stale/invalid Allen cache')
        return row['cell_id'], _DEVICE
    pool, _ = geometry(row); rng = np.random.default_rng(20261112+row['inference_index']*17)
    points = pool[rng.integers(len(pool), size=8192)].copy()
    with torch.no_grad():
        features = chunked_encode(_MODEL.encoder, torch.tensor(points[None], device=_DEVICE),
                                  chunk=512 if _DEVICE == 'cpu' else 256, return_features=True).cpu()
        values = {key: head(features).norm(dim=-1).numpy()[0].astype(float) for key, head in _HEADS.items()}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError('Nonfinite Allen output')
    save(path, **values, features=features.numpy()[0], source_hash=row['source_mesh_sha256'], device=_DEVICE)
    return row['cell_id'], _DEVICE


def allen(missing, device, cpu_workers=2):
    rows = missing.to_dict('records'); started = time.monotonic(); done = 0
    with ProcessPoolExecutor(max_workers=cpu_workers, initializer=initialize_allen, initargs=('cpu',)) as cpu:
        gpu = ProcessPoolExecutor(max_workers=1, initializer=initialize_allen, initargs=('mps',)) if device == 'mps' else None
        try:
            pending = deque(rows); futures = {}
            for executor in [cpu]*cpu_workers + ([gpu] if gpu else []):
                if pending: futures[executor.submit(allen_one, pending.popleft())] = executor
            while futures:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in ready:
                    executor = futures.pop(future); cid, worker = future.result(); done += 1
                    if done % 5 == 0 or done == len(rows):
                        status('Allen full8192', done, len(rows), started, last_cell=cid, device=worker)
                    if pending: futures[executor.submit(allen_one, pending.popleft())] = executor
        finally:
            if gpu: gpu.shutdown(wait=True, cancel_futures=True)
    values = {key: [] for key in ('pretrained', 'head_decoder')}
    for row in rows:
        with np.load(OUT/'allen'/f"{row['original_index']:04d}.npz") as data:
            for key in values: values[key].append(data[key])
    save(OUT/'allen_codes.npz', ids=missing.cell_id.to_numpy(dtype=str), **{k: np.asarray(v) for k, v in values.items()})


def assemble():
    from export_atlas_shape_benchmarks import benchmark_readouts
    from evaluate_spatial_benchmark import residual_apply
    verify_frozen()
    original_frame, specs, _ = benchmark_readouts()
    frame = pd.read_csv(OUT/'manifest.csv')
    missing = frame[frame.benchmark_index < 0].copy()
    ids = missing.cell_id.tolist(); raw = {}; fits = {}; artifacts = []

    def load(name):
        path = OUT/name; artifacts.append(name)
        values = np.load(path)
        if values['ids'].tolist() != ids:
            raise ValueError('Inference source identities differ: '+name)
        return values

    allen_values = load('allen_codes.npz')
    for arm in ('pretrained', 'head_decoder'):
        key = 'allen_'+arm
        raw[key] = allen_values[arm]
        fits[key] = ROOT/'results/allen_reference/evaluation'/(arm+'_readout_fit.npz')
    for arm in ('pretrained', 'adapted'):
        values = load('point_mae_'+arm+'.npz')
        for kind in ('native', 'pose_average'):
            key = 'point_mae_'+arm+'_'+kind; raw[key] = values[kind]
            fits[key] = ROOT/'results/point_mae'/(arm+'_'+kind+'_fit.npz')
    for kind in ('invariant', 'cnn'):
        key = 'shapeembed3d_'+kind; raw[key] = load('lite_'+kind+'.npz')['mean']
        fits[key] = ROOT/'results/shapeembed3d'/(kind+'_fit.npz')
    original_values = {meta['id']: values for meta, values in specs}
    lookup = {cid: i for i, cid in enumerate(frame.cell_id)}
    old_rows = [lookup[cid] for cid in original_frame.cell_id]; new_rows = [lookup[cid] for cid in ids]
    output = {}
    for base, values in raw.items():
        values = values.astype(original_values[base+'_raw'].dtype)
        fit = dict(np.load(fits[base]))
        adjusted = residual_apply(missing, (values-fit['center'])/fit['scale'], fit)
        for suffix, additional in [('raw', values), ('adjusted', adjusted)]:
            key = base+'_'+suffix; original = original_values[key]
            full = np.empty((len(frame), original.shape[1]), dtype=original.dtype)
            full[old_rows] = original; full[new_rows] = additional
            if not np.isfinite(full).all(): raise ValueError('Non-finite full embedding: '+key)
            np.testing.assert_array_equal(full[old_rows], original)
            output[key] = full
    save(OUT/'vectors.npz', ids=frame.cell_id.to_numpy(dtype=str), **output)
    artifacts += ['vectors.npz', 'manifest.csv', 'protocol.json', 'frozen_inputs.json']
    complete = dict(status='complete', cells=len(frame), existing=len(original_frame), additional=len(missing),
                    methods=sorted(output), artifacts={name:sha256(OUT/name) for name in artifacts},
                    preserved_benchmark=True, retrained=False)
    (OUT/'completed.json').write_text(json.dumps(complete, indent=2))
    print(f'Full frozen inference assembled: {len(output)} readouts × {len(frame)} cells.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'prepare', 'lite', 'point-mae', 'allen', 'assemble'], default='all')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--cpu-workers', type=int, choices=[1, 2, 3, 4], default=2)
    args = parser.parse_args(); torch.set_num_threads(4)
    missing = prepare()
    if args.stage in ('all', 'lite'): lite(missing, args.device)
    if args.stage in ('all', 'point-mae'): point_mae(missing, args.device)
    if args.stage in ('all', 'allen'): allen(missing, args.device, args.cpu_workers)
    if args.stage in ('all', 'assemble'): assemble()
    verify_frozen()
    print('Frozen inference stage complete: '+args.stage, flush=True)


if __name__ == '__main__': main()
