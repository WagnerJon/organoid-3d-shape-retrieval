"""Complete transferable representations for any exported Atlas dataset.

Frozen inference only; existing cell vectors, measurements and geometry are retained.
Each method family runs in a fresh process. Successful families are published
atomically, with backups and input checks, so cancellation can be resumed safely.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for key, value in {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '2',
                   'LOKY_MAX_CPU_COUNT': '2', 'PYTORCH_ENABLE_MPS_FALLBACK': '1'}.items():
    os.environ.setdefault(key, value)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')))
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def catalog():
    return json.loads((ROOT / 'config/atlas_representation_methods.json').read_text())


def families():
    specs = catalog()
    groups = {
        'spatial': ['spatial_robust', 'surface_signature', 'surface_signature_adjusted', 'spatial_adjusted',
                    'morph_controls', 'morph_size', 'morph_shape', 'morph_all', 'morph_adjusted'],
        'harmonics': ['spherical_harmonics', 'spherical_harmonics_adjusted'],
        'lite': [m['id'] for m in specs if m.get('family') == 'shapeembed3d'],
        'pointnet': ['pointnet_shape128'],
        'point-mae': [m['id'] for m in specs if m.get('family') == 'point_mae'],
        'allen': [m['id'] for m in specs if m.get('family') == 'allen'],
        'image': ['simclr3d', 'simclr3d_raw', 'simclr3d_raw_isolated', 'icone3d_raw_isolated', 'icone3d_raw_context'],
        'cajal': ['cajal_qgw', 'cajal_profile_adjusted'],
    }
    return groups


def requirements(group):
    paths = {
        'spatial': ['results/spatial_benchmark/aligned_pca.npz', 'results/spatial_benchmark/freeze.json',
                    'results/spatial_benchmark/dev_manifest.csv', 'results/pointnet_shape/data/manifest.csv',
                    'config/pointnet_shape.json', 'results/atlas_spatial/signature_encoder.npz',
                    'results/atlas_spatial/signature_nuisance.npz', 'results/atlas_spatial/spatial_nuisance.npz'],
        'harmonics': ['results/harmonics_cajal/harmonics_adjustment.npz'],
        'lite': ['results/shapeembed3d/protocol.json'] + [f'results/shapeembed3d/{k}/best.pt' for k in ('invariant', 'cnn')]
                + [f'results/shapeembed3d/{k}_fit.npz' for k in ('invariant', 'cnn')],
        'pointnet': ['results/pointnet_shape/baseline/best.pt', 'config/pointnet_shape.json'],
        'point-mae': ['.cache/point-mae-official/models/Point_MAE.py', 'results/point_mae/checkpoints/pretrain.pth',
                      'results/point_mae/best.pt', 'results/point_mae/protocol.json']
                     + [f'results/point_mae/{arm}_{kind}_fit.npz' for arm in ('pretrained', 'adapted') for kind in ('native', 'pose_average')],
        'allen': ['results/allen_reference/checkpoints/Rotation_invariant_pointcloud_SDF.ckpt',
                  'results/allen_reference/head_decoder/best.pt', 'results/allen_reference/manifest.csv',
                  '.cache/allen-cyto-dl/cyto_dl/nn/point_cloud/vnn.py',
                  '.cache/allen-cyto-dl/cyto_dl/nn/point_cloud/graph_functions.py',
                  '.cache/allen-cyto-dl/cyto_dl/nn/point_cloud/dgcnn.py',
                  '.cache/allen-benchmarking/src/pointcloudutils/networks/simple_inner.py']
                 + [f'results/allen_reference/evaluation/{arm}_readout_fit.npz' for arm in ('pretrained', 'head_decoder')],
        'image': [f'results/{k}/best.pt' for k in families()['image']],
        'cajal': ['.cache/cajal-env/bin/python', 'results/harmonics_cajal/surfaces.npz',
                  'results/harmonics_cajal/cajal_profile_adjustment.npz'],
    }
    return [ROOT / p for p in paths[group]]


def resolve(folder, path):
    path = Path(path)
    return path if path.is_absolute() else folder / path


def mesh_path(folder, cell):
    sample = str(cell['sample'])
    if sample in ('', '.', '..') or '/' in sample or '\\' in sample or int(cell['label']) <= 0:
        raise ValueError('Invalid cell mesh path')
    return folder / sample / f"{int(cell['label'])}.mesh"


def image_sources(data, folder):
    """Use original processing grids, never the downsampled viewer buffers."""
    sources = {}
    for sample in data['samples']:
        raw = resolve(folder, sample['sourceImage'])
        run = resolve(folder, sample['sourceLabels']).parent
        needed = [raw, run / 'run.json', run / 'processing.json', run / 'labels_processed.npy']
        if not all(p.is_file() for p in needed):
            raise ValueError(f"{sample['id']}: original raw image or processing-grid files are unavailable")
        config = json.loads((run / 'run.json').read_text())
        grid = json.loads((run / 'processing.json').read_text())
        if len(config.get('spacingZYXum', [])) != 3 or len(grid.get('effectiveSpacingZYXum', [])) != 3:
            raise ValueError(f"{sample['id']}: original physical calibration is unavailable")
        import numpy as np
        expected_offset = (np.array(grid['effectiveSpacingZYXum']) - np.array(config['spacingZYXum'])) / 2
        if not np.allclose(grid.get('rawGridOffsetZYXum', expected_offset), expected_offset):
            raise ValueError(f"{sample['id']}: cropped label grids are not supported by the frozen intensity encoders")
        # The established frozen preprocessing reads a memory-mappable TIFF.
        import tifffile
        try:
            image = tifffile.memmap(raw, mode='r')
            if image.ndim != 3:
                raise ValueError('Expected a single 3D image')
            del image
        except (ValueError, OSError) as exc:
            raise ValueError(f"{sample['id']}: frozen intensity inference requires an uncompressed 3D TIFF") from exc
        sources[sample['id']] = dict(raw=str(raw), run=str(run))
    return sources


def complete(data, method):
    import math
    spec = next((m for m in data['methods'] if m['id'] == method), None)
    if spec is None:
        return False
    for cell in data['cells']:
        vector = cell.get('representations', {}).get(method, {}).get('vector', [])
        if len(vector) != spec['dimensions'] or not vector or not all(math.isfinite(x) for x in vector):
            return False
    return True


def inspect(folder):
    folder = Path(folder).resolve()
    data = json.loads((folder / 'dataset.json').read_text())
    if not data.get('cells') or len({c['id'] for c in data['cells']}) != len(data['cells']):
        raise ValueError('The dataset needs a nonempty, unique cell inventory')
    missing_mesh = next((c['id'] for c in data['cells'] if not mesh_path(folder, c).is_file()), None)
    specs = {m['id']: m for m in catalog()}
    groups = []
    for group, keys in families().items():
        missing = [key for key in keys if not complete(data, key)]
        if not missing:
            continue
        unavailable = [p for p in requirements(group) if not p.is_file()]
        reason = f"Original surface mesh missing for {missing_mesh}" if missing_mesh else ''
        if unavailable:
            reason = 'Missing model or reference file: ' + str(unavailable[0].relative_to(ROOT))
        if not reason and group == 'image':
            try:
                image_sources(data, folder)
            except ValueError as exc:
                reason = str(exc)
        groups.append(dict(id=group, methods=missing, names=[specs[k]['name'] for k in missing],
                           available=not reason, reason=reason))
    return dict(dataset=str(folder), cells=len(data['cells']), organoids=len(data['organoids']),
                groups=groups, runnable=sum(len(g['methods']) for g in groups if g['available']),
                missing=sum(len(g['methods']) for g in groups),
                note='Frozen inference; no training. The original 48-cell CAJAL validation reference is not transferable.')


def event(stage, fraction, **extra):
    print('@methods-progress ' + json.dumps(dict(stage=stage, fraction=fraction, **extra)), flush=True)


def prepare_workspace(folder, data, groups):
    import numpy as np
    import pandas as pd
    from spatial_benchmark_core import CLASSICAL
    rows = []
    for index, cell in enumerate(data['cells']):
        rows.append(dict(cell_id=cell['id'], sample=cell['sample'], label=cell['label'], original_index=index,
                         inference_index=200000 + index, source_mesh_sha256=digest(mesh_path(folder, cell)), **cell['metrics']))
    # Keep the cache identity independent of which families are still missing.
    # Otherwise publishing one family would invalidate unfinished work on resume.
    files = {str(p.relative_to(ROOT)): digest(p) for g in families() for p in requirements(g) if p.is_file()}
    files.update({str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'scripts').glob('*.py')})
    reference = ROOT / 'results/pointnet_shape/data/manifest.csv'
    if reference.exists():
        files[str(reference.relative_to(ROOT))] = digest(reference)
    files['config/atlas_representation_methods.json'] = digest(ROOT / 'config/atlas_representation_methods.json')
    try:
        sources = image_sources(data, folder)
    except ValueError:
        sources = {}
        if 'image' in groups:
            raise
    raw_stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for source in sources.values()
                 for p in [Path(source['raw']), Path(source['run']) / 'labels_processed.npy',
                           Path(source['run']) / 'run.json', Path(source['run']) / 'processing.json']}
    identity = hashlib.sha256(json.dumps([rows, files, raw_stats], sort_keys=True).encode()).hexdigest()
    work = folder / '.atlas-methods' / identity[:20]
    atlas, out = work / 'atlas_dataset', work / 'representations'
    out.mkdir(parents=True, exist_ok=True)
    atlas.mkdir(exist_ok=True)
    save(atlas / 'dataset.json', data)
    save(work / 'sources.json', dict(files=files, rawStats=raw_stats, images=sources))
    for sample in {c['sample'] for c in data['cells']}:
        target = atlas / sample
        if not target.exists():
            target.symlink_to(folder / sample, target_is_directory=True)
    frame = pd.DataFrame(rows)
    development = ROOT / 'results/spatial_benchmark/dev_manifest.csv'
    medians = {}
    if reference.exists() and development.exists():
        original, splits = pd.read_csv(reference), pd.read_csv(development)
        training_ids = set(splits.loc[splits.split == 'train', 'cell_id'])
        medians = original.loc[original.cell_id.isin(training_ids), CLASSICAL].median()
    imputed = []
    for key in CLASSICAL:
        if key not in frame:
            frame[key] = np.nan
        imputed.extend(dict(cell=str(cid), feature=key, value=float(medians[key]))
                       for cid in frame.loc[frame[key].isna(), 'cell_id'] if key in medians)
        frame[key] = frame[key].fillna(medians.get(key, np.nan))
    frame.to_csv(out / 'manifest.csv', index=False)
    save(work / 'comparison-inputs.json', dict(imputed=imputed,
         rule='Frozen original training-data medians in descriptor inputs only; physical measurements are unchanged.'))
    return work


def stage(work, group, device):
    import numpy as np
    import pandas as pd
    import infer_joint_organoid_methods as engine
    from pointnet_shape_core import sample_surface
    from prepare_pointnet_shape import read_mesh
    from spatial_benchmark_core import surface_frame
    work = Path(work)
    engine.JOINT, engine.ATLAS, engine.OUT, engine.DEVICE = work, work / 'atlas_dataset', work / 'representations', device
    frame = pd.read_csv(engine.OUT / 'manifest.csv')
    if group in ('lite', 'point-mae', 'allen'):
        (engine.OUT / 'geometry').mkdir(exist_ok=True)
        (engine.OUT / 'allen').mkdir(exist_ok=True)
        for index, row in enumerate(frame.to_dict('records')):
            path = engine.OUT / 'geometry' / f'{index:04d}.npz'
            if path.exists():
                continue
            v, f = read_mesh(engine.ATLAS / row['sample'] / f"{row['label']}.mesh")
            v = v.astype(float)
            center, scale = (v.min(0) + v.max(0)) / 2, np.ptp(v, axis=0).max()
            if scale <= 0:
                raise ValueError('Degenerate surface: ' + row['cell_id'])
            points = sample_surface((v - center) / scale, f, 32768, np.random.default_rng(20261110 + row['inference_index']))
            _, axes, _, gap = surface_frame(v, f)
            engine.npz(path, points=points, axes=axes, gap=gap, center=center, scale=scale, source_hash=row['source_mesh_sha256'])
            engine.progress('Preparing surfaces', index + 1, len(frame))
    if group == 'spatial':
        from encode_standalone_atlas import main
        main(engine.ATLAS)
    elif group == 'harmonics':
        engine.harmonic_codes(frame)
    elif group == 'cajal':
        engine.harmonic_codes(frame)
        engine.cajal_codes(frame)
    elif group == 'pointnet':
        engine.pointnet(frame)
    elif group == 'image':
        engine.image_inputs(frame, json.loads((work / 'sources.json').read_text())['images'])
        engine.image_codes(frame)
    else:
        engine.learned(frame, group)


def values_for(work, group, keys):
    import numpy as np
    import pandas as pd
    from encode_standalone_atlas import residual
    out = work / 'representations'
    frame = pd.read_csv(out / 'manifest.csv')
    def load(name):
        with np.load(out / name) as a:
            if a['ids'].tolist() != frame.cell_id.tolist():
                raise ValueError('Cached cell order changed')
            return {k: a[k].copy() for k in a.files}
    result, reconstructions = {}, None
    if group == 'spatial':
        staged = json.loads((work / 'atlas_dataset/dataset.json').read_text())
        for key in keys:
            result[key] = np.array([c['representations'][key]['vector'] for c in staged['cells']])
    elif group == 'harmonics':
        raw = load('surfaces.npz')['harmonics']
        result = dict(spherical_harmonics=raw, spherical_harmonics_adjusted=residual(frame, raw, np.load(ROOT / 'results/harmonics_cajal/harmonics_adjustment.npz')))
    elif group == 'cajal':
        a = load('cajal_codes.npz')
        result = dict(cajal_qgw=np.c_[np.arange(len(frame)), a['distances']],
                      cajal_profile_adjusted=residual(frame, a['profile'], np.load(ROOT / 'results/harmonics_cajal/cajal_profile_adjustment.npz')))
    elif group == 'pointnet':
        a = load('pointnet_codes.npz')
        result['pointnet_shape128'], reconstructions = a['codes'], a['paths']
    elif group == 'image':
        result = {key: load(f'{key}_codes.npz')['codes'] for key in keys}
    else:
        arms = {}
        if group == 'lite':
            for kind in ('invariant', 'cnn'):
                arms[f'shapeembed3d_{kind}'] = (load(f'lite_{kind}.npz')['mean'], ROOT / f'results/shapeembed3d/{kind}_fit.npz')
        elif group == 'allen':
            a = load('allen_codes.npz')
            for kind in ('pretrained', 'head_decoder'):
                arms[f'allen_{kind}'] = (a[kind], ROOT / f'results/allen_reference/evaluation/{kind}_readout_fit.npz')
        else:
            for arm in ('pretrained', 'adapted'):
                a = load(f'point_mae_{arm}.npz')
                for kind in ('native', 'pose_average'):
                    arms[f'point_mae_{arm}_{kind}'] = (a[kind], ROOT / f'results/point_mae/{arm}_{kind}_fit.npz')
        for base, (raw, path) in arms.items():
            fit = dict(np.load(path))
            result[base + '_raw'] = raw
            result[base + '_adjusted'] = residual(frame, (raw - fit['center']) / fit['scale'], fit)
    return {key: result[key] for key in keys}, reconstructions


def publish(folder, work, group, keys, expected):
    import numpy as np
    from organoid_membership import project
    path = folder / 'dataset.json'
    if digest(path) != expected:
        raise RuntimeError('Dataset changed while calculating. Reopen it and retry; cached work is retained.')
    data = json.loads(path.read_text())
    values, reconstructions = values_for(work, group, keys)
    specs = {m['id']: m for m in catalog()}
    for key, vectors in values.items():
        if vectors.ndim != 2 or len(vectors) != len(data['cells']) or not np.isfinite(vectors).all():
            raise ValueError('Invalid representation: ' + key)
        existing = next((m for m in data['methods'] if m['id'] == key), None)
        spec = copy.deepcopy(existing or specs[key])
        if existing and existing['dimensions'] != vectors.shape[1]:
            raise ValueError('Existing method dimensions differ: ' + key)
        spec.update(dimensions=vectors.shape[1], coverage='all_cells')
        if not existing:
            spec['description'] = spec.get('description', spec['name']) + ' Frozen inference on this dataset; no retraining or biological validation.'
        # Precomputed vectors refer to the entire ordered cohort and cannot be
        # spliced from a different subset. Reject rather than corrupt distances.
        if spec['metric'] == 'precomputed' and existing:
            for i, cell in enumerate(data['cells']):
                old = cell.get('representations', {}).get(key)
                if old and not np.allclose(old['vector'], vectors[i], rtol=1e-6, atol=1e-9):
                    raise ValueError('Existing CAJAL distances use a different cohort; retained without replacement')
        for index, cell in enumerate(data['cells']):
            reps = cell.setdefault('representations', {})
            old = reps.get(key)
            if old is None or len(old.get('vector', [])) != vectors.shape[1] or not np.isfinite(old['vector']).all():
                reps[key] = dict(vector=vectors[index].tolist(), point=[0., 0.], cluster=0)
                if key == 'pointnet_shape128' and reconstructions is not None:
                    # Reconstruction assets remain inside this dataset's cache.
                    relative = (work / 'atlas_dataset' / str(reconstructions[index])).relative_to(folder)
                    cell.setdefault('pointReconstructions', {})[key] = str(relative)
        included = [c for c in data['cells'] if c.get('inMainOrganoid', True)]
        spec['variance'] = project(included, key, spec['metric'])
        eligible = spec.get('scope') != 'cells' and spec['metric'] not in ('precomputed', 'shape_hybrid95')
        for organoid in data['organoids']:
            reps = organoid.setdefault('representations', {})
            cells = [c for c in included if c['sample'] == organoid['sample'] and c['qc']]
            if eligible and cells:
                vector = np.mean([c['representations'][key]['vector'] for c in cells], axis=0)
                if spec['metric'] == 'cosine':
                    vector /= max(np.linalg.norm(vector), 1e-12)
                reps[key] = dict(vector=vector.tolist(), point=[0., 0.], cluster=0)
            else:
                reps.pop(key, None)
        spec['stackVariance'] = project([s for s in data['organoids'] if key in s['representations']], key) if eligible else [0., 0.]
        data['methods'] = [spec if m['id'] == key else m for m in data['methods']] if existing else data['methods'] + [spec]
    sources = json.loads((work / 'sources.json').read_text())
    for name, value in sources['files'].items():
        if digest(ROOT / name) != value:
            raise RuntimeError('Frozen inference source changed: ' + name)
    import pandas as pd
    for cell, row in zip(data['cells'], pd.read_csv(work / 'representations/manifest.csv').to_dict('records')):
        if cell['id'] != row['cell_id'] or digest(mesh_path(folder, cell)) != row['source_mesh_sha256']:
            raise RuntimeError('Source cell geometry changed during calculation')
    for name, stat in sources['rawStats'].items():
        p = Path(name)
        if [p.stat().st_size, p.stat().st_mtime_ns] != stat:
            raise RuntimeError('Source image or labels changed during calculation')
    data.setdefault('methodCompletion', []).append(dict(group=group, methods=keys, cache=str(work.relative_to(folder)),
                                                       retrained=False, sources='sources.json', comparisonInputs='comparison-inputs.json', time=time.time()))
    backup = folder / '.atlas-methods' / f'dataset.before-{expected[:16]}.json'
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    if digest(path) != expected:
        raise RuntimeError('Dataset changed before publication')
    save(path, data)
    return digest(path)


def run(folder, output, only=None, device='auto'):
    folder = Path(folder).resolve()
    output = Path(output)
    plan = inspect(folder)
    groups = [g for g in plan['groups'] if g['available'] and (not only or g['id'] in only)]
    result = dict(plan, completed=[], failed=[], status='complete')
    save(output, dict(result, status='running'))
    if not groups:
        save(output, result)
        event('No runnable methods are missing', 1)
        return result
    # An OS advisory lock is released even if Atlas or Python is interrupted.
    import fcntl
    lock = folder / '.atlas-methods/processing.lock'
    lock.parent.mkdir(exist_ok=True)
    with lock.open('w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Missing methods are already being calculated for this dataset') from exc
        expected = digest(folder / 'dataset.json')
        data = json.loads((folder / 'dataset.json').read_text())
        event('Checking surfaces and frozen models', 0)
        work = prepare_workspace(folder, data, [g['id'] for g in groups])
        if device == 'auto':
            import torch
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        current = None
        def cancel(signum, frame):
            if current is not None and current.poll() is None:
                os.killpg(current.pid, signal.SIGTERM)
                try:
                    current.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(current.pid, signal.SIGKILL)
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, cancel)
        signal.signal(signal.SIGINT, cancel)
        try:
            for index, group in enumerate(groups):
                label = ', '.join(group['names'])
                event(label, index / len(groups))
                python = ROOT / '.cache/cajal-env/bin/python' if group['id'] == 'cajal' else Path(sys.executable)
                log_path = work / f"{group['id']}.log"
                with log_path.open('w') as log:
                    current = subprocess.Popen([str(python), '-u', str(Path(__file__).resolve()), 'stage',
                                                '--dataset', str(work), '--group', group['id'], '--device', device],
                                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                    for line in current.stdout:
                        log.write(line); log.flush()
                        print(line, end='', flush=True)
                        try:
                            value = json.loads(line)
                            fraction = min(.95, value['completed'] / max(1, value['total']))
                            event(value['stage'], (index + fraction) / len(groups))
                        except (ValueError, KeyError, TypeError):
                            pass
                    code = current.wait()
                    current.stdout.close()
                if code:
                    result['failed'].append(dict(group=group['id'], reason=str(log_path), details=log_path.read_text()[-1800:]))
                    continue
                # Publication conflicts are fatal; never continue against stale data.
                expected = publish(folder, work, group['id'], group['methods'], expected)
                result['completed'].extend(group['methods'])
                save(output, dict(result, status='running'))
            result['status'] = 'partial' if result['failed'] else 'complete'
            save(output, result)
            event('Calculation finished', 1)
            return result
        finally:
            if current is not None and current.poll() is None:
                os.killpg(current.pid, signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['inspect', 'run', 'stage'])
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--group', choices=list(families()))
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    args = parser.parse_args()
    if args.operation == 'stage':
        stage(args.dataset, args.group, args.device)
    elif args.operation == 'inspect':
        save(args.output, inspect(args.dataset))
    else:
        run(args.dataset, args.output, [args.group] if args.group else None, args.device)


if __name__ == '__main__':
    main()
