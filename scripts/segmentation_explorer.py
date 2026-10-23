"""Image-only segmentation jobs and native-grid comparisons for Organoid Atlas.

Run manifests are published atomically only after all artifacts are complete.
Preview volumes never enter either prediction pipeline or agreement statistics.
"""
import argparse
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('PANSEG_HOME', str(ROOT / '.cache/panseg'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache/numba'))
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(ROOT / 'vendor/panseg'))
import numpy as np
import tifffile
from skimage.transform import resize
from skimage import measure
from scipy.optimize import linear_sum_assignment
from organoid_membership import classify, included_labels


def progress(stage, fraction, span=0):
    """Structured stage progress; inference spans can be refined by completed units."""
    print('\n@atlas-progress ' + json.dumps(dict(stage=stage, fraction=fraction, span=span)), flush=True)


class PansegProgress:
    def __init__(self):
        self.total = 0
        self._progress = 0

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value):
        self._progress = value
        # PanSeg increments before each patch; only preceding patches are finished.
        completed = max(0, value - 1)
        progress(f'PanSeg: boundary inference · {completed}/{self.total} patches',
                 .1 + .6 * completed / max(self.total, 1))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_raw(path):
    with tifffile.TiffFile(path) as tiff:
        axes = tiff.series[0].axes
        if any(a in axes for a in 'CST'):
            raise ValueError('Channel, sample/color and time axes are not supported. Select a single-channel ZYX stack.')
        raw = tiff.asarray()
    if raw.ndim != 3 or min(raw.shape) < 2 or not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise ValueError('Select a finite, single-channel 3D TIFF in ZYX order (no time/channel axes).')
    return raw


def check_spacing(spacing):
    if len(spacing) != 3 or not all(np.isfinite(v) and v > 0 for v in spacing):
        raise ValueError('Voxel spacing Z, Y and X must be positive micrometers.')


def prepare(path, spacing, output):
    check_spacing(spacing)
    raw = read_raw(path)
    digest = fingerprint(path)
    shape = np.maximum(2, np.rint(np.array(raw.shape) * min(1, 144 / max(raw.shape)))).astype(int)
    small = resize(raw.astype(np.float32), tuple(shape), preserve_range=True, anti_aliasing=True)
    low, high = np.percentile(raw, [1, 99.8])
    normalized = np.clip((small - low) / max(high - low, 1e-6), 0, 1)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'raw.u8').write_bytes((normalized * 255).astype(np.uint8).tobytes())
    meta = dict(path=str(Path(path).resolve()), fingerprint=digest, shape=shape.tolist(), nativeShape=list(raw.shape),
                spacing=spacing, low=float(low), high=float(high), dtype=str(raw.dtype), folder=str(output.resolve()))
    save(output / 'source.json', meta)
    if output.parent.name == 'sources':
        annotate_history(output.parent.parent / 'runs', meta)
    return meta


def verify_source(source):
    check_spacing(source['spacing'])
    if fingerprint(source['path']) != source['fingerprint']:
        raise ValueError('The raw stack has changed. Reload it before running or comparing.')
    raw = read_raw(source['path'])
    if list(raw.shape) != source['nativeShape']:
        raise ValueError('Raw dimensions changed.')
    return raw


def annotate_history(folder, source):
    """Upgrade saved run membership when opening its raw stack; prediction files stay intact."""
    for path in sorted(folder.glob('*/result.json')):
        result = json.loads(path.read_text())
        if result.get('source') != source['fingerprint'] or result.get('spacing') != source['spacing']:
            continue
        if all('inMainOrganoid' in c for c in result['cells']):
            continue
        native = path.parent / 'labels_native.tif'
        if fingerprint(native) != result['labelHash']:
            raise ValueError('Saved labels changed; re-import the run before membership classification.')
        membership = classify(tifffile.imread(native))
        for cell in result['cells']:
            cell.update(membership[cell['label']])
        save(path, result)


def publish(labels, source, folder, method, settings, seconds, provenance, extras=None):
    if list(labels.shape) != source['nativeShape']:
        raise ValueError('Labels must match the original raw grid.')
    if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() > np.iinfo(np.uint32).max:
        raise ValueError('Labels must be nonnegative uint32 instances, with background 0.')
    labels = labels.astype(np.uint32)
    tifffile.imwrite(folder / 'labels_native.tif', labels, photometric='minisblack', metadata={'axes': 'ZYX', 'spacing_zyx_um': source['spacing']})
    small = resize(labels, source['shape'], order=0, preserve_range=True, anti_aliasing=False).astype('<u4')
    (folder / 'labels.u32').write_bytes(small.tobytes())
    membership = classify(labels)
    cells = []
    # Compact ids before regionprops to avoid allocating by potentially sparse imported ids.
    ids, compact = np.unique(labels, return_inverse=True)
    if 0 not in ids:
        compact += 1
        ids = np.r_[0, ids]
    for r in measure.regionprops(compact.reshape(labels.shape).astype(np.int32)):
        cells.append(dict(label=int(ids[r.label]), volume=float(r.area * np.prod(source['spacing'])),
                          **membership[int(ids[r.label])], centroid=list(r.centroid), border=any(a == 0 for a in r.bbox[:3]) or any(b == n for b, n in zip(r.bbox[3:], labels.shape))))
    result = dict(id=folder.name, method=method, source=source['fingerprint'], shape=source['nativeShape'],
                  spacing=source['spacing'], folder=str(folder.resolve()), cells=cells, settings=settings,
                  seconds=seconds, provenance=provenance, created=time.strftime('%Y-%m-%d %H:%M:%S'), labelHash=fingerprint(folder / 'labels_native.tif'))
    result.update(extras or {})
    save(folder / 'result.json', result)
    return result


def run(source, method, settings, folder, reuse=None, purpose=None, comparison_side=None):
    progress("Verifying raw stack", .01)
    raw = verify_source(source)
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / 'request.json', dict(source=source, method=method, settings=settings))
    start = time.time()
    runtime = {}
    for package in ('torch','numpy','cellpose','dinov3','u-Segment3D','python-elf','nifty','vigra',
                    'scipy','scikit-image','tifffile','zarr','PyYAML','imagecodecs'):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            pass
    settings = dict(settings, runtime=runtime, workerSHA256=fingerprint(__file__))
    if method == 'cellpose':
        from reconstruct import predict
        model = settings.get('model', 'cpsam_v2')
        from atlas_cellpose_models import MODEL_NAMES, model_metadata, weights_path, local_fields
        if model not in MODEL_NAMES:
            raise ValueError('Cellpose model must be one of '+', '.join(MODEL_NAMES)+'.')
        settings['model'] = model
        settings.update(local_fields(settings,freeze=True))
        config = dict(spacing_zyx_um=source['spacing'], scale=float(settings['scale']), model=model,
                      diameter=float(settings['diameter']), gpu=bool(settings['gpu']), background_correction=bool(settings['background']), gradient_decay=.05)
        from atlas_cellpose_3d import settings as native_settings
        config.update(native_settings(settings))
        config.update(local_fields(settings))
        if not 0.05 <= config['scale'] <= 1 or not 2 <= config['diameter'] <= 500:
            raise ValueError('Cellpose scale must be 0.05–1 and diameter 2–500 processed pixels.')
        _, labels, _ = predict(Path(source['path']), config, folder, progress=progress)
        weights = weights_path(model,settings)
        if weights.is_file(): settings['checkpointSHA256'] = fingerprint(weights)
        settings['modelRuntime'] = model_metadata(model,settings)
        provenance = f'Cellpose {model}, '+('native do_3D=True; MPS prediction with CPU 3D masks when using Apple GPU' if config['segmentationMode']=='cellpose3d' else 'XY/XZ/YZ inference + u-Segment3D aggregation')+'. Image only; no reference masks.'
    elif method == 'panseg':
        from panseg_explorer import run_panseg
        labels, settings, provenance, prediction = run_panseg(raw, source, settings, folder, progress, PansegProgress, fingerprint, reuse=reuse)
    elif method == 'focus3d':
        from focus3d_explorer import run_focus3d
        labels, settings, provenance = run_focus3d(raw, source, settings, folder, progress, fingerprint)
    else:
        raise ValueError('Unknown method')
    progress("Verifying source and preparing final outputs", .92)
    verify_source(source)
    progress("Writing native labels, previews and cell measurements", .95)
    extras = {'purpose':purpose}
    if method == 'cellpose': extras.update(cellposeModel=model, cellposeMode=config['segmentationMode'], comparisonSide=comparison_side)
    if method == 'panseg': extras['prediction'] = prediction
    return publish(labels, source, folder, method, settings, time.time()-start, provenance, extras=extras)


def compare(a, b, source):
    if a['source'] != b['source'] or a['source'] != source['fingerprint'] or a['shape'] != b['shape'] or a['spacing'] != b['spacing'] or a['spacing'] != source['spacing']:
        raise ValueError('Compare runs from the same raw stack, voxel grid and calibration.')
    for item in (a, b):
        if fingerprint(Path(item['folder']) / 'labels_native.tif') != item['labelHash']:
            raise ValueError('A saved label file changed. Re-import it before comparison.')
    x = tifffile.imread(Path(a['folder']) / 'labels_native.tif'); y = tifffile.imread(Path(b['folder']) / 'labels_native.tif')
    if list(x.shape) != source['nativeShape'] or x.shape != y.shape:
        raise ValueError('Label grids differ.')
    included_x, included_y = included_labels(x), included_labels(y)
    result = agreement(included_x, included_y)
    result['excludedA'] = sorted(set(np.unique(x)) - set(np.unique(included_x)) - {0})
    result['excludedB'] = sorted(set(np.unique(y)) - set(np.unique(included_y)) - {0})
    result['excludedA'] = [int(v) for v in result['excludedA']]
    result['excludedB'] = [int(v) for v in result['excludedB']]
    result['note'] += ' Detached cells excluded using each run’s largest connected tissue component (not an accuracy assessment).'
    return result


def refine_run(source, parent, settings, folder):
    from membrane_refinement import refine
    progress('Refinement: verifying source and parent labels',.01)
    raw = verify_source(source)
    if parent['source'] != source['fingerprint'] or parent['shape'] != source['nativeShape'] or parent['spacing'] != source['spacing']:
        raise ValueError('Refine a saved run from the same stack and calibration.')
    if parent['method'] == 'refined': raise ValueError('Select an original segmentation to refine, not an already refined run.')
    path = Path(parent['folder'])/'labels_native.tif'
    if fingerprint(path) != parent['labelHash']: raise ValueError('Parent labels changed; re-import them first.')
    labels = tifffile.imread(path)
    folder.mkdir(parents=True,exist_ok=False)
    save(folder/'parent.json',parent)
    save(folder/'request.json',dict(source=source,parent=parent,settings=settings))
    started = time.time()
    output,opts,stats,rendered,slices = refine(raw,labels,source['spacing'],settings,progress)
    verify_source(source)
    if fingerprint(path) != parent['labelHash']: raise ValueError('Parent labels changed while refinement was running.')
    metadata = dict(parentID=parent['id'],parentFolder=parent['folder'],parentMethod=parent['method'],**stats)
    # Rendered prediction is a fitting diagnostic, not acquired microscopy.
    tifffile.imwrite(folder/'fitted_membrane_roi.tif',rendered.astype(np.float32),photometric='minisblack',metadata={'axes':'ZYX','spacing_zyx_um':source['spacing'],'roi_start_zyx':stats['roiStart'],'description':'Continuous optical fit before discretization and safety guards; not acquired microscopy.'})
    save(folder/'refinement.json',metadata)
    save(folder/'optimization.json',stats['history'])
    progress('Refinement: writing original-grid labels and measurements',.93)
    settings = dict(opts,workerSHA256=fingerprint(__file__),adapterSHA256=fingerprint(ROOT/'scripts/membrane_refinement.py'),runtime={p:version(p) for p in ('torch','numpy','scipy')})
    result = publish(output,source,folder,'refined',settings,time.time()-started,
                     'Shape-preserving spatial warp (v2), bounded displacement/Jacobian, PSF-rendered membrane fitting, shape/volume priors, topology and roughness guards. Original native planes only; fixed instance identities.',
                     extras=dict(purpose='refinement',refinement=metadata,cellposeModel=parent.get('cellposeModel'),comparisonSide='B'))
    return result


def agreement(x, y):
    if x.shape != y.shape:
        raise ValueError('Label grids differ.')
    ax, ix = np.unique(x, return_inverse=True); ay, iy = np.unique(y, return_inverse=True)
    if len(ax) * len(ay) > 25000000:
        raise ValueError('Too many instances for interactive dense matching (>25M pairs).')
    counts = np.bincount(ix * len(ay) + iy, minlength=len(ax)*len(ay)).reshape(len(ax), len(ay))
    inter = counts[ax != 0][:, ay != 0]
    sizesx = counts.sum(1)[ax != 0]; sizesy = counts.sum(0)[ay != 0]
    iou = inter / np.maximum(sizesx[:, None] + sizesy[None, :] - inter, 1)
    rows, cols = linear_sum_assignment(-(iou >= .5).astype(float) - iou / (2 * max(min(iou.shape), 1)))
    pairs = [dict(a=int(ax[ax != 0][r]), b=int(ay[ay != 0][c]), iou=float(iou[r,c])) for r,c in zip(rows,cols) if iou[r,c] > 0]
    valid = [p for p in pairs if p['iou'] >= .5]
    fgx=x>0; fgy=y>0; denom=int(fgx.sum()+fgy.sum())
    return dict(dice=2*int((fgx & fgy).sum())/denom if denom else 1., matched=len(valid), countA=len(sizesx), countB=len(sizesy),
                meanIoU=float(np.mean([p['iou'] for p in valid])) if valid else 0., pairs=pairs,
                note='Method agreement, not accuracy. One-to-one matches at IoU ≥ 0.5; computed on the full native grid.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare','run','import','compare','refine'])
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); request=json.loads(args.request.read_text())
    try:
        os.setsid()  # permit the app to cancel worker descendants as one process group
    except OSError:
        pass
    if args.operation == 'prepare':
        result=prepare(request['path'], request['spacing'], args.output.parent)
    elif args.operation == 'run':
        result=run(request['source'], request['method'], request['settings'], args.output.parent, reuse=request.get('reuse'), purpose=request.get('purpose'), comparison_side=request.get('comparisonSide'))
    elif args.operation == 'refine':
        result=refine_run(request['source'],request['parent'],request['settings'],args.output.parent)
    elif args.operation == 'import':
        source=request['source']; verify_source(source)
        labels=tifffile.imread(request['labels'])
        if list(labels.shape) != source['nativeShape']:
            raise ValueError('Imported labels must match the native raw grid. Select labels_native.tif.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        metadata_path=next((Path(request['labels']).parent/name for name in ['settings.json','run.json'] if (Path(request['labels']).parent/name).is_file()),None)
        original = json.loads(metadata_path.read_text()) if metadata_path else None
        model = (original.get('config', original).get('model', 'unknown') if original else 'unknown')
        result=publish(labels, source, args.output.parent, request['method'], {'importedFrom':request['labels'], 'originalSettings':original}, 0., 'Imported existing native-grid labels; original settings remain alongside their source file.', extras={'cellposeModel':model if request['method'] == 'cellpose' else None, 'cellposeMode':original.get('config',original).get('segmentationMode','usegment3d') if original and request['method']=='cellpose' else None, 'comparisonSide':request.get('comparisonSide')})
    else:
        verify_source(request['source']); result=compare(request['a'],request['b'],request['source'])
    save(args.output,result)
    progress("Complete", 1.)
    print('Finished: '+str(args.output),flush=True)


if __name__ == '__main__':
    main()
