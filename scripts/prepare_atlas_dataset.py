"""Prepare an Atlas dataset from raw TIFF, NumPy or OME-Zarr stacks; resumable.

The request is immutable once a job starts. Raw files are referenced, never edited.
Only the final atomic dataset.json makes the completed experiment visible to Atlas.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for key, value in {'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'2', 'LOKY_MAX_CPU_COUNT':'2',
                   'MPLBACKEND':'Agg', 'PYTORCH_ENABLE_MPS_FALLBACK':'1'}.items():
    os.environ.setdefault(key, value)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2))
    temporary.replace(path)


def event(stage, fraction):
    print('@dataset-progress ' + json.dumps(dict(stage=stage, fraction=max(0, min(1, fraction)))), flush=True)


def inspect(path):
    import tifffile
    path = Path(path).resolve()
    if path.suffix.lower() == '.npy':
        from atlas_numpy import inspect as inspect_numpy
        return inspect_numpy(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if len(tif.series) != 1 or series.axes not in ('ZYX', 'QYX', 'IYX') or len(series.shape) != 3 or min(series.shape) < 2:
            raise ValueError(f'{path.name}: select a single-channel 3D Z/Y/X TIFF; time, channels and multiple series are not supported.')
        if len(tif.pages) != series.shape[0]:
            raise ValueError(f'{path.name}: TIFF must contain one page per Z plane.')
        spacing, source = None, 'Enter voxel spacing in micrometers'
        if tif.ome_metadata:
            import xml.etree.ElementTree as ET
            pixels = next((e for e in ET.fromstring(tif.ome_metadata).iter() if e.tag.split('}')[-1] == 'Pixels'), None)
            if pixels is not None:
                units = {'µm':1., 'μm':1., 'um':1., 'nm':.001, 'mm':1000.}
                try:
                    spacing = [float(pixels.attrib['PhysicalSize'+axis])*units[pixels.attrib.get('PhysicalSize'+axis+'Unit', 'µm')] for axis in 'ZYX']
                    source = 'OME physical voxel sizes'
                except (KeyError, ValueError):
                    spacing = None
        if spacing is None:
            metadata = tif.imagej_metadata or {}
            if metadata.get('unit') in ('micron', 'microns', 'um', 'µm', 'μm') and 'spacing' in metadata:
                try:
                    def inverse(key):
                        a, b = tif.pages[0].tags[key].value
                        return b/a
                    spacing = [float(metadata['spacing']), inverse('YResolution'), inverse('XResolution')]
                    source = 'ImageJ spacing and resolution tags'
                except (KeyError, ValueError, ZeroDivisionError):
                    spacing = None
        if spacing is not None and not all(math.isfinite(v) and v > 0 for v in spacing):
            spacing = None
        if spacing is None:
            spacing = [1.,1.,1.]
            source = 'No calibration metadata; default 1 × 1 × 1 µm (editable)'
        stat = path.stat()
        return dict(path=str(path), name=path.name, shape=list(series.shape), axes=series.axes,
                    dtype=str(series.dtype), spacing=spacing, calibrationSource=source,
                    sourceStat=[stat.st_size, stat.st_mtime_ns])


def validate(request):
    if not request.get('title', '').strip() or not request.get('inputs'):
        raise ValueError('Enter a dataset name and select at least one stack.')
    execution=request.get('execution') or {}
    if not execution.get('useLocal',True) and not execution.get('workers'):
        raise ValueError('Select this Mac or at least one paired worker.')
    if execution.get('splitStacks'):
        size=execution.get('planesPerBatch',16)
        if type(size) is not int or not 1<=size<=128:raise ValueError('Planes per batch must be between 1 and 128')
    settings = request['settings']
    from atlas_streamed_linking import validate_mode
    validate_mode(request.get('linkingMode') or settings.get('linkingMode'))
    if settings['model'] not in ('cpsam_v2', 'cpdino', 'cpdino-vitb'):
        raise ValueError('Unsupported Cellpose model.')
    if not .05 <= float(settings['scale']) <= 1 or not 2 <= float(settings['diameter']) <= 500:
        raise ValueError('Scale must be 0.05–1 and diameter 2–500 processed pixels.')
    paths = set();zarr_cache={}
    for entry in request['inputs']:
        if entry.get('omeZarr'):
            from atlas_ome_zarr import inspect_entry
            metadata=inspect_entry(entry,zarr_cache)
            key=(metadata['path'],entry['omeZarr']['time'],entry['omeZarr']['channel'],entry['omeZarr']['level'])
        else:
            metadata=inspect(entry['path']);key=metadata['path']
        if key in paths:raise ValueError('A stack or OME-Zarr T/C/level selection was selected more than once.')
        paths.add(key)
        spacing = entry.get('spacing') or []
        if len(spacing) != 3 or not all(math.isfinite(v) and v > 0 for v in spacing):
            raise ValueError(f"Confirm positive Z/Y/X spacing for {metadata['name']}.")
        if entry.get('shape') != metadata['shape']:raise ValueError('Input dimensions changed since selection.')
        if entry.get('sourceStat') != metadata['sourceStat']:
            raise ValueError(f"{metadata['name']} changed since it was selected. Add the stack again.")
        if entry.get('labels'):
            label_stat = Path(entry['labels']).stat()
            expected = entry.get('labelStat') or []
            if len(expected) != 2 or expected[0] != label_stat.st_size or abs(expected[1]-label_stat.st_mtime_ns) > 1000:
                raise ValueError('Imported labels changed since selection. Select the label stack again.')
            if Path(entry['labels']).suffix.lower() == '.npy':
                from atlas_numpy import validate_labels
                validate_labels(entry['labels'], metadata['shape'])
            else:
                import tifffile
                import numpy as np
                with tifffile.TiffFile(entry['labels']) as tif:
                    if tuple(tif.series[0].shape) != tuple(metadata['shape']) or not np.issubdtype(tif.series[0].dtype, np.integer):
                        raise ValueError('Imported labels must be integer instances on the same Z/Y/X grid as the raw stack.')


def call(arguments, log, progress=None):
    with Path(log).open('a') as handle:
        process = subprocess.Popen([str(a) for a in arguments], cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                handle.write(line); handle.flush()
                print(line, end='', flush=True)
                if progress:
                    progress(line)
            if process.wait() != 0:
                raise RuntimeError(f'Processing failed. See {log}')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate(); process.wait()


def imported(entry, run, settings):
    import numpy as np
    import tifffile
    from run_standalone_cellpose import export
    run.mkdir(parents=True, exist_ok=True)
    labels = tifffile.imread(entry['labels'])
    if np.any(labels < 0) or labels.max() > 65535:
        raise ValueError('Imported labels must be nonnegative IDs up to 65535; background is zero.')
    np.save(run/'labels_processed.npy', labels.astype(np.uint32))
    from run_standalone_cellpose import digest
    cfg = dict(source=entry['path'], sourceSHA256=digest(entry['path']), importedLabels=entry['labels'], importedLabelsSHA256=digest(entry['labels']), nativeShape=list(labels.shape), spacingZYXum=entry['spacing'],
               model='imported labels', scale=1., calibrationSource='User-confirmed spacing')
    save(run/'run.json', cfg)
    save(run/'processing.json', dict(effectiveSpacingZYXum=entry['spacing'], processedShape=list(labels.shape)))
    summary = export(run, cfg)
    save(run/'complete.json', dict(status='complete', summary=summary))


def controls(data):
    import numpy as np
    groups = [('morph_size', 'Size features', ['volume_um3', 'surface_area_um2', 'major_axis_um', 'middle_axis_um', 'minor_axis_um']),
              ('morph_shape', 'Shape features', ['sphericity', 'elongation', 'flatness', 'mesh_solidity']),
              ('morph_controls', 'Volume + elongation + sphericity', ['volume_um3', 'elongation', 'sphericity'])]
    for key, name, fields in groups:
        fields = [f for f in fields if all(f in c['metrics'] for c in data['cells'])]
        values = np.array([[c['metrics'][f] for f in fields] for c in data['cells']])
        cohort = [i for i, c in enumerate(data['cells']) if c['inMainOrganoid']]
        fit = values[cohort] if cohort else values
        vectors = (values-fit.mean(0))/np.maximum(fit.std(0), 1e-12)
        for cell, vector in zip(data['cells'], vectors):
            cell['representations'][key] = dict(vector=vector.tolist(), point=[0., 0.], cluster=0)
        data['methods'].append(dict(id=key, name=name, metric='euclidean', dimensions=len(fields), variance=[0.,0.], stackVariance=[0.,0.], family='traditional', description=', '.join(fields)))


def prepare_stack(entry, settings, run, logs, progress, export_atlas=True):
    from atlas_worker import compute_slot
    with compute_slot():
        run,logs=Path(run),Path(logs)
        logs.mkdir(parents=True,exist_ok=True)
        if not (run/'complete.json').exists():
            if entry.get('labels'):
                imported(entry,run,settings)
            else:
                args=[sys.executable,'-u',ROOT/'scripts/run_standalone_cellpose.py','--input',entry['path'],'--output',run,
                      '--spacing',*map(str,entry['spacing']),'--scale',settings['scale'],'--diameter',settings['diameter'],'--model',settings['model']]
                if not settings['gpu']:args.append('--cpu')
                if not settings['background']:args.append('--no-background')
                if settings.get('linkingMode')=='streamed':args.extend(['--linking-mode','streamed'])
                call(args,logs/f'{run.name}_segmentation.log',progress)
        if not (run/'qc_summary.json').exists():
            call([sys.executable,ROOT/'scripts/qc_standalone_cellpose.py',run],logs/f'{run.name}_qc.log')
        if export_atlas:
            staging=run/'atlas_export'
            if not (staging/'dataset.json').exists():
                call([sys.executable,ROOT/'scripts/export_standalone_atlas.py',run,staging,'--sample-id',run.name],logs/f'{run.name}_export.log')
            return staging


def build(request, output):
    # Never adopt or overwrite another experiment, or treat partial output as complete.
    validate(request)
    original_request=request
    output = Path(output).resolve()
    for entry in request['inputs']:
        if output == Path(entry['path']).resolve() or output in Path(entry['path']).resolve().parents or (entry.get('omeZarr') and Path(entry['path']).resolve() in output.parents):
            raise ValueError('Choose a dedicated results folder that does not contain the raw input files.')
    analysis_request = {k:v for k,v in request.items() if k != 'execution'}
    identity = hashlib.sha256(json.dumps(analysis_request, sort_keys=True).encode()).hexdigest()
    manifest = output/'preparation.json'
    if output.exists() and any(output.iterdir()):
        if not manifest.exists() or json.loads(manifest.read_text()).get('identity') != identity:
            raise ValueError('This folder contains other data or different preparation settings. Choose a new, empty folder.')
    output.mkdir(parents=True, exist_ok=True)
    save(manifest, dict(identity=identity, request=request))
    (output/'logs').mkdir(exist_ok=True)
    if (output/'complete.json').exists() and (output/'atlas_dataset/dataset.json').exists():
        event('Dataset already prepared', 1)
        return output/'atlas_dataset'
    mode=request.get('linkingMode') or request['settings'].get('linkingMode') or 'standard'
    if mode=='streamed':
        import copy
        request=copy.deepcopy(request)
        request['settings']['linkingMode']='streamed'
    settings = request['settings']
    from atlas_preprocessing import check_memory
    required_disk=0
    for index,entry in enumerate(request['inputs']):
        run=output/'stacks'/f'Stack_{index+1:03d}'
        if not entry.get('labels') and not (entry.get('omeZarr') or {}).get('label') and not (run/'aggregate.completed.json').exists():
            plan=check_memory(entry['shape'],entry['spacing'],settings['scale'],linking_mode=mode)
            if mode=='streamed':
                from atlas_streamed_linking import check_space
                cached=sum(p.stat().st_size for p in run.rglob('*') if p.is_file() and not p.is_symlink()) if run.exists() else 0
                required_disk+=max(0,plan['estimatedDiskGiB']*1024**3-cached)
    if required_disk:
        check_space(output,required_disk,'Dataset preparation across all stacks')
    if request.get('harmonics', True):
        from harmonics_cajal_core import harmonics  # Check optional dependencies before segmentation.
    from atlas_numpy import is_numpy
    if any(entry.get('omeZarr') or is_numpy(entry['path']) or (entry.get('labels') and is_numpy(entry['labels'])) for entry in request['inputs']):
        import copy
        from atlas_ome_zarr import materialize
        request=copy.deepcopy(request)
        for index,entry in enumerate(original_request['inputs']):
            if entry.get('omeZarr'):
                request['inputs'][index]=materialize(entry,output/'inputs'/f'Stack_{index+1:03d}',lambda line:event(line,.01))
            elif is_numpy(entry['path']) or (entry.get('labels') and is_numpy(entry['labels'])):
                from atlas_numpy import materialize as materialize_numpy
                request['inputs'][index]=materialize_numpy(entry,output/'inputs'/f'Stack_{index+1:03d}',lambda line:event(line,.01))
    if (request.get('execution') or {}).get('splitStacks'):
        from atlas_inference_batches import distribute_stack
        for index, entry in enumerate(request['inputs']): distribute_stack(request, output, index, entry)
    elif (request.get('execution') or {}).get('workers'):
        from atlas_worker_client import distribute
        distribute(request, output)
    parts = []
    count = len(request['inputs'])
    for i, entry in enumerate(request['inputs']):
        sample = f'Stack_{i+1:03d}'
        run = output/'stacks'/sample
        stage_fraction = .7/count
        def stage_progress(line):
            from run_standalone_cellpose import STAGES
            if line.startswith('@segmentation-progress '):
                value=json.loads(line[len('@segmentation-progress '):]);phase=value.get('stage')
                if phase in STAGES:
                    n=STAGES.index(phase);fraction=max(0,min(1,float(value['fraction'])))
                    event(f'{i+1}/{count} · {entry["name"]} · {value["label"]}',(i+(n+fraction)/len(STAGES))*stage_fraction)
            for n, stage in enumerate(STAGES):
                if line.startswith(('Starting '+stage, 'Using completed stage: '+stage)):
                    label = dict(preprocess='Preparing image', xy='Cellpose XY', xz='Cellpose XZ', yz='Cellpose YZ', aggregate='Linking cells in 3D', export='Building cell surfaces')[stage]
                    event(f'{i+1}/{count} · {entry["name"]} · {label}', (i+n/len(STAGES))*stage_fraction)
        event(f'{i+1}/{count} · {entry["name"]}', i*stage_fraction)
        staging = prepare_stack(entry,settings,run,output/'logs',stage_progress)
        parts.append((staging, json.loads((staging/'dataset.json').read_text())))
    event('Combining stacks and classifying organoid membership', .72)
    import shutil
    import numpy as np
    from organoid_membership import classify, rebuild
    atlas = output/'atlas_dataset'
    atlas.mkdir(exist_ok=True)
    data = dict(parts[0][1], title=request['title'].strip(), cells=[c for _, d in parts for c in d['cells']],
                organoids=[s for _, d in parts for s in d['organoids']], samples=[s for _, d in parts for s in d['samples']])
    for index,entry in enumerate(original_request['inputs']):
        if entry.get('omeZarr'):
            data['samples'][index]['name']=entry['name']
            data['samples'][index]['sourceOMEZarr']=dict(path=entry['path'],**entry['omeZarr'])
        elif is_numpy(entry['path']) or (entry.get('labels') and is_numpy(entry['labels'])):
            data['samples'][index]['name']=entry['name']
            source=json.loads((output/'inputs'/f'Stack_{index+1:03d}'/'source.json').read_text())
            data['samples'][index]['sourceNumPy']=dict(path=entry['path'],labels=entry.get('labels'),axes='ZYX',
                shape=entry['shape'],spacingZYXum=entry['spacing'],sourceStat=entry['sourceStat'],sourcesSHA256=source['sourcesSHA256'],
                calibration='User-confirmed voxel spacing; NumPy axes interpreted as Z/Y/X')
    # A joint feature set must be finite for every object, without imputing measurements.
    fields = [f for f in data['traditionalFeatures'] if all(f in c['metrics'] for c in data['cells'])]
    data['traditionalFeatures'] = fields
    data['methods'][0].update(dimensions=len(fields), description=', '.join(fields))
    for i, (staging, part) in enumerate(parts):
        sample = part['samples'][0]['id']
        destination = atlas/sample
        shutil.copytree(staging/sample, destination, dirs_exist_ok=True)
        labels = np.load(output/'stacks'/sample/'labels_processed.npy', mmap_mode='r')
        membership = classify(labels)
        for cell in data['cells']:
            if cell['sample'] != sample: continue
            auto = membership[cell['label']]
            cell.update(auto, automaticMainOrganoid=auto['inMainOrganoid'])
            if not request.get('excludeDetached', True):
                cell.update(inMainOrganoid=True, membershipOverride=True)
            cell['representations']['traditional']['vector'] = [0.]*len(fields)
    data['membershipStackTemplates'] = data['organoids']
    controls(data)
    if request.get('harmonics', True):
        from harmonics_cajal_core import harmonics, angular_basis
        from prepare_pointnet_shape import read_mesh
        basis = angular_basis()
        cache = output/'harmonics'
        cache.mkdir(exist_ok=True)
        for i, cell in enumerate(data['cells']):
            from run_standalone_cellpose import digest
            mesh_path = atlas/cell['sample']/f"{cell['label']}.mesh"
            mesh_hash = digest(mesh_path)
            path = cache/f"{cell['sample']}_{cell['label']}.npz"
            vector = None
            if path.exists():
                try:
                    with np.load(path) as cached:
                        if str(cached['mesh_hash']) == mesh_hash:
                            vector = cached['vector'].copy()
                except (ValueError, KeyError, OSError):
                    pass
            if vector is None:
                v, f = read_mesh(mesh_path)
                vector = harmonics(v, f, basis)
                np.savez_compressed(path, vector=vector, mesh_hash=mesh_hash)
            if not np.isfinite(vector).all(): raise ValueError(f"Nonfinite descriptor for {cell['id']}")
            cell['representations']['spherical_harmonics'] = dict(vector=vector.tolist(), point=[0.,0.], cluster=0)
            event(f'Shape descriptors · {i+1}/{len(data["cells"])} cells', .75+.2*(i+1)/len(data['cells']))
        data['methods'].append(dict(id='spherical_harmonics', name='Raw multi-shell harmonics', metric='euclidean', dimensions=len(vector), variance=[0.,0.], stackVariance=[0.,0.], family='shape', description='Rotation-invariant shell powers; size normalized; same descriptor as the five-organoid benchmark. No training.'))
    event('Calculating joint feature plots and organoid measurements', .96)
    data['provenance'] = 'Prepared from selected 3D image volumes (TIFF, NumPy Z/Y/X arrays or OME-Zarr T/C/resolution selections). OME-Zarr time points are independent frames, not biological replicates or linked cell tracks. Original surfaces in calibrated micrometers. Voxel spacing was confirmed by the user. Main-organoid membership is classified on the processed segmentation grid. TIFF inputs remain referenced in their original locations. NumPy arrays and selected OME-Zarr volumes are cached as calibrated TIFFs inside the results folder, with original source provenance retained. NumPy files contain no physical calibration; their three axes are interpreted as Z/Y/X without squeezing or transposing. Automated QC flags need review. No biological validation or learned model training is implied.'
    rebuild(data)
    data['membershipAnalysis']['rule'] = 'Largest 26-connected foreground component on the processed grid; strict majority per cell. Manual overrides take precedence.'
    validate(original_request)
    save(atlas/'dataset.json', data)
    save(output/'complete.json', dict(dataset=str(atlas), cells=len(data['cells']), stacks=count, identity=identity))
    event('Dataset ready', 1)
    return atlas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['inspect', 'inspect-zarr', 'build', 'slice', 'test-slice'])
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if args.operation == 'inspect-zarr':
        from atlas_ome_zarr import discover
        save(args.output,discover(request['paths']))
    elif args.operation == 'inspect':
        save(args.output, dict(inputs=[inspect(p) for p in request['paths']]))
    else:
        try: os.setsid()
        except OSError: pass
        if args.operation in ('slice', 'test-slice'):
            from atlas_slice_test import run
            run(request, args.output, segment=args.operation == 'test-slice')
        else:
            build(request, args.output)


if __name__ == '__main__':
    main()
