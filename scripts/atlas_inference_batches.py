"""Independent plane inference with one global preprocess and one global 3D merge."""
import argparse
import json
from pathlib import Path
import run_standalone_cellpose as pipeline  # Runtime environment and vendor path.

ORDERS = {'xy': (0, 1, 2), 'xz': (1, 0, 2), 'yz': (2, 0, 1)}


def infer(source, settings, output):
    import numpy as np
    import torch
    from cellpose import models
    from atlas_worker import atomic, compute_slot
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    planes = np.load(source, mmap_mode='r', allow_pickle=False)
    if planes.ndim != 3 or planes.dtype != np.float32 or not 1 <= planes.shape[0] <= 128 or min(planes.shape) < 1:
        raise ValueError('Expected 1–128 float32 image planes')
    if not np.isfinite(planes).all(): raise ValueError('Nonfinite batch intensities')
    with compute_slot():
        gpu = settings['gpu'] and (torch.backends.mps.is_available() or torch.cuda.is_available())
        model = models.CellposeModel(pretrained_model=settings['model'], gpu=gpu)
        prob = np.lib.format.open_memmap(output/'prob.npy', mode='w+', dtype='float32', shape=planes.shape)
        flow = np.lib.format.open_memmap(output/'flow.npy', mode='w+', dtype='float32', shape=(2,)+planes.shape)
        for i, image in enumerate(planes):
            print(f'Starting plane {i+1}/{len(planes)}', flush=True)
            # Matches u-Segment3D's supplied-diameter, hist_norm=False Cellpose v4 path.
            _, fields, _ = model.eval([image.copy()], batch_size=32, channels=[0,0], diameter=settings['diameter'], compute_masks=False)
            prob[i] = fields[0][2]; flow[:,i] = fields[0][1]
            atomic(output/'status.json', dict(stage=f'Plane {i+1}/{len(planes)}'))
        prob.flush(); flow.flush()
        if not pipeline.finite(prob) or not pipeline.finite(flow): raise ValueError('Nonfinite predictions')
        atomic(output/'batch.completed.json', dict(shape=list(planes.shape), device='GPU' if gpu else 'CPU'))


def prepare(entry, settings, run):
    from types import SimpleNamespace
    from atlas_worker import compute_slot
    args = SimpleNamespace(input=Path(entry['path']), output=Path(run), spacing=entry['spacing'], scale=settings['scale'], diameter=settings['diameter'], model=settings['model'], cpu=not settings['gpu'], no_background=not settings['background'], through_stage='preprocess',linking_mode=settings.get('linkingMode','standard'))
    with compute_slot():
        pipeline.run(args)


def assemble(run, jobs):
    import numpy as np
    from atlas_worker import atomic, sha
    run = Path(run); shape = tuple(json.loads((run/'processing.json').read_text())['processedShape'])
    for view, order in ORDERS.items():
        if (run/f'{view}.completed.json').exists(): continue
        prob = np.lib.format.open_memmap(run/'predictions'/f'{view}_prob.npy', mode='w+', dtype='float32', shape=shape)
        flow = np.lib.format.open_memmap(run/'predictions'/f'{view}_flow.npy', mode='w+', dtype='float32', shape=(2,)+shape)
        plane_prob = prob.transpose(order); plane_flow = flow.transpose((0,)+tuple(i+1 for i in order))
        covered = np.zeros(shape[order[0]], dtype=bool)
        provenance = []
        for job in jobs:
            if job['view'] != view: continue
            start, stop = job['start'], job['stop']; folder = Path(job['result'])
            p = np.load(folder/'prob.npy', mmap_mode='r'); f = np.load(folder/'flow.npy', mmap_mode='r')
            expected = (stop-start,)+plane_prob.shape[1:]
            if p.shape != expected or f.shape != (2,)+expected or covered[start:stop].any() or not pipeline.finite(p) or not pipeline.finite(f):
                raise ValueError('Invalid or overlapping inference batch')
            plane_prob[start:stop] = p; plane_flow[:,start:stop] = f; covered[start:stop] = True
            provenance.append(dict(start=start, stop=stop, probabilitySHA256=sha(folder/'prob.npy'), flowSHA256=sha(folder/'flow.npy')))
        if not covered.all(): raise ValueError('Missing planes; cannot link incomplete predictions')
        prob.flush(); flow.flush()
        atomic(run/f'{view}.completed.json', dict(stage=view, shape=list(shape), execution='Distributed independent planes; global preprocessing and aggregation', batches=provenance))


def distribute_stack(request, output, index, entry):
    import numpy as np
    import sys
    from atlas_worker import atomic
    from atlas_worker_client import distribute
    from prepare_atlas_dataset import call, event
    output = Path(output); run = output/'stacks'/f'Stack_{index+1:03d}'
    if entry.get('labels') or (run/'complete.json').exists(): return
    size = (request.get('execution') or {}).get('planesPerBatch', 16)
    if type(size) is not int or not 1 <= size <= 128: raise ValueError('Planes per batch must be between 1 and 128')
    event(f'Stack {index+1} · preparing image for distributed batches', .02)
    req = output/'logs'/f'{run.name}-batch-prepare.json'; atomic(req, dict(input=entry, settings=request['settings']))
    call([sys.executable, Path(__file__), 'prepare', '--request', req, '--output', run], output/'logs'/f'{run.name}-batch-prepare.log')
    processed = np.load(run/'processed.npy', mmap_mode='r')
    batch_root = run/'inference_batches'; batch_root.mkdir(exist_ok=True); jobs = []
    for view, order in ORDERS.items():
        if (run/f'{view}.completed.json').exists(): continue
        planes = processed.transpose(order)
        for start in range(0,len(planes),size):
            stop = min(start+size,len(planes)); folder = batch_root/f'{view}_{start:06d}_{stop:06d}'
            folder.mkdir(exist_ok=True); source = folder/'planes.npy'
            result = folder/'result'
            if not (result/'batch.completed.json').exists() and not source.exists(): pipeline.save_array(source, np.ascontiguousarray(planes[start:stop]))
            jobs.append(dict(path=str(source), view=view, start=start, stop=stop, result=str(result), spacing=entry['spacing'], name=f'{view.upper()} planes {start+1}–{stop}'))
    if jobs:
        dispatch = dict(request, inputs=jobs, _inferenceBatches=True)
        distribute(dispatch, output)
        assemble(run, jobs)
        # Retain compact predictions/checkpoints, remove transfer-only input duplicates.
        for job in jobs: Path(job['path']).unlink(missing_ok=True)
    event(f'Stack {index+1} · all batches ready; linking cells across the whole volume', .65)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('operation', choices=['prepare','infer']); p.add_argument('--request', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); req = json.loads(a.request.read_text())
    if a.operation == 'prepare': prepare(req['input'], req['settings'], a.output)
    else: infer(req['input']['path'], req['settings'], a.output)
