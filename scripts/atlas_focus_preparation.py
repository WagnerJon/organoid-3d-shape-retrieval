"""FOCUS-3D initial segmentation, restored to the raw grid for Atlas export.

Inference runs in the existing dedicated FOCUS environment; measurements/export
run in the project environment. No Cellpose predictions or u-Segment3D linking.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from run_standalone_cellpose import save, digest, export
from focus3d_explorer import inference_options, DEFAULT_CONFIG


def memory_plan(shape, spacing, settings, total_bytes=None):
    """Conservative planning, including dense whole-volume upstream buffers.

    This is not a measured peak guarantee. Matches upstream radius resampling;
    a small patch/batch does not eliminate global volume arrays.
    """
    options=inference_options(settings,spacing)
    if len(shape)!=3 or any(type(x) is not int or x<2 for x in shape):raise ValueError('FOCUS-3D needs a 3D stack')
    factor=options['cell_radius']/15
    factors=(1 if options['z_ratio']>1 else factor,factor,factor)
    grid=[max(1,round(n/f)) for n,f in zip(shape,factors)]
    # Padded grid is at least one model patch; extra overlap allowance.
    padded=[max(p,n+2*s) for n,p,s in zip(grid,options['patch_size'],options['stride'])]
    if total_bytes is None:
        import psutil
        total_bytes=psutil.virtual_memory().total
    gib=1024**3;budget=min(11.5*gib,.65*total_bytes)
    peak=6*gib+32*math.prod(padded)+16*math.prod(shape)
    return dict(processedShape=grid,estimatedPeakGiB=peak/gib,budgetGiB=budget/gib,fits=peak<=budget,
                note='Conservative FOCUS-3D model/buffer estimate, not a guarantee; upstream keeps global arrays.')


def validate_settings(settings, spacing):
    inference_options(settings,spacing)
    checkpoint=Path(settings.get('checkpoint','')).expanduser()
    config=Path(settings.get('config','') or DEFAULT_CONFIG).expanduser()
    if not checkpoint.is_file():raise ValueError('Choose a downloaded FOCUS-3D checkpoint (.pth).')
    if not config.is_file() or not (ROOT/'.cache/focus3d-env/bin/python').is_file():
        raise ValueError('FOCUS-3D is not installed. Run scripts/setup_focus3d.sh.')
    return checkpoint,config


def progress(stage,label,fraction):
    print('@segmentation-progress '+json.dumps(dict(stage=stage,label=label,fraction=fraction)),flush=True)


def infer(entry, settings, output):
    import numpy as np
    from focus3d_explorer import run_focus3d
    labels,metadata,note=run_focus3d(SimpleNamespace(shape=tuple(entry['shape'])),
        dict(path=entry['path'],spacing=entry['spacing']),settings,output,
        lambda label,fraction:progress('focus3d',label,fraction),digest)
    if labels.max(initial=0)>65535:
        raise ValueError('Atlas currently supports at most65535 label IDs per stack.')
    if not np.any(labels):raise ValueError('FOCUS-3D found no cells. Review cell radius, background and detection thresholds before retrying.')
    temporary=output/'labels_processed.pending.npy'
    np.save(temporary,labels.astype(np.uint32,copy=False));temporary.replace(output/'labels_processed.npy')
    save(output/'focus3d.json',dict(metadata=metadata,note=note))
    progress('focus3d','FOCUS-3D labels restored to the original grid',1.)


def guarded(arguments, folder, name):
    """Relay progress without a blocking pipe; stop sustained memory pressure."""
    import psutil
    log=folder/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
    with log.open('a') as writer,log.open() as reader,(folder/'memory.jsonl').open('a') as samples:
        reader.seek(0,2);pending='';low=0;peak=0
        process=subprocess.Popen(list(map(str,arguments)),cwd=ROOT,stdout=writer,stderr=subprocess.STDOUT)
        try:
            while True:
                pending+=reader.read();lines=pending.split('\n');pending=lines.pop()
                for line in lines:print(line,flush=True)
                if process.poll() is not None:break
                try:rss=psutil.Process(process.pid).memory_info().rss/1024**3
                except psutil.NoSuchProcess:rss=0
                available=psutil.virtual_memory().available/1024**3;peak=max(peak,rss)
                samples.write(json.dumps(dict(stage=name,rssGiB=rss,availableGiB=available,time=time.time()))+'\n');samples.flush()
                low=low+1 if available<.5 or rss>11.5 else 0
                if low>=3:raise RuntimeError('Memory safeguard stopped FOCUS-3D. Use a smaller crop or close other memory-heavy applications. Completed stages remain cached.')
                time.sleep(1)
            if pending:print(pending,flush=True)
            if process.returncode:raise RuntimeError(f'{name} failed; see {log}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    return peak


def run(entry, settings, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    checkpoint,config=validate_settings(settings,entry['spacing'])
    from prepare_atlas_dataset import inspect
    info=inspect(entry['path'])
    if info['shape']!=entry['shape'] or info['sourceStat']!=entry['sourceStat']:raise ValueError('Raw stack changed since selection')
    identity=dict(source=str(Path(entry['path']).resolve()),sourceSHA256=digest(entry['path']),nativeShape=entry['shape'],
        spacingZYXum=entry['spacing'],focusSettings=settings,checkpointSHA256=digest(checkpoint),configSHA256=digest(config),
        adapterSHA256=digest(ROOT/'scripts/focus3d_explorer.py'),preparationSHA256=digest(__file__))
    cfg_path=output/'run.json'
    if cfg_path.exists():
        old=json.loads(cfg_path.read_text())
        if old.get('focusIdentity')!=identity:raise ValueError('FOCUS-3D inputs, weights or settings changed. Choose a new results folder.')
        cfg=old
    else:
        cfg=dict(source=identity['source'],sourceSHA256=identity['sourceSHA256'],nativeShape=entry['shape'],spacingZYXum=entry['spacing'],
            model='FOCUS-3D',segmentationMode='focus3d',scale=1.,focusIdentity=identity,calibrationSource=entry.get('calibrationSource','User-confirmed spacing'))
        save(cfg_path,cfg)
    if (output/'complete.json').exists():return
    if not (output/'aggregate.completed.json').exists():
        plan=memory_plan(entry['shape'],entry['spacing'],settings)
        if not plan['fits']:raise ValueError(f"FOCUS-3D memory estimate {plan['estimatedPeakGiB']:.1f}GiB exceeds budget {plan['budgetGiB']:.1f}GiB. Use a smaller crop; patch inference still needs whole-volume buffers.")
        request=output/'focus-request.json';save(request,dict(input=entry,settings=settings))
        peak=guarded([ROOT/'.cache/focus3d-env/bin/python','-u',Path(__file__),'infer','--request',request,'--output',output],output,'focus3d')
        save(output/'processing.json',dict(processedShape=entry['shape'],effectiveSpacingZYXum=entry['spacing'],
            rawGridOffsetZYXum=[0,0,0],resamplingZoomZYX=[1,1,1],segmentationMode='focus3d',memoryPlan=plan))
        save(output/'aggregate.completed.json',dict(method='focus3d',restoredToOriginalGrid=True,peakSampledRSSGiB=peak))
    if not (output/'export.completed.json').exists():
        progress('export','Building calibrated cell surfaces and measurements',0.)
        peak=guarded([sys.executable,'-u',Path(__file__),'export','--output',output],output,'export')
        save(output/'export.completed.json',dict(method='focus3d',peakSampledRSSGiB=peak))
    save(output/'complete.json',dict(status='complete',segmentationMode='focus3d',summary=json.loads((output/'summary.json').read_text())))
    progress('export','FOCUS-3D stack ready',1.)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['run','infer','export'])
    parser.add_argument('--request',type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if args.operation=='export':
        export(args.output,json.loads((args.output/'run.json').read_text()))
    else:
        request=json.loads(args.request.read_text());globals()[args.operation](request['input'],request['settings'],args.output)
