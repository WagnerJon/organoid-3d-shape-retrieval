"""Checkpointed fine-grid linking and plane-streamed Cellpose for Atlas preparation."""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import time

VERSION=1
ARRAYS=('probability.npy','gradient.npy','binary-threshold.npy','binary-closed.npy',
        'foreground-components.npy','binary-filtered.npy','binary.npy','endpoints.npy',
        'endpoint-occupancy.npy','votes.npy','attractor-mask.npy','centers.npy','labels.npy')


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def runtime_signature():
    from importlib.metadata import version
    return {name:version(name) for name in ('numpy','scipy','scikit-image','connected-components-3d','cellpose','torch')}


def progress_event(stage,label,fraction):
    print('@segmentation-progress '+json.dumps(dict(stage=stage,label=label,fraction=max(0.,min(1.,float(fraction))))),flush=True)


def check_space(path,required,description='Processing'):
    path=Path(path)
    while not path.exists():path=path.parent
    available=shutil.disk_usage(path).free
    if available<required:
        raise ValueError(f'{description} needs approximately {required/2**30:.1f} GiB of additional free disk space; {available/2**30:.1f} GiB is available. Choose a results folder on a larger drive or lower the processing scale. Completed checkpoints are retained.')


def validate_mode(value):
    mode=value or 'standard'
    if mode not in ('standard','streamed'):raise ValueError('Choose standard or memory-saving fine-grid linking.')
    return mode


def aggregate(run,params,progress=None):
    import numpy as np
    import atlas_streaming_core as core
    run=Path(run);folder=run/'streamed_linking';destination=run/'labels_processed.npy'
    if folder.is_symlink():raise ValueError('Refusing a linked scratch directory')
    folder.mkdir(exist_ok=True)
    signature_files=[run/'predictions'/f'{v}_{kind}.npy' for v in ['xy','xz','yz'] for kind in ['prob','flow']]
    source=[dict(name=p.name,size=p.stat().st_size,mtime=p.stat().st_mtime_ns) for p in signature_files]
    implementation={p.name:digest(p) for p in [Path(__file__),Path(core.__file__)]}
    identity=hashlib.sha256(json.dumps(dict(version=VERSION,source=source,parameters=params,implementation=implementation,runtime=runtime_signature()),sort_keys=True).encode()).hexdigest()
    manifest=folder/'state.json'
    if manifest.exists():
        state=json.loads(manifest.read_text())
        if state.get('identity')!=identity:raise ValueError('Linking inputs or implementation changed. Keep these checkpoints and choose a new results folder.')
    else:
        if any(folder.iterdir()):raise ValueError('Refusing to adopt an unrecognized linking scratch folder')
        state=dict(identity=identity,version=VERSION,implementation=implementation,fusionDone=False,trajectoryNextZ=0,trajectoriesDone=False,complete=False)
        save(manifest,state)
    for name in ARRAYS:
        if (folder/name).is_symlink():raise ValueError('Refusing a linked scratch file')
    def clean(names):
        for name in names:(folder/name).unlink(missing_ok=True)
    def report(label,fraction):
        progress_event('aggregate',label,fraction)
        if progress:progress(label,fraction)
    def final_valid():
        if not destination.is_file():return False
        labels=np.load(destination,mmap_mode='r',allow_pickle=False)
        valid=list(labels.shape)==state['result']['shape'] and labels.dtype==np.uint32
        labels._mmap.close()
        return valid and digest(destination)==state['result']['labelSHA256']
    if state['complete']:
        if not final_valid():raise ValueError('Completed streamed labels are missing or changed; restore them or choose a new results folder.')
        clean(ARRAYS);report('Using completed fine-grid linking',1.)
        return state['result']
    header=np.load(signature_files[0],mmap_mode='r');shape=header.shape;header._mmap.close()
    existing=sum((folder/name).stat().st_size for name in ARRAYS if (folder/name).is_file())
    check_space(folder,max(0,24*math.prod(shape)-existing)+512*1024**2,'Fine-grid linking scratch files')
    if not state['fusionDone']:
        clean(ARRAYS)
        info=core.fuse_predictions(run/'predictions',folder,params,progress=lambda text,f:report(text,.35*f))
        state.update(fusionDone=True,fusion=info);save(manifest,state)
    # Also release fusion temporaries if cancellation happened just after its checkpoint.
    clean(('probability.npy','binary-threshold.npy','binary-closed.npy','foreground-components.npy','binary-filtered.npy'))
    if not state['trajectoriesDone']:
        def checkpoint(z):
            state['trajectoryNextZ']=int(z);save(manifest,state)
        report('Following fine-grid cell flows',.35)
        core.trajectories(folder,params['gradient_descent'],start_z=state['trajectoryNextZ'],checkpoint=checkpoint,
                          progress=lambda text,f:report(text,.35+.4*f))
        state['trajectoriesDone']=True;save(manifest,state)
    clean(('gradient.npy',))
    info=core.cluster_endpoints(folder,params['connected_component']['smooth_sigma'],progress=lambda text,f:report(text,.75+.24*f))
    result=folder/'labels.npy'
    labels=np.load(result,mmap_mode='r');assert labels.shape==shape and labels.dtype==np.uint32;labels._mmap.close()
    result.replace(destination)
    details=dict(**info,shape=list(shape),linkingMode='streamed',labelSHA256=digest(destination),
                 execution='Fine grid; haloed disk-backed fusion, batched trajectories, global connected components; no coarse resampling')
    state.update(complete=True,result=details);save(manifest,state)
    clean(ARRAYS)
    report('Fine-grid linking complete',1.)
    return details


def infer_view(run,view,settings,model_factory=None):
    """Only one image plane and one prediction are materialized at a time."""
    import numpy as np
    from atlas_streaming_core import disk
    order={'xy':(0,1,2),'xz':(1,0,2),'yz':(2,0,1)}[view]
    from atlas_cellpose_models import local_fields
    settings=dict(settings,**local_fields(settings,freeze=True))
    run=Path(run);processed=np.load(run/'processed.npy',mmap_mode='r')
    shape=processed.shape;plane_shape=tuple(shape[a] for a in order)
    checkpoint=run/'predictions'/f'{view}.streamed.json'
    signature=hashlib.sha256(json.dumps(dict(settings=settings,shape=shape,source=[(run/'processed.npy').stat().st_size,(run/'processed.npy').stat().st_mtime_ns],version=VERSION,implementation=digest(__file__),modelAdapter=digest(Path(__file__).with_name('atlas_cellpose_models.py')),runtime=runtime_signature()),sort_keys=True).encode()).hexdigest()
    prob_path=run/'predictions'/f'{view}_prob.npy';flow_path=run/'predictions'/f'{view}_flow.npy'
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(identity=signature,nextPlane=0)
    if state.get('identity')!=signature:raise ValueError('Cellpose plane checkpoint settings changed; choose a new results folder.')
    start=int(state['nextPlane'])
    if not 0<=start<=plane_shape[0]:raise ValueError('Invalid Cellpose plane checkpoint')
    if start:
        probability=np.load(prob_path,mmap_mode='r+');flow=np.load(flow_path,mmap_mode='r+')
        if probability.shape!=shape or flow.shape!=(2,)+shape or probability.dtype!=np.float32 or flow.dtype!=np.float32:raise ValueError('Invalid Cellpose checkpoint predictions')
    else:
        probability=disk(prob_path,shape);flow=disk(flow_path,(2,)+shape)
    if start==plane_shape[0]:return dict(view=view,shape=list(shape),execution='Using completed streamed view')
    if model_factory is None:
        from atlas_cellpose_models import make_model
        model_factory=make_model
    from atlas_cellpose_models import factory_arguments
    model=model_factory(**factory_arguments(settings))
    plane_prob=probability.transpose(order);plane_flow=flow.transpose((0,)+tuple(a+1 for a in order))
    planes=processed.transpose(order)
    for index in range(start,plane_shape[0]):
        _,fields,_=model.eval([np.asarray(planes[index]).copy()],batch_size=32,channels=[0,0],diameter=settings['diameter'],compute_masks=False)
        p,f=fields[0][2],fields[0][1]
        if p.shape!=plane_shape[1:] or f.shape!=(2,)+plane_shape[1:] or not np.isfinite(p).all() or not np.isfinite(f).all():raise ValueError('Invalid Cellpose prediction')
        plane_prob[index]=p;plane_flow[:,index]=f
        if (index+1)%8==0 or index+1==plane_shape[0]:
            probability.flush();flow.flush();state['nextPlane']=index+1;save(checkpoint,state)
            del plane_prob,plane_flow
            probability._mmap.close();flow._mmap.close()
            probability=np.load(prob_path,mmap_mode='r+');flow=np.load(flow_path,mmap_mode='r+')
            plane_prob=probability.transpose(order);plane_flow=flow.transpose((0,)+tuple(a+1 for a in order))
        progress_event(view,f'Cellpose {view.upper()} · plane {index+1}/{plane_shape[0]}',(index+1)/plane_shape[0])
    return dict(view=view,shape=list(shape),device=str(getattr(model,'device','unknown')),execution='Plane-streamed Cellpose; checkpoint every8 planes')
