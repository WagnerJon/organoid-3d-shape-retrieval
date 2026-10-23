"""Select modern Cellpose or the official classic CNN without silent fallback.

Cellpose4 no longer implements cyto CNNs. Load the pinned official Cellpose3
Python sources under a private package name, sharing Torch/dependencies but
never replacing/import-shadowing the installed Cellpose4 package.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types
import urllib.request
import urllib.error
import zipfile
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('CELLPOSE_LOCAL_MODELS_PATH',str(ROOT/'.cache/cellpose'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.cache/numba'))
LEGACY_MODELS=('cyto3','nuclei')
MODEL_NAMES=('cpsam_v2','cpdino','cpdino-vitb',*LEGACY_MODELS,'local')
VERSION='3.1.1.3'
WHEEL=f'cellpose-{VERSION}-py3-none-any.whl'
WHEEL_SHA256='0c309f306a8356ba2779054dfd8a1b4a88ea7d6c3388a6fdac99b91b21ce7ff2'


def local_fields(values, freeze=False):
    """Validate explicit checkpoints and optionally bind their content identity."""
    if values.get('model') != 'local': return {}
    runtime=values.get('localModelRuntime','cellpose3')
    if runtime not in ('cellpose3','cellpose4'): raise ValueError('Choose Cellpose 3 CNN or Cellpose 4 SAM/DINO for the local model.')
    raw=values.get('localModelPath','')
    if not isinstance(raw,str) or not raw.strip(): raise ValueError('Choose a local Cellpose checkpoint file.')
    path=Path(raw).expanduser().resolve()
    if not path.is_file() or path.stat().st_size==0: raise ValueError('Local Cellpose checkpoint is missing or empty: '+str(path))
    result=dict(localModelPath=str(path),localModelRuntime=runtime)
    expected=values.get('localModelSHA256')
    if freeze or expected:
        actual=digest(path)
        if expected and expected!=actual: raise ValueError('Local Cellpose checkpoint changed; choose a new results folder or start a new run.')
        result['localModelSHA256']=actual
    return result


def factory_arguments(values):
    args=dict(pretrained_model=values.get('model','cpsam_v2'),gpu=bool(values.get('gpu',True)))
    if args['pretrained_model']=='local': args.update(local_settings=local_fields(values,freeze=True))
    return args


def explicit_inference(values):
    return values.get('model') in (*LEGACY_MODELS,'local')


def model_metadata(model, values=None):
    from importlib.metadata import version
    values=values or {}
    classic=model in LEGACY_MODELS or (model=='local' and values.get('localModelRuntime','cellpose3')=='cellpose3')
    result=dict(model=model,engine='Cellpose3 CNN' if classic else 'Cellpose4 transformer',
                version=VERSION if classic else version('cellpose'),adapterSHA256=digest(__file__))
    if classic:result['wheelSHA256']=WHEEL_SHA256
    path=weights_path(model,values)
    if path.is_file():result.update(checkpoint=str(path),checkpointSHA256=digest(path))
    return result


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def weights_path(model, values=None):
    if model not in MODEL_NAMES:raise ValueError('Unsupported Cellpose model: '+str(model))
    if model=='local': return Path(local_fields(dict(values or {},model='local'))['localModelPath'])
    filename='nucleitorch_0' if model=='nuclei' else model
    return Path(os.environ.get('CELLPOSE_LOCAL_MODELS_PATH',ROOT/'.cache/cellpose'))/filename


def ensure_legacy():
    """Download a checksum-pinned wheel once; no pip/dependency modifications."""
    import fcntl
    cache=ROOT/'.cache/cellpose3-runtime';cache.mkdir(parents=True,exist_ok=True)
    with (cache/'setup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        wheel=cache/WHEEL
        if not wheel.exists():
            downloaded=ROOT/'.cache/cellpose3-download'/WHEEL
            temporary=cache/(WHEEL+'.pending')
            if downloaded.is_file():temporary.write_bytes(downloaded.read_bytes())
            else:
                print('Downloading official Cellpose3 runtime for classic models…',flush=True)
                with urllib.request.urlopen(f'https://pypi.org/pypi/cellpose/{VERSION}/json',timeout=60) as response:
                    metadata=json.load(response)
                entry=next(x for x in metadata['urls'] if x['filename']==WHEEL)
                if entry['digests']['sha256']!=WHEEL_SHA256:raise ValueError('Unexpected Cellpose3 package checksum')
                with urllib.request.urlopen(entry['url'],timeout=120) as response,temporary.open('wb') as handle:
                    while block:=response.read(1024**2):handle.write(block)
            if digest(temporary)!=WHEEL_SHA256:raise ValueError('Cellpose3 download checksum mismatch')
            temporary.replace(wheel)
        if digest(wheel)!=WHEEL_SHA256:raise ValueError('Cellpose3 runtime archive changed; restore the official pinned wheel')
        destination=cache/VERSION;destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if not (name.startswith('cellpose/') or 'LICENSE' in name) or name.endswith('/'):continue
                relative=Path(name)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid Cellpose3 archive path')
                target=destination/relative;expected=archive.read(name)
                if target.exists():
                    if target.read_bytes()!=expected:raise ValueError('Cellpose3 runtime source changed: '+name)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    temporary=target.with_suffix(target.suffix+'.pending');temporary.write_bytes(expected);temporary.replace(target)
    return destination/'cellpose'


@lru_cache(maxsize=1)
def legacy_backend():
    source=ensure_legacy();name='_atlas_cellpose3'
    # Bypass upstream __init__'s absolute `cellpose.version` import. All model
    # inference imports are relative and resolve inside this private namespace.
    package=types.ModuleType(name);package.__path__=[str(source)];package.__package__=name
    package.version=VERSION;package.version_str=VERSION
    sys.modules[name]=package
    return importlib.import_module(name+'.models')


def make_model(pretrained_model='cpsam_v2',gpu=False,local_settings=None,**kwargs):
    if pretrained_model not in MODEL_NAMES:raise ValueError('Unsupported Cellpose model: '+str(pretrained_model))
    if pretrained_model=='local':
        fields=local_fields(dict(local_settings or {},model='local'),freeze=True)
        path=fields['localModelPath']
        try:
            if fields['localModelRuntime']=='cellpose3':
                model=legacy_backend().CellposeModel(pretrained_model=path,gpu=gpu,**kwargs)
            else:
                from cellpose.models import CellposeModel
                model=CellposeModel(pretrained_model=path,gpu=gpu,**kwargs)
        except Exception as exc:
            raise ValueError('Could not load the local Cellpose checkpoint. Check that its training version matches the selected runtime: '+str(exc)) from exc
        if Path(model.pretrained_model).resolve()!=Path(path): raise ValueError('Cellpose loaded unexpected weights; refusing local-model fallback')
        local_fields(dict(fields,model='local'))  # Detect checkpoint replacement during loading.
        model.atlas_runtime=model_metadata('local',fields)
        return model
    if pretrained_model in LEGACY_MODELS:
        backend=legacy_backend()
        try:
            model=backend.CellposeModel(model_type=pretrained_model,gpu=gpu,**kwargs)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f'Could not download official {pretrained_model} weights. The Cellpose server may be unavailable. Place the official {weights_path(pretrained_model).name} checkpoint at {weights_path(pretrained_model)}.') from exc
        expected=weights_path(pretrained_model)
        if Path(model.pretrained_model).resolve()!=expected.resolve():
            raise ValueError('Classic Cellpose loaded unexpected weights; refusing fallback')
        model.atlas_runtime=dict(engine='Cellpose3 CNN',version=VERSION,wheelSHA256=WHEEL_SHA256,
                                 checkpointSHA256=digest(expected),checkpoint=str(expected))
    else:
        from cellpose.models import CellposeModel
        model=CellposeModel(pretrained_model=pretrained_model,gpu=gpu,**kwargs)
    return model


def legacy_view(processed,view,model_name,gpu,diameter,local_settings=None):
    """Raw per-plane probabilities/flows for the existing global u-Segment3D merge."""
    import numpy as np
    order={'xy':(0,1,2),'xz':(1,0,2),'yz':(2,0,1)}[view]
    planes=np.asarray(processed).transpose(order)
    model=make_model(**factory_arguments(dict(local_settings or {},model=model_name,gpu=gpu)))
    probs=np.empty(planes.shape,np.float32);flows=np.empty((2,)+planes.shape,np.float32);styles=[]
    for index,image in enumerate(planes):
        _,fields,style=model.eval([image.copy()],channels=[0,0],diameter=diameter,compute_masks=False,batch_size=32)
        probs[index]=fields[0][2];flows[:,index]=fields[0][1];styles.append(style[0])
    inverse=np.argsort(order)
    return None,probs.transpose(inverse),flows.transpose((0,)+tuple(inverse+1)),np.asarray(styles)


if __name__=='__main__':
    print('Classic Cellpose runtime ready:',ensure_legacy())
