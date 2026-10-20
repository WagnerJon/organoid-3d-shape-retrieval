"""PanSeg workbench settings, boundary caching and native-grid output.

Uses the same functional APIs as PanSeg's napari tasks. A prediction cache is
valid only for the same raw fingerprint, calibration and inference settings.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage.transform import resize
import tifffile

DEFAULTS = dict(modelSource='zoo',model='generic_confocal_3D_unet',modelID='',configPath='',weightsPath='',
                rescaleZ=1.,rescaleY=1.,rescaleX=1.,gaussianSigma=0.,medianRadius=0,
                automaticPatch=True,patchZ=80,patchY=128,patchX=128,haloMode='tiled',haloZ=44,haloY=44,haloX=44,
                singleBatch=True,device='auto',gpu=True,boundaryChannel=0,
                threshold=.5,minSize=100,sigmaSeeds=1.,sigmaWeights=2.,alpha=1.,stacked=False,
                anisotropy=True,nonmax=False,blockwise=False,threads=4,clustering='multicut',beta=.5,
                postMinSize=100,linkage='average',foreground=0.)
INFERENCE_KEYS = ['modelSource','model','modelID','configPath','weightsPath','rescaleZ','rescaleY','rescaleX',
                  'gaussianSigma','medianRadius','automaticPatch','patchZ','patchY','patchX','haloMode',
                  'haloZ','haloY','haloX','singleBatch','device','gpu','boundaryChannel']


def normalized_settings(settings):
    result=dict(DEFAULTS,**{k:v for k,v in settings.items() if k in DEFAULTS})
    # Keep old comparisons usable while migrating their basic settings.
    if 'clustering' not in settings and 'multicut' in settings:
        result['clustering']='multicut' if settings['multicut'] else 'watershed'
        result['postMinSize']=settings.get('minSize',100)
    for key in ('rescaleZ','rescaleY','rescaleX'):
        if not np.isfinite(result[key]) or not .05 <= result[key] <= 8:
            raise ValueError('Resampling factors must be finite and between 0.05 and 8.')
    for key in ('threshold','alpha','beta','foreground'):
        if not np.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f'{key} must be between 0 and 1.')
    for key in ('automaticPatch','singleBatch','gpu','stacked','anisotropy','nonmax','blockwise'):
        if not isinstance(result[key],bool):raise ValueError(f'{key} must be a boolean.')
    if not 0 < result['beta'] < 1:
        raise ValueError('Clustering beta must be strictly between 0 and 1.')
    if not 0 < result['threshold'] < 1:
        raise ValueError('Watershed threshold must be strictly between 0 and 1.')
    for key in ('gaussianSigma','sigmaSeeds','sigmaWeights'):
        if not np.isfinite(result[key]) or not 0 <= result[key] <= 100:
            raise ValueError(f'{key} must be finite and between 0 and 100.')
    bounds=dict(medianRadius=(0,10),boundaryChannel=(0,64),minSize=(1,100000000),postMinSize=(0,100000000),threads=(1,64),
                patchZ=(16,2048),patchY=(16,2048),patchX=(16,2048),haloZ=(0,512),haloY=(0,512),haloX=(0,512))
    for key,(low,high) in bounds.items():
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value != int(value) or not low <= value <= high:
            raise ValueError(f'{key} must be an integer from {low} to {high}.')
        result[key]=int(value)
    for key,choices in dict(modelSource=['zoo','bioimageio','custom'],device=['auto','cpu','mps','cuda'],haloMode=['tiled','model','manual'],
                            clustering=['watershed','multicut','gasp','mutex_ws'],linkage=['average','sum','max','min','abs_max']).items():
        if result[key] not in choices:
            raise ValueError(f'Unsupported {key}: {result[key]}')
    if result['modelSource']=='zoo' and not result['model'].strip():
        raise ValueError('Select a model from the PanSeg zoo.')
    if result['modelSource']=='bioimageio' and not result['modelID'].strip():
        raise ValueError('Enter a compatible BioImage.IO U-Net model ID.')
    if result['modelSource']=='custom' and not result['configPath'].strip():
        raise ValueError('Select a custom U-Net YAML configuration.')
    return result


def prediction_key(source, settings, fingerprint):
    s=normalized_settings(settings)
    identity={k:s[k] for k in INFERENCE_KEYS}
    # File contents matter, even when a local model path remains unchanged.
    if s['modelSource']=='custom':
        for k in ('configPath','weightsPath'):
            if s[k]: identity[k+'SHA256']=fingerprint(s[k])
    identity.update(source=source['fingerprint'],spacing=source['spacing'],shape=source['nativeShape'],protocol=1)
    return hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()


def cached_boundary(result_path, source, settings, fingerprint):
    result=json.loads(Path(result_path).read_text())
    key=prediction_key(source,settings,fingerprint)
    info=result.get('prediction',{})
    if result.get('source') != source['fingerprint'] or result.get('spacing') != source['spacing'] or info.get('key') != key:
        raise ValueError('Boundary prediction does not match this raw stack, calibration or inference settings. Run Predict + segment again.')
    expected=normalized_settings(settings)
    expected_shape=np.maximum(1,np.rint(np.array(source['nativeShape'])*[expected['rescaleZ'],expected['rescaleY'],expected['rescaleX']])).astype(int)
    expected_spacing=np.array(source['spacing'])*np.array(source['nativeShape'])/expected_shape
    if info.get('shape') != expected_shape.tolist() or not np.allclose(info.get('spacing',[]),expected_spacing):
        raise ValueError('Saved boundary calibration or working grid differs from the requested inference.')
    path=Path(result['folder'])/'boundaries.tif'
    if fingerprint(path) != info.get('hash'):
        raise ValueError('Saved boundary data changed. Run Predict + segment again.')
    boundary=tifffile.imread(path)
    if list(boundary.shape) != info.get('shape') or not np.isfinite(boundary).all():
        raise ValueError('Invalid saved boundary prediction.')
    return boundary.astype(np.float32),info


def segment_boundaries(boundary, source, settings, spacing, progress, api=None):
    if api is None:
        from panseg.functionals.segmentation.segmentation import dt_watershed, multicut, gasp, mutex_ws
        api=dict(watershed=dt_watershed,multicut=multicut,gasp=gasp,mutex_ws=mutex_ws)
    s=normalized_settings(settings)
    progress('PanSeg: distance-transform watershed',.72)
    distance_spacing=np.asarray(spacing[1:] if s['stacked'] else spacing)
    labels=api['watershed'](boundary,threshold=s['threshold'],min_size=s['minSize'],sigma_seeds=s['sigmaSeeds'],
                  sigma_weights=s['sigmaWeights'],alpha=s['alpha'],stacked=s['stacked'],
                  pixel_pitch=tuple(distance_spacing/distance_spacing.min()) if s['anisotropy'] else None,
                  apply_nonmax_suppression=s['nonmax'],blockwise=s['blockwise'],n_threads=s['threads'])
    mode=s['clustering']
    if mode != 'watershed':
        progress('PanSeg: '+mode.replace('_',' ')+' clustering',.82)
        kwargs=dict(superpixels=labels,beta=s['beta'],post_minsize=s['postMinSize'])
        if mode != 'multicut':kwargs['n_threads']=s['threads']
        if mode=='gasp':kwargs['gasp_linkage_criteria']=s['linkage']
        # Graph labels use zero as a valid instance index. Reserve it for background.
        labels=api[mode](boundary,**kwargs).astype(np.uint32)+1
    return resize(labels,source['nativeShape'],order=0,preserve_range=True,anti_aliasing=False).astype(np.uint32)


def run_panseg(raw, source, supplied, folder, progress, tracker, fingerprint, reuse=None):
    s=normalized_settings(supplied)
    progress('PanSeg: loading libraries and model',.05)
    key=prediction_key(source,s,fingerprint)
    shape=np.maximum(1,np.rint(np.array(raw.shape)*[s['rescaleZ'],s['rescaleY'],s['rescaleX']])).astype(int)
    if min(shape)<16 or np.prod(shape)>128000000:
        raise ValueError('Working volume must have at least 16 voxels per axis and at most 128 million voxels. Adjust resampling.')
    spacing=(np.array(source['spacing'])*np.array(raw.shape)/shape).tolist()
    if reuse:
        boundary,info=cached_boundary(reuse,source,s,fingerprint)
        progress('PanSeg: reusing verified boundary prediction',.7)
        device=info.get('device','cached')
    else:
        import torch
        from panseg.functionals.prediction.prediction import unet_prediction
        device=s['device']
        if device=='auto':
            device='cpu'
            if s['gpu']:
                if torch.cuda.is_available(): device='cuda'
                elif torch.backends.mps.is_available():
                    try:
                        with torch.no_grad():
                            probe=torch.nn.Conv3d(1,2,3,padding=1).to('mps')
                            probe(torch.zeros((1,1,16,16,16),device='mps'));torch.mps.synchronize()
                        del probe
                        device='mps'
                    except (RuntimeError,NotImplementedError):
                        print('MPS Conv3d unavailable; using CPU.',flush=True)
        if device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS is unavailable. Choose Auto or CPU.')
        if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is unavailable. Choose Auto or CPU.')
        torch.set_num_threads(s['threads'])
        progress('PanSeg: resampling and smoothing raw volume',.07)
        processed=resize(raw.astype(np.float32),tuple(shape),preserve_range=True,anti_aliasing=True) if list(shape)!=list(raw.shape) else raw.astype(np.float32)
        if s['gaussianSigma']>0:processed=ndimage.gaussian_filter(processed,s['gaussianSigma'])
        if s['medianRadius']>0:processed=ndimage.median_filter(processed,size=2*s['medianRadius']+1)
        if s['automaticPatch']:patch=tuple(min(n,limit) for n,limit in zip(shape,(80,128,128)))
        else:
            patch=(s['patchZ'],s['patchY'],s['patchX'])
            if any(p>n for p,n in zip(patch,shape)):raise ValueError('Manual patch dimensions must fit the resampled volume.')
        halo=None if s['haloMode']=='model' else tuple(s[k] for k in ('haloZ','haloY','haloX')) if s['haloMode']=='manual' else tuple(44 if p<n else 0 for p,n in zip(patch,shape))
        progress('PanSeg: loading weights and predicting boundaries on '+device,.1)
        output=unet_prediction(processed,'ZYX',s['model'] if s['modelSource']=='zoo' else None,
                s['modelID'] if s['modelSource']=='bioimageio' else None,patch=patch,patch_halo=halo,
                device=device,single_batch_mode=s['singleBatch'],tracker=tracker(),
                config_path=Path(s['configPath']) if s['modelSource']=='custom' else None,
                model_weights_path=Path(s['weightsPath']) if s['modelSource']=='custom' and s['weightsPath'] else None)
        if output.ndim==4:
            if s['boundaryChannel']>=output.shape[0]:raise ValueError('Selected boundary channel does not exist in model output.')
            boundary=output[s['boundaryChannel']]
        else:
            if s['boundaryChannel']!=0:raise ValueError('This model has only one output channel.')
            boundary=output
        if list(boundary.shape)!=list(shape) or not np.isfinite(boundary).all() or boundary.min()<0 or boundary.max()>1:
            raise ValueError('Model output must be a finite 3D boundary probability map in [0,1].')
        info=dict(key=key,shape=list(map(int,shape)),spacing=spacing,device=device,patch=list(map(int,patch)),halo=list(map(int,halo)) if halo else None)
    tifffile.imwrite(folder/'boundaries.tif',boundary.astype(np.float32),photometric='minisblack',metadata={'axes':'ZYX'})
    info=dict(info,hash=fingerprint(folder/'boundaries.tif'),reusedFrom=str(reuse) if reuse else None)
    preview=resize(boundary,source['shape'],preserve_range=True,anti_aliasing=True)
    (folder/'boundaries.u8').write_bytes((np.clip(preview,0,1)*255).astype(np.uint8).tobytes())
    labels=segment_boundaries(boundary,source,s,spacing,progress)
    if s['foreground']>0:
        progress('PanSeg: masking image-derived background',.9)
        normalized=np.clip((raw.astype(np.float32)-source['low'])/max(source['high']-source['low'],1e-6),0,1)
        labels[~ndimage.binary_fill_holes(normalized>s['foreground'])]=0
    provenance='PanSeg 2.1.2 / a0c7123: '+s['modelSource']+' U-Net boundary inference + watershed + '+s['clustering']+'. Model domain and preprocessing are recorded in settings. No annotation inputs.'
    return labels,dict(s,runtime=supplied.get('runtime',{}),workerSHA256=supplied.get('workerSHA256','')),provenance,info
