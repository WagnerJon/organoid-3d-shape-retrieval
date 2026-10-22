"""Memory planning and plane-streamed TIFF preprocessing for large uint8/uint16 stacks."""
import math
from pathlib import Path

GIB = 1024**3
# Planning allowance for dense gradients, coordinates and aggregation workspaces.
# This is deliberately conservative; actual peak depends on foreground occupancy.
BYTES_PER_VOXEL = 256
RUNTIME_BYTES = 1.25*GIB


def memory_plan(shape, spacing, scale, total_bytes=None):
    if len(shape) != 3 or len(spacing) != 3 or not all(math.isfinite(float(x)) and x > 0 for x in [*shape,*spacing,scale]):
        raise ValueError('Positive finite Z/Y/X dimensions, spacing and scale required')
    if total_bytes is None:
        import psutil
        total_bytes = psutil.virtual_memory().total
    minimum = min(spacing)
    processed = tuple(math.floor(n*(scale*(s/minimum))) for n,s in zip(shape,spacing))
    voxels = math.prod(processed)
    budget = min(11.5*GIB, .65*total_bytes)
    peak = RUNTIME_BYTES+BYTES_PER_VOXEL*voxels
    valid = min(processed) >= 2 and peak <= budget
    suggested = None
    for step in range(min(100,math.floor(scale*100)),4,-1):
        candidate=step/100
        dims=[math.floor(n*(candidate*(s/minimum))) for n,s in zip(shape,spacing)]
        if min(dims)>=2 and RUNTIME_BYTES+BYTES_PER_VOXEL*math.prod(dims)<=budget:
            suggested=candidate;break
    return dict(processedShape=list(processed),singleArrayGiB=voxels*4/GIB,
                estimatedPeakGiB=peak/GIB,budgetGiB=budget/GIB,fits=valid,suggestedScale=suggested)


def check_memory(shape,spacing,scale,total_bytes=None):
    plan=memory_plan(shape,spacing,scale,total_bytes)
    if not plan['fits']:
        suggestion=(f"Try processing scale {plan['suggestedScale']:.2f} or lower and test the cell diameter again. Choose a new results folder when changing scale."
                    if plan['suggestedScale'] is not None else 'Use a crop or a machine with more memory.')
        raise ValueError(f"Processing scale {scale:g} produces Z/Y/X {' × '.join(map(str,plan['processedShape']))}; one float32 image alone is {plan['singleArrayGiB']:.1f} GiB. "
                         f"The conservative whole-volume linking estimate is {plan['estimatedPeakGiB']:.1f} GiB (planning budget {plan['budgetGiB']:.1f} GiB). "
                         f"Distributed Cellpose batches do not split preprocessing or final 3D linking. {suggestion} Raw data are unchanged.")
    return plan


def positive_median(tif, progress=None):
    """Exact positive-pixel median for integer input, with bounded memory."""
    import numpy as np
    dtype=np.dtype(tif.series[0].dtype)
    if dtype not in (np.dtype('uint8'),np.dtype('uint16')):raise ValueError('Streamed preprocessing requires uint8 or uint16 TIFFs')
    hist=np.zeros(1 << (8*dtype.itemsize),dtype=np.int64)
    pages=tif.series[0].pages
    for index,page in enumerate(pages):
        image=page.asarray()
        if image.ndim!=2:raise ValueError('Expected one XY plane per TIFF page')
        hist+=np.bincount(image.ravel(),minlength=len(hist))
        if progress and (index%32==0 or index==len(pages)-1):progress(f'Reading intensity statistics · plane {index+1}/{len(pages)}')
    hist[0]=0;total=int(hist.sum())
    if not total:return 0
    cumulative=hist.cumsum()
    a=int(np.searchsorted(cumulative,(total-1)//2+1));b=int(np.searchsorted(cumulative,total//2+1))
    # u-Segment3D fills into an integer copy, truncating a fractional median.
    return (a+b)//2


def resize_tiff(tif, zoom, output, fill, progress=None):
    """Separable, global-grid trilinear interpolation; no per-slab rescaling."""
    import numpy as np
    import torch
    from torch.nn.functional import interpolate
    shape=tif.series[0].shape;target=tuple(math.floor(n*z) for n,z in zip(shape,zoom))
    result=np.lib.format.open_memmap(output,mode='w+',dtype='float32',shape=target)
    cache={}
    for z in range(target[0]):
        coordinate=max(0.,(z+.5)/zoom[0]-.5)
        lo=min(shape[0]-1,math.floor(coordinate));hi=min(shape[0]-1,lo+1);weight=coordinate-math.floor(coordinate)
        needed={lo,hi};cache={key:value for key,value in cache.items() if key in needed}
        for key in sorted(needed):
            if key not in cache:
                plane=tif.series[0].pages[key].asarray().astype(np.float32)
                plane[plane<=0]=fill
                with torch.inference_mode():
                    cache[key]=interpolate(torch.from_numpy(plane)[None,None],scale_factor=list(zoom[1:]),mode='bilinear',align_corners=False)[0,0].numpy()
        result[z]=(1-weight)*cache[lo]+weight*cache[hi]
        if progress and (z%16==0 or z==target[0]-1):progress(f'Resampling image · plane {z+1}/{target[0]}')
    result.flush()
    return result


def preprocess(source, folder, params, progress=print):
    """Keep raw TIFF access and resize buffers bounded; normalize the global image."""
    import numpy as np
    import tifffile
    from scipy import ndimage
    from segment3D import gpu
    folder=Path(folder);temporary=folder/'preprocess-resized.tmp.npy'
    destination=folder/'processed.tmp.npy'
    try:
        with tifffile.TiffFile(source) as tif:
            median=positive_median(tif,progress)
            zoom=[params['factor']*s for s in params['voxel_res']]
            resized=resize_tiff(tif,zoom,temporary,median,progress)
        if progress:progress('Applying global background correction and intensity normalization')
        if params['do_bg_correction']:
            # Same global low-resolution background and percentile operation as
            # the established Mac (torch CPU) preprocessing path.
            corrected=gpu.bg_normalize_torch(resized,bg_ds=params['bg_ds'],bg_sigma=params['bg_sigma'])
        else:corrected=resized
        normalized=gpu.num_normalize(corrected,pmin=params['normalize_min'],pmax=params['normalize_max'],clip=True)
        del corrected,resized
        shape=normalized.shape
        result=np.lib.format.open_memmap(destination,mode='w+',dtype='float32',shape=shape)
        # Median with an exact one-plane halo; only real volume edges reflect.
        depth=max(1,min(16,(32*1024**2)//max(1,shape[1]*shape[2]*4)))
        for start in range(0,shape[0],depth):
            stop=min(start+depth,shape[0]);lo=max(0,start-1);hi=min(shape[0],stop+1)
            block=ndimage.median_filter(normalized[lo:hi],size=3)
            result[start:stop]=block[start-lo:stop-lo]
        result.flush();del result,normalized
        destination.replace(folder/'processed.npy')
        return dict(processedShape=list(shape),mode='Plane-streamed TIFF; global-grid trilinear resize, global normalization, halo-correct median',zeroFillPositiveMedian=median)
    finally:
        temporary.unlink(missing_ok=True);destination.unlink(missing_ok=True)
