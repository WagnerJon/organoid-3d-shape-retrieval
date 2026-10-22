"""Memory-bounded single-plane Cellpose settings preview for dataset preparation.

Uses raw intensities only. Slice-local preprocessing approximates volume
preprocessing; 2D masks are never presented as u-Segment3D reconstruction.
"""
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CELLPOSE_LOCAL_MODELS_PATH', str(ROOT/'.cache/cellpose'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.cache/numba'))
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

PLANES = {'xy': (0, (1, 2)), 'xz': (1, (0, 2)), 'yz': (2, (0, 1))}


def read_slice(entry, plane, index):
    import numpy as np
    import tifffile
    from prepare_atlas_dataset import inspect
    if entry.get('omeZarr'):
        from atlas_ome_zarr import read_slice as zarr_slice
        if plane not in PLANES:raise ValueError('Choose XY, XZ or YZ.')
        return zarr_slice(entry,plane,index)
    metadata = inspect(entry['path'])
    if metadata['sourceStat'] != entry['sourceStat']:
        raise ValueError('Raw stack changed since selection. Add it again.')
    if plane not in PLANES:
        raise ValueError('Choose XY, XZ or YZ.')
    axis, dimensions = PLANES[plane]
    if not isinstance(index, int) or not 0 <= index < metadata['shape'][axis]:
        raise ValueError('Slice index is outside the stack.')
    shape = [metadata['shape'][i] for i in dimensions]
    if np.prod(shape) > 32_000_000:
        raise ValueError('This plane is too large for the quick preview (32 million pixels maximum).')
    with tifffile.TiffFile(entry['path']) as tif:
        if axis == 0:
            image = tif.pages[index].asarray()
        else:
            # XZ/YZ stream one native XY page at a time, retaining just its row/column.
            image = np.empty(shape, dtype=tif.series[0].dtype)
            for z, page in enumerate(tif.pages):
                image[z] = np.take(page.asarray(), index, axis=axis-1)
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError('The selected slice must contain finite numeric intensities.')
    return image, dimensions


def preprocess(image, spacing, dimensions, settings):
    import numpy as np
    from scipy.ndimage import gaussian_filter, median_filter
    from skimage.transform import resize
    if len(spacing) != 3 or not all(np.isfinite(v) and v > 0 for v in spacing):
        raise ValueError('Confirm positive Z/Y/X voxel spacing before testing.')
    scale, diameter = float(settings['scale']), float(settings['diameter'])
    if not .05 <= scale <= 1 or not 2 <= diameter <= 500:
        raise ValueError('Scale must be 0.05–1 and diameter 2–500 processed pixels.')
    target = np.maximum(2, np.rint(np.array(image.shape)*scale*np.array(spacing)[list(dimensions)]/min(spacing))).astype(int)
    if np.prod(target) > 8_000_000:
        raise ValueError('The rescaled plane is too large for a quick test. Reduce the processing scale.')
    processed = resize(image.astype(np.float32), tuple(target), order=1, preserve_range=True, anti_aliasing=False)
    if settings['background']:
        small = resize(processed, tuple(np.maximum(1, target//8)), preserve_range=True)
        background = resize(gaussian_filter(small, sigma=5), processed.shape, preserve_range=True)
        processed = processed * (background.mean()/(background+.1))
    lo, hi = np.percentile(processed, [2, 99.8])
    processed = np.clip((processed-lo)/max(float(hi-lo), 1e-20), 0, 1)
    return median_filter(processed, size=3).astype(np.float32)


def run(request, output, segment=False, model_factory=None):
    import numpy as np
    import tifffile
    from PIL import Image
    from skimage.transform import resize
    from skimage.segmentation import find_boundaries
    from prepare_atlas_dataset import save, event
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entry, plane, index = request['input'], request['plane'], request['index']
    event('Reading selected slice', .05)
    image, dimensions = read_slice(entry, plane, index)
    spacing = entry['spacing']
    if len(spacing) != 3 or not all(np.isfinite(v) and v > 0 for v in spacing):
        raise ValueError('Confirm positive Z/Y/X voxel spacing before testing.')
    # Match physical aspect ratios in both raw and label previews.
    aspect = np.array(image.shape)*np.array(spacing)[list(dimensions)]
    display_shape = np.maximum(2, np.rint(aspect*min(1536/aspect.max(), 1/min(spacing)))).astype(int)
    small = resize(image, tuple(display_shape), preserve_range=True, anti_aliasing=True)
    lo, hi = np.percentile(small, [1,99.8])
    raw = np.clip((small-lo)/max(float(hi-lo), 1e-20)*255, 0,255).astype(np.uint8)
    Image.fromarray(raw).save(output/'raw.png')
    result = dict(path=entry['path'], plane=plane, index=index, nativeShape=list(image.shape),
                  spacing=spacing, settings=request.get('settings'), cells=None, device=None, seconds=None,
                  note='2D Cellpose preview with slice-local preprocessing; no cross-slice linking or 3D reconstruction.')
    if segment:
        started = time.monotonic()
        settings = request['settings']
        if settings['model'] not in ('cpsam_v2','cpdino','cpdino-vitb'):
            raise ValueError('Unsupported Cellpose model.')
        event('Preparing slice', .15)
        processed = preprocess(image, spacing, dimensions, settings)
        event('Loading Cellpose model', .3)
        if model_factory is None:
            from cellpose.models import CellposeModel
            model_factory = CellposeModel
        model = model_factory(pretrained_model=settings['model'], gpu=bool(settings['gpu']))
        event('Segmenting this slice', .5)
        masks, _, _ = model.eval(processed, diameter=float(settings['diameter']), do_3D=False, compute_masks=True)
        if masks.shape != processed.shape or not np.issubdtype(masks.dtype, np.integer) or np.any(masks < 0):
            raise ValueError('Cellpose returned invalid 2D labels.')
        native_labels = resize(masks, image.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)
        tifffile.imwrite(output/'labels.tif', native_labels, photometric='minisblack')
        labels = resize(masks, tuple(display_shape), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)
        rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
        for label in np.unique(labels):
            if label:
                color = np.random.default_rng(int(label)).integers(70, 256, size=3, dtype=np.uint8)
                rgba[labels==label,:3] = color
                rgba[labels==label,3] = 170
        rgba[find_boundaries(labels, mode='inner'),3] = 255
        Image.fromarray(rgba).save(output/'labels.png')
        result.update(cells=int(np.count_nonzero(np.unique(masks))), device=str(model.device),
                      processedShape=list(processed.shape), seconds=time.monotonic()-started)
    save(output/'slice.json', result)
    event('Slice test ready' if segment else 'Raw slice ready', 1)
    return result
