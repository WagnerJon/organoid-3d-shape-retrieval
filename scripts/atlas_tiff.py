"""Explicit TIFF series/T/C selection with bounded page reads and Z/Y/X exports.

Never merge time with depth or guess unnamed non-spatial dimensions. Selected
frames are independent samples, not tracked cells or biological replicates.
"""
import hashlib
import json
import math
from pathlib import Path

MAX_PAGE_BYTES = 256 * 1024**2
VERSION = 1


def source_stat(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns]


def normalized_axes(axes, shape):
    # Historic plain TIFF stacks label their page dimension Q or I.
    if axes in ('QYX', 'IYX'): return 'ZYX'
    if 'C' not in axes: axes = axes.replace('S', 'C')
    if len(axes) != len(shape) or len(set(axes)) != len(axes):
        raise ValueError('TIFF axes must be unique and match its dimensions')
    if not set('ZYX').issubset(axes) or any(n > 1 and a not in 'TCZYX' for a, n in zip(axes, shape)):
        raise ValueError(f'Unsupported TIFF axes {axes}: declare Z/Y/X and any time/channel axes explicitly')
    return axes


def calibration(tif, series_index, shape):
    from atlas_calibration import assumed_cubic_spacing, ESTIMATED_SOURCE
    spacing, source = None, ESTIMATED_SOURCE
    if tif.ome_metadata:
        import xml.etree.ElementTree as ET
        images = [e for e in ET.fromstring(tif.ome_metadata) if e.tag.split('}')[-1] == 'Image']
        pixels = next((e for e in images[series_index] if e.tag.split('}')[-1] == 'Pixels'), None) if series_index < len(images) else None
        if pixels is not None:
            units = {'µm':1., 'μm':1., 'um':1., 'nm':.001, 'mm':1000., 'cm':10000., 'm':1e6}
            try:
                spacing = [float(pixels.attrib['PhysicalSize'+a])*units[pixels.attrib.get('PhysicalSize'+a+'Unit', 'µm')] for a in 'ZYX']
                source = 'OME physical voxel sizes'
            except (KeyError, ValueError): spacing = None
    if spacing is None:
        info = tif.imagej_metadata or {}
        # Retain the existing ImageJ calibration policy; unsupported units must
        # be confirmed, never silently treated as micrometers.
        if info.get('unit') in ('micron', 'microns', 'um', 'µm', 'μm') and 'spacing' in info:
            try:
                page = tif.series[series_index].pages[0]
                def inverse(key):
                    a,b = page.tags[key].value
                    return b/a
                spacing = [float(info['spacing']), inverse('YResolution'), inverse('XResolution')]
                source = 'ImageJ spacing and resolution tags'
            except (KeyError, ValueError, ZeroDivisionError): spacing = None
    if spacing is not None and not all(math.isfinite(v) and v > 0 for v in spacing): spacing = None
    return (spacing, source) if spacing is not None else (assumed_cubic_spacing(shape), ESTIMATED_SOURCE)


def describe_series(tif, index, path):
    import numpy as np
    series = tif.series[index]
    axes = normalized_axes(series.axes, series.shape)
    shape = [series.shape[axes.index(a)] for a in 'ZYX']
    if min(shape) < 2: raise ValueError('Select a 3D image with at least two voxels along Z, Y and X')
    if series.dtype.kind not in 'uif' or series.dtype.itemsize > 8: raise ValueError('Only real numeric TIFF intensities are supported')
    page = series.pages[0]
    if page is None: raise ValueError('TIFF has missing image planes')
    page_shape = tuple(page.shape)
    # tifffile flattens the leading dimensions into pages in C order. A page
    # can include samples or a volumetric tile; reject unsupported layouts.
    if len(page_shape) > len(series.shape) or tuple(series.shape[-len(page_shape):]) != page_shape:
        raise ValueError('TIFF page layout does not match the declared series dimensions')
    leading = len(series.shape)-len(page_shape)
    if not set('YX').issubset(axes[leading:]): raise ValueError('TIFF pages must contain both X and Y spatial axes')
    if math.prod(series.shape[:leading]) != len(series.pages): raise ValueError('TIFF has missing or unsupported image planes')
    if math.prod(page_shape)*series.dtype.itemsize > MAX_PAGE_BYTES: raise ValueError('A decoded TIFF page exceeds 256 MiB; crop or retile before import')
    spacing, source = calibration(tif, index, shape)
    count_c = series.shape[axes.index('C')] if 'C' in axes else 1
    channels = [f'Channel {c}' for c in range(count_c)]
    if tif.ome_metadata:
        import xml.etree.ElementTree as ET
        images = [e for e in ET.fromstring(tif.ome_metadata) if e.tag.split('}')[-1] == 'Image']
        if index < len(images):
            names = [e.attrib.get('Name') for e in images[index].iter() if e.tag.split('}')[-1] == 'Channel']
            if len(names) == count_c: channels = [n or channels[i] for i,n in enumerate(names)]
    return dict(path=str(path), name=path.name, series=index, seriesName=series.name or f'Series {index}',
                axes=axes, sourceAxes=series.axes, sourceShape=list(series.shape), shapeZYX=shape,
                dtype=str(series.dtype), channels=channels, timepoints=series.shape[axes.index('T')] if 'T' in axes else 1,
                spacing=spacing, calibrationSource=source, sourceStat=source_stat(path), pageBytes=math.prod(page_shape)*series.dtype.itemsize)


def describe(path):
    import tifffile
    path = Path(path).resolve()
    images, ignored = [], []
    with tifffile.TiffFile(path) as tif:
        for index in range(len(tif.series)):
            try: images.append(describe_series(tif,index,path))
            except ValueError as error: ignored.append(f'{path.name} · series {index}: {error}')
    if not images: raise ValueError('; '.join(ignored) or 'TIFF contains no 3D image series')
    return dict(images=images, ignored=ignored)


def selection_entry(image, time, channel):
    if type(time) is not int or not 0 <= time < image['timepoints'] or type(channel) is not int or not 0 <= channel < len(image['channels']):
        raise ValueError('TIFF time point or channel outside the selected series')
    selection = dict(series=image['series'], time=time, channel=channel, axes=image['axes'], sourceAxes=image['sourceAxes'], sourceShape=image['sourceShape'])
    return dict(path=image['path'], name=f"{image['name']} · series {image['series']} · T{time} · {image['channels'][channel]}",
                shape=image['shapeZYX'], axes='ZYX', dtype=image['dtype'], spacing=image['spacing'],
                calibrationSource=image['calibrationSource'], sourceStat=image['sourceStat'], tiffSelection=selection)


def inspect_simple(path):
    import tifffile
    path = Path(path).resolve()
    with tifffile.TiffFile(path) as tif:
        series=tif.series[0]
        if len(tif.series) != 1 or series.axes not in ('ZYX','QYX','IYX') or len(series.shape) != 3 or len(series.pages) != series.shape[0] or tuple(series.pages[0].shape) != tuple(series.shape[1:]):
            raise ValueError(f'{path.name}: select a TIFF series, time point and channel in the import window to obtain a single-channel Z/Y/X stack.')
        image=describe_series(tif,0,path)
        result=selection_entry(image,0,0);result.pop('tiffSelection');result.update(name=path.name,axes=series.axes)
        return result


def inspect_entry(entry):
    import tifffile
    path=Path(entry['path']).resolve()
    if source_stat(path) != entry.get('sourceStat'): raise ValueError('TIFF changed since selection. Add it again.')
    s=entry['tiffSelection']
    with tifffile.TiffFile(path) as tif:
        if type(s.get('series')) is not int or not 0 <= s['series'] < len(tif.series): raise ValueError('TIFF series outside the image')
        image=describe_series(tif,s['series'],path)
    result=selection_entry(image,s['time'],s['channel'])
    if s != result['tiffSelection'] or entry.get('shape') != result['shape']: raise ValueError('TIFF dimensions or axes changed since selection. Add it again.')
    return result


def plane(series, selection, z):
    """Decode a single page, select its channel, and return an owned Y/X plane."""
    import numpy as np
    axes=selection['axes'];coords={'T':selection['time'],'C':selection['channel'],'Z':z}
    page_shape=tuple(series.pages[0].shape);leading=len(series.shape)-len(page_shape)
    prefix=tuple(coords.get(a,0) for a in axes[:leading])
    index=int(np.ravel_multi_index(prefix,series.shape[:leading])) if leading else 0
    page=series.pages[index]
    if page is None: raise ValueError('TIFF has a missing selected image plane')
    if math.prod(page.shape)*series.dtype.itemsize > MAX_PAGE_BYTES: raise ValueError('Decoded TIFF page exceeds the 256 MiB limit')
    data=page.asarray(maxworkers=1)
    indexes=tuple(slice(None) if a in 'YX' else coords.get(a,0) for a in axes[leading:])
    remaining=[a for a in axes[leading:] if a in 'YX']
    if set(remaining) != set('YX'): raise ValueError('TIFF page must contain X and Y axes')
    return np.array(data[indexes].transpose([remaining.index(a) for a in 'YX']),copy=True)


def read_slice(entry, view, index):
    import numpy as np
    import tifffile
    from atlas_slice_test import PLANES
    metadata=inspect_entry(entry)
    if view not in PLANES: raise ValueError('Choose XY, XZ or YZ.')
    axis,dimensions=PLANES[view]
    if type(index) is not int or not 0 <= index < metadata['shape'][axis]: raise ValueError('Slice index is outside the selected TIFF volume')
    shape=[metadata['shape'][i] for i in dimensions]
    if math.prod(shape) > 32_000_000: raise ValueError('This plane is too large for the quick preview (32 million pixels maximum).')
    with tifffile.TiffFile(entry['path']) as tif:
        series=tif.series[entry['tiffSelection']['series']]
        if axis==0: image=plane(series,entry['tiffSelection'],index)
        else:
            image=np.empty(shape,dtype=series.dtype)
            for z in range(metadata['shape'][0]): image[z]=np.take(plane(series,entry['tiffSelection'],z),index,axis=axis-1)
    if not np.isfinite(image).all(): raise ValueError('The selected TIFF slice contains nonfinite intensities')
    return image,dimensions


def materialize(entry, destination, report=None):
    """Export only the chosen frame/channel, with immutable source provenance."""
    import numpy as np
    import tifffile
    from prepare_atlas_dataset import save
    from run_standalone_cellpose import digest
    from atlas_streamed_linking import check_space
    inspect_entry(entry)
    root=Path(destination)
    if root.is_symlink(): raise ValueError('TIFF import cache must not be a symlink')
    identity=hashlib.sha256(json.dumps(dict(version=VERSION,entry=entry),sort_keys=True).encode()).hexdigest()
    owner=root/'import.request.json';done=root/'import.completed.json'
    allowed={'import.request.json','import.request.json.tmp','import.completed.json','import.completed.json.tmp','source.json','source.json.tmp','raw.ome.tif','raw.ome.tif.tmp'}
    if root.exists() and any(root.iterdir()):
        if not owner.is_file() or json.loads(owner.read_text()).get('identity') != identity: raise ValueError('TIFF import cache belongs to another selection; choose a new results folder')
        if any(p.name not in allowed or p.is_symlink() for p in root.iterdir()): raise ValueError('TIFF import cache contains unrelated files')
    source_hash=digest(entry['path'])
    if done.exists():
        saved=json.loads(done.read_text())
        if saved.get('identity') != identity or saved.get('sourceSHA256') != source_hash or digest(root/'raw.ome.tif') != saved.get('sha256'):
            raise ValueError('TIFF source or imported cache changed; choose a new results folder')
    else:
        spacing=entry['spacing']
        if len(spacing)!=3 or not all(math.isfinite(v) and v>0 for v in spacing): raise ValueError('Confirm positive Z/Y/X voxel spacing')
        shape=tuple(entry['shape']);dtype=np.dtype(entry['dtype'])
        check_space(root,math.prod(shape)*dtype.itemsize+shape[0]*16384+512*1024**2,'TIFF selection import')
        root.mkdir(parents=True,exist_ok=True);save(owner,dict(identity=identity))
        temporary=root/'raw.ome.tif.tmp'
        try:
            with tifffile.TiffFile(entry['path']) as tif:
                series=tif.series[entry['tiffSelection']['series']]
                def planes():
                    for z in range(shape[0]):
                        image=plane(series,entry['tiffSelection'],z)
                        if not np.isfinite(image).all(): raise ValueError('TIFF selection contains nonfinite intensities')
                        if report: report(f'Importing TIFF · Z {z+1}/{shape[0]}')
                        yield image
                with tifffile.TiffWriter(temporary,ome=True,bigtiff=True) as writer:
                    writer.write(planes(),shape=shape,dtype=dtype,photometric='minisblack',metadata=dict(axes='ZYX',Name=entry['name'],PhysicalSizeZ=spacing[0],PhysicalSizeY=spacing[1],PhysicalSizeX=spacing[2],PhysicalSizeZUnit='µm',PhysicalSizeYUnit='µm',PhysicalSizeXUnit='µm'))
            inspect_entry(entry)
            if digest(entry['path']) != source_hash: raise ValueError('TIFF changed while importing')
            temporary.replace(root/'raw.ome.tif')
            save(root/'source.json',dict(source=entry,sourceSHA256=source_hash,selection=entry['tiffSelection'],spacingZYXum=spacing,
                 tracking='Time points are independent samples, not biological replicates or linked cell tracks'))
            save(done,dict(identity=identity,sourceSHA256=source_hash,sha256=digest(root/'raw.ome.tif')))
        finally: temporary.unlink(missing_ok=True)
    result=inspect_simple(root/'raw.ome.tif');result.update(name=entry['name'],spacing=entry['spacing'])
    for key in ('labels','labelStat'):
        if entry.get(key) is not None: result[key]=entry[key]
    return result
