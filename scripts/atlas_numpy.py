"""Read-only NumPy Z/Y/X imports and bounded, calibrated TIFF cache creation.

A .npy file has no physical calibration or reliable axis names. Atlas accepts
exactly one numeric 3D array, interprets its axes as Z/Y/X, and requires the
user to confirm voxel spacing. Object/pickled arrays are never loaded.
"""
import hashlib
import json
import math
from pathlib import Path

VERSION = 1
MAX_PLANE_BYTES = 256 * 1024**2


def is_numpy(path):
    return Path(path).suffix.lower() == '.npy'


def stat(path):
    value = Path(path).stat()
    return [value.st_size, value.st_mtime_ns]


def open_array(path):
    import numpy as np
    try:
        array = np.load(path, mmap_mode='r', allow_pickle=False)
    except (ValueError, TypeError, EOFError) as error:
        raise ValueError(f'{Path(path).name}: select a numeric .npy array; object/pickled arrays are not supported.') from error
    if not isinstance(array, np.memmap):
        if hasattr(array, 'close'): array.close()
        raise ValueError('Select a single .npy array, not a NumPy archive.')
    if array.ndim != 3 or min(array.shape) < 2 or array.dtype.kind not in 'uif' or array.dtype.itemsize > 8:
        array._mmap.close()
        raise ValueError(f'{Path(path).name}: NumPy input must be a numeric, noncomplex 3D Z/Y/X array with at least two voxels along each axis; axes are never squeezed or reordered.')
    return array


def inspect(path):
    from atlas_calibration import assumed_cubic_spacing, ESTIMATED_SOURCE
    path = Path(path).resolve()
    array = open_array(path)
    try:
        return dict(path=str(path), name=path.name, shape=list(array.shape), axes='ZYX', dtype=str(array.dtype),
                    spacing=assumed_cubic_spacing(array.shape), calibrationSource='NumPy axes assumed Z/Y/X; '+ESTIMATED_SOURCE,
                    sourceStat=stat(path))
    finally:
        array._mmap.close()


def validate_labels(path, shape):
    import numpy as np
    array = open_array(path)
    try:
        if list(array.shape) != list(shape) or not np.issubdtype(array.dtype, np.integer):
            raise ValueError('Imported labels must be integer instances on the same Z/Y/X grid as the raw stack.')
    finally:
        array._mmap.close()


def check_entry(entry):
    from prepare_atlas_dataset import inspect as inspect_raw
    metadata = inspect_raw(entry['path'])
    if metadata['shape'] != entry.get('shape') or metadata['sourceStat'] != entry.get('sourceStat'):
        raise ValueError('Raw stack changed since selection. Add it again.')
    if entry.get('labels'):
        current = stat(entry['labels'])
        selected = entry.get('labelStat') or []
        if len(selected) != 2 or current[0] != selected[0] or abs(current[1]-selected[1]) > 1000:
            raise ValueError('Imported labels changed since selection. Select the label stack again.')
        if is_numpy(entry['labels']): validate_labels(entry['labels'], metadata['shape'])
    return metadata


def copy_plane(array, indexes):
    """Copy a 2D view without retaining a volume's worth of strided mmap pages."""
    import numpy as np
    view = array[indexes]
    def mapped():
        return np.memmap(array.filename, mode='r', dtype=array.dtype, shape=array.shape,
                         offset=array.offset, order='F' if array.flags.f_contiguous else 'C')
    if view.flags.c_contiguous:
        source = mapped()
        try: return np.array(source[indexes], copy=True, order='C')
        finally: source._mmap.close()
    result = np.empty(view.shape, dtype=view.dtype)
    # Fortran arrays and orthogonal slices can touch one page per pixel. At
    # most 8192 source elements are touched before unmapping those pages.
    # Unmapping is intentional: macOS MADV_DONTNEED is only advisory and did
    # not bound resident pages during an orthogonal real-stack preview.
    for row in range(0, view.shape[0], 32):
        for column in range(0, view.shape[1], 256):
            tile = (slice(row,row+32), slice(column,column+256))
            source = mapped()
            try: result[tile] = source[indexes][tile]
            finally: source._mmap.close()
    return result


def read_slice(entry, plane, index):
    import numpy as np
    from atlas_slice_test import PLANES
    metadata = check_entry(entry)
    if plane not in PLANES: raise ValueError('Choose XY, XZ or YZ.')
    axis, dimensions = PLANES[plane]
    if type(index) is not int or not 0 <= index < metadata['shape'][axis]:
        raise ValueError('Slice index is outside the stack.')
    if math.prod(metadata['shape'][i] for i in dimensions) > 32_000_000:
        raise ValueError('This plane is too large for the quick preview (32 million pixels maximum).')
    array = open_array(entry['path'])
    try:
        indexes = tuple(index if i == axis else slice(None) for i in range(3))
        image = copy_plane(array, indexes)
    finally:
        array._mmap.close()
    if not np.isfinite(image).all(): raise ValueError('The selected slice must contain finite numeric intensities.')
    return image, dimensions


def materialize(entry, destination, report=None):
    """Stream NumPy raw/label volumes to owned TIFF caches, retaining source hashes."""
    import numpy as np
    import tifffile
    from prepare_atlas_dataset import save, inspect as inspect_raw
    from run_standalone_cellpose import digest
    from atlas_streamed_linking import check_space
    metadata = check_entry(entry)
    spacing = entry.get('spacing') or []
    if len(spacing) != 3 or not all(math.isfinite(float(v)) and v > 0 for v in spacing):
        raise ValueError('Confirm positive Z/Y/X spacing in micrometers before importing NumPy data.')
    objects = []
    if is_numpy(entry['path']): objects.append(('raw.ome.tif', entry['path'], False))
    if entry.get('labels') and is_numpy(entry['labels']): objects.append(('labels.ome.tif', entry['labels'], True))
    if not objects: return dict(entry)
    required = 0
    for _, source, _ in objects:
        array = open_array(source)
        try:
            plane_bytes = math.prod(array.shape[1:]) * array.dtype.itemsize
            if plane_bytes > MAX_PLANE_BYTES:
                raise ValueError('A NumPy XY plane exceeds the 256 MiB import limit. Crop or resample it before importing on this Mac.')
            required += array.nbytes + 16 * array.shape[0] * 1024 + 1024**2
        finally: array._mmap.close()
    root = Path(destination)
    if root.is_symlink(): raise ValueError('The NumPy import cache must not be a symlink.')
    identity = hashlib.sha256(json.dumps(dict(version=VERSION, entry=entry), sort_keys=True).encode()).hexdigest()
    owner = root/'import.request.json'
    done = root/'import.completed.json'
    allowed = {'import.request.json', 'import.request.json.tmp', 'import.completed.json', 'import.completed.json.tmp', 'source.json', 'source.json.tmp'}
    allowed.update(name for name, _, _ in objects)
    allowed.update(name+'.tmp' for name, _, _ in objects)
    if root.exists() and any(root.iterdir()):
        if not owner.is_file() or json.loads(owner.read_text()).get('identity') != identity:
            raise ValueError('NumPy import cache belongs to different inputs or settings. Choose a new results folder.')
        if any(p.name not in allowed or p.is_symlink() for p in root.iterdir()):
            raise ValueError('NumPy import cache contains unrelated files; choose a new results folder.')
    if not done.exists():
        # Check before hashing/reading image data, not just before allocating.
        # A fresh atomic replacement needs room alongside any partial caches.
        check_space(root, required+512*1024**2, 'NumPy import')
    source_hashes = {str(Path(source).resolve()): digest(source) for _, source, _ in objects}
    if done.exists():
        saved = json.loads(done.read_text())
        if saved.get('identity') != identity or saved.get('sources') != source_hashes:
            raise ValueError('NumPy source content changed since import. Choose a new results folder.')
        for name, checksum in saved['files'].items():
            if name not in allowed or not (root/name).is_file() or digest(root/name) != checksum:
                raise ValueError('Cached NumPy import checksum mismatch; choose a new results folder.')
    else:
        root.mkdir(parents=True, exist_ok=True)
        save(owner, dict(identity=identity, version=VERSION))
        files = {}
        for filename, source, labels in objects:
            temporary = root/(filename+'.tmp')
            array = open_array(source)
            shape, dtype = array.shape, array.dtype
            def planes():
                for z in range(shape[0]):
                    plane = copy_plane(array, (z, slice(None), slice(None)))
                    if not np.isfinite(plane).all(): raise ValueError('NumPy input contains nonfinite values.')
                    if labels and (dtype.kind not in 'ui' or plane.min() < 0 or plane.max() > 65535):
                        raise ValueError('Atlas instance labels must be integers in 0–65535; background is zero.')
                    if report: report(f'Importing {filename} · Z {z+1}/{shape[0]}')
                    yield plane
            try:
                with tifffile.TiffWriter(temporary, ome=True, bigtiff=True) as writer:
                    writer.write(planes(), shape=shape, dtype=dtype, photometric='minisblack', metadata=dict(
                        axes='ZYX', Name=entry['name'], PhysicalSizeZ=spacing[0], PhysicalSizeY=spacing[1], PhysicalSizeX=spacing[2],
                        PhysicalSizeZUnit='µm', PhysicalSizeYUnit='µm', PhysicalSizeXUnit='µm'))
                temporary.replace(root/filename)
                files[filename] = digest(root/filename)
            finally:
                array._mmap.close()
                temporary.unlink(missing_ok=True)
        check_entry(entry)
        if any(digest(source) != checksum for source, checksum in source_hashes.items()):
            raise ValueError('NumPy source content changed while importing. Choose a new results folder.')
        save(root/'source.json', dict(source=entry, sourcesSHA256=source_hashes, axes='ZYX',
             axisInterpretation='NumPy axes explicitly interpreted as Z/Y/X; no squeezing, transposing, intensity scaling or resampling',
             spacingZYXum=spacing, calibration='User-confirmed physical voxel spacing; NumPy files have no embedded calibration'))
        save(done, dict(identity=identity, sources=source_hashes, files=files))
    result = inspect_raw(root/'raw.ome.tif') if is_numpy(entry['path']) else dict(metadata)
    result.update(name=entry['name'], spacing=spacing)
    if entry.get('labels'):
        label = root/'labels.ome.tif' if is_numpy(entry['labels']) else Path(entry['labels'])
        result.update(labels=str(label), labelStat=stat(label))
    return result
