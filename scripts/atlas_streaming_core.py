"""Atlas fine-grid fusion and linking, derived from the validated offline control.

Preserves the recorded u-Segment3D equations and global instance IDs. Dense
volumes are disk-backed; filters use halos and dynamics use bounded batches.
Coarse linking and experimental local shape refinement are intentionally absent.
"""
from pathlib import Path
import mmap
import math
import numpy as np
from scipy import ndimage as ndi


def notify(progress, label, fraction):
    if progress: progress(label, float(fraction))


def disk(path, shape, dtype='float32'):
    return np.lib.format.open_memmap(Path(path), mode='w+', shape=tuple(shape), dtype=dtype)


def retire(*arrays):
    """Flush shared writes and tell the OS earlier mapped pages may be reclaimed."""
    for array in arrays:
        if isinstance(array, np.memmap) and array._mmap is not None:
            if array.mode != 'r': array.flush()
            if hasattr(array._mmap, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
                array._mmap.madvise(mmap.MADV_DONTNEED)


def read_block(path, region, component=None):
    """Scope input mappings to one copied slab, avoiding retained mapped pages."""
    array = np.load(path, mmap_mode='r')
    result = np.array(array[region] if component is None else array[component, region], copy=True)
    array._mmap.close()
    return result


def remap(array, path):
    array.flush(); array._mmap.close()
    return np.load(path, mmap_mode='r+')


def slabs(shape, depth=8, halo=0):
    for start in range(0, shape[0], depth):
        stop = min(shape[0], start+depth)
        lo, hi = max(0, start-halo), min(shape[0], stop+halo)
        yield slice(start, stop), slice(lo, hi), slice(start-lo, stop-lo)


def gaussian_disk(source, path, sigma, depth=8, dtype='float32'):
    sigma = np.broadcast_to(sigma, (3,)).astype(float)
    result = disk(path, source.shape, dtype)
    halo = int(4*sigma[0]+.5)
    for dest, region, core in slabs(source.shape, depth, halo):
        result[dest] = ndi.gaussian_filter(np.asarray(source[region], dtype=dtype), sigma)[core]
    result.flush()
    return result


def morph_disk(source, path, operation, radius=1, depth=8):
    from skimage import morphology
    footprint = morphology.ball(radius)
    result = disk(path, source.shape, 'bool')
    halo = (2 if operation == 'closing' else 1)*radius
    function = getattr(morphology, 'binary_'+operation)
    for dest, region, core in slabs(source.shape, depth, halo):
        result[dest] = function(np.asarray(source[region]), footprint)[core]
    result.flush()
    return result


def components_disk(source, path, connectivity=26, minimum=0, depth=8, progress=None):
    """Exact global 18/26-connected components with bounded slab workspaces."""
    import cc3d
    if connectivity not in (18, 26):
        raise ValueError('Only 18/26 connectivity is supported')
    provisional = disk(path, source.shape, 'uint32')
    parent = [0]
    counts = [0]

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for dest, _, _ in slabs(source.shape, depth):
        block, number = cc3d.connected_components(np.asarray(source[dest]), connectivity=connectivity, return_N=True)
        offset = len(parent)-1
        histogram = np.bincount(block.ravel(), minlength=number+1)
        parent.extend(range(offset+1, offset+number+1))
        counts.extend(histogram[1:].tolist())
        mapped = block.astype('uint32')
        mapped[block > 0] += offset
        provisional[dest] = mapped
        if dest.start:
            previous, current = provisional[dest.start-1], mapped[0]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if connectivity == 18 and abs(dy)+abs(dx) > 1:
                        continue
                    ya, yb = (slice(max(0, dy), min(source.shape[1], source.shape[1]+dy)),
                              slice(max(0, -dy), min(source.shape[1], source.shape[1]-dy)))
                    xa, xb = (slice(max(0, dx), min(source.shape[2], source.shape[2]+dx)),
                              slice(max(0, -dx), min(source.shape[2], source.shape[2]-dx)))
                    a, b = previous[ya, xa], current[yb, xb]
                    valid = (a > 0) & (b > 0)
                    for i, j in np.unique(np.stack([a[valid], b[valid]], axis=1), axis=0):
                        ri, rj = root(int(i)), root(int(j))
                        if ri != rj:
                            parent[max(ri, rj)] = min(ri, rj)
        notify(progress,'Joining components',.7*dest.stop/source.shape[0])
        if dest.stop % 32 == 0: provisional=remap(provisional,path)
    roots = np.array([root(i) for i in range(len(parent))], dtype='uint32')
    totals = np.bincount(roots, weights=counts, minlength=len(parent))
    keep = (roots == np.arange(len(parent))) & (totals >= max(1, minimum))
    keep[0] = False
    compact = np.zeros(len(parent), dtype='uint32')
    compact[keep] = np.arange(1, np.count_nonzero(keep)+1, dtype='uint32')
    lookup = compact[roots]
    for dest, _, _ in slabs(source.shape, depth):
        provisional[dest] = lookup[provisional[dest]]
        notify(progress,'Resolving global cell IDs',.7+.3*dest.stop/source.shape[0])
        if dest.stop % 32 == 0: provisional=remap(provisional,path)
    provisional.flush()
    return provisional


def fuse_predictions(predictions, output, params, depth=8, progress=None):
    """Halo-correct fusion for the existing ksize=1, sigma=1 pipeline."""
    from skimage.filters import threshold_multiotsu
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    views = ['xy', 'xz', 'yz']
    probabilities = [Path(predictions)/f'{v}_prob.npy' for v in views]
    flows = [Path(predictions)/f'{v}_flow.npy' for v in views]
    header = np.load(probabilities[0], mmap_mode='r'); shape = header.shape; header._mmap.close()
    if params['combine_cell_probs']['ksize'] != 1 or params['combine_cell_gradients']['ksize'] != 1:
        raise ValueError('Memory-saving linking requires neighborhood size 1')
    if params['combine_cell_gradients']['smooth_sigma'] != 1 or params['combine_cell_probs']['smooth_sigma'] != 0:
        raise ValueError('Unsupported smoothing in this exact-parity prototype')
    if not params['combine_cell_probs']['cellpose_prob_mask'] or params['postprocess_binary']['binary_fill_holes'] or params['postprocess_binary']['extra_erode']:
        raise ValueError('Memory-saving linking requires raw Cellpose probabilities without hole filling or extra erosion')
    for path,expected in [(p,shape) for p in probabilities]+[(p,(2,)+shape) for p in flows]:
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        if array.shape!=expected or array.dtype!=np.float32: raise ValueError('Invalid prediction shape or dtype')
        array._mmap.close()
    combined = disk(output/'probability.npy', shape)
    gradients = disk(output/'gradient.npy', (3,)+shape)
    pairs = [((1, 0), (2, 0)), ((0, 0), (2, 1)), ((0, 1), (1, 1))]
    # ksize=1 variance is exactly zero: preserve upstream operation order.
    weight_p = np.float32(1/params['combine_cell_probs']['alpha'])
    weight_g = np.float32(1/params['combine_cell_gradients']['alpha'])
    for dest, region, core in slabs(shape, depth, halo=8):
        sigmoids = [1./(1.+np.exp(-np.clip(read_block(p, dest), -88.72, 88.72))) for p in probabilities]
        combined[dest] = np.sum(np.stack(sigmoids)*weight_p, axis=0)/(3*weight_p)
        fields = []
        for pair in pairs:
            smoothed = [ndi.gaussian_filter(read_block(flows[v], region, component), 1.) for v, component in pair]
            fused = np.sum(np.stack(smoothed)*weight_g, axis=0)/(2*weight_g)
            fields.append(ndi.gaussian_filter(fused, 1.)[core])
        field = np.stack(fields)
        gradients[:, dest] = field/(np.linalg.norm(field, axis=0)[None]+1e-20)
        if not np.isfinite(combined[dest]).all() or not np.isfinite(field).all(): raise ValueError('Nonfinite Cellpose predictions')
        notify(progress,'Combining Cellpose predictions',.65*dest.stop/shape[0])
        if dest.stop % 32 == 0:
            combined = remap(combined, output/'probability.npy')
            gradients = remap(gradients, output/'gradient.npy')
    combined.flush(); gradients.flush(); gradients._mmap.close()
    lo = min(float(combined[s].min()) for s, _, _ in slabs(shape, depth))
    hi = max(float(combined[s].max()) for s, _, _ in slabs(shape, depth))
    edges = np.linspace(lo, hi, 257, dtype='float32')
    histogram = np.zeros(256, dtype='int64')
    for dest, _, _ in slabs(shape, depth):
        histogram += np.histogram(combined[dest], bins=edges)[0]
    controls = params['combine_cell_probs']
    threshold = controls['prob_thresh']
    if threshold is None:
        threshold = float(threshold_multiotsu(classes=controls['threshold_n_levels'], hist=(histogram, (edges[:-1]+edges[1:])/2))[controls['threshold_level']])
        if controls['apply_one_d_p_thresh']:
            threshold = int(threshold*10)/10.
        threshold = max(controls['min_prob_thresh'], threshold)
    binary = disk(output/'binary-threshold.npy', shape, 'bool')
    for dest, _, _ in slabs(shape, depth):
        binary[dest] = combined[dest] >= threshold
    binary.flush()
    post = params['postprocess_binary']
    if post['binary_closing']:
        binary = morph_disk(binary, output/'binary-closed.npy', 'closing', post['binary_closing'], depth)
    if post['remove_small_objects']:
        components = components_disk(binary, output/'foreground-components.npy', 18, post['remove_small_objects'], depth,
                                     lambda text,fraction:notify(progress,'Checking foreground · '+text,.7+.2*fraction))
        binary = disk(output/'binary-filtered.npy', shape, 'bool')
        for dest, _, _ in slabs(shape, depth):
            binary[dest] = components[dest] > 0
        binary.flush(); del components
    if post['binary_dilation']:
        binary = morph_disk(binary, output/'binary.npy', 'dilation', post['binary_dilation'], depth)
    else:
        target = disk(output/'binary.npy', shape, 'bool')
        for dest, _, _ in slabs(shape, depth): target[dest] = binary[dest]
        target.flush(); binary = target
    if post['binary_fill_holes'] or post['extra_erode']:
        raise ValueError('Hole filling/extra erosion are not part of the recorded benchmark')
    notify(progress,'Foreground ready',1)
    return dict(shape=list(shape), threshold=float(threshold), foregroundVoxels=sum(int(binary[s].sum()) for s, _, _ in slabs(shape, depth)))


def follow(points, field, params):
    position = points.astype('float32', copy=True)
    old = np.zeros_like(position)
    flat = field.reshape(3, -1)
    shape = field.shape[1:]
    delta, mu = params['delta'], params['momenta']
    for i in range(params['n_iter']):
        if params.get('interp'):
            direction = np.stack([ndi.map_coordinates(field[a], position.T, order=1, mode='nearest', prefilter=False) for a in range(3)], axis=1)
        else:
            direction = flat[:, np.ravel_multi_index(position.astype('int32').T, shape)].T
        if params.get('renorm_gradient'):
            direction = direction/(np.linalg.norm(direction, axis=1)[:, None]+1e-20)
        position = position+(delta*direction+mu*old)*(1./(delta+mu))*(delta/(1.+i*params['gradient_decay']))
        old = direction.copy()
        for axis, size in enumerate(shape):
            position[:, axis] = np.clip(position[:, axis], 0, size-1)
    return position


def trajectories(folder, params, depth=8, batch=150000, start_z=0, checkpoint=None, progress=None):
    folder = Path(folder)
    binary = np.load(folder/'binary.npy', mmap_mode='r')
    field = np.load(folder/'gradient.npy', mmap_mode='r')
    if params['use_connectivity'] or params['binary_mask_gradient'] or params['track_percent']:
        raise ValueError('Unsupported optional dynamics in memory-saving linking')
    endpoint_dtype='uint32' if math.prod(binary.shape)<=2**32 else 'uint64'
    if start_z:
        endpoints=np.load(folder/'endpoints.npy',mmap_mode='r+')
        occupied=np.load(folder/'endpoint-occupancy.npy',mmap_mode='r+')
        if endpoints.shape!=binary.shape or endpoints.dtype!=np.dtype(endpoint_dtype) or occupied.shape!=binary.shape or occupied.dtype!=np.bool_:
            raise ValueError('Invalid trajectory checkpoint arrays')
    else:
        endpoints = disk(folder/'endpoints.npy', binary.shape, endpoint_dtype)
        occupied = disk(folder/'endpoint-occupancy.npy', binary.shape, 'bool')
    for dest, _, _ in slabs(binary.shape, depth):
        if dest.stop<=start_z: continue
        if dest.start<start_z:raise ValueError('Invalid trajectory checkpoint boundary')
        points = np.argwhere(binary[dest]).astype('float32'); points[:, 0] += dest.start
        for start in range(0, len(points), batch):
            original = points[start:start+batch]
            final = follow(original, field, params)
            indexes = np.ravel_multi_index(final.astype('int32').T, binary.shape)
            endpoints[tuple(original.astype('int32').T)] = indexes
            occupied.reshape(-1)[indexes] = True
        # Occupancy sets are idempotent: repeating a partially written chunk is safe.
        endpoints.flush();occupied.flush()
        if checkpoint:checkpoint(dest.stop)
        notify(progress,'Following fine-grid cell flows',dest.stop/binary.shape[0])
        if dest.start % 32 == 0:
            print('Trajectories', dest.stop, '/', binary.shape[0], flush=True)
            retire(binary, field, endpoints, occupied)
    endpoints.flush(); occupied.flush()


def cluster_endpoints(folder, sigma=1., depth=8, progress=None):
    folder = Path(folder)
    votes = gaussian_disk(np.load(folder/'endpoint-occupancy.npy', mmap_mode='r'), folder/'votes.npy', sigma, depth, 'float64')
    notify(progress,'Finding cell centers',.25)
    mean = sum(float(np.sum(votes[s], dtype='float64')) for s, _, _ in slabs(votes.shape, depth))/votes.size
    active = disk(folder/'attractor-mask.npy', votes.shape, 'bool')
    for dest, _, _ in slabs(votes.shape, depth): active[dest] = votes[dest] > mean
    active.flush()
    centers = components_disk(active, folder/'centers.npy', 26, depth=depth,
                              progress=lambda text,fraction:notify(progress,text,.3+.4*fraction))
    endpoints = np.load(folder/'endpoints.npy', mmap_mode='r')
    foreground = np.load(folder/'binary.npy', mmap_mode='r')
    result = disk(folder/'labels.npy', votes.shape, 'uint32')
    for dest, _, _ in slabs(votes.shape, depth):
        result[dest] = centers.reshape(-1)[endpoints[dest]]*foreground[dest]
        notify(progress,'Writing reconstructed cell labels',.7+.3*dest.stop/votes.shape[0])
        if dest.stop % 32 == 0:result=remap(result,folder/'labels.npy')
    result.flush()
    # Pinned upstream currently does not apply its commented-out min_area filter.
    used=set()
    for dest,_,_ in slabs(votes.shape,depth):used.update(np.unique(result[dest]).tolist())
    return dict(attractorThreshold=mean, cells=len(used-{0}))
