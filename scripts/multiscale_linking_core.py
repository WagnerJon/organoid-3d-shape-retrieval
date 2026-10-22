"""Offline, disk-backed u-Segment3D experiments. Not used by Atlas.

The fine-grid control follows the pinned upstream fusion/dynamics equations and
binary endpoint occupancy (not a new shape prior). All dense fine volumes live
in .npy files; morphology/filtering/CCL use haloed slabs, dynamics use point batches.
"""
from pathlib import Path
import math
import mmap
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix


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


def components_disk(source, path, connectivity=26, minimum=0, depth=8):
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
    roots = np.array([root(i) for i in range(len(parent))], dtype='uint32')
    totals = np.bincount(roots, weights=counts, minlength=len(parent))
    keep = (roots == np.arange(len(parent))) & (totals >= max(1, minimum))
    keep[0] = False
    compact = np.zeros(len(parent), dtype='uint32')
    compact[keep] = np.arange(1, np.count_nonzero(keep)+1, dtype='uint32')
    lookup = compact[roots]
    for dest, _, _ in slabs(source.shape, depth):
        provisional[dest] = lookup[provisional[dest]]
    provisional.flush()
    return provisional


def fuse_predictions(predictions, output, params, depth=8):
    """Halo-correct fusion for the existing ksize=1, sigma=1 pipeline."""
    from skimage.filters import threshold_multiotsu
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    views = ['xy', 'xz', 'yz']
    probabilities = [Path(predictions)/f'{v}_prob.npy' for v in views]
    flows = [Path(predictions)/f'{v}_flow.npy' for v in views]
    header = np.load(probabilities[0], mmap_mode='r'); shape = header.shape; header._mmap.close()
    if params['combine_cell_probs']['ksize'] != 1 or params['combine_cell_gradients']['ksize'] != 1:
        raise ValueError('This controlled prototype supports the recorded ksize=1 pipeline only')
    if params['combine_cell_gradients']['smooth_sigma'] != 1 or params['combine_cell_probs']['smooth_sigma'] != 0:
        raise ValueError('Unsupported smoothing in this exact-parity prototype')
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
        components = components_disk(binary, output/'foreground-components.npy', 18, post['remove_small_objects'], depth)
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
    return dict(shape=list(shape), threshold=float(threshold), foregroundVoxels=sum(int(binary[s].sum()) for s, _, _ in slabs(shape, depth)))


def area_weights(source, target):
    rows, cols, values = [], [], []
    width = source/target
    for i in range(target):
        lo, hi = i*width, (i+1)*width
        for j in range(math.floor(lo), min(source, math.ceil(hi))):
            rows.append(i); cols.append(j); values.append((min(hi, j+1)-max(lo, j))/width)
    return csr_matrix((np.array(values, dtype='float32'), (rows, cols)), shape=(target, source))


def area_reduce(source, path, target, depth=8):
    weights = [area_weights(a, b) for a, b in zip(source.shape, target)]
    result = disk(path, target)
    for dest, _, _ in slabs(target, depth):
        wz = weights[0][dest]
        lo, hi = int(wz.indices.min()), int(wz.indices.max())+1
        block = wz[:, lo:hi] @ np.asarray(source[lo:hi]).reshape(hi-lo, -1)
        block = block.reshape(dest.stop-dest.start, source.shape[1], source.shape[2])
        block = (weights[2] @ block.reshape(-1, source.shape[2]).T).T.reshape(len(block), source.shape[1], target[2])
        block = (weights[1] @ block.transpose(1, 0, 2).reshape(source.shape[1], -1)).reshape(target[1], len(block), target[2]).transpose(1, 0, 2)
        result[dest] = block
    result.flush()
    return result


def coarsen(fine, output, target):
    fine, output = Path(fine), Path(output); output.mkdir(parents=True, exist_ok=True)
    gradient = np.load(fine/'gradient.npy', mmap_mode='r')
    ratios = np.array(target)/gradient.shape[1:]
    field = disk(output/'gradient.npy', (3,)+tuple(target))
    for axis in range(3):
        part = area_reduce(gradient[axis], output/'reduced.tmp.npy', target)
        # Affine displacement transform: x_c=(x_f+0.5)*ratio-0.5.
        # NO renormalization afterwards: preserve physical travel per iteration.
        field[axis] = part*ratios[axis]
        del part
    field.flush(); (output/'reduced.tmp.npy').unlink()
    fraction = area_reduce(np.load(fine/'binary.npy', mmap_mode='r'), output/'foreground-fraction.npy', target)
    binary = disk(output/'binary.npy', target, 'bool'); binary[:] = fraction >= .5; binary.flush()
    return ratios


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


def trajectories(folder, params, depth=8, batch=150000):
    folder = Path(folder)
    binary = np.load(folder/'binary.npy', mmap_mode='r')
    field = np.load(folder/'gradient.npy', mmap_mode='r')
    endpoints = disk(folder/'endpoints.npy', binary.shape, 'uint64')
    occupied = disk(folder/'endpoint-occupancy.npy', binary.shape, 'float64')
    for dest, _, _ in slabs(binary.shape, depth):
        points = np.argwhere(binary[dest]).astype('float32'); points[:, 0] += dest.start
        for start in range(0, len(points), batch):
            original = points[start:start+batch]
            final = follow(original, field, params)
            indexes = np.ravel_multi_index(final.astype('int32').T, binary.shape)
            endpoints[tuple(original.astype('int32').T)] = indexes
            occupied.reshape(-1)[indexes] = 1.
        if dest.start % 32 == 0:
            print('Trajectories', dest.stop, '/', binary.shape[0], flush=True)
            retire(binary, field, endpoints, occupied)
    endpoints.flush(); occupied.flush()


def cluster_endpoints(folder, sigma=1., depth=8):
    folder = Path(folder)
    votes = gaussian_disk(np.load(folder/'endpoint-occupancy.npy', mmap_mode='r'), folder/'votes.npy', sigma, depth, 'float64')
    mean = sum(float(np.sum(votes[s], dtype='float64')) for s, _, _ in slabs(votes.shape, depth))/votes.size
    active = disk(folder/'attractor-mask.npy', votes.shape, 'bool')
    for dest, _, _ in slabs(votes.shape, depth): active[dest] = votes[dest] > mean
    active.flush()
    centers = components_disk(active, folder/'centers.npy', 26, depth=depth)
    endpoints = np.load(folder/'endpoints.npy', mmap_mode='r')
    foreground = np.load(folder/'binary.npy', mmap_mode='r')
    result = disk(folder/'labels.npy', votes.shape, 'uint32')
    for dest, _, _ in slabs(votes.shape, depth):
        result[dest] = centers.reshape(-1)[endpoints[dest]]*foreground[dest]
    result.flush()
    # Pinned upstream currently does not apply its commented-out min_area filter.
    return dict(attractorThreshold=mean, cells=int(centers.max()))


def recover_ids(fine, coarse, output, depth=8):
    """Joint fine trajectories classified by coarse proposals; one ID per voxel."""
    fine, coarse, output = Path(fine), Path(coarse), Path(output)
    endpoints = np.load(fine/'endpoints.npy', mmap_mode='r')
    binary = np.load(fine/'binary.npy', mmap_mode='r')
    labels = np.load(coarse/'labels.npy', mmap_mode='r')
    ratios = np.array(labels.shape)/np.array(binary.shape)
    # Fill background proposal gaps by the closest coarse foreground cell.
    # This does not create new identities or repair coarse merges.
    nearest = ndi.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
    proposals = labels[tuple(nearest)]
    result = disk(output, binary.shape, 'uint32')
    for dest, _, _ in slabs(binary.shape, depth):
        positions = np.stack(np.unravel_index(endpoints[dest], binary.shape))
        coords = [np.minimum(labels.shape[a]-1, ((positions[a]+.5)*ratios[a]).astype('int32')) for a in range(3)]
        result[dest] = proposals[tuple(coords)]*binary[dest]
    result.flush()
    return result


def refine_neighborhood(binary, field, initial, params, next_id):
    """Joint local fine relinking. Freeze all objects touching the patch edge.

    This uses the original fine foreground/flow, not independent per-cell masks.
    Local endpoint density uses the upstream local mean rule. It is deliberately
    evaluated as a heuristic: local context need not reproduce global clustering.
    """
    import cc3d
    from scipy.optimize import linear_sum_assignment
    points = np.argwhere(binary).astype('float32')
    final = follow(points, field, params)
    occupancy = np.zeros(binary.shape, dtype='float64')
    occupancy[tuple(final.astype('int32').T)] = 1.
    density = ndi.gaussian_filter(occupancy, 1.)
    attractors = cc3d.connected_components(density > density.mean(), connectivity=26)
    proposed = np.zeros(binary.shape, dtype='uint32')
    proposed[tuple(points.astype('int32').T)] = attractors[tuple(final.astype('int32').T)]
    boundary_ids = set(np.unique(np.concatenate([initial[0].ravel(),initial[-1].ravel(),initial[:,0].ravel(),initial[:,-1].ravel(),initial[:,:,0].ravel(),initial[:,:,-1].ravel()])).tolist())
    candidates = np.array(sorted(set(np.unique(initial).tolist())-boundary_ids-{0}),dtype='uint32')
    if not len(candidates):return initial.copy(),next_id,dict(editableObjects=0)
    editable = np.isin(initial,candidates)
    maximum = int(proposed.max())+1
    table = np.bincount((initial.astype('int64')*maximum+proposed).ravel(),minlength=(int(initial.max())+1)*maximum).reshape(-1,maximum)
    overlap = table[candidates,1:]
    rows,cols = linear_sum_assignment(-overlap)
    mapping = np.zeros(maximum,dtype='uint32')
    for row,col in zip(rows,cols):
        if overlap[row,col]>0:mapping[col+1]=candidates[row]
    for label in range(1,maximum):
        # New split identities need a nontrivial exclusive fine-mask support.
        if mapping[label]==0 and np.count_nonzero((proposed==label)&editable)>=1000:
            next_id+=1;mapping[label]=next_id
    replacement=mapping[proposed]
    result=initial.copy();change=editable&(replacement>0)
    result[change]=replacement[change]
    assert np.array_equal(result>0,initial>0)
    return result,next_id,dict(editableObjects=len(candidates),localAttractors=maximum-1,changedVoxels=int(np.count_nonzero(result!=initial)))


def local_correction(fine, initial_path, output_path, params, limit=12, maximum_voxels=3000000):
    """Limited pilot chosen by proposal size outliers, never reference agreement."""
    fine=Path(fine)
    initial=np.load(initial_path,mmap_mode='r')
    result=disk(output_path,initial.shape,'uint32')
    for dest,_,_ in slabs(initial.shape):result[dest]=initial[dest]
    result.flush()
    counts=np.zeros(int(initial.max())+1,dtype='int64')
    for dest,_,_ in slabs(initial.shape):counts+=np.bincount(initial[dest].ravel(),minlength=len(counts))
    positive=counts[1:][counts[1:]>0];median=float(np.median(positive))
    # Fixed volume-outlier heuristic. No tuning against the reference labels.
    large=sorted([i for i in range(1,len(counts)) if counts[i]>2*median],key=lambda i:counts[i],reverse=True)
    small=sorted([i for i in range(1,len(counts)) if 0<counts[i]<median/4],key=lambda i:counts[i])
    # Balance both tails; otherwise tiny fragments occupy the entire pilot.
    ids=large[:limit//2]+small[:limit-limit//2]
    boxes=ndi.find_objects(initial)
    field=np.load(fine/'gradient.npy',mmap_mode='r')
    foreground=np.load(fine/'binary.npy',mmap_mode='r')
    next_id=int(initial.max());records=[]
    for label in ids[:limit]:
        original=boxes[label-1]
        box=tuple(slice(max(0,s.start-24),min(n,s.stop+24)) for s,n in zip(original,initial.shape))
        shape=tuple(s.stop-s.start for s in box)
        if math.prod(shape)>maximum_voxels:
            records.append(dict(label=label,status='skipped',reason='Patch exceeds fixed 3-million-voxel budget',shape=shape));continue
        local_field=np.asarray(field[(slice(None),)+box]).copy()
        current=np.asarray(result[box]).copy()
        updated,next_id,details=refine_neighborhood(np.asarray(foreground[box]),local_field,current,params,next_id)
        result[box]=updated;result.flush()
        records.append(dict(label=label,status='processed',shape=shape,**details))
        print('Local relinking',len(records),'/',min(limit,len(ids)),records[-1],flush=True)
    return dict(selection='At most6 large (>2x median) and6 small (<1/4 median) predicted-volume outliers. No reference labels used to select patches. Joint local endpoint clustering with boundary-touching IDs frozen; maximum3M patch voxels. Local coordinate precision/context and order can differ from global linking.',
                candidateObjects=len(large)+len(small),candidateLargeObjects=len(large),candidateSmallObjects=len(small),processed=sum(r['status']=='processed' for r in records),records=records)
