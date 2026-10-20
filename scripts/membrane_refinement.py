"""Joint, bounded level-set refinement against a PSF-rendered membrane image.

Uses the acquired native planes only. Instance identities and deep interior
anchors are retained; this is boundary refinement, not automatic cell relinking.
"""
import math
import numpy as np
from scipy import ndimage as ndi
from skimage import measure


def options(settings):
    defaults = dict(iterations=30, membraneWeight=1., shapeWeight=.5, smoothness=.5,
                    ellipsoidWeight=0., maxDisplacement=.5, controlSpacing=2.,
                    psfZ=.7, psfXY=.25, learningRate=.03, maxVolumeChange=.1,
                    maxRoughnessIncrease=.05,maxChangedFraction=.1,device='auto')
    limits = dict(iterations=(1,500),membraneWeight=(.01,100),shapeWeight=(0,10),smoothness=(0,10),
                  ellipsoidWeight=(0,10),maxDisplacement=(.05,10),controlSpacing=(.1,10),
                  psfZ=(0,10),psfXY=(0,10),learningRate=(.001,1),maxVolumeChange=(.01,2),
                  maxRoughnessIncrease=(0,1),maxChangedFraction=(.001,2))
    result = {}
    for key,(low,high) in limits.items():
        value = float(settings.get(key,defaults[key]))
        if not math.isfinite(value) or not low <= value <= high or (key == 'iterations' and value != int(value)):
            raise ValueError(f'Invalid refinement {key}: expected {low}–{high}.')
        result[key] = int(value) if key == 'iterations' else value
    result['device'] = settings.get('device','auto')
    result['preserveConnectivity'] = settings.get('preserveConnectivity',True)
    result['preserveTopology'] = settings.get('preserveTopology',True)
    result['algorithmVersion'] = 2
    if not isinstance(result['preserveTopology'],bool): raise ValueError('preserveTopology must be a boolean.')
    if not isinstance(result['preserveConnectivity'],bool): raise ValueError('preserveConnectivity must be a boolean.')
    if result['device'] not in ('auto','cpu','mps','cuda:0'):
        raise ValueError('Unknown refinement device.')
    return result


def shape_quality(mask, spacing):
    """Topology plus physical surface area normalized by volume^(2/3)."""
    points = np.where(mask)
    if not len(points[0]): return dict(components=0,cavities=0,handles=0,roughness=0.)
    crop = np.pad(mask[tuple(slice(int(p.min()),int(p.max())+1) for p in points)],1)
    components = ndi.label(crop,structure=ndi.generate_binary_structure(3,3))[1]
    cavities = ndi.label(ndi.binary_fill_holes(crop) & ~crop)[1]
    euler = int(measure.euler_number(crop,connectivity=3))
    vertices,faces,_,_ = measure.marching_cubes(crop,.5,spacing=spacing)
    area = float(measure.mesh_surface_area(vertices,faces))
    volume = float(mask.sum()*np.prod(spacing))
    return dict(components=components,cavities=cavities,handles=max(0,components+cavities-euler),
                roughness=area/max(volume**(2/3),1e-9))


def quality_failures(original, candidate, spacing, opts, original_quality=None):
    before = original_quality or shape_quality(original,spacing)
    after = shape_quality(candidate,spacing)
    reasons = []
    if opts['preserveTopology'] and any(after[key] > before[key] for key in ('components','cavities','handles')):
        reasons.append('new topology defects')
    if after['roughness'] > before['roughness']*(1+opts['maxRoughnessIncrease'])+1e-6:
        reasons.append('excess surface roughness')
    if np.count_nonzero(original != candidate)/max(1,original.sum()) > opts['maxChangedFraction']:
        reasons.append('excess boundary changes')
    return reasons,after


def blur(volume, sigma, spacing):
    import torch
    import torch.nn.functional as F
    result = volume[None,None]
    for axis,(physical,step) in enumerate(zip(sigma,spacing)):
        width = physical/step
        if width < .05: continue
        radius = min(int(math.ceil(width*3)),volume.shape[axis]-1)
        positions = torch.arange(-radius,radius+1,device=volume.device,dtype=volume.dtype)
        kernel = torch.exp(-.5*(positions/width)**2); kernel /= kernel.sum()
        shape = [1,1,1,1,1]; shape[axis+2] = len(kernel)
        padding = [0,0,0]; padding[axis] = radius
        result = F.conv3d(result,kernel.reshape(shape),padding=tuple(padding))
    return result[0,0]


def refine(raw, labels, spacing, settings, progress=lambda *args:None):
    import torch
    import torch.nn.functional as F
    from focus3d_explorer import resolve_device
    opts = options(settings)
    spacing = np.asarray(spacing,dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Refinement requires positive physical ZYX voxel spacing.')
    if raw.shape != labels.shape or raw.ndim != 3 or min(raw.shape) < 2 or not np.isfinite(raw).all():
        raise ValueError('Refinement requires finite raw intensities and matching native 3D labels.')
    if not np.issubdtype(labels.dtype,np.integer) or labels.min() < 0 or labels.max() > np.iinfo(np.uint32).max:
        raise ValueError('Refinement labels must be nonnegative integers.')
    ids = np.unique(labels); ids = ids[ids > 0]
    if not len(ids): raise ValueError('The selected segmentation has no cells to refine.')
    low,high = np.percentile(raw,[1,99.8])
    if high <= low: raise ValueError('No intensity contrast for membrane fitting.')
    positions = np.where(labels > 0)
    margin = np.ceil((opts['maxDisplacement'] + 3*max(opts['psfZ'],opts['psfXY'])) / spacing).astype(int)+1
    starts = np.maximum(0,[p.min() for p in positions]-margin)
    ends = np.minimum(raw.shape,np.array([p.max()+1 for p in positions])+margin)
    slices = tuple(slice(int(a),int(b)) for a,b in zip(starts,ends))
    initial = labels[slices]
    shape = initial.shape
    if len(ids)*np.prod(shape) > 32_000_000:
        raise ValueError('Refinement exceeds the interactive memory limit (32M cell voxels). Use a smaller stack/ROI.')
    device = resolve_device(opts['device']); opts['resolvedDevice'] = device
    progress('Refinement: initializing physical cell surfaces',.05)
    tau = float(min(spacing)*.6)
    fields,anchors,ellipsoids = [],[],[]
    structure = ndi.generate_binary_structure(3,3)
    for index,label in enumerate(ids):
        mask = initial == label
        inside = ndi.distance_transform_edt(mask,sampling=spacing)
        outside = ndi.distance_transform_edt(~mask,sampling=spacing)
        signed = inside-outside
        signed = np.sign(signed)*np.maximum(0,np.abs(signed)-.5*min(spacing))
        fields.append(np.clip(signed,-opts['maxDisplacement']-10*tau,opts['maxDisplacement']+10*tau).astype(np.float32))
        components,count = ndi.label(mask,structure=structure)
        anchors.append([np.unravel_index(np.where(components == component,inside,-1).argmax(),shape) for component in range(1,count+1)])
        if opts['ellipsoidWeight']:
            coords = np.argwhere(mask)*spacing
            center = coords.mean(0); covariance = np.cov(coords,rowvar=False) if len(coords)>3 else np.diag(spacing**2)
            vals,vectors = np.linalg.eigh(covariance)
            radii = np.maximum(np.sqrt(np.maximum(vals,0)*5),spacing)
            grids = np.stack(np.meshgrid(*[np.arange(n)*s for n,s in zip(shape,spacing)],indexing='ij'),axis=-1)
            q = ((grids-center)@vectors / radii)**2
            ellipsoids.append(((1-np.sqrt(q.sum(-1)))*radii.min()).astype(np.float32))
        progress(f'Refinement: surface {index+1}/{len(ids)}',.05+.1*(index+1)/len(ids))
    base = torch.tensor(np.stack(fields),device=device)
    target = torch.tensor(np.clip((raw[slices].astype(np.float32)-low)/(high-low),0,1),device=device)
    weights = target+.05; weights = weights/weights.mean()
    control_shape = tuple(min(n,max(2,int(math.ceil((n-1)*s/opts['controlSpacing']))+1)) for n,s in zip(shape,spacing))
    displacement = torch.nn.Parameter(torch.zeros((len(ids),3,*control_shape),device=device))
    z,y,x = torch.meshgrid(*[torch.linspace(-1,1,n,device=device) for n in shape],indexing='ij')
    reference_grid = torch.stack((x,y,z),dim=-1)[None]
    original_quality = {int(label):shape_quality(initial == label,spacing) for label in ids}
    amplitude = torch.nn.Parameter(torch.tensor(.5,device=device))
    background = torch.nn.Parameter(torch.tensor(-4.,device=device))
    optimizer = torch.optim.Adam([displacement,amplitude,background],lr=opts['learningRate'])
    band = base.abs() < opts['maxDisplacement']+2*tau
    ellipsoid = torch.tensor(np.stack(ellipsoids),device=device) if ellipsoids else None
    original_volumes = torch.tensor([(initial == label).sum() for label in ids],device=device,dtype=torch.float32)
    def objective():
        coarse = displacement.tanh()*(opts['maxDisplacement']/math.sqrt(3))
        # Bound the displacement Jacobian to prevent folding the shape warp.
        bounds = []
        smooth = torch.tensor(0.,device=device)
        for axis in (2,3,4):
            step = (shape[axis-2]-1)*spacing[axis-2]/(control_shape[axis-2]-1)
            derivative = torch.diff(coarse,dim=axis)/step
            bounds.extend(derivative[:,component].abs().amax() for component in range(3))
            smooth = smooth+derivative.square().mean()
        bound = torch.sqrt(sum(value.square() for value in bounds)+1e-9)
        coarse = coarse*torch.clamp(.45/bound,max=1.)
        warp = F.interpolate(coarse,size=shape,mode='trilinear',align_corners=True)
        grid = reference_grid+torch.stack([warp[:,i]*2/((shape[i]-1)*spacing[i]) for i in (2,1,0)],dim=-1)
        phi = F.grid_sample(base[:,None],grid,mode='bilinear',padding_mode='border',align_corners=True)[:,0]
        delta = phi-base
        occupancy = torch.sigmoid(phi/tau)
        shell = 4*occupancy*(1-occupancy)
        membrane = 1-torch.prod(1-shell,dim=0)
        predicted = F.softplus(amplitude)*blur(membrane,(opts['psfZ'],opts['psfXY'],opts['psfXY']),spacing)+.2*torch.sigmoid(background)
        residual = predicted-target
        data = ((torch.sqrt(residual.square()+.01**2)-.01)*weights).mean()
        prior = (delta[band]/opts['maxDisplacement']).square().mean()
        volume = ((occupancy.sum((1,2,3))-original_volumes)/original_volumes.clamp_min(1)).square().mean()
        overlap = F.relu(occupancy.sum(0)-1).square().mean()
        # Smooth spatial warping retains the original distance field instead
        # of learning arbitrary offsets that can carve new internal shells.
        gradients_u = [(occupancy[:,1:,:-1,:-1]-occupancy[:,:-1,:-1,:-1])/spacing[0],
                       (occupancy[:,:-1,1:,:-1]-occupancy[:,:-1,:-1,:-1])/spacing[1],
                       (occupancy[:,:-1,:-1,1:]-occupancy[:,:-1,:-1,:-1])/spacing[2]]
        area = torch.sqrt(sum(g.square() for g in gradients_u)+1e-6).mean()
        normal_shape = ((torch.sigmoid(ellipsoid/tau)-occupancy).square().mean()) if ellipsoid is not None else torch.tensor(0.,device=device)
        loss = opts['membraneWeight']*data+opts['shapeWeight']*(prior+volume)+opts['smoothness']*(smooth+area)+opts['ellipsoidWeight']*normal_shape+overlap
        return loss,phi,predicted,data
    best = None; best_loss = float('inf'); history = []
    for iteration in range(opts['iterations']+1):
        optimizer.zero_grad(set_to_none=True)
        loss,phi,predicted,data = objective()
        value = float(loss.detach().cpu()); image_loss = float(data.detach().cpu())
        if not math.isfinite(value): raise ValueError('Refinement diverged; reduce learning rate or movement limit.')
        if iteration == 0: initial_loss,initial_image_loss = value,image_loss
        if value < best_loss:
            best_loss = value; best = (displacement.detach().clone(),amplitude.detach().clone(),background.detach().clone()); best_image_loss = image_loss
        history.append(dict(iteration=iteration,loss=value,membraneLoss=image_loss))
        progress(f'Refinement: membrane/shape fitting {iteration}/{opts["iterations"]} · loss {value:.5f}',.15+.7*iteration/opts['iterations'])
        if iteration < opts['iterations']:
            loss.backward(); optimizer.step()
    with torch.no_grad():
        displacement.copy_(best[0]); amplitude.copy_(best[1]); background.copy_(best[2])
        _,phi,predicted,_ = objective()
        values = phi.cpu().numpy()
        winner = values.argmax(0); maximum = values.max(0)
        refined = np.where(maximum > 0,ids[winner],0).astype(np.uint32)
        for i,cell_anchors in enumerate(anchors):
            for anchor in cell_anchors: refined[anchor] = ids[i]
        rendered = predicted.cpu().numpy()
    # If a fitted cell changes volume excessively, lock it to its original
    # region. Repeat because protecting it can affect adjacent competitors.
    rejected = set()
    reasons = {}
    pruned_voxels = 0
    for _ in ids:
        bad = set()
        if opts['preserveConnectivity']:
            for index,label in enumerate(ids):
                components,_ = ndi.label(refined == label,structure=structure)
                keep = {int(components[anchor]) for anchor in anchors[index]}-{0}
                disconnected = (refined == label) & ~np.isin(components,list(keep))
                # Prune spurious islands only if they contain at most 1% of the
                # original cell. Larger lost portions trigger a full rollback.
                if np.count_nonzero(disconnected & (initial == label)) <= .01*np.count_nonzero(initial == label):
                    pruned_voxels += int(disconnected.sum()); refined[disconnected] = 0
                else:
                    bad.add(int(label)); reasons[int(label)] = 'new disconnected components'
        volume_bad = {int(label) for label in ids if abs(np.count_nonzero(refined == label)/np.count_nonzero(initial == label)-1) > opts['maxVolumeChange']}
        bad |= volume_bad
        for label in volume_bad: reasons[label] = 'volume change'
        for label in ids:
            failures,_ = quality_failures(initial == label,refined == label,spacing,opts,original_quality[int(label)])
            if failures: bad.add(int(label)); reasons[int(label)] = ', '.join(failures)
        new = bad-rejected
        if not new: break
        rejected |= new
        refined[np.isin(refined,list(rejected))] = 0
        locked = np.isin(initial,list(rejected)); refined[locked] = initial[locked]
    output = labels.astype(np.uint32).copy(); output[slices] = refined
    if set(np.unique(output))-{0} != set(ids): raise ValueError('Refinement would remove a cell identity; output rejected.')
    stats = dict(initialLoss=initial_loss,finalLoss=best_loss,initialMembraneLoss=initial_image_loss,finalMembraneLoss=best_image_loss,
                 algorithmVersion=2,
                 changedVoxels=int(np.count_nonzero(output != labels)),roiStart=starts.tolist(),roiEnd=ends.tolist(),
                 rejectedCells=sorted(rejected),prunedDisconnectedVoxels=pruned_voxels,rejectionReasons=[dict(label=label,reason=reasons[label]) for label in sorted(rejected)],history=history,
                 note='Boundary refinement on acquired native planes; fixed cell IDs. Fitted-image loss is not segmentation accuracy. Volume guards may roll back fitted cells; losses describe the continuous fit before discrete guards.')
    stats['shapeQuality'] = [dict(label=int(label),before=original_quality[int(label)],after=shape_quality(refined == label,spacing)) for label in ids]
    return output,opts,stats,rendered,slices
