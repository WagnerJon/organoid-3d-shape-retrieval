"""Offline benchmark. Never modifies source data, Atlas datasets, or app settings."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for key, value in dict(MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='2', CELLPOSE_LOCAL_MODELS_PATH=str(ROOT/'.cache/cellpose'), PYTORCH_ENABLE_MPS_FALLBACK='1').items():
    os.environ.setdefault(key, value)
import argparse
import csv
import hashlib
import json
import resource
import subprocess
import sys
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import multiscale_linking_core as core

DEFAULT_OUTPUT = ROOT/'results/multiscale-linking-benchmark'


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x:x.tolist() if hasattr(x, 'tolist') else str(x)))


def reference_info(source):
    source = Path(source)
    cfg = json.loads((source/'run.json').read_text())
    process = json.loads((source/'processing.json').read_text())
    params = json.loads((source/'aggregation.json').read_text())['parameters']
    # Audit actual explicit-scale sampling instead of stretching floor-rounded
    # dimensions back to the full input extent. Source files stay unchanged.
    pre = process.get('parameters',{})
    if 'factor' in pre and 'voxel_res' in pre:
        native_spacing=np.array(cfg['spacingZYXum'])
        actual=native_spacing/(pre['factor']*np.array(pre['voxel_res']))
        process['reportedReferenceSpacingZYXum']=process['effectiveSpacingZYXum']
        process['effectiveSpacingZYXum']=actual.tolist()
        cfg.setdefault('physicalOriginZYXum',((actual-native_spacing)/2).tolist())
    return cfg, process, params


def stage(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    cfg, info, params = reference_info(args.source)
    fine = output/'fine'; shape = np.array(info['processedShape'])
    started = time.monotonic(); details = {}
    name = args.stage
    if name == 'fuse':
        details = core.fuse_predictions(args.source/'predictions', fine, params)
        save(output/'protocol.json', dict(source=str(args.source.resolve()), sourceSHA256=cfg.get('sourceSHA256'),
            originalImage=cfg['source'], shape=shape, spacingZYXum=info['effectiveSpacingZYXum'],
            sourceReportedSpacingZYXum=info.get('reportedReferenceSpacingZYXum'),physicalOriginZYXum=cfg['physicalOriginZYXum'],
            parameters=params, reference='Previously saved full-grid automatic segmentation, not manual ground truth',
            controls='Same frozen Cellpose predictions for every linking variant; fine binary and pre/post smoothing remain in original physical units.',
            implementation='Exact streamed fine fusion and endpoint dynamics; global disk-backed connected components. Coarse fields are area-averaged fused fine fields transformed as index displacements, without renormalizing away the coordinate transform. Fine foreground morphology is applied before coarsening. Coarse vote-kernel sigma scales by each grid ratio; iteration/momentum/decay are retained because vector components already carry the displacement transform.',
            limitation='Only recorded ksize=1/sigma=1, no-fill-holes pipeline. Coarse proposals may lose identities. Full fine flow correction is an expensive exact control, not a learned prior. No Atlas integration or two-physical-Mac speedup claim.',
            scriptHashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/'scripts/multiscale_linking_core.py']}))
    elif name == 'fine-link':
        core.trajectories(fine, params['gradient_descent'])
        details = core.cluster_endpoints(fine, params['connected_component']['smooth_sigma'])
    elif name == 'coarse':
        target = tuple(np.ceil(shape/args.factor).astype(int))
        folder = output/f'coarse-{args.factor:g}x'
        ratios = core.coarsen(fine, folder, target)
        core.trajectories(folder, params['gradient_descent'])
        details = core.cluster_endpoints(folder, np.array([params['connected_component']['smooth_sigma']]*3)*ratios)
        details.update(targetShape=target, effectiveSpacingZYXum=np.array(info['effectiveSpacingZYXum'])/ratios,
                       displacementRatio=ratios, vectorRenormalization=False)
        name = folder.name
    elif name == 'recover':
        coarse = output/f'coarse-{args.factor:g}x'
        result = core.recover_ids(fine, coarse, output/f'recovered-{args.factor:g}x.npy')
        ids = set()
        for z in range(len(result)): ids.update(np.unique(result[z]).tolist())
        details = dict(cells=len(ids-{0}), scope='Fine Cellpose-derived foreground and fine trajectories, coarse global cell IDs; single exclusive label per voxel. Coarse merges cannot be corrected by reassigning those same IDs.')
        name = f'recovered-{args.factor:g}x'
    elif name == 'local':
        details=core.local_correction(fine,output/f'recovered-{args.factor:g}x.npy',output/f'local-{args.factor:g}x.npy',params['gradient_descent'])
        name=f'local-{args.factor:g}x'
    elif name == 'reference':
        sys.path.insert(0, str(ROOT/'vendor/u-segment3D'))
        from segment3D import usegment3d
        probabilities = [np.load(args.source/'predictions'/f'{v}_prob.npy', mmap_mode='r') for v in ['xy', 'xz', 'yz']]
        flows = [np.load(args.source/'predictions'/f'{v}_flow.npy', mmap_mode='r') for v in ['xy', 'xz', 'yz']]
        labels, _ = usegment3d.aggregate_2D_to_3D_segmentation_direct_method(probabilities, flows, params)
        np.save(output/'fresh-reference.npy', labels.astype('uint32'))
        details = dict(cells=len(np.unique(labels))-1)
    elif name == 'evaluate':
        details = evaluate(args)
    elif name == 'morphology':
        details = morphology(args)
    elif name == 'export':
        details = export(args)
    elif name == 'figure':
        figure(args); details = dict(image=str(output/'comparison.png'))
    elif name == 'infer-crop':
        details = infer_crop(args)
    else:
        raise ValueError(name)
    details.update(seconds=time.monotonic()-started, peakProcessRSSGiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30,
                   implementationHashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/multiscale_linking_core.py']})
    save(output/f'{name}.completed.json', details)
    print(json.dumps(details, default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)), flush=True)


def method_paths(output):
    result = [('streamed-fine', output/'fine/labels.npy', False)]
    for factor in (1.5, 2., 4.):
        p = output/f'coarse-{factor:g}x/labels.npy'
        if p.exists(): result.append((f'coarse-{factor:g}x', p, True))
        p = output/f'recovered-{factor:g}x.npy'
        if p.exists(): result.append((f'recovered-{factor:g}x', p, False))
        p = output/f'local-{factor:g}x.npy'
        if p.exists(): result.append((f'local-{factor:g}x', p, False))
        p = output/f'local-size-rank-{factor:g}x.npy'
        if p.exists(): result.append((f'local-size-rank-{factor:g}x', p, False))
    return result


def plane(array, shape, z):
    if tuple(array.shape) == tuple(shape): return np.asarray(array[z])
    indexes = [np.minimum(n-1, ((np.arange(m)+.5)*n/m).astype(int)) for n,m in zip(array.shape[1:], shape[1:])]
    iz = min(array.shape[0]-1, int((z+.5)*array.shape[0]/shape[0]))
    return array[iz][indexes[0][:,None], indexes[1][None,:]]


def confusion(reference, candidate):
    n = int(reference.max())+1; m = int(candidate.max())+1
    table = np.zeros((n, m), dtype='int64')
    for z in range(reference.shape[0]):
        a, b = reference[z], plane(candidate, reference.shape, z)
        table += np.bincount((a.astype('int64')*m+b).ravel(), minlength=n*m).reshape(n, m)
    return table


def object_geometry(mask, spacing):
    from skimage.measure import marching_cubes
    from scipy.spatial import cKDTree
    vertices, faces, _, _ = marching_cubes(np.pad(mask, 1), .5, spacing=spacing)
    triangles = vertices[faces]
    area = float(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1).sum()/2)
    volume = float(np.count_nonzero(mask)*np.prod(spacing))
    sphericity = float(np.pi**(1/3)*(6*volume)**(2/3)/area)
    return dict(volume=volume, area=area, sphericity=sphericity), vertices


def evaluate(args):
    output = args.output; _, info, _ = reference_info(args.source)
    reference = load_reference(args)
    spacing = np.array(info['effectiveSpacingZYXum']); summaries = []
    for name, path, upsampled in method_paths(output):
        candidate = np.load(path, mmap_mode='r'); table = confusion(reference, candidate)
        va, vb = table.sum(1), table.sum(0); intersection = table[1:, 1:]
        iou = intersection/np.maximum(1, va[1:,None]+vb[None,1:]-intersection)
        rows, cols = linear_sum_assignment(-iou); scores = iou[rows, cols]; good = scores >= .5
        frac_ref = intersection/np.maximum(1, va[1:,None])
        split = np.count_nonzero((frac_ref >= .2).sum(1) >= 2)
        merge = np.count_nonzero((frac_ref >= .2).sum(0) >= 2)
        volume_error = np.abs(va[rows[good]+1]-vb[cols[good]+1])/va[rows[good]+1]
        summary = dict(method=name, referenceCells=len(va)-1, predictedCells=int(np.count_nonzero(vb[1:])),
            matchedAtIoU05=int(good.sum()), referenceRecall=float(good.sum()/max(1,len(va)-1)),
            precision=float(good.sum()/max(1,np.count_nonzero(vb[1:]))),
            foregroundDice=float(2*intersection.sum()/max(1,va[1:].sum()+vb[1:].sum())),
            meanMatchedIoU=float(scores[good].mean()) if good.any() else 0.,
            medianVolumeRelativeError=float(np.median(volume_error)) if good.any() else None,
            splitReferenceObjects=int(split), mergedPredictedObjects=int(merge),
            segmentationEquivalentUpToIDs=bool(good.sum()==len(va)-1 and len(va)==len(vb) and scores.min()>=1.))
        save(output/f'{name}.agreement.json', summary); summaries.append(summary)
        print(name, json.dumps(summary), flush=True)
    save(output/'agreement.json', summaries)
    return dict(methods=summaries, reference='Agreement to automatic fine-grid segmentation; not manual accuracy')


def load_reference(args):
    fresh = args.output/'fresh-reference.npy'
    return np.load(fresh if fresh.exists() else args.source/'labels_processed.npy', mmap_mode='r')


def morphology(args):
    from scipy.spatial import cKDTree
    reference = load_reference(args)
    _, info, _ = reference_info(args.source)
    spacing = np.array(info['effectiveSpacingZYXum'])
    ref_boxes = ndi.find_objects(reference)
    summaries = []
    for name,path,_ in method_paths(args.output):
        candidate = np.load(path,mmap_mode='r'); table = confusion(reference,candidate)
        va,vb = table.sum(1),table.sum(0)
        iou = table[1:,1:]/np.maximum(1,va[1:,None]+vb[None,1:]-table[1:,1:])
        rows,cols = linear_sum_assignment(-iou)
        boxes = ndi.find_objects(candidate); ratio=np.array(candidate.shape)/np.array(reference.shape)
        values=[]
        for row,col in zip(rows,cols):
            if iou[row,col]<.5:continue
            ref_label,pred_label=row+1,col+1
            a=ref_boxes[row]; b=boxes[col]
            if a is None or b is None:continue
            starts=np.maximum(0,np.ceil(np.array([s.start for s in b])/ratio-.5)).astype(int)
            stops=np.minimum(reference.shape,np.ceil(np.array([s.stop for s in b])/ratio-.5)).astype(int)
            box=tuple(slice(min(s.start,int(u)),max(s.stop,int(v))) for s,u,v in zip(a,starts,stops))
            ref_mask=reference[box]==ref_label
            pred_mask=np.stack([plane(candidate,reference.shape,z)[box[1],box[2]]==pred_label for z in range(box[0].start,box[0].stop)])
            r,rv=object_geometry(ref_mask,spacing); p,pv=object_geometry(pred_mask,spacing)
            # Deterministic evenly spaced vertices for the distance query only.
            # Trees retain all vertices; this is a sampled surface-distance proxy.
            rd=cKDTree(pv).query(rv[np.linspace(0,len(rv)-1,min(2048,len(rv)),dtype=int)])[0]
            pd=cKDTree(rv).query(pv[np.linspace(0,len(pv)-1,min(2048,len(pv)),dtype=int)])[0]
            distances=np.r_[rd,pd]
            values.append(dict(referenceLabel=ref_label,predictedLabel=pred_label,IoU=float(iou[row,col]),
                volumeRelativeError=abs(p['volume']-r['volume'])/r['volume'],
                sphericityAbsoluteError=abs(p['sphericity']-r['sphericity']),
                surfaceAreaRelativeError=abs(p['area']-r['area'])/r['area'],
                meanSurfaceVertexDistanceUm=float(distances.mean()),surfaceVertexDistance95Um=float(np.percentile(distances,95))))
        with (args.output/f'{name}.morphology.csv').open('w') as stream:
            if values:
                writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        summary=dict(method=name,matchedCells=len(values))
        for field in ['volumeRelativeError','sphericityAbsoluteError','surfaceAreaRelativeError','meanSurfaceVertexDistanceUm','surfaceVertexDistance95Um']:
            summary['median_'+field]=float(np.median([v[field] for v in values])) if values else None
        summaries.append(summary);print(json.dumps(summary),flush=True)
    save(args.output/'morphology.json',summaries)
    return dict(methods=summaries,scope='Metrics among IoU>=0.5 matched cells only; unmatched/split/merged counts are reported separately. Surface distances are sampled vertex proxies, not exact continuous-surface distances.')


def export(args):
    import tifffile
    import trimesh
    labels = np.load(args.output/'fine/labels.npy', mmap_mode='r')
    cfg, info, _ = reference_info(args.source)
    spacing = np.array(info['effectiveSpacingZYXum'])
    origin = np.array(cfg.get('physicalOriginZYXum',[0.,0.,0.]))
    folder = args.output/'recovered-fine-export'; folder.mkdir(exist_ok=True)
    tifffile.imwrite(folder/'labels.ome.tif', data=(np.asarray(labels[z]) for z in range(len(labels))),
        shape=labels.shape, dtype='uint32', bigtiff=True, ome=True, photometric='minisblack',
        metadata=dict(axes='ZYX', PhysicalSizeZ=spacing[0], PhysicalSizeY=spacing[1], PhysicalSizeX=spacing[2],
                      PhysicalSizeZUnit='µm', PhysicalSizeYUnit='µm', PhysicalSizeXUnit='µm',
                      Plane=dict(PositionZ=(origin[0]+np.arange(len(labels))*spacing[0]).tolist(),
                                 PositionY=[float(origin[1])]*len(labels),PositionX=[float(origin[2])]*len(labels),
                                 PositionZUnit=['µm']*len(labels),PositionYUnit=['µm']*len(labels),PositionXUnit=['µm']*len(labels))))
    boxes = ndi.find_objects(labels); rows = []; meshes = folder/'meshes'; meshes.mkdir(exist_ok=True)
    from skimage.measure import marching_cubes
    for label, box in enumerate(boxes, 1):
        if box is None: continue
        mask = labels[box] == label; metrics, _ = object_geometry(mask, spacing)
        v, f, _, _ = marching_cubes(np.pad(mask,1), .5, spacing=spacing)
        v += (np.array([s.start for s in box])-1)*spacing+origin
        mesh = trimesh.Trimesh(v[:,::-1], f[:,::-1], process=False)
        mesh.export(meshes/f'cell_{label:04d}.ply')
        centroid=(np.array(ndi.center_of_mass(mask))+np.array([s.start for s in box]))*spacing+origin
        rows.append(dict(label=label, voxel_count=int(mask.sum()), volume_um3=metrics['volume'], surface_area_um2=metrics['area'], sphericity=metrics['sphericity'],
                         centroid_z_um=centroid[0],centroid_y_um=centroid[1],centroid_x_um=centroid[2],
                         touches_label_grid_boundary=any(s.start==0 or s.stop==n for s,n in zip(box,labels.shape))))
    with (folder/'measurements.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = dict(labelGridZYX=list(labels.shape), labelSpacingZYXum=spacing, labelOriginZYXum=cfg.get('physicalOriginZYXum',[0.,0.,0.]),
        source=args.source, method='Streamed full fine-grid recovery/control using original Cellpose evidence; not upsampled coarse labels',
        objects=len(rows), nativeRawGridWasNotClaimed=True)
    save(folder/'label-grid.json', metadata)
    with tifffile.TiffFile(folder/'labels.ome.tif') as tif:
        assert tuple(tif.series[0].shape)==tuple(labels.shape) and tif.series[0].dtype==np.uint32
    return metadata


def figure(args):
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    reference = load_reference(args)
    raw = np.load(args.source/'processed.npy', mmap_mode='r')
    methods = [('Reference', reference), ('Streamed fine', np.load(args.output/'fine/labels.npy',mmap_mode='r'))]
    for name,path,_ in method_paths(args.output)[1:]:
        if name in ('coarse-2x','recovered-2x','local-2x'): methods.append((name, np.load(path,mmap_mode='r')))
    fig, axes = plt.subplots(3, len(methods), figsize=(4*len(methods), 10), facecolor='white')
    for row,z in enumerate([int(len(reference)*f) for f in [.35,.5,.65]]):
        image = raw[z]; lo,hi = np.percentile(image,[1,99.7])
        for col,(name, labels) in enumerate(methods):
            ax=axes[row,col]; lab=plane(labels,reference.shape,z)
            ax.imshow(image,cmap='gray',vmin=lo,vmax=hi)
            boundary=find_boundaries(lab,mode='inner')
            overlay=np.zeros((*lab.shape,4));overlay[boundary]=[0.,.85,1.,.9]
            ax.imshow(overlay);ax.set_title(f'{name} · Z {z}');ax.axis('off')
    fig.suptitle('Same Cellpose predictions · global cell boundaries\nAutomatic-reference agreement, not manual accuracy',fontsize=16)
    fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=160);plt.close(fig)


def infer_crop(args):
    """Actual MPS inference, one view plane at a time, into disk-backed outputs."""
    import torch
    from cellpose import models
    processed=np.load(args.source/'processed.npy',mmap_mode='r')
    predictions=args.source/'predictions';predictions.mkdir(exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError('This Mac GPU benchmark requires Metal access; refusing a silent CPU fallback.')
    print('Using Metal GPU for streamed Cellpose inference',flush=True)
    model=models.CellposeModel(pretrained_model='cpsam_v2',gpu=torch.backends.mps.is_available())
    for view,order in [('xy',(0,1,2)),('xz',(1,0,2)),('yz',(2,0,1))]:
        if (args.source/f'{view}.completed.json').exists():continue
        planes=processed.transpose(order)
        probability=core.disk(predictions/f'{view}_prob.npy',processed.shape)
        flow=core.disk(predictions/f'{view}_flow.npy',(2,)+processed.shape)
        pp=probability.transpose(order);ff=flow.transpose((0,)+tuple(a+1 for a in order))
        for i,image in enumerate(planes):
            _, fields, _=model.eval([np.asarray(image).copy()],batch_size=32,channels=[0,0],diameter=args.diameter,compute_masks=False)
            pp[i]=fields[0][2];ff[:,i]=fields[0][1]
            if i%20==0: print(view,i+1,'/',len(planes),flush=True)
        probability.flush();flow.flush()
        save(args.source/f'{view}.completed.json',dict(planes=len(planes),diameter=args.diameter,computeMasks=False))
    return dict(shape=processed.shape,execution='One Cellpose plane at a time; each result written directly to disk',gpu=torch.backends.mps.is_available())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fuse','fine-link','coarse','recover','local','reference','evaluate','morphology','export','figure','infer-crop'])
    parser.add_argument('--source',type=Path,required=True,help='Completed standalone run containing cached predictions and aggregation.json')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--factor',type=float,default=2.)
    parser.add_argument('--diameter',type=float,default=60.)
    stage(parser.parse_args())
