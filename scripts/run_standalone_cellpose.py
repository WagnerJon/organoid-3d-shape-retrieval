"""Memory-conscious, checkpointed Cellpose + u-Segment3D run outside Atlas.

Each stage runs in a fresh process. Float32 predictions are saved uncompressed for
read-only memory mapping; native-grid labels are streamed one plane at a time.
"""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for k,v in {'MPLBACKEND':'Agg','MPLCONFIGDIR':str(ROOT/'.cache/matplotlib'),'CELLPOSE_LOCAL_MODELS_PATH':str(ROOT/'.cache/cellpose'),'NUMBA_CACHE_DIR':str(ROOT/'.cache/numba'),'XDG_CACHE_HOME':str(ROOT/'.cache'),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','NUMBA_NUM_THREADS':'2','PYTORCH_ENABLE_MPS_FALLBACK':'1'}.items():os.environ.setdefault(k,v)
import argparse,json,hashlib,subprocess,sys,time,csv,resource,signal
sys.path.insert(0,str(ROOT/'vendor/u-segment3D'))

STAGES=['preprocess','xy','xz','yz','aggregate','export']

def save(p,value):
    def encode(x):
        if hasattr(x,'tolist'):return x.tolist()
        if callable(x):return x.__module__+'.'+x.__name__
        if isinstance(x,Path):return str(x)
        raise TypeError('Unsupported metadata: '+type(x).__name__)
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=encode));tmp.replace(p)

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def source_stat(p):
    s=Path(p).stat();return [s.st_size,s.st_mtime_ns]

def calibration(path, spacing_override=None):
    import tifffile
    with tifffile.TiffFile(path) as t:
        s=t.series[0];assert s.axes in ('ZYX','QYX','IYX') and len(s.shape)==3,'Single-channel ZYX TIFF required'
        if spacing_override is not None:
            assert len(spacing_override)==3 and all(float(x)>0 for x in spacing_override), 'Positive Z/Y/X spacing required'
            return dict(nativeShape=list(s.shape),dtype=str(s.dtype),spacingZYXum=list(spacing_override),calibrationSource='User-confirmed Z/Y/X voxel spacing in micrometers')
        md=t.imagej_metadata or {};assert md.get('unit') in ('micron','um','µm'),'Expected micrometer calibration'
        def inverse(name):
            a,b=t.pages[0].tags[name].value;return b/a
        spacing=[float(md['spacing']),inverse('YResolution'),inverse('XResolution')]
        assert all(x>0 for x in spacing)
        return dict(nativeShape=list(s.shape),dtype=str(s.dtype),spacingZYXum=spacing,calibrationSource='TIFF ImageJ spacing and X/Y resolution tags, unit=micron')

def setup(args):
    from importlib.metadata import version,PackageNotFoundError
    folder=args.output.resolve();source=args.input.resolve();meta=calibration(source, getattr(args, "spacing", None))
    core=dict(source=str(source),sourceStat=source_stat(source),scale=args.scale,diameterProcessedPx=args.diameter,model=getattr(args,'model','cpsam_v2'),gpu=not getattr(args,'cpu',False),backgroundCorrection=not getattr(args,'no_background',False),binaryFillHoles=False,gradientDecay=.05,**meta)
    from atlas_cellpose_3d import settings as native_settings
    native = native_settings(dict(segmentationMode=getattr(args,'segmentation_mode','usegment3d'), flow3DSmooth=getattr(args,'flow_3d_smooth',0.), cellprobThreshold=getattr(args,'cellprob_threshold',0.), minSize3D=getattr(args,'min_size_3d',15)))
    if native['segmentationMode']=='cellpose3d':core.update(native)
    from atlas_streamed_linking import validate_mode,check_space
    mode=validate_mode(getattr(args,'linking_mode','standard'))
    if native['segmentationMode']=='cellpose3d' and mode!='standard':
        raise ValueError('Native Cellpose 3D requires whole-volume mode, not streamed u-Segment3D linking.')
    if mode=='streamed':core['linkingMode']=mode
    if not (folder/'aggregate.completed.json').exists():
        from atlas_preprocessing import check_memory
        plan=check_memory(meta['nativeShape'],meta['spacingZYXum'],args.scale,linking_mode=mode)
        if mode=='streamed':
            cached=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink()) if (folder/'run.json').exists() else 0
            check_space(folder,max(0,plan['estimatedDiskGiB']*2**30-cached),'Dataset preparation')
    folder.mkdir(parents=True,exist_ok=True)
    p=folder/'run.json'
    if p.exists():
        current=json.loads(p.read_text());assert all(current.get(k)==v for k,v in core.items()),'Output folder belongs to different settings or source'
        assert current.get('linkingMode','standard')==mode,'Output folder belongs to a different linking mode'
        assert current.get('segmentationMode','usegment3d')==native['segmentationMode'],'Output folder belongs to a different segmentation mode'
        return current
    if list(folder.iterdir()):raise ValueError('Refusing to reuse a nonempty, unrecognized output folder')
    runtime={}
    for name in ['numpy','scipy','scikit-image','cellpose','torch','u-Segment3D','tifffile','trimesh']:
        try:runtime[name]=version(name)
        except PackageNotFoundError:pass
    config=dict(**core,sourceSHA256=digest(source),checkpointSHA256=digest(ROOT/'.cache/cellpose'/core['model']) if (ROOT/'.cache/cellpose'/core['model']).is_file() else 'Downloaded by Cellpose on first inference',runtime=runtime,
                scriptSHA256=digest(__file__),created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),execution='Native Cellpose 3D; whole-volume masks' if native['segmentationMode']=='cellpose3d' else 'Separate processes; sequential views; '+('disk-backed fine-grid linking' if mode=='streamed' else 'whole-volume aggregation'))
    save(p,config);(folder/'logs').mkdir();(folder/'predictions').mkdir();(folder/'script_snapshot.py').write_bytes(Path(__file__).read_bytes())
    segmentation_note='Native Cellpose do_3D=True on the isotropic processing grid; no u-Segment3D linking. MPS prediction uses CPU 3D mask formation.' if native['segmentationMode']=='cellpose3d' else 'XY, XZ and YZ predictions are merged by u-Segment3D (200 gradient-descent iterations, decay 0.05). Optional whole-volume binary hole filling is disabled.'
    (folder/'README.md').write_text(f'''# Standalone Cellpose 3D analysis

Input: `{source.name}`. The raw file is unchanged.

- Native ZYX: {meta['nativeShape']}; TIFF spacing ZYX: {meta['spacingZYXum']} µm.
- Cellpose {core['model']}, processing scale {args.scale}, diameter {args.diameter} processed pixels.
- Original full field of view; no crop. Background correction enabled.
- {segmentation_note}
- Each stage has a separate process and checkpoint; logs and memory samples are retained.

Outputs after completion: calibrated `labels_processed.ome.tif`, streamed original-grid `labels_native.ome.tif`, `meshes/`, `cell_measurements.csv`, `preview.png`, `summary.json` and `complete.json`.

Measurements and meshes are computed on the processed segmentation using effective physical spacing; the native label file is a nearest-neighbor resampling, not additional recovered detail. These are unvalidated model predictions, not manual ground truth. Check overlays and border-touching objects before analysis. All detected objects are retained; no Atlas main-organoid membership rule is applied to this new specimen.
''')
    return config

def finite(array):
    import numpy as np
    return all(np.isfinite(array[...,i:i+8,:,:] if array.ndim==4 else array[i:i+8]).all() for i in range(0,array.shape[-3],8))

def save_array(p,a):
    import numpy as np
    tmp=p.with_name(p.stem+'.tmp.npy');np.save(tmp,a);tmp.replace(p)

def native_plane(labels,native_shape,z,zoom=None):
    import numpy as np
    # Pixel-center nearest-neighbor mapping, matching skimage resize(order=0).
    factors=np.asarray(zoom) if zoom is not None else np.asarray(labels.shape)/native_shape
    iz=min(labels.shape[0]-1,int((z+.5)*factors[0]))
    iy=np.minimum(labels.shape[1]-1,((np.arange(native_shape[1])+.5)*factors[1]).astype(int))
    ix=np.minimum(labels.shape[2]-1,((np.arange(native_shape[2])+.5)*factors[2]).astype(int))
    return labels[iz][iy[:,None],ix[None,:]]

def worker(folder,stage):
    import numpy as np,tifffile
    folder=Path(folder);cfg=json.loads((folder/'run.json').read_text());assert source_stat(cfg['source'])==cfg['sourceStat'],'Source changed'
    start=time.time();details={}
    if stage=='preprocess':
        import dask
        from scipy import ndimage
        from segment3D import parameters,usegment3d
        dask.config.set(num_workers=1,scheduler='threads')
        pre=parameters.get_preprocess_params();spacing=np.array(cfg['spacingZYXum'])
        pre.update(factor=cfg['scale'],voxel_res=(spacing/spacing.min()).tolist(),do_bg_correction=cfg['backgroundCorrection'],bg_ds=8)
        from atlas_preprocessing import check_memory,preprocess
        plan=check_memory(cfg['nativeShape'],cfg['spacingZYXum'],cfg['scale'],linking_mode=cfg.get('linkingMode','standard'))
        print('Preprocessing',cfg['nativeShape'],'spacing',spacing,'scale',cfg['scale'],flush=True)
        if cfg['dtype'] in ('uint8','uint16'):
            streamed=preprocess(cfg['source'],folder,pre,lambda line:print(line,flush=True))
            processed=np.load(folder/'processed.npy',mmap_mode='r')
        else:
            raw=tifffile.imread(cfg['source'])
            processed=np.squeeze(usegment3d.preprocess_imgs(raw,pre));del raw
            processed=ndimage.median_filter(processed,size=3).astype(np.float32,copy=False)
            save_array(folder/'processed.npy',processed)
            streamed=dict(mode='Original u-Segment3D preprocessing (non-uint8/uint16 input)')
        assert processed.ndim==3 and finite(processed)
        zoom=pre['factor']*np.asarray(pre['voxel_res']);actual_spacing=spacing/zoom
        details=dict(processedShape=list(processed.shape),effectiveSpacingZYXum=actual_spacing.tolist(),resamplingZoomZYX=zoom.tolist(),rawGridOffsetZYXum=((actual_spacing-spacing)/2).tolist(),parameters=pre,memoryPlan=plan,preprocessing=streamed)
        save(folder/'processing.json',details)
    elif stage in ['xy','xz','yz']:
        import torch
        from segment3D import parameters
        use_gpu=cfg.get('gpu',True) and (torch.backends.mps.is_available() or torch.cuda.is_available())
        if cfg.get('linkingMode')=='streamed':
            from atlas_streamed_linking import infer_view
            details=infer_view(folder,stage,dict(model=cfg['model'],gpu=use_gpu,diameter=cfg['diameterProcessedPx']))
        else:
            from segment3D import usegment3d
            processed=np.load(folder/'processed.npy',mmap_mode='r');cp=parameters.get_Cellpose_autotune_params()
            cp.update(cellpose_modelname=cfg['model'],gpu=use_gpu,best_diam=cfg['diameterProcessedPx'],debug_viz=False)
            print('Running Cellpose',stage,'processed shape',processed.shape,'diameter',cfg['diameterProcessedPx'],'MPS available',torch.backends.mps.is_available(),flush=True)
            auto,prob,flow,styles=usegment3d.Cellpose2D_model_auto(processed[...,None],stage,cp)
            assert prob.shape==processed.shape and flow.shape==(2,)+processed.shape
            assert finite(prob) and finite(flow)
            save_array(folder/'predictions'/f'{stage}_prob.npy',prob.astype(np.float32,copy=False));save_array(folder/'predictions'/f'{stage}_flow.npy',flow.astype(np.float32,copy=False))
            details=dict(view=stage,shape=list(prob.shape),parameters=cp,device='GPU' if use_gpu else 'CPU')
    elif stage=='aggregate':
        if cfg.get('segmentationMode')=='cellpose3d':
            from atlas_cellpose_3d import predict
            from atlas_streamed_linking import progress_event
            processed=np.load(folder/'processed.npy',mmap_mode='r')
            labels, details=predict(processed,dict(cfg,diameter=cfg['diameterProcessedPx']),lambda label,fraction:progress_event('aggregate',label,fraction))
            save_array(folder/'labels_processed.npy',labels)
            save(folder/'aggregation.json',details)
            details.update(stage=stage,seconds=time.time()-start,peakProcessRSSGiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30,scriptSHA256=digest(__file__))
            save(folder/f'{stage}.completed.json',details)
            print('STAGE COMPLETE',stage,json.dumps(details),flush=True)
            return
        from segment3D import parameters
        agg=parameters.get_2D_to_3D_aggregation_params();agg['gradient_descent']['gradient_decay']=cfg['gradientDecay'];agg['gradient_descent']['do_mp']=False
        agg['postprocess_binary']['binary_fill_holes']=cfg['binaryFillHoles']
        if cfg.get('linkingMode')=='streamed':
            from atlas_streamed_linking import aggregate
            details=dict(**aggregate(folder,agg),parameters=agg)
        else:
            from segment3D import usegment3d
            probs=[np.load(folder/'predictions'/f'{v}_prob.npy',mmap_mode='r') for v in ['xy','xz','yz']]
            flows=[np.load(folder/'predictions'/f'{v}_flow.npy',mmap_mode='r') for v in ['xy','xz','yz']]
            print('Aggregating',probs[0].shape,'with disk-backed input arrays; no raw stack or Cellpose model loaded',flush=True)
            labels,auxiliary=usegment3d.aggregate_2D_to_3D_segmentation_direct_method(probs,flows,agg)
            del auxiliary,probs,flows
            assert labels.shape==tuple(json.loads((folder/'processing.json').read_text())['processedShape'])
            save_array(folder/'labels_processed.npy',labels.astype(np.uint32,copy=False))
            unique=np.unique(labels);details=dict(cells=int(np.count_nonzero(unique)),parameters=agg)
        save(folder/'aggregation.json',details)
    elif stage=='export':
        details=export(folder,cfg)
    details.update(stage=stage,seconds=time.time()-start,peakProcessRSSGiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30,scriptSHA256=digest(__file__))
    save(folder/f'{stage}.completed.json',details)
    print('STAGE COMPLETE',stage,json.dumps({k:v for k,v in details.items() if k!='parameters'}),flush=True)

def export(folder,cfg):
    import numpy as np,tifffile,trimesh
    from skimage import measure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb
    labels=np.load(folder/'labels_processed.npy',mmap_mode='r');info=json.loads((folder/'processing.json').read_text());spacing=np.array(info['effectiveSpacingZYXum']);native=cfg['nativeShape']
    def md(sp):return dict(axes='ZYX',PhysicalSizeZ=float(sp[0]),PhysicalSizeY=float(sp[1]),PhysicalSizeX=float(sp[2]),PhysicalSizeZUnit='µm',PhysicalSizeYUnit='µm',PhysicalSizeXUnit='µm')
    tifffile.imwrite(folder/'labels_processed.ome.tif',data=(labels[z] for z in range(labels.shape[0])),shape=labels.shape,dtype=np.uint32,bigtiff=True,ome=True,metadata=md(spacing),photometric='minisblack')
    tifffile.imwrite(folder/'labels_native.ome.tif',data=(native_plane(labels,native,z,info.get('resamplingZoomZYX')) for z in range(native[0])),shape=tuple(native),dtype=np.uint32,bigtiff=True,ome=True,metadata=md(cfg['spacingZYXum']),photometric='minisblack')
    (folder/'meshes').mkdir(exist_ok=True);rows=[]
    for reg in measure.regionprops(labels):
        v,f,_,_=measure.marching_cubes(np.pad(reg.image,1),.5,spacing=spacing);v+=(np.array(reg.bbox[:3])-1)*spacing
        mesh=trimesh.Trimesh(vertices=v[:,::-1],faces=f[:,::-1],process=False);mesh.fix_normals(multibody=True);mesh.export(folder/'meshes'/f'cell_{reg.label:04d}.ply')
        inertia=reg.inertia_tensor;cov=np.trace(inertia)/2*np.eye(3)-inertia;cov=spacing[:,None]*cov*spacing[None,:]+np.diag(spacing**2/12)
        axes=np.sqrt(20*np.maximum(np.linalg.eigvalsh(cov)[::-1],1e-12));volume=float(reg.area*np.prod(spacing));area=float(mesh.area)
        row=dict(label=int(reg.label),voxel_count=int(reg.area),volume_um3=volume,surface_area_um2=area,sphericity=float(np.pi**(1/3)*(6*volume)**(2/3)/area),major_axis_um=float(axes[0]),middle_axis_um=float(axes[1]),minor_axis_um=float(axes[2]),elongation=float(axes[0]/axes[1]),flatness=float(axes[1]/axes[2]),touches_border=bool(any(a==0 for a in reg.bbox[:3]) or any(b==s for b,s in zip(reg.bbox[3:],labels.shape))),mesh_watertight=bool(mesh.is_watertight))
        row.update(zip(['centroid_z_um','centroid_y_um','centroid_x_um'],(np.array(reg.centroid)*spacing).tolist()));rows.append(row)
        if len(rows)%25==0:print('Exported',len(rows),'cell surfaces',flush=True)
    fields=list(rows[0]) if rows else ['label','volume_um3','surface_area_um2','sphericity','elongation','flatness','touches_border']
    with (folder/'cell_measurements.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(2,3,figsize=(13,8),facecolor='white');rng=np.random.default_rng(42);colors=rng.uniform(.25,1,(max(1,len(rows)),3))
    for axis,name in enumerate(['XY','XZ','YZ']):
        # Stack axes 0,1,2 respectively; only two-dimensional slices are loaded.
        idx=native[axis]//2
        with tifffile.TiffFile(cfg['source']) as raw_tif:
            if axis==0: im=raw_tif.pages[idx].asarray()
            else: im=np.stack([np.take(page.asarray(),idx,axis=axis-1) for page in raw_tif.pages])
        if 'resamplingZoomZYX' in info:
            zoom=np.array(info['resamplingZoomZYX'])
            coordinates=[np.minimum(labels.shape[a]-1,((np.arange(native[a])+.5)*zoom[a]).astype(int)) for a in range(3)]
            if axis==0:lab=native_plane(labels,native,idx,zoom)
            elif axis==1:lab=labels[coordinates[0][:,None],coordinates[1][idx],coordinates[2][None,:]]
            else:lab=labels[coordinates[0][:,None],coordinates[1][None,:],coordinates[2][idx]]
        else:
            pi=min(labels.shape[axis]-1,int((idx+.5)*labels.shape[axis]/native[axis]))
            from skimage.transform import resize
            lab=resize(np.take(labels,pi,axis=axis),im.shape,order=0,preserve_range=True,anti_aliasing=False).astype(np.uint32)
        lo,hi=np.percentile(im,[1,99.8]);norm=np.clip((im.astype(np.float32)-lo)/max(float(hi-lo),1),0,1)
        axs[0,axis].imshow(norm,cmap='gray');axs[1,axis].imshow(label2rgb(lab,image=norm,colors=colors,alpha=.45,bg_label=0));axs[0,axis].set_title(f'{name} raw · middle slice');axs[1,axis].set_title(f'{name} prediction overlay')
        for row in [0,1]:axs[row,axis].axis('off')
    method='Native Cellpose 3D' if cfg.get('segmentationMode')=='cellpose3d' else 'Cellpose + u-Segment3D'
    fig.suptitle(f'{method} · {len(rows)} predicted objects · scale {cfg["scale"]}\nTIFF calibration; review predictions before quantitative use',fontsize=13);fig.tight_layout();fig.savefig(folder/'preview.png',dpi=150);plt.close(fig)
    for name,expected in [('labels_native.ome.tif',tuple(native)),('labels_processed.ome.tif',labels.shape)]:
        with tifffile.TiffFile(folder/name) as t:assert t.series[0].shape==expected and t.series[0].dtype==np.uint32 and t.is_ome
    summary=dict(predictedObjects=len(rows),borderTouchingObjects=sum(r['touches_border'] for r in rows),nonWatertightMeshes=sum(not r['mesh_watertight'] for r in rows),effectiveSpacingZYXum=spacing.tolist(),processedShape=list(labels.shape),nativeShape=native,validation='No manual/reference labels supplied; overlays and morphology are descriptive model outputs')
    save(folder/'summary.json',summary);return summary

def run(args):
    import psutil
    cfg=setup(args);folder=args.output.resolve();print('Output folder:',folder,flush=True)
    from atlas_cellpose_3d import stage_sequence
    all_stages=stage_sequence(cfg)
    stages=all_stages[:all_stages.index(args.through_stage)+1] if getattr(args,'through_stage',None) else all_stages
    for stage in stages:
        checkpoint=folder/f'{stage}.completed.json'
        if checkpoint.exists():print('Using completed stage:',stage,flush=True);continue
        started=time.time();save(folder/'status.json',dict(status='running',stage=stage,started=started))
        print('Starting',stage,flush=True)
        with (folder/'logs'/f'{stage}.log').open('a') as log, (folder/'logs'/f'{stage}.log').open() as reader:
            reader.seek(0,2);pending='';inner_fraction=0.;inner_label=stage
            def relay():
                nonlocal pending,inner_fraction,inner_label
                pending+=reader.read()
                lines=pending.split('\n');pending=lines.pop()
                for line in lines:
                    if line.startswith('@segmentation-progress '):
                        event=json.loads(line.split(' ',1)[1]);inner_fraction=float(event['fraction']);inner_label=event['label']
                        print(line,flush=True)
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--output',str(folder),'--stage',stage],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            peak=0.;minimum_available=100.;low_count=0
            with (folder/'memory.jsonl').open('a') as samples:
                while process.poll() is None:
                    relay()
                    try:rss=psutil.Process(process.pid).memory_info().rss/2**30
                    except psutil.NoSuchProcess:rss=0
                    available=psutil.virtual_memory().available/2**30;peak=max(peak,rss);minimum_available=min(minimum_available,available)
                    samples.write(json.dumps(dict(time=time.time(),stage=stage,rssGiB=rss,availableGiB=available))+'\n');samples.flush()
                    low_count=low_count+1 if available<.5 or rss>11.5 else 0
                    if low_count>=3:
                        process.terminate();process.wait();save(folder/'status.json',dict(status='memory_guard_stopped',stage=stage,peakRSSGiB=peak,minAvailableGiB=minimum_available));raise RuntimeError(f'Memory safeguard stopped {stage}: peak sampled process memory {peak:.1f} GiB, lowest available memory {minimum_available:.1f} GiB. Close other memory-heavy apps, or lower processing scale and choose a new results folder. Completed stages remain cached.')
                    save(folder/'status.json',dict(status='running',stage=inner_label,phase=stage,taskFraction=(all_stages.index(stage)+inner_fraction)/len(all_stages),elapsedSeconds=time.time()-started,rssGiB=rss,peakRSSGiB=peak,availableGiB=available))
                    time.sleep(2)
            relay()
            if process.returncode!=0:
                save(folder/'status.json',dict(status='failed',stage=stage,returncode=process.returncode,log=str(folder/'logs'/f'{stage}.log')));raise RuntimeError(f'{stage} failed; see log')
        assert checkpoint.exists();print('Finished',stage,'peak sampled RSS',round(peak,2),'GiB',flush=True)
    if getattr(args,'through_stage',None):return
    assert source_stat(cfg['source'])==cfg['sourceStat'] and digest(cfg['source'])==cfg['sourceSHA256']
    summary=json.loads((folder/'summary.json').read_text());save(folder/'complete.json',dict(status='complete',sourceUnchanged=True,summary=summary,stages=[json.loads((folder/f'{s}.completed.json').read_text()) for s in all_stages]));save(folder/'status.json',dict(status='complete',**summary));print('COMPLETE',json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--scale',type=float,default=.35);p.add_argument('--diameter',type=float,default=30);p.add_argument('--stage',choices=STAGES);p.add_argument('--through-stage',choices=STAGES);p.add_argument('--spacing',type=float,nargs=3);p.add_argument('--model',choices=['cpsam_v2','cpdino','cpdino-vitb'],default='cpsam_v2');p.add_argument('--cpu',action='store_true');p.add_argument('--no-background',action='store_true');p.add_argument('--linking-mode',choices=['standard','streamed'],default='standard')
    p.add_argument('--segmentation-mode',choices=['usegment3d','cellpose3d'],default='usegment3d');p.add_argument('--flow-3d-smooth',type=float,default=0.);p.add_argument('--cellprob-threshold',type=float,default=0.);p.add_argument('--min-size-3d',type=int,default=15)
    args=p.parse_args()
    if args.stage:worker(args.output,args.stage)
    else:
        assert args.input is not None and 0<args.scale<=1 and args.diameter>0
        run(args)

if __name__=='__main__':main()
