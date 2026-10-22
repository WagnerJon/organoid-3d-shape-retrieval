"""Local OME-NGFF 0.4/0.5 import: explicit T/C/level selection, bounded ZYX reads.

Time points are independent Atlas samples, never inferred biological replicates
or track identities. Original OME-Zarrs remain read-only.
"""
import hashlib
import json
import math
import os
from pathlib import Path

UNITS={'micrometer':1.,'µm':1.,'μm':1.,'um':1.,'nanometer':.001,'nm':.001,'millimeter':1000.,'mm':1000.,'meter':1e6,'m':1e6}
MAX_CHUNK=256*1024**2


def child(root,relative):
    root=Path(root).resolve();p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('OME-Zarr paths must stay inside the selected image store')
    result=(root/p).resolve()
    if not result.is_relative_to(root):raise ValueError('OME-Zarr path escapes its store')
    return result


def attrs(path):
    path=Path(path)
    if (path/'zarr.json').is_file():return json.loads((path/'zarr.json').read_text()).get('attributes',{})
    if (path/'.zattrs').is_file():return json.loads((path/'.zattrs').read_text())
    return {}


def ome(path):
    value=attrs(path);return value.get('ome',value)


def fingerprint(root):
    """Detect changed chunks as well as metadata, without reading all image bytes."""
    root=Path(root).resolve();h=hashlib.sha256();size=0;modified=0
    # GEFF tracks and Parquet tables are unrelated to the imported image/labels.
    files=[]
    for directory,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in ('tracks','tables') and not d.startswith('._'))
        files.extend(Path(directory)/name for name in names)
    for p in sorted(files):
        if not p.is_file() or p.name.startswith('._') or p.name=='.DS_Store':continue
        if p.is_symlink():raise ValueError('Use a local OME-Zarr store without symlinked files')
        s=p.stat();size+=s.st_size;modified=max(modified,s.st_mtime_ns)
        h.update(json.dumps([p.relative_to(root).as_posix(),s.st_size,s.st_mtime_ns],separators=(',',':')).encode())
        if p.name in ('zarr.json','.zattrs','.zarray','.zgroup','.zmetadata'):h.update(p.read_bytes())
    return h.hexdigest(),[size,modified]


def transform(axes,transforms):
    scale=[1.]*len(axes);offset=[0.]*len(axes)
    for t in transforms:
        kind=t.get('type')
        if kind=='identity':continue
        if kind not in ('scale','translation') or kind not in t or len(t[kind])!=len(axes):raise ValueError('Unsupported OME coordinate transformation')
        values=[float(x) for x in t[kind]]
        if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite OME coordinate transformation')
        if kind=='scale':
            scale=[a*b for a,b in zip(scale,values)];offset=[a*b for a,b in zip(offset,values)]
        else:offset=[a+b for a,b in zip(offset,values)]
    return scale,offset


def describe(root,allow_labels=False):
    import numpy as np
    root=Path(root).resolve();metadata=ome(root);multi=metadata.get('multiscales',[])
    if len(multi)!=1:raise ValueError('Select an image group containing exactly one named multiscale image')
    if 'image-label' in metadata and not allow_labels:raise ValueError('Select the intensity image; its labels can be selected in the import window')
    m=multi[0];axes=m['axes'];axes=[a if isinstance(a,dict) else dict(name=a) for a in axes]
    names=[a['name'].lower() for a in axes]
    if len(names)!=len(set(names)) or not set('zyx').issubset(names) or set(names)-set('tczyx'):raise ValueError('OME-Zarr must declare unique T/C/Z/Y/X axes with all three spatial dimensions')
    levels=[]
    for index,ds in enumerate(m['datasets']):
        path=child(root,ds['path'])
        meta_path=path/'zarr.json' if (path/'zarr.json').is_file() else path/'.zarray'
        info=json.loads(meta_path.read_text());shape=info['shape']
        if len(shape)!=len(names) or any(type(n) is not int or n<1 for n in shape):raise ValueError('Array shape does not match OME axes')
        if info.get('dimension_names') is not None and [a.lower() for a in info['dimension_names']]!=names:raise ValueError('Array dimension names do not match declared OME axes')
        dtype=np.dtype(info.get('data_type',info.get('dtype')))
        if dtype.kind not in 'uif' or dtype.itemsize>8:raise ValueError('Only numeric intensity/label arrays are supported')
        chunks=info.get('chunks') or info['chunk_grid']['configuration']['chunk_shape']
        scale,origin=transform(axes,ds.get('coordinateTransformations',[])+m.get('coordinateTransformations',[]))
        units=[UNITS.get(axes[names.index(a)].get('unit','')) for a in 'zyx']
        spacing=[scale[names.index(a)]*u for a,u in zip('zyx',units)] if all(u is not None for u in units) else None
        offset=[origin[names.index(a)]*u for a,u in zip('zyx',units)] if spacing else None
        if spacing and not all(x>0 for x in spacing):raise ValueError('Positive physical voxel sizes required')
        levels.append(dict(index=index,path=ds['path'],shape=shape,shapeZYX=[shape[names.index(a)] for a in 'zyx'],dtype=str(dtype),spacing=spacing,origin=offset,
                           chunkBytes=math.prod(chunks)*dtype.itemsize,chunkShape=chunks))
    if not levels:raise ValueError('OME-Zarr image has no resolution levels')
    first=levels[0];count_t=first['shape'][names.index('t')] if 't' in names else 1
    count_c=first['shape'][names.index('c')] if 'c' in names else 1
    for level in levels:
        if ('t' in names and level['shape'][names.index('t')]!=count_t) or ('c' in names and level['shape'][names.index('c')]!=count_c):raise ValueError('Pyramid levels must retain the same time points and channels')
    channel_metadata=metadata.get('omero',{}).get('channels',[])
    dca=attrs(root).get('dca',{})
    channel_labels=[str(channel_metadata[c].get('label',f'Channel {c}')) if c<len(channel_metadata) else f'Channel {c}' for c in range(count_c)]
    times=dca.get('acquisition',{}).get('timestamps',[])
    timestamps=times if len(times)==count_t and all(isinstance(x,(int,float)) and math.isfinite(x) for x in times) else []
    return dict(path=str(root),name=root.name,axes=names,axisMetadata=axes,levels=levels,channels=channel_labels,timepoints=count_t,timestamps=timestamps,labels=[])


def inspect_store(root):
    info=describe(root)
    labels=ome(Path(root)/'labels').get('labels',[])
    for name in labels:
        p=child(root,'labels/'+name)
        label=describe(p,allow_labels=True)
        semantic=any(s in str(name).lower() for s in ['lumen','epithelium'])
        info['labels'].append(dict(name=str(name),path=str(p),levels=label['levels'],axes=label['axes'],timepoints=label['timepoints'],instanceEligible=not semantic))
    info['fingerprint'],info['sourceStat']=fingerprint(root)
    return info


def discover(paths):
    images=[];ignored=[]
    for given in paths:
        root=Path(given).resolve()
        candidates=[root] if ome(root).get('multiscales') else [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith('.') and ome(p).get('multiscales')]
        for path in candidates:
            try:
                preview=describe(path)
                if min(preview['levels'][0]['shapeZYX'])<2:ignored.append(path.name+' · 2D projection, not a 3D stack');continue
                images.append(inspect_store(path))
            except ValueError as e:ignored.append(path.name+' · '+str(e))
    if not images:raise ValueError('No 3D OME-Zarr images found. Select an extracted .ome.zarr directory or its immediate parent. ZIP files and projections are not 3D inputs.')
    return dict(images=images,ignored=ignored)


def selection_entry(image,time,channel,level,label=None):
    if type(time) is not int or not 0<=time<image['timepoints'] or type(channel) is not int or not 0<=channel<len(image['channels']):raise ValueError('Time point or channel outside the image')
    if type(level) is not int or not 0<=level<len(image['levels']):raise ValueError('Resolution level outside the image')
    grid=image['levels'][level]
    if min(grid['shapeZYX'])<2:raise ValueError('Select a resolution with at least two samples along every spatial axis')
    if grid['chunkBytes']>MAX_CHUNK:raise ValueError('Decoded chunks exceed 256 MiB. Choose a coarser pyramid level or rechunk the source before importing on this Mac.')
    if label:
        labels=next((x for x in image['labels'] if x['name']==label),None)
        if not labels or not labels['instanceEligible']:raise ValueError('Choose cell or nucleus instance labels; lumen/epithelium masks are semantic classes')
        if level>=len(labels['levels']):raise ValueError('Label resolution level is unavailable')
        mask=labels['levels'][level]
        import numpy as np
        if np.dtype(mask['dtype']).kind not in 'ui':raise ValueError('Instance label arrays must use an integer dtype')
        if labels['timepoints'] not in (1,image['timepoints']):raise ValueError('Label time points do not match the image')
        if mask['chunkBytes']>MAX_CHUNK:raise ValueError('Label chunks exceed 256 MiB; choose a coarser level')
        import numpy as np
        if mask['shapeZYX']!=grid['shapeZYX'] or not grid['spacing'] or not mask['spacing'] or not np.allclose(grid['spacing'],mask['spacing'],rtol=1e-6,atol=1e-8) or not np.allclose(grid['origin'],mask['origin'],rtol=1e-6,atol=1e-8):raise ValueError('Labels must share the image spatial grid, voxel sizes and origin at the chosen level')
        if 'c' in labels['axes'] and mask['shape'][labels['axes'].index('c')]!=1:raise ValueError('Instance labels must have a singleton channel axis')
    selection=dict(time=time,channel=channel,level=level,label=label,fingerprint=image['fingerprint'],origin=grid['origin'],timestamp=image['timestamps'][time] if image['timestamps'] else None)
    name=f"{image['name']} · T{time} · {image['channels'][channel]} · level {level}"
    return dict(path=image['path'],name=name,shape=grid['shapeZYX'],axes='ZYX',dtype=grid['dtype'],spacing=grid['spacing'],calibrationSource='OME-NGFF level-specific spatial transforms (µm)' if grid['spacing'] else 'Enter voxel spacing in micrometers; OME spatial units are missing or unsupported',sourceStat=image['sourceStat'],omeZarr=selection)


def inspect_entry(entry,cache=None):
    path=entry['path'];cache={} if cache is None else cache
    if path not in cache:cache[path]=inspect_store(path)
    info=cache[path];s=entry['omeZarr']
    if s['fingerprint']!=info['fingerprint']:raise ValueError('OME-Zarr metadata or chunks changed since selection. Import it again.')
    return selection_entry(info,s['time'],s['channel'],s['level'],s.get('label'))


def open_array(path):
    try:import zarr
    except ImportError as e:raise RuntimeError('OME-Zarr reader is missing. Install the pinned zarr==3.0.10 and numcodecs==0.15.1 runtime requirements.') from e
    zarr.config.set({'async.concurrency':2,'threading.max_workers':2})
    return zarr.open_array(str(path),mode='r')


def read_spatial(array,axes,time,channel,spatial):
    import numpy as np
    indexes=tuple(time if a=='t' else channel if a=='c' else spatial.get(a,slice(None)) for a in axes)
    remaining=[a for a,index in zip(axes,indexes) if isinstance(index,slice)]
    target=[a for a in 'zyx' if a in remaining]
    data=np.asarray(array[indexes])
    return data.transpose([remaining.index(a) for a in target])


def read_slice(entry,plane,index):
    from atlas_slice_test import PLANES
    import numpy as np
    info=inspect_store(entry['path']);inspect_entry(entry,{entry['path']:info})
    s=entry['omeZarr'];grid=info['levels'][s['level']];axis,dimensions=PLANES[plane]
    if type(index) is not int or not 0<=index<grid['shapeZYX'][axis]:raise ValueError('Slice index outside the selected Zarr volume')
    if math.prod(grid['shapeZYX'][i] for i in dimensions)>32_000_000:raise ValueError('Preview exceeds 32 million pixels')
    array=open_array(child(entry['path'],grid['path']))
    data=read_spatial(array,info['axes'],s['time'],s['channel'],{'zyx'[axis]:index})
    if not np.isfinite(data).all():raise ValueError('Slice contains nonfinite intensities')
    return data,dimensions


def materialize(entry,destination,report=None):
    """Create just the selected T/C/level as streamed, calibrated OME-TIFFs."""
    import numpy as np
    import tifffile
    import shutil
    from prepare_atlas_dataset import save,inspect
    from run_standalone_cellpose import digest
    info=inspect_store(entry['path']);inspect_entry(entry,{entry['path']:info});s=entry['omeZarr'];grid=info['levels'][s['level']]
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest();done=root/'import.completed.json'
    if done.exists():
        saved=json.loads(done.read_text())
        if saved['identity']!=identity:raise ValueError('Import cache belongs to a different selection')
        for name,checksum in saved['files'].items():
            if digest(root/name)!=checksum:raise ValueError('Cached OME-Zarr import checksum mismatch; choose a new results folder')
    else:
        files={};objects=[('raw.ome.tif',info,s['time'],s['channel'])]
        if s.get('label'):
            label=next(x for x in info['labels'] if x['name']==s['label']);objects.append(('labels.ome.tif',label,s['time'] if label['timepoints']>1 else 0,0))
        required=sum(math.prod(x['levels'][s['level']]['shapeZYX'])*np.dtype(x['levels'][s['level']]['dtype']).itemsize for _,x,_,_ in objects)
        if required+512*1024**2>shutil.disk_usage(root).free:raise ValueError('Insufficient disk space for selected OME-Zarr volumes')
        for filename,obj,t,c in objects:
            level=obj['levels'][s['level']];array=open_array(child(obj['path'],level['path']));shape=level['shapeZYX'];dtype=np.dtype(level['dtype'])
            depth=max(1,min(level['chunkShape'][obj['axes'].index('z')],(64*1024**2)//max(1,shape[1]*shape[2]*dtype.itemsize)))
            def planes():
                for start in range(0,shape[0],depth):
                    stop=min(start+depth,shape[0])
                    block=read_spatial(array,obj['axes'],t,c,dict(z=slice(start,stop)))
                    if not np.isfinite(block).all():raise ValueError('OME-Zarr selection contains nonfinite values')
                    if filename.startswith('labels') and (dtype.kind not in 'ui' or block.min()<0 or block.max()>65535):raise ValueError('Atlas instance labels must be integers in 0–65535')
                    if report:report(f"Importing {filename} · Z {start+1}–{stop}/{shape[0]}")
                    yield from block
            temporary=root/(filename+'.tmp')
            try:
                spacing=entry['spacing']
                with tifffile.TiffWriter(temporary,ome=True,bigtiff=True) as writer:
                    writer.write(planes(),shape=tuple(shape),dtype=dtype,photometric='minisblack',metadata=dict(axes='ZYX',Name=entry['name'],PhysicalSizeZ=spacing[0],PhysicalSizeY=spacing[1],PhysicalSizeX=spacing[2],PhysicalSizeZUnit='µm',PhysicalSizeYUnit='µm',PhysicalSizeXUnit='µm'))
                temporary.replace(root/filename);files[filename]=digest(root/filename)
            finally:temporary.unlink(missing_ok=True)
        inspect_entry(entry)  # Never commit a mixed-time snapshot of a changed store.
        save(root/'source.json',dict(source=entry,axes=info['axisMetadata'],omeMetadata=ome(entry['path']),array=grid['path'],spatialOriginZYXum=grid['origin'],timestamps=info['timestamps'],coordinates='Exported cell coordinates are local to this selected volume; original spatial origin retained here',tracking='Frames are independent samples; GEFF lineage and persistent track identities are not imported'))
        save(done,dict(identity=identity,files=files))
    result=inspect(root/'raw.ome.tif');result.update(name=entry['name'],spacing=entry['spacing'])
    if s.get('label'):
        label=root/'labels.ome.tif';st=label.stat();result.update(labels=str(label),labelStat=[st.st_size,st.st_mtime_ns])
    return result
