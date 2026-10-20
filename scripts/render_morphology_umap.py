"""Render original segmented cell surfaces at reproducible, full-distance UMAP positions.

No training, Atlas edits, mesh smoothing, or point-position displacement. A common
coverage subset makes the showcase legible; all-cell companion views retain everyone.
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'.cache/numba'))
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
sys.path.insert(0, str(ROOT/'.cache/umap-render-deps'))
import argparse, csv, hashlib, json, struct
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
from atlas_interactive import comparison_distances
from spatial_benchmark_core import surface_frame

OUT = ROOT/'results/morphology_umap'
PALETTE = LinearSegmentedColormap.from_list('shape', ['#f5a36b','#b5a0e9','#62dcd4'])
BG = (12,20,31,255)
FONT = font_manager.findfont('DejaVu Sans')
BOLD = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))
RENDER_VERSION = '3-isotropic-display-smooth-lighting-original-triangles'

def font(n, bold=False): return ImageFont.truetype(BOLD if bold else FONT,n)

def read_mesh(path):
    payload=path.read_bytes(); nv,nf=struct.unpack('<II',payload[:8])
    if len(payload)!=8+nv*24+nf*12: raise ValueError('Invalid source mesh')
    v=np.frombuffer(payload,dtype='<f4',count=nv*3,offset=8).reshape(-1,3).astype(float)
    f=np.frombuffer(payload,dtype='<u4',count=nf*3,offset=8+nv*24).reshape(-1,3)
    return v,f,hashlib.sha256(payload).hexdigest()

def sprite(job):
    cell, folder, out = job
    path=Path(folder)/cell['sample']/f"{cell['label']}.mesh"
    v,f,digest=read_mesh(path)
    dest=Path(out)/(cell['id'].replace(':','_')+'.png')
    metadata=dest.with_suffix('.json')
    if dest.exists() and metadata.exists() and json.loads(metadata.read_text()).get('cacheKey')==[digest,RENDER_VERSION,cell['metrics']['sphericity']]:
        return dict(id=cell['id'], meshSHA256=digest)
    center, axes, _, _ = surface_frame(v,f)
    # Only a proper rigid display transform; each cell is uniformly scaled to
    # comparable glyph extent. Original segmentation terraces are retained.
    v=(v-center)@axes
    for k in (0,1):
        if v[np.argmax(np.abs(v[:,k])),k] < 0:
            v[:,k]*=-1; v[:,2]*=-1
    az,el=np.deg2rad(-32),np.deg2rad(24)
    right=np.array([np.cos(az),np.sin(az),0])
    up=np.array([-np.sin(az)*np.sin(el),np.cos(az)*np.sin(el),np.cos(el)])
    eye=np.cross(right,up)
    q=v@np.stack([right,up,eye],axis=1)
    q[:,:2]-=(q[:,:2].min(0)+q[:,:2].max(0))/2
    q/=max(np.ptp(q[:,:2],axis=0).max(),1e-12)
    tri=q[f]; face_normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    # Smooth LIGHTING only: no vertex positions, segmentation boundaries or
    # mesh topology change. Area-weighted vertex normals soften voxel facets.
    vertex_normals=np.zeros_like(q)
    for k in range(3): np.add.at(vertex_normals,f[:,k],face_normals)
    vertex_normals/=np.maximum(np.linalg.norm(vertex_normals,axis=1,keepdims=True),1e-12)
    normals=vertex_normals[f].mean(1)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    light=np.array([-.45,.65,1.]); light/=np.linalg.norm(light)
    lambert=np.maximum(normals@light,0)
    brightness=.55+.50*lambert
    spec=.20*np.maximum(normals@np.array([-.2,.4,.8944]),0)**18
    t=np.clip((cell['metrics']['sphericity']-.55)/.4,0,1)
    base=np.array(PALETTE(float(t))[:3])*255
    colors=np.clip(base[None,:]*brightness[:,None]+255*spec[:,None],0,255).astype(np.uint8)
    # Back-to-front orthographic triangle rendering, supersampled for soft edges.
    size=640; image=Image.new('RGBA',(size,size),(0,0,0,0)); draw=ImageDraw.Draw(image)
    xy=tri[:,:,:2]*size*.83; xy[:,:,1]*=-1; xy+=size/2
    for k in np.argsort(tri[:,:,2].mean(1)):
        draw.polygon([tuple(p) for p in xy[k]],fill=tuple(int(a) for a in colors[k])+(255,))
    image=image.resize((320,320),Image.Resampling.LANCZOS); image.save(dest)
    metadata.write_text(json.dumps(dict(meshSHA256=digest,cacheKey=[digest,RENDER_VERSION,cell['metrics']['sphericity']],display='proper PCA frame; equal projected longest extent; original triangles; smooth lighting only')))
    return dict(id=cell['id'],meshSHA256=digest)

def coverage_subset(points,count):
    # Deterministic farthest-point coverage of BOTH independently fitted maps;
    # same cell IDs on each showcase, without choosing by cell shape or outcome.
    maps=[(p-p.min(0))/np.maximum(np.ptp(p,axis=0).max(),1e-12) for p in points]
    selected=[int(np.argmin(sum(np.square(p-np.median(p,axis=0)).sum(1) for p in maps)))]
    nearest=[np.full(len(maps[0]),np.inf) for _ in maps]
    while len(selected)<min(count,len(maps[0])):
        for p,best in zip(maps,nearest): np.minimum(best,np.square(p-p[selected[-1]]).sum(1),out=best)
        score=np.maximum.reduce(nearest);score[selected]=-1
        selected.append(int(np.argmax(score)))
    return sorted(selected)

def draw_map(points,cells,selected,key,title,subtitle,out,all_cells=False):
    size=4800; im=Image.new('RGBA',(size,size),BG); draw=ImageDraw.Draw(im)
    draw.text((200,130),'CELL SHAPE ATLAS',font=font(37,True),fill='#62dcd4')
    draw.text((195,205),title,font=font(106,True),fill='#f4f6fa')
    draw.text((200,355),subtitle,font=font(39),fill='#aab9c9')
    draw.text((200,415),f"{len(selected)} original cell surfaces shown · {len(cells)} cells used to fit UMAP",font=font(34),fill='#aab9c9')
    rect=np.array([240,650,4320,3550],float)
    span=np.ptp(points,axis=0); scale=min(rect[2]/span[0],rect[3]/span[1])
    pos=(points-(points.min(0)+points.max(0))/2)*scale
    pos[:,1]*=-1;pos+=rect[:2]+rect[2:]/2
    p=pos[selected]; near=cdist(p,p);np.fill_diagonal(near,np.inf)
    # At exact UMAP coordinates. Do not repel, tile or snap the meshes.
    glyph=100 if all_cells else 235
    widths=np.minimum(glyph,np.maximum(34 if all_cells else 95,near.min(1)*.92))
    for ix in np.argsort(widths)[::-1]:
        i=selected[int(ix)]; w=int(widths[ix]);cell=cells[i]
        s=Image.open(out/'sprites'/(cell['id'].replace(':','_')+'.png')).convert('RGBA').resize((w,w),Image.Resampling.LANCZOS)
        x,y=pos[i];im.alpha_composite(s,(int(x-w/2),int(y-w/2)))
    draw=ImageDraw.Draw(im)
    draw.line([(230,4210),(470,4210)],fill='#8093a8',width=3)
    draw.line([(230,4210),(230,3970)],fill='#8093a8',width=3)
    draw.text((245,4230),'UMAP 1',font=font(28),fill='#8093a8')
    draw.text((80,3920),'UMAP 2',font=font(28),fill='#8093a8')
    # Shared, explicit physical-shape color scale; no invented group labels.
    for x in range(1050):
        color=tuple(int(c*255) for c in PALETTE(x/1049)[:3])
        draw.line([(200+x,4435),(200+x,4470)],fill=color,width=1)
    draw.text((200,4370),'SPHERICITY',font=font(29,True),fill='#dbe5ef')
    draw.text((200,4490),'0.55',font=font(28),fill='#aab9c9');draw.text((1160,4490),'0.95',font=font(28),fill='#aab9c9')
    draw.text((1500,4380),'Normalized cell size · smaller glyphs in crowded regions · common pose',font=font(28),fill='#aab9c9')
    draw.text((1500,4430),'Included, QC-passing cells only · n_neighbors 30 · min_dist 0.30 · seed 42',font=font(28),fill='#aab9c9')
    draw.text((200,4630),'Exploratory layout; distances between islands do not establish biological cell types.',font=font(30),fill='#8295aa')
    target=out/(key+('_all_cells' if all_cells else '')+'.png')
    im.convert('RGB').save(target,dpi=(300,300))
    im.convert('RGB').save(target.with_suffix('.pdf'),resolution=300.)
    print('Rendered',target.name,flush=True)
    return target

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--glyphs',type=int,default=180)
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True);(OUT/'sprites').mkdir(exist_ok=True)
    path=ROOT/'results/explorer_dataset/dataset.json';payload=path.read_bytes();data=json.loads(payload)
    keys=['morph_all','spatial_robust']; methods={m['id']:m for m in data['methods']}
    cells=[c for c in data['cells'] if c.get('inMainOrganoid',True) and c['qc'] and all(k in c['representations'] for k in keys)]
    cells.sort(key=lambda c:c['id']);points=[];matrices=[]
    for key in keys:
        d=comparison_distances([c['representations'][key]['vector'] for c in cells],methods[key]['metric']);matrices.append(d)
        cache=OUT/(key+'_embedding.npz');fingerprint=hashlib.sha256(d.tobytes()).hexdigest()
        if cache.exists() and str(np.load(cache)['fingerprint'])==fingerprint:
            xy=np.load(cache)['xy']
        else:
            from umap import UMAP
            print('Fitting UMAP:',key,len(cells),flush=True)
            xy=UMAP(n_components=2,n_neighbors=30,min_dist=.30,metric='precomputed',random_state=42,n_jobs=1,init='spectral').fit_transform(d)
            np.savez(cache,xy=xy,fingerprint=fingerprint)
        points.append(xy.astype(float))
    selected=coverage_subset(points,args.glyphs)
    with ProcessPoolExecutor(max_workers=3) as pool:
        meshes=[]
        for i,result in enumerate(pool.map(sprite,[(c,str(path.parent),str(OUT/'sprites')) for c in cells],chunksize=4)):
            meshes.append(result)
            if i%75==0:print('Rendered cell surfaces',i+1,'/',len(cells),flush=True)
    titles=['Traditional morphometry','Spatial shape descriptors']
    subtitles=['9 standardized size + shape features · Euclidean distance','95D spatial PCA · proper sign frames + ambiguity fallback']
    files=[]
    for key,p,title,subtitle in zip(keys,points,titles,subtitles):
        files.append(draw_map(p,cells,selected,key,title,subtitle,OUT))
        draw_map(p,cells,list(range(len(cells))),key,title,subtitle,OUT,all_cells=True)
    with (OUT/'coordinates.csv').open('w') as h:
        writer=csv.writer(h);writer.writerow(['cell_id','showcase','sphericity','traditional_umap1','traditional_umap2','spatial_umap1','spatial_umap2'])
        for i,c in enumerate(cells):writer.writerow([c['id'],i in selected,c['metrics']['sphericity'],*points[0][i],*points[1][i]])
    # Comparison at readable preview resolution; full-resolution panels separate.
    previews=[Image.open(p).resize((1800,1800),Image.Resampling.LANCZOS) for p in files]
    pair=Image.new('RGB',(3600,1800));pair.paste(previews[0],(0,0));pair.paste(previews[1],(1800,0));pair.save(OUT/'comparison.png')
    previews=[Image.open(OUT/(k+'_all_cells.png')).resize((1800,1800),Image.Resampling.LANCZOS) for k in keys]
    pair=Image.new('RGB',(3600,1800));pair.paste(previews[0],(0,0));pair.paste(previews[1],(1800,0));pair.save(OUT/'comparison_all_cells.png')
    provenance=dict(datasetSHA256=hashlib.sha256(payload).hexdigest(),cells=len(cells),showcaseCells=len(selected),
                    selectedIDs=[cells[i]['id'] for i in selected],methods=[methods[k] for k in keys],
                    umapVersion=version('umap-learn'),n_neighbors=30,min_dist=.30,seed=42,metric='precomputed',n_jobs=1,
                    display='original triangles, smooth lighting only; proper PCA display pose; equal projected extent; exact UMAP coordinates',
                    selection='common deterministic farthest-coverage subset of both maps; all cells in companion figures',
                    methodRationale='spatial_robust had lowest observed matched-validation geometry retrieval cost, 0.022884; not a biological ranking',
                    color='sphericity shared 0.55–0.95, clipped',meshes=meshes,
                    renderScriptSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==provenance['datasetSHA256']
    print('Finished; source dataset unchanged.',flush=True)

if __name__=='__main__':main()
