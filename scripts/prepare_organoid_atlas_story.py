"""Build a data-linked organoid -> shape atlas -> nearest-neighbor storyboard.

Illustrative rigid transitions only: original surfaces, real UMAP coordinates,
full-descriptor neighbor ranking, and unchanged Atlas/benchmark inputs.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,struct,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from render_morphology_umap import ROOT,OUT,BG,font,read_mesh
from spatial_benchmark_core import surface_frame
from atlas_interactive import comparison_distances

STORY=OUT/'organoid_story'
FPS=20;FRAMES=320;WIDTH=1920;HEIGHT=1080

def smooth(t):
    t=np.clip(t,0,1);return t*t*(3-2*t)

def phase(t):
    if t<2:return 0,'Reconstruct the organoid','12 included cell surfaces · X206 · original relative positions'
    if t<3.5:return 1,'Meet the individual cells','Separate the surfaces while keeping each cell’s identity'
    if t<8:return 2,'Find each cell’s place in shape space','Spatial shape descriptors · UMAP fitted on 744 included, QC-passing cells'
    if t<13:return 3,'Find similar shapes across stacks','Query X206:L10 · 3 nearest matches ranked in the full descriptor space'
    return 0,'Every shape keeps its context','Return to the original reconstruction · illustrative motion, unchanged cell geometry'

def main():
    STORY.mkdir(parents=True,exist_ok=True);(STORY/'meshes').mkdir(exist_ok=True);(STORY/'overlays').mkdir(exist_ok=True)
    datafile=ROOT/'results/explorer_dataset/dataset.json';payload=datafile.read_bytes();data=json.loads(payload)
    provenance=json.loads((OUT/'provenance.json').read_text());assert hashlib.sha256(payload).hexdigest()==provenance['datasetSHA256']
    cells=sorted([c for c in data['cells'] if c.get('inMainOrganoid',True) and c['qc'] and all(k in c['representations'] for k in ['morph_all','spatial_robust'])],key=lambda c:c['id'])
    lookup={c['id']:i for i,c in enumerate(cells)};org=[i for i,c in enumerate(cells) if c['sample']=='X206'];qi=lookup['X206:L10']
    d=comparison_distances([c['representations']['spatial_robust']['vector'] for c in cells],'shape_hybrid95')
    nearest=[int(i) for i in np.argsort(d[qi],kind='stable') if cells[i]['sample']!='X206'][:3]
    selected=sorted({lookup[c] for c in provenance['selectedIDs']}|set(org)|set(nearest))
    original={};centers={};radii={};orientations={};hashes={}
    for i in selected:
        c=cells[i];v,f,digest=read_mesh(datafile.parent/c['sample']/f"{c['label']}.mesh");hashes[c['id']]=digest
        assert digest==next(m['meshSHA256'] for m in provenance['meshes'] if m['id']==c['id'])
        center=(v.min(0)+v.max(0))/2;radius=np.linalg.norm(v-center,axis=1).max();centers[i]=center;radii[i]=radius
        original[i]=v if i in org else None
        _,axes,_,_=surface_frame(v,f);canonical=(v-center)@axes
        for k in (0,1):
            if canonical[np.argmax(np.abs(canonical[:,k])),k]<0:axes[:,k]*=-1;axes[:,2]*=-1;canonical=(v-center)@axes
        # Original mesh orientation -> canonical display pose, proper rotations only.
        display=Rotation.from_euler('xy',[.32,-.4]).as_matrix();orientations[i]=Rotation.from_matrix(display@axes.T).as_quat().tolist()
        local=(v-center)/radius;t=local[f];ns=np.zeros_like(local);face=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
        for k in range(3):np.add.at(ns,f[:,k],face)
        ns/=np.maximum(np.linalg.norm(ns,axis=1,keepdims=True),1e-12)
        (STORY/'meshes'/(c['id'].replace(':','_')+'.mesh')).write_bytes(struct.pack('<II',len(v),len(f))+local.astype('<f4').tobytes()+ns.astype('<f4').tobytes()+f.astype('<u4').tobytes())
    cloud=np.concatenate([original[i] for i in org]);orgcenter=(cloud.min(0)+cloud.max(0))/2
    orgscale=310/np.linalg.norm(cloud-orgcenter,axis=1).max()
    xy=np.load(OUT/'spatial_robust_embedding.npz')['xy'].astype(float)
    rect=np.array([130,220,1660,650]);scale=min(rect[2:]/np.ptp(xy,axis=0));pos=(xy-(xy.min(0)+xy.max(0))/2)*scale
    pos[:,1]*=-1;pos+=rect[:2]+rect[2:]/2
    colors=['#77c9cb','#aaa0ee','#e49bb7','#85b5eb','#b8d293','#d0aaec','#89cdb1','#e2bc82','#99bedf','#b3a3dd','#83b4b4','#cca6b4']
    def rgb(s):return [int(s[k:k+2],16)/255 for k in [1,3,5]]
    nodes=[]
    for i in selected:
        c=cells[i];source=i in org;match=nearest.index(i)+1 if i in nearest else 0
        color=rgb('#f6a26e') if i==qi else rgb(colors[org.index(i)%len(colors)]) if source else rgb(['#74ded5','#b3a0ef','#8fc3f3'][match-1]) if match else rgb('#759cae')
        nodes.append(dict(id=c['id'],mesh=str(STORY/'meshes'/(c['id'].replace(':','_')+'.mesh')),
            source=source,sourceOrder=org.index(i) if source else -1,query=i==qi,match=match,color=color,
            organoidPosition=((centers[i]-orgcenter)*orgscale).tolist() if source else [0,0,0],
            organoidRadius=float(radii[i]*orgscale) if source else 1.,
            target=[float(pos[i,0]-WIDTH/2),float(HEIGHT/2-pos[i,1]),0.],targetRadius=25. if source or match else 16.,
            quaternion=orientations[i]))
    spec=dict(width=WIDTH,height=HEIGHT,fps=FPS,frames=FRAMES,nodes=nodes,output=str(STORY/'frames'),
        datasetSHA256=hashlib.sha256(payload).hexdigest(),sourceMeshes=hashes,organoid='X206',query=cells[qi]['id'],
        neighbors=[dict(id=cells[i]['id'],distance=float(d[qi,i])) for i in nearest],
        atlasShown=len(selected),atlasFitted=len(cells),method='spatial_robust',
        umapCoordinatesSHA256=hashlib.sha256((OUT/'spatial_robust_embedding.npz').read_bytes()).hexdigest(),
        geometry='Original triangles; rigid rotations, translations and uniform display scaling only; no morphing',
        motion='Illustrative staged transitions, not observed biological cell motion',
        querySelection='X206:L10 chosen as the narrative query, not by retrieval outcome')
    (STORY/'scene.json').write_text(json.dumps(spec,indent=2))
    # Per-frame text and selection graphics; map geometry is rendered separately.
    for frame in range(FRAMES):
        t=frame/FPS;stage,title,subtitle=phase(t)
        if t>15.2:stage,title,subtitle=phase(0)
        image=Image.new('RGBA',(WIDTH,HEIGHT),(0,0,0,0));draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,WIDTH,177),fill=BG)
        draw.text((65,30),'FROM ORGANOID TO SHAPE ATLAS',font=font(18,True),fill='#6bdad4')
        draw.text((60,67),title,font=font(46,True),fill='#f4f6fa')
        draw.text((65,133),subtitle,font=font(21),fill='#aab9c9')
        draw.rectangle((0,950,WIDTH,HEIGHT),fill=BG)
        names=['RECONSTRUCT','SEPARATE','MAP SHAPES','FIND MATCHES']
        for j,name in enumerate(names):
            x=65+j*460;color='#6bdad4' if j==stage else '#3b4a5d'
            draw.rounded_rectangle((x,984,x+410,989),radius=2,fill=color)
            draw.text((x,1005),f'0{j+1}  {name}',font=font(17,True),fill=color)
        draw.text((65,1043),'Actual segmented surfaces · display size normalized in the atlas · illustrative transitions, not cell motion',font=font(16),fill='#8194a9')
        # Ring and links identify full-distance neighbors at their REAL UMAP positions.
        a=float(smooth((t-8)/.6)*(1-smooth((t-9.5)/.4)))
        if a>0:
            qp=tuple(pos[qi]);draw.ellipse((qp[0]-36,qp[1]-36,qp[0]+36,qp[1]+36),outline=(246,162,110,int(a*255)),width=3)
            for ni in nearest:
                np_=tuple(pos[ni]);draw.line((qp,np_),fill=(113,218,211,int(a*160)),width=2)
                draw.ellipse((np_[0]-32,np_[1]-32,np_[0]+32,np_[1]+32),outline=(113,218,211,int(a*255)),width=2)
        card_alpha=float(smooth((t-10.6)/.4)*(1-smooth((t-13)/.4)))
        if card_alpha>0:
            for j,i in enumerate([qi]+nearest):
                cx=300+j*440
                draw.rounded_rectangle((cx-195,298,cx+195,878),radius=22,outline=(98,120,145,int(card_alpha*180)),width=2)
                color=('#f6a26e' if j==0 else '#73d9d2')
                draw.text((cx-165,326),'QUERY' if j==0 else f'MATCH 0{j}',font=font(17,True),fill=(*rgb255(color),int(card_alpha*255)))
                draw.text((cx-165,774),cells[i]['id'],font=font(29,True),fill=(238,243,250,int(card_alpha*255)))
                draw.text((cx-165,824),'X206 · source organoid' if j==0 else 'Different image stack',font=font(18),fill=(166,185,205,int(card_alpha*255)))
        image.save(STORY/'overlays'/f'{frame:04d}.png')
    print('Prepared',len(org),'organoid cells,',len(selected),'atlas cells,',FRAMES,'frames',flush=True)
    print('Nearest matches:',spec['neighbors'],flush=True)

def rgb255(s):return tuple(int(s[k:k+2],16) for k in [1,3,5])
if __name__=='__main__':main()
