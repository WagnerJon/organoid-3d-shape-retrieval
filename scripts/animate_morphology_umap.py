"""Prepare fixed UMAP positions and original surfaces for an offscreen SceneKit turntable."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,struct,hashlib,argparse
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image,ImageDraw
from render_morphology_umap import ROOT,OUT,BG,PALETTE,font,read_mesh
from spatial_benchmark_core import surface_frame


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all-cells',action='store_true');args=parser.parse_args()
    provenance=json.loads((OUT/'provenance.json').read_text())
    datafile=ROOT/'results/explorer_dataset/dataset.json';payload=datafile.read_bytes()
    assert hashlib.sha256(payload).hexdigest()==provenance['datasetSHA256']
    data=json.loads(payload);cells=sorted([c for c in data['cells'] if c.get('inMainOrganoid',True) and c['qc'] and all(k in c['representations'] for k in ['morph_all','spatial_robust'])],key=lambda c:c['id'])
    selected=list(range(len(cells))) if args.all_cells else [i for i,c in enumerate(cells) if c['id'] in provenance['selectedIDs']]
    mode='all_cells' if args.all_cells else 'showcase';folder=OUT/'animation'/mode;folder.mkdir(parents=True,exist_ok=True)
    meshdir=OUT/'animation/meshes';meshdir.mkdir(exist_ok=True)
    width,height=(2560,1600) if args.all_cells else (1920,1200)
    scale=width/1920
    background=Image.new('RGB',(width,height),BG[:3]);draw=ImageDraw.Draw(background)
    def text(x,y,value,size,color='#aab9c9',bold=False):draw.text((int(x*scale),int(y*scale)),value,font=font(int(size*scale),bold),fill=color)
    text(60,26,'3D ORGANOID SHAPE EXPLORATION',17,'#62dcd4',True)
    text(60,65,'Traditional morphometry',38,'#f4f6fa',True)
    text(1020,65,'Spatial shape descriptors',38,'#f4f6fa',True)
    text(60,120,'9 standardized size + shape features',19)
    text(1020,120,'Spatial PCA + ambiguity fallback',19)
    text(60,157,f'{len(selected)} cells shown in each map · 744 cells used for both UMAP fits',17)
    draw.line((int(960*scale),int(65*scale),int(960*scale),int(1080*scale)),fill='#253243',width=1)
    text(60,1100,'Original segmented surfaces · fixed UMAP positions · full 360° turn',18,'#dbe5ef')
    text(60,1136,'Size normalized for display · colors show sphericity · exploratory geometry, not cell types',16)
    text(1435,1100,'SPHERICITY',15,'#aab9c9',True)
    for x in range(350):
        col=tuple(int(c*255) for c in PALETTE(x/349)[:3]);xx=int((1435+x)*scale)
        draw.line((xx,int(1132*scale),xx,int(1145*scale)),fill=col,width=max(1,int(scale)))
    text(1435,1153,'0.55',13);text(1750,1153,'0.95',13)
    background.save(folder/'background.png')
    overlay=Image.new('RGBA',(width,height),(0,0,0,0));mask=Image.new('L',(width,height),0);md=ImageDraw.Draw(mask)
    # Headers and footer are static and composited after GPU rendering.
    md.rectangle((0,0,width,int(195*scale)),fill=255);md.rectangle((0,int(1080*scale),width,height),fill=255)
    overlay.paste(background,(0,0),mask);overlay.save(folder/'overlay.png')
    geometry={};source_hashes={}
    for i in selected:
        c=cells[i];v,f,digest=read_mesh(datafile.parent/c['sample']/f"{c['label']}.mesh")
        center,axes,_,_=surface_frame(v,f);v=(v-center)@axes
        for k in (0,1):
            if v[np.argmax(np.abs(v[:,k])),k]<0:v[:,k]*=-1;v[:,2]*=-1
        # Fixed center and isotropic radius across every frame: no breathing,
        # shape fitting, axis-wise scaling or frame-dependent recentering.
        v-=(v.min(0)+v.max(0))/2;v/=max(np.linalg.norm(v,axis=1).max(),1e-12)
        ns=np.zeros_like(v);t=v[f];face=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
        for k in range(3):np.add.at(ns,f[:,k],face)
        ns/=np.maximum(np.linalg.norm(ns,axis=1,keepdims=True),1e-12)
        dest=meshdir/(c['id'].replace(':','_')+'.mesh')
        dest.write_bytes(struct.pack('<II',len(v),len(f))+v.astype('<f4').tobytes()+ns.astype('<f4').tobytes()+f.astype('<u4').tobytes())
        geometry[i]=str(dest);source_hashes[c['id']]=digest
    nodes=[]
    for j,key in enumerate(['morph_all','spatial_robust']):
        points=np.load(OUT/(key+'_embedding.npz'))['xy'].astype(float)
        rect=np.array([55+j*960,215,850,815])*scale
        span=np.ptp(points,axis=0);s=min(rect[2]/span[0],rect[3]/span[1]);pos=(points-(points.min(0)+points.max(0))/2)*s
        pos[:,1]*=-1;pos+=rect[:2]+rect[2:]/2
        near=cdist(pos[selected],pos[selected]);np.fill_diagonal(near,np.inf)
        sizes=np.minimum((24 if args.all_cells else 53)*scale,np.maximum((10 if args.all_cells else 22)*scale,near.min(1)*.92))
        for k,i in enumerate(selected):
            c=cells[i];rgb=PALETTE(float(np.clip((c['metrics']['sphericity']-.55)/.4,0,1)))[:3]
            nodes.append(dict(id=c['id'],mesh=geometry[i],x=float(pos[i,0]-width/2),y=float(height/2-pos[i,1]),radius=float(sizes[k]*.46),color=list(rgb)))
    spec=dict(width=width,height=height,frames=120 if not args.all_cells else 72,fps=20 if not args.all_cells else 12,
              nodes=nodes,output=str(folder/'frames'),datasetSHA256=provenance['datasetSHA256'],sourceMeshes=source_hashes,
              displayedCellsPerMap=len(selected),fittedCells=744,rotation='one full proper rigid turn about the display vertical axis; no last-frame duplication',
              sourceCoordinatesSHA256=hashlib.sha256((OUT/'coordinates.csv').read_bytes()).hexdigest())
    (folder/'scene.json').write_text(json.dumps(spec,indent=2));print(folder/'scene.json')

if __name__=='__main__':main()
