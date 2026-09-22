"""Render representative cells and check grid sensitivity on 12 filename-spaced stacks."""
from analyze_all import ROOT,OUT
from analyze_morphology import properties
import json
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from skimage.measure import marching_cubes
from scipy import ndimage


def main():
    df=pd.read_csv(OUT/'all_cells.csv')
    paths={p.parent.name:p for root in [ROOT/'results/cpsam_v2_first5',ROOT/'results/cpsam_v2_remaining'] for p in root.glob('run_*/*/complete.json')}
    examples=[('X001:L3','Smallest matched cell'),('X206:L3','Largest matched cell'),('X212:L2','Most sphere-like'),('X216:L6','Most elongated')]
    geometries=[]
    for ident,title in examples:
        sample,label=ident.split(':L');path=paths[sample];meta=json.loads(path.read_text())
        labels=tifffile.imread(path.parent/'labels_isotropic.tif');mask=labels==int(label)
        crop=ndimage.find_objects(mask.astype(np.uint8))[0]
        verts,faces,_,_=marching_cubes(np.pad(mask[crop],2),.5,spacing=meta['effective_spacing_zyx_um'],step_size=2)
        verts=verts[:,::-1];verts-=verts.mean(0)
        geometries.append((ident,title,verts,faces))
    limit=max(np.abs(v).max() for _,_,v,_ in geometries)*1.08
    fig=plt.figure(figsize=(16,5))
    for i,(ident,title,verts,faces) in enumerate(geometries):
        ax=fig.add_subplot(1,4,i+1,projection='3d')
        poly=Poly3DCollection(verts[faces],facecolors=plt.get_cmap('tab10')(i),linewidths=0,shade=True,lightsource=LightSource(azdeg=315,altdeg=45))
        ax.add_collection3d(poly)
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit),xlabel='X (µm)',ylabel='Y (µm)',zlabel='Z (µm)')
        ax.set_box_aspect((1,1,1));ax.view_init(25,35);ax.tick_params(labelsize=7)
        row=df[df.cell_id==ident].iloc[0]
        ax.set_title(f'{title}\n{ident}\n{row.volume_um3:.0f} µm³ • axis ratio {row.aspect_ratio:.2f}',fontsize=10)
    fig.suptitle('Representative matched cells at a common physical scale',fontsize=16)
    fig.text(.5,.02,'Centered for display; no claim of shared identity. Coarser rendering only; measurements use full-resolution working labels.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.02,right=.98,bottom=.12,top=.80,wspace=.12)
    fig.savefig(OUT/'representative_cells.png',dpi=180);fig.savefig(OUT/'representative_cells.pdf');plt.close(fig)
    names=sorted(paths);selected=[names[i] for i in np.linspace(0,len(names)-1,12).round().astype(int)]
    rows=[]
    for sample in selected:
        path=paths[sample]
        spacing=json.loads((path.parent/'settings.json').read_text())['config']['spacing_zyx_um']
        native=properties(tifffile.imread(path.parent/'labels_native.tif'),spacing)
        for r in native:
            original=df[(df['sample']==sample)&(df.label==r['label'])].iloc[0]
            rows.append(dict(cell_id=original.cell_id,sample=sample,reference_matched=original.reference_matched,shape_qc_pass=original.shape_qc_pass,
                **{key+'_native_change_pct':100*(r[key]/original[key]-1) for key in ['volume_um3','surface_area_um2','sphericity']}))
        print('Grid sensitivity:',sample,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'grid_sensitivity_12_stacks.csv',index=False)


if __name__=='__main__':main()
