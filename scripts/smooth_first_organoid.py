"""Non-destructive mesh-smoothing pilot for all six X001 predictions."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import json
import hashlib
import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SOURCE=ROOT/'results/cpsam_v2_first5/run_47ae813661eb/X001/meshes'
OUT=ROOT/'results/smoothing_X001'
VARIANTS={'original':0,'gentle':30,'stronger':100}


def display(ax,meshes,bounds,title,labels=False):
    for label,mesh in meshes:
        color=plt.get_cmap('tab10')((label-1)%10)
        ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolors=color,linewidths=0,
            shade=True,lightsource=LightSource(azdeg=315,altdeg=45)))
        if labels:
            center=mesh.vertices.mean(0)
            ax.text(*center,f'L{label}',fontsize=9)
    lo,hi=bounds
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='X (µm)',ylabel='Y (µm)',zlabel='Z (µm)')
    ax.set_box_aspect(hi-lo);ax.view_init(25,35);ax.set_title(title,fontsize=12,pad=16);ax.tick_params(labelsize=7)


def main():
    OUT.mkdir(exist_ok=True)
    files=sorted(SOURCE.glob('cell_*.ply'))
    assert len(files)==6
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    variants={key:[] for key in VARIANTS}
    rows=[]
    for file in files:
        label=int(file.stem.split('_')[-1]);original=trimesh.load(file,process=False)
        assert original.is_watertight and original.volume>0
        for variant,passes in VARIANTS.items():
            mesh=original.copy()
            if passes:
                trimesh.smoothing.filter_taubin(mesh,lamb=.5,nu=.53,iterations=passes)
                (OUT/variant).mkdir(exist_ok=True)
                mesh.export(OUT/variant/file.name)
                reloaded=trimesh.load(OUT/variant/file.name,process=False)
                assert reloaded.is_watertight and reloaded.is_winding_consistent and reloaded.volume>0
                assert np.array_equal(reloaded.faces,original.faces)
                assert mesh.euler_number==original.euler_number
            moved=np.linalg.norm(mesh.vertices-original.vertices,axis=1)
            variants[variant].append((label,mesh))
            rows.append(dict(label=label,reference_matched=label<=4,variant=variant,passes=passes,
                mesh_volume_um3=mesh.volume,area_um2=mesh.area,
                volume_change_pct=100*(mesh.volume/original.volume-1),area_change_pct=100*(mesh.area/original.area-1),
                vertex_movement_mean_um=moved.mean(),vertex_movement_p95_um=np.percentile(moved,95),vertex_movement_max_um=moved.max(),
                watertight=mesh.is_watertight,winding_consistent=mesh.is_winding_consistent))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'comparison_metrics.csv',index=False)
    for variant in VARIANTS:
        scene=trimesh.Scene()
        for label,m in variants[variant]:
            m=m.copy();m.visual.face_colors=(np.array(plt.get_cmap('tab10')((label-1)%10))*255).astype(np.uint8)
            scene.add_geometry(m,node_name=f'cell_{label:04d}')
        scene.export(OUT/f'{variant}_all_cells.glb')
    # Cell one, full-resolution surfaces, identical camera and shading across all panels.
    original=variants['original'][0][1]
    bounds=original.bounds.copy();pad=(bounds[1]-bounds[0])*.08;bounds[0]-=pad;bounds[1]+=pad
    fig=plt.figure(figsize=(15,5.7))
    for i,(variant,passes) in enumerate(VARIANTS.items()):
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        display(ax,[variants[variant][0]],bounds,f'{variant.title()}'+(f' • {passes} passes' if passes else ''))
    fig.suptitle('X001 cell L1 — actual mesh geometry before and after smoothing',fontsize=16)
    fig.text(.5,.03,'Full-resolution triangles • Same physical scale, camera and flat lighting • No decimation or smoothing by the renderer',ha='center',fontsize=9)
    fig.subplots_adjust(left=.015,right=.98,bottom=.12,top=.82,wspace=.08)
    fig.savefig(OUT/'single_cell_comparison.png',dpi=200);fig.savefig(OUT/'single_cell_comparison.pdf');plt.close(fig)
    # Main four-cell group; retain and export unmatched L5/L6 without hiding their existence.
    original_vertices=np.vstack([m.vertices for label,m in variants['original'] if label<=4])
    bounds=np.array([original_vertices.min(0),original_vertices.max(0)]);pad=(bounds[1]-bounds[0])*.08;bounds[0]-=pad;bounds[1]+=pad
    fig=plt.figure(figsize=(15,5.7))
    for i,variant in enumerate(VARIANTS):
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        display(ax,[(label,m) for label,m in variants[variant] if label<=4],bounds,variant.title())
    fig.suptitle('X001 main four-cell group — original, gentle and stronger smoothing',fontsize=16)
    fig.text(.5,.03,'All six predicted cells were processed; the two distant unmatched objects are included in the GLB scenes and PLY folders.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.015,right=.98,bottom=.12,top=.82,wspace=.08)
    fig.savefig(OUT/'organoid_comparison.png',dpi=200);plt.close(fig)
    # Cross-section through the first cell reveals geometric stair steps independently of shading.
    center=original.center_mass
    fig,ax=plt.subplots(figsize=(7,7))
    for variant,color in zip(VARIANTS,['#737373','#1673b1','#d16b22']):
        mesh=variants[variant][0][1]
        section=mesh.section(plane_origin=center,plane_normal=[0,1,0])
        for i,line in enumerate(section.discrete):
            ax.plot(line[:,0],line[:,2],color=color,label=variant.title() if i==0 else None,lw=1.4)
    ax.set_aspect('equal');ax.set_xlabel('X (µm)');ax.set_ylabel('Z (µm)');ax.legend()
    ax.set_title('Cell L1: common XZ mesh section\nOriginal stair steps versus smoothed contours');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'section_comparison.png',dpi=200);plt.close(fig)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (OUT/'provenance.json').write_text(json.dumps(dict(source_mesh_sha256=hashes,method='Taubin',lamb=.5,nu=.53,
        variants=VARIANTS,units='micrometers',volume_correction=False,segmentation_tiffs_modified=False),indent=2))
    print(metrics.to_string(index=False))


if __name__=='__main__':main()
