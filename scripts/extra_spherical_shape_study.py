"""Arbitrary-shape stress controls and full shell-coefficient reconstructions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.special import expit
from skimage.measure import marching_cubes
from spherical_shape_study_core import *
from harmonics_cajal_core import angular_basis,RADII,WIDTH
from check_cajal_validation import candidate_sets
from spatial_benchmark_core import mesh_for


def shell_reconstruct(v,f,res=40):
    directions,T,bands=angular_basis(32,64,12)
    q=(RADII[:,None,None]*directions).reshape(-1,3)
    sdf=igl.signed_distance(q,v,f,igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0]
    occ=expit(-sdf.reshape(len(RADII),-1)/WIDTH)
    c=occ@T
    axis=np.linspace(-2.6,2.6,res)
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    radii=np.linalg.norm(grid,axis=1);dirs=grid/np.maximum(radii[:,None],1e-12)
    y=basis(dirs,12)
    high=np.searchsorted(RADII,radii).clip(1,len(RADII)-1);low=high-1
    w=((radii-RADII[low])/(RADII[high]-RADII[low])).clip(0,1)
    prediction=np.sum(y*(c[low]*(1-w[:,None])+c[high]*w[:,None]),axis=1).real
    prediction[radii>RADII[-1]]=0
    outv,outf,_,_=marching_cubes(prediction.reshape((res,)*3),.5,spacing=(axis[1]-axis[0],)*3);outv+=axis[0]
    points,_=trimesh.sample.sample_surface(trimesh.Trimesh(outv,outf,process=False),4096,seed=42)
    return outv,outf,reconstruction_metrics(points,v,f)


def controls():
    axis=np.linspace(-2.2,2.2,64);g=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1)
    theta=np.linspace(-2.3,2.3,65);curve=np.c_[.9*np.cos(theta),.9*np.sin(theta),np.zeros_like(theta)]
    cases={
        'sphere':np.linalg.norm(g,axis=-1)-1,
        'elongated':np.linalg.norm(g/np.array([.65,.65,1.4]),axis=-1)-1,
        'curved_tube':np.min(np.stack([np.linalg.norm(g-c,axis=-1) for c in curve]),axis=0)-.26,
        'torus':np.sqrt((np.linalg.norm(g[...,:2],axis=-1)-.9)**2+g[...,2]**2)-.28}
    # Genus-zero tube with genuinely bent three-dimensional branches, not just
    # radial lobes. A surface can re-enter a ray because of the curved arm.
    t=np.linspace(0,1,40)
    paths=[np.c_[0*t,0*t,-1+2*t],np.c_[1.1*np.sin(2*t),0*t,.15+1.1*(1-np.cos(2*t))],np.c_[0*t,-.9*t,.2+.7*t]]
    cases['bent_branched_tubes']=np.min(np.stack([np.linalg.norm(g-c,axis=-1) for path in paths for c in path]),axis=0)-.26
    d,Y=fib_basis(2048,12);s=trimesh.creation.icosphere(subdivisions=4);Ye=basis(s.vertices,12)
    results=[];(OUT/'arbitrary_controls').mkdir(exist_ok=True)
    for name,sdf in cases.items():
        v,f,_,_=marching_cubes(sdf,0,spacing=(axis[1]-axis[0],)*3);v+=axis[0]
        c,qc,v,scale=radial_coefficients(v,f,2048,12)
        r=s.vertices*np.maximum((Ye@c).real,.001)[:,None]
        row=dict(shape=name,radial_qc=qc,radial=reconstruction_metrics(r,v,f))
        mapped,info=map_sphere(v,f);row['mapping']=info
        general=np.zeros((0,3))
        if info['valid']:
            _,value,sample_qc=ray_samples(mapped,f,d,attribute=v)
            gen=coefficients(value,Y);general=(Ye@gen).real
            row['general']=reconstruction_metrics(general,v,f)
        sv,sf,sm=shell_reconstruct(v,f);row['shell_full_coefficients']=sm
        np.savez_compressed(OUT/'arbitrary_controls'/f'{name}.npz',vertices=v,faces=f,radial=r,general=general,shell_vertices=sv,shell_faces=sf)
        results.append(row);print('Arbitrary-shape control',name,info['reason'],flush=True)
    save_json(OUT/'arbitrary_controls.json',results)


def real_shell():
    frame,_,_=candidate_sets();(OUT/'shell_reconstructions').mkdir(exist_ok=True)
    records=[]
    for i,row in enumerate(frame.to_dict('records')):
        path=OUT/'shell_reconstructions'/f'{i:03d}.json'
        if path.exists():records.append(json.loads(path.read_text()));continue
        v,f=mesh_for(row);v,scale=normalize_mesh(v,f)
        sv,sf,metric=shell_reconstruct(v,f)
        result=dict(cell_id=row['cell_id'],metrics=metric,resolution=40,L=12,radial_shells=32)
        np.savez_compressed(path.with_suffix('.npz'),vertices=sv,faces=sf)
        save_json(path,result);records.append(result)
        if i%6==0:print(f'Shell-coefficient reconstruction {i+1}/48',flush=True)
    save_json(OUT/'shell_reconstruction.json',records)
if __name__=='__main__':
    controls();real_shell();save_json(OUT/'extras.complete.json',dict(status='complete'))
