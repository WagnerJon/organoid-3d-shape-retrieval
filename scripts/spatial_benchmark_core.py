"""Geometry/pose helpers for the frozen spatial-shape benchmark."""
import itertools
import numpy as np
import igl
from scipy.spatial import cKDTree
from pointnet_shape_core import ROOT,sha256
from prepare_pointnet_shape import read_mesh

SIGNS=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=float)
AXIS=np.linspace(-1.15,1.15,32,dtype=np.float64)
GRID=np.stack(np.meshgrid(AXIS,AXIS,AXIS,indexing='ij'),-1).reshape(-1,3)
CONTROLS=['volume_um3','sphericity','elongation']
CLASSICAL=['volume_um3','surface_area_um2','sphericity','elongation','flatness','mesh_solidity','major_axis_um','middle_axis_um','minor_axis_um']


def mesh_for(row):
    path=ROOT/'results/explorer_dataset'/row['sample']/f"{row['label']}.mesh"
    if sha256(path)!=row['source_mesh_sha256']:raise ValueError('Source mesh changed')
    v,f=read_mesh(path);center=np.array([row['normalization_center_'+a+'_um'] for a in 'xyz'])
    return (v.astype(float)-center)/row['normalization_radius_um'],f.astype(np.int64)


def surface_frame(v,f):
    """Exact area-weighted surface covariance; independent of triangulation density."""
    triangles=v[f];s=triangles.sum(1)
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    weight=area/area.sum();center=(weight[:,None]*s/3).sum(0)
    moment=(np.einsum('t,tvi,tvj->ij',weight,triangles,triangles)+np.einsum('t,ti,tj->ij',weight,s,s))/12
    covariance=moment-np.outer(center,center)
    return frame_from_cov(center,covariance)


def frame_from_cov(center,covariance):
    values,axes=np.linalg.eigh(covariance);values=values[::-1];axes=axes[:,::-1].copy()
    if np.linalg.det(axes)<0:axes[:,-1]*=-1
    gap=float(min(values[0]-values[1],values[1]-values[2])/max(values[0],1e-12))
    return np.asarray(center),axes,values,gap


def sampled_frame(points):
    center=points.mean(0);p=points-center
    return frame_from_cov(center,p.T@p/len(p))


def sdf_grid(v,f,center=None,axes=None):
    queries=GRID if axes is None else GRID@axes.T+center
    return igl.signed_distance(queries,v,f,igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0].clip(-.15,.15).reshape(32,32,32).astype(np.float32)


def orbit_grids(v,f,frame=None):
    center,axes,_,_=surface_frame(v,f) if frame is None else frame
    return np.stack([sdf_grid(v,f,center,axes*sign) for sign in SIGNS])


def orbit_distance(a,b):
    # Min over proper sign frames; no reflection or arbitrary eigenvector sign choice.
    a=np.asarray(a);b=np.asarray(b)
    return np.maximum((a*a).sum(-1)[:,None,:,None]+(b*b).sum(-1)[None,:,None,:]
        -2*np.einsum('afd,bgd->abfg',a,b),0).min((2,3))


def summary_features(z):return np.concatenate([z.mean(1),z.std(1)],axis=1)


def morphology(frame,columns):
    values=frame[columns].to_numpy(float).copy()
    for i,name in enumerate(columns):
        if name in ['volume_um3','surface_area_um2','major_axis_um','middle_axis_um','minor_axis_um']:
            values[:,i]=np.log(values[:,i].clip(1e-12))
    return values


def geometry_cost(a,b):
    """Proper-rotation surface ICP, 24 axis seeds; no scale/reflection fitting."""
    a=a-a.mean(0);b=b-b.mean(0)
    _,aa=np.linalg.eigh(a.T@a);_,bb=np.linalg.eigh(b.T@b);a=a@aa;b=b@bb
    ta=cKDTree(a);best=float('inf')
    for permutation in itertools.permutations(range(3)):
        base=np.eye(3)[:,permutation]
        for signs in itertools.product([-1,1],repeat=3):
            r=base*np.array(signs)
            # aa and bb may have different handedness, so account for their bases.
            if np.linalg.det(aa@r@bb.T)<0:continue
            moved=b@r.T
            for _ in range(8):
                distances,indices=ta.query(moved)
                u,_,vt=np.linalg.svd(moved.T@a[indices]);step=u@vt
                if np.linalg.det(step)<0:u[:,-1]*=-1;step=u@vt
                moved=moved@step
            value=float(np.square(ta.query(moved)[0]).mean()+np.square(cKDTree(moved).query(a)[0]).mean())
            best=min(best,value)
    return best
