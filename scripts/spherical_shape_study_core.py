"""Offline spherical ShapeEmbed experiments: transparent numerical reference code.

No input dataset, mesh, segmentation, Atlas registration, or measurements are edited.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import trimesh
import igl
from scipy.special import sph_harm_y
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve
from sympy.physics.wigner import clebsch_gordan
from harmonics_cajal_core import normalize_mesh

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/spherical_shape_study'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))

def fibonacci(n):
    z=1-2*(np.arange(n)+.5)/n
    phi=np.arange(n)*np.pi*(3-np.sqrt(5))
    r=np.sqrt(1-z*z)
    return np.c_[r*np.cos(phi),r*np.sin(phi),z]

def basis(directions,L):
    theta=np.arccos(np.clip(directions[:,2],-1,1));phi=np.arctan2(directions[:,1],directions[:,0])
    return np.column_stack([sph_harm_y(l,m,theta,phi) for l in range(L+1) for m in range(-l,l+1)])

@lru_cache(maxsize=8)
def fib_basis(n=2048,L=12):
    d=fibonacci(n);return d,basis(d,L)

def coefficients(values,Y):return (4*np.pi/len(Y))*(Y.conj().T@values)

def power(c,L=12):
    return np.array([np.linalg.norm(c[l*l:(l+1)**2]) for l in range(L+1)])/max(abs(c[0]),1e-12)

@lru_cache(maxsize=4)
def cg_terms(L=8):
    terms=[]
    for a in range(1,L+1):
        for b in range(a,L+1):
            for c in range(b,min(L,a+b)+1):
                ia,ib,ic,w=[],[],[],[]
                for ma in range(-a,a+1):
                    for mb in range(-b,b+1):
                        mc=ma+mb
                        if abs(mc)>c:continue
                        v=float(clebsch_gordan(a,b,c,ma,mb,mc))
                        if abs(v)>1e-14:
                            ia.append(a*a+ma+a);ib.append(b*b+mb+b);ic.append(c*c+mc+c);w.append(v)
                terms.append((a,b,c,np.array(ia),np.array(ib),np.array(ic),np.array(w)))
    return terms

def bispectrum(c,L=8):
    a=c/max(abs(c[0]),1e-12)
    result=[]
    for l1,l2,l3,i,j,k,w in cg_terms(L):
        z=np.sum(w*a[i]*a[j]*a[k].conj())
        # For real spherical signals, even parity is real, odd parity imaginary.
        # Keep the allowed component; omit analytically zero redundant slots.
        if (l1+l2+l3)%2==0:result.append(z.real)
        elif l1!=l2 and l2!=l3:result.append(z.imag)
    return np.asarray(result)

def all_features(c):return np.r_[power(c,12),bispectrum(c,8)]

def ray_samples(v,f,directions,attribute=None):
    """Outermost positive surface intersection, with explicit missing/multihit QC.

    Triangle-edge duplicate intersections are merged by distance, not counted as
    a second boundary. A radial envelope deliberately discards inner crossings.
    """
    mesh=trimesh.Trimesh(v,f,process=False)
    engine=trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations,rays,faces=engine.intersects_location(np.zeros_like(directions),directions,multiple_hits=True)
    t=np.einsum('ij,ij->i',locations,directions[rays])
    order=np.lexsort((t,rays));rays=rays[order];t=t[order];faces=faces[order];locations=locations[order]
    keep=t>1e-8
    rays,t,faces,locations=rays[keep],t[keep],faces[keep],locations[keep]
    distinct=np.ones(len(t),bool)
    if len(t)>1:distinct[1:]=(rays[1:]!=rays[:-1])|(abs(t[1:]-t[:-1])>1e-6)
    counts=np.bincount(rays[distinct],minlength=len(directions))
    radius=np.full(len(directions),np.nan);values=None
    if len(t):
        last=np.r_[rays[1:]!=rays[:-1],True]
        radius[rays[last]]=t[last]
        if attribute is not None:
            tri=v[f[faces[last]]]
            bary=trimesh.triangles.points_to_barycentric(tri,locations[last])
            values=np.full((len(directions),attribute.shape[1]),np.nan)
            values[rays[last]]=np.einsum('ni,nij->nj',bary,attribute[f[faces[last]]])
    if np.isnan(radius).any() and np.isfinite(radius).any():
        missing=~np.isfinite(radius);_,ix=cKDTree(directions[~missing]).query(directions[missing])
        radius[missing]=radius[~missing][ix]
        if values is not None:values[missing]=values[~missing][ix]
    return radius,values,dict(coverage=float((counts>0).mean()),multiple_crossing_fraction=float((counts>1).mean()))

def radial_coefficients(v,f,n=2048,L=12):
    v,scale=normalize_mesh(v,f);d,Y=fib_basis(n,L)
    r,_,qc=ray_samples(v,f,d)
    if not np.isfinite(r).all():raise ValueError('No usable radial intersections')
    return coefficients(r,Y),qc,v,scale

def topology(v,f):
    mesh=trimesh.Trimesh(v,f,process=False)
    components=len(mesh.split(only_watertight=False))
    return dict(watertight=bool(mesh.is_watertight),euler=int(mesh.euler_number),components=components,
                genus_zero_eligible=bool(mesh.is_watertight and mesh.euler_number==2 and components==1 and mesh.is_winding_consistent))

def map_sphere(v,f,steps=45,dt=.04):
    """Independent cMCF-inspired finite-element implementation, not SPHARM-OT.

    Freeze initial cotangent stiffness, update barycentric mass; center and area
    normalize each step. Projection used only for spherical parameter coordinates.
    The original xyz positions remain untouched and provide the mapped signal.
    """
    info=topology(v,f)
    if not info['genus_zero_eligible']:return None,dict(**info,valid=False,reason='topology_ineligible')
    x=v.copy();L=igl.cotmatrix(x,f)
    for k in range(steps):
        mass=igl.massmatrix(x,f,igl.MassMatrixType.MASSMATRIX_TYPE_BARYCENTRIC)
        x=spsolve(mass-dt*L,mass@x)
        weights=mass.diagonal();x-=np.average(x,axis=0,weights=weights)
        area=trimesh.triangles.area(x[f]).sum()
        x*=np.sqrt(4*np.pi/max(area,1e-15))
        if not np.isfinite(x).all():return None,dict(**info,valid=False,reason='nonfinite_flow')
    radius=np.linalg.norm(x,axis=1)
    sphere=x/np.maximum(radius[:,None],1e-12)
    tri=sphere[f]
    triple=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))
    angles=2*np.arctan2(triple,1+np.einsum('ij,ij->i',tri[:,0],tri[:,1])+np.einsum('ij,ij->i',tri[:,1],tri[:,2])+np.einsum('ij,ij->i',tri[:,2],tri[:,0]))
    sign=1 if angles.sum()>0 else -1
    flips=int((triple*sign<=1e-13).sum())
    winding=abs(angles.sum())/(4*np.pi)
    valid=bool(flips==0 and abs(winding-1)<1e-5)
    info.update(valid=valid,reason='valid' if valid else 'flips_or_degree_failure',flipped_faces=flips,
                mapping_degree=float(winding),radial_cv=float(radius.std()/radius.mean()),steps=steps)
    return sphere,info

def reconstruction_metrics(points,v,f):
    dist=igl.point_mesh_squared_distance(np.asarray(points,float),v,f)[0]
    source,_=trimesh.sample.sample_surface(trimesh.Trimesh(v,f,process=False),4096,seed=42)
    reverse=cKDTree(points).query(source)[0]
    return dict(chamfer_squared=float(dist.mean()+np.square(reverse).mean()),p95=float(np.percentile(np.r_[np.sqrt(dist),reverse],95)))

def synthetic_radius(d,family,amount,width=.3,seed=0):
    rng=np.random.default_rng(seed)
    axes=rng.normal(size=(3,3));q,_=np.linalg.qr(axes);u=d@q
    angle=np.arccos(np.clip(u[:,2],-1,1))
    if family=='elongation':return 1/np.sqrt(u[:,0]**2+u[:,1]**2+(u[:,2]/(1+amount))**2)
    if family=='protrusion':return 1+amount*np.exp(-.5*(angle/width)**2)
    if family=='indentation':return 1-.65*amount*np.exp(-.5*(angle/width)**2)
    if family=='branching':
        # Three smooth radial lobes: controlled branching proxy, not arbitrary topology.
        angles=np.arccos(np.clip(u@np.array([[0,0,1],[.94,0,-.342],[-.47,.814,-.342]]).T,-1,1))
        return 1+amount*np.exp(-.5*(angles/width)**2).sum(1)
    if family=='roughness':
        degree=int(round(2+4/width));signal=sph_harm_y(degree,3,np.arccos(np.clip(u[:,2],-1,1)),np.arctan2(u[:,1],u[:,0])).real
        return 1+.15*amount*signal*np.sqrt(8*np.pi)
    raise ValueError(family)

def random_radial(seed,L=8):
    rng=np.random.default_rng(seed);c=np.zeros((L+1)**2,complex);c[0]=np.sqrt(4*np.pi)
    for l in range(2,L+1):
        scale=.24/(l**1.15)
        c[l*l+l]=rng.normal()*scale
        for m in range(1,l+1):
            z=(rng.normal()+1j*rng.normal())*scale/np.sqrt(2)
            c[l*l+l+m]=z;c[l*l+l-m]=(-1)**m*z.conjugate()
    return c


def balance_sphere(mapped,v,f):
    """Möbius-center parameter coordinates against the original surface area.

    This preserves the spherical surface and changes only the parameterization.
    It is an area-centering normalization, not an equal-area/optimal-transport map.
    """
    from scipy.optimize import least_squares
    weights=np.zeros(len(v));area=trimesh.triangles.area(v[f])
    for k in range(3):np.add.at(weights,f[:,k],area/3)
    weights/=weights.sum()
    def parameter(u):
        length=np.linalg.norm(u)
        return .999*np.tanh(length)*u/max(length,1e-15)
    def transform(a):
        delta=mapped-a
        return (1-np.dot(a,a))*delta/np.maximum(np.square(delta).sum(1)[:,None],1e-15)-a
    def residual(u):return np.sum(transform(parameter(u))*weights[:,None],axis=0)
    fit=least_squares(residual,np.zeros(3),xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=80)
    corrected=transform(parameter(fit.x));corrected/=np.linalg.norm(corrected,axis=1)[:,None]
    tri=corrected[f];triple=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))
    angle=2*np.arctan2(triple,1+np.einsum('ij,ij->i',tri[:,0],tri[:,1])+np.einsum('ij,ij->i',tri[:,1],tri[:,2])+np.einsum('ij,ij->i',tri[:,2],tri[:,0]))
    sign=1 if angle.sum()>0 else -1;flips=int((triple*sign<=1e-13).sum());degree=abs(angle.sum())/(4*np.pi)
    return corrected,dict(valid=bool(flips==0 and abs(degree-1)<1e-5),flipped_faces=flips,mapping_degree=float(degree),original_area_barycenter_norm=float(np.linalg.norm(residual(np.zeros(3)))),balanced_area_barycenter_norm=float(np.linalg.norm(residual(fit.x))),moebius_parameter_norm=float(np.linalg.norm(parameter(fit.x))))
