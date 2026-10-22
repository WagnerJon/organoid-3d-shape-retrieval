"""Controlled 64D full shell-coefficient study. Independent of Atlas exports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['VECLIB_MAXIMUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import numpy as np
import igl,trimesh
from scipy.special import expit
from spherical_shape_study_core import sha,save_json,random_radial,basis
from harmonics_cajal_core import angular_basis,RADII,WIDTH
from spatial_benchmark_core import surface_frame
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/spectral_autoencoder64'
RCOUNT=32;LMAX=12;K=169;DIM=5408

def pack(c):
    out=np.zeros(c.shape,float)
    for l in range(LMAX+1):
        zero=l*l+l;out[...,zero]=c[...,zero].real
        for m in range(1,l+1):
            out[...,zero+m]=np.sqrt(2)*c[...,zero+m].real
            out[...,zero-m]=-np.sqrt(2)*c[...,zero+m].imag
    return out

def unpack(x):
    c=np.zeros(x.shape,complex)
    for l in range(LMAX+1):
        zero=l*l+l;c[...,zero]=x[...,zero]
        for m in range(1,l+1):
            c[...,zero+m]=(x[...,zero+m]-1j*x[...,zero-m])/np.sqrt(2)
            c[...,zero-m]=(-1)**m*c[...,zero+m].conj()
    return c

def canonical(v,f):
    center,axes,values,gap=surface_frame(v,f);scale=np.sqrt(np.maximum(values,0).sum());p=(v-center)/scale
    area=trimesh.triangles.area(p[f]);tc=p[f].mean(1)
    for k in [0,1]:
        project=tc@axes[:,k];skew=np.sum(area*project**3)/area.sum()
        sign=skew if abs(skew)>1e-5 else (p@axes[:,k])[np.argmax(abs(p@axes[:,k]))]
        if sign<0:axes[:,k]*=-1;axes[:,2]*=-1
    return p@axes,dict(center=center.tolist(),scale=float(scale),axes=axes.tolist(),axis_gap=float(gap))

def field_coefficients(v,f):
    d,T,_=angular_basis(32,64,LMAX)
    q=(RADII[:,None,None]*d[None]).reshape(-1,3)
    sdf=igl.signed_distance(q,v.astype(float),f.astype(np.int64),igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0]
    c=expit(-sdf.reshape(32,-1)/WIDTH)@T
    return pack(c).astype(np.float32)

def powers(x):
    x=np.asarray(x).reshape(-1,32,K);p=[]
    for l in range(LMAX+1):p.append(np.linalg.norm(x[...,l*l:(l+1)**2],axis=-1))
    p=np.stack(p,axis=-1)*RADII[None,:,None];p=p.reshape(len(x),-1)
    return p/np.linalg.norm(p,axis=1,keepdims=True).clip(1e-12)

def weighted(x):return (np.asarray(x).reshape(-1,32,K)*RADII[None,:,None]).reshape(-1,DIM)
def unweighted(x):return np.asarray(x).reshape(-1,32,K)/RADII[None,:,None]

def synthetic_mesh(seed,mirror=False):
    sphere=trimesh.creation.icosphere(subdivisions=3);c=random_radial(seed,12)
    r=(basis(sphere.vertices,12)@c).real
    v=sphere.vertices*r[:,None];f=sphere.faces.copy()
    if mirror:v[:,0]*=-1;f=f[:,[0,2,1]]
    return v,f
