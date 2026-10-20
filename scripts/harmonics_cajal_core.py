"""Rotation-invariant shell harmonic descriptor and CAJAL surface preparation.

Independent implementation of the spherical-shell power-spectrum construction
(Kazhdan et al., 2003), not SPHARM-PDM or a reproduction of Dynamic SPHARM.
Original segmented surfaces are never modified.
"""
import numpy as np
import igl
from scipy.special import sph_harm_y, expit
from scipy.spatial.distance import cdist
from spatial_benchmark_core import surface_frame

LMAX=12
RADII=np.linspace(.04,2.6,32)
WIDTH=.035


def angular_basis(n_theta=32,n_phi=64,lmax=LMAX):
    u,w=np.polynomial.legendre.leggauss(n_theta)
    theta,phi=np.meshgrid(np.arccos(u),np.arange(n_phi)*2*np.pi/n_phi,indexing='ij')
    theta=theta.ravel();phi=phi.ravel();weights=np.repeat(w,n_phi)*(2*np.pi/n_phi)
    directions=np.c_[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]
    basis=[];bands=[]
    for degree in range(lmax+1):
        bands.append(slice(degree*degree,(degree+1)**2))
        basis.extend([sph_harm_y(degree,m,theta,phi).conj()*weights for m in range(-degree,degree+1)])
    return directions,np.asarray(basis).T,bands


def normalize_mesh(v,f):
    center,_,values,_=surface_frame(v,f)
    scale=float(np.sqrt(np.maximum(values,0).sum()))
    if not scale>1e-12:raise ValueError('Degenerate surface')
    return (np.asarray(v)-center)/scale,scale


def harmonics(v,f,basis=None):
    v,scale=normalize_mesh(v,f)
    directions,transform,bands=angular_basis() if basis is None else basis
    query=(RADII[:,None,None]*directions[None]).reshape(-1,3)
    sdf=igl.signed_distance(query.astype(np.float64),v.astype(np.float64),f.astype(np.int64),
            igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0]
    occupancy=expit(-sdf.reshape(len(RADII),-1)/WIDTH)
    coeff=occupancy@transform
    power=np.stack([np.linalg.norm(coeff[:,band],axis=1) for band in bands],1)
    # Radial integration measure r^2 dr. Soft occupancy affects only descriptor
    # sampling, not source masks, meshes, or physical measurements.
    vector=(power*RADII[:,None]).ravel()
    vector/=max(np.linalg.norm(vector),1e-12)
    return vector


def surface_points(v,f,count=128,seed=0):
    """Reproducible area-weighted sampling; rotation-covariant for fixed mesh."""
    v,_=normalize_mesh(v,f);tri=v[f]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    rng=np.random.default_rng(seed)
    # Stratified areas improve coverage while retaining the surface measure.
    face=np.searchsorted(np.cumsum(area)/area.sum(),(np.arange(count)+rng.random(count))/count)
    u=np.sqrt(rng.random(count));w=rng.random(count)
    weights=np.c_[1-u,u*(1-w),u*w]
    return (tri[face]*weights[:,:,None]).sum(1)


def pairwise_subset(vectors):
    """Packed [global_index, distance_row...] vectors retain exact cached GW."""
    x=np.asarray(vectors,float);indices=x[:,0].astype(int)
    if not np.all(indices==x[:,0]) or np.any(indices<0) or np.any(indices>=x.shape[1]-1):raise ValueError('Invalid pairwise identity')
    d=x[:,1:][:,indices]
    if not np.isfinite(d).all() or np.min(d)<-1e-10 or not np.allclose(d,d.T,atol=1e-8):raise ValueError('Invalid pairwise distances')
    return d


def project_pairwise(records,key):
    from sklearn.cluster import AgglomerativeClustering
    if not records:return [0.,0.]
    d=pairwise_subset([r['representations'][key]['vector'] for r in records])
    b=-.5*d*d;b-=b.mean(0)[None,:]+b.mean(1)[:,None]-b.mean()
    values,vectors=np.linalg.eigh(b);ids=np.argsort(values)[::-1][:2];positive=np.maximum(values[ids],0)
    points=np.pad(vectors[:,ids]*np.sqrt(positive),((0,0),(0,2-len(ids))))
    variance=np.pad(positive/max(np.maximum(values,0).sum(),1e-12),(0,2-len(ids))).tolist()
    groups=AgglomerativeClustering(n_clusters=min(6,len(records)),metric='precomputed',linkage='average').fit_predict(d) if len(records)>1 else [0]
    for r,p,g in zip(records,points,groups):r['representations'][key].update(point=p.tolist(),cluster=int(g))
    return variance
