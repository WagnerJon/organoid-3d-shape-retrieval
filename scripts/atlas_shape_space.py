"""Atlas shape comparison: frozen spatial PCA, ambiguity fallback, no retraining.

Fallback signatures are deterministic rotation-invariant geometric descriptors,
not neural embeddings. Invariance assumes unchanged triangulation/face ordering.
"""
import numpy as np
from scipy.spatial.distance import pdist, cdist
from sklearn.cluster import AgglomerativeClustering
from spatial_benchmark_core import surface_frame

GAP = .05
ORBIT_DIM = 95
SIGNATURE_DIM = 95
HYBRID_DIM = 1 + 4 * ORBIT_DIM + SIGNATURE_DIM


def signature(v, f, count=1024):
    """Area-stratified triangle quadrature; SO(3) and translation/scale invariant.

Pair-distance distributions also discard chirality and are not complete shape
invariants. Face ordering affects quadrature under remeshing, not rigid rotation.
"""
    t = np.asarray(v, float)[f]
    area = np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)/2
    if area.sum() <= 0: raise ValueError('Empty surface')
    indices = np.searchsorted(np.cumsum(area)/area.sum(), (np.arange(count)+.5)/count)
    k = np.arange(count)+1
    u = np.sqrt((k * .7548776662466927) % 1)
    w = (k * .5698402909980532) % 1
    weights = np.stack([1-u, u*(1-w), u*w], 1)
    p = (t[indices] * weights[:, :, None]).sum(1)
    center, _, _, _ = surface_frame(v, f)
    p -= center
    scale = np.sqrt(np.mean(np.square(p).sum(1)))
    p /= max(scale, 1e-12)
    return np.r_[np.quantile(np.linalg.norm(p, axis=1), np.linspace(.01,.99,64)),
                 np.quantile(pdist(p), np.linspace(.005,.995,192))]


def pack(gaps, orbits, signatures):
    return np.c_[np.asarray(gaps) < GAP, np.asarray(orbits).reshape(-1,380), signatures]


def distances(vectors, metric):
    x = np.asarray(vectors, float)
    if metric == 'precomputed':
        from harmonics_cajal_core import pairwise_subset
        return pairwise_subset(x)
    if metric != 'shape_hybrid95': return cdist(x,x)
    if x.shape[1] != HYBRID_DIM: raise ValueError('Invalid hybrid vector size')
    z = x[:,1:381].reshape(-1,4,95)
    # Keep intermediate allocation bounded for the complete cohort.
    d = np.full((len(x),len(x)), np.inf)
    for a in range(4):
        for b in range(4): d = np.minimum(d, cdist(z[:,a],z[:,b]))
    fallback = (x[:,0] > .5)[:,None] | (x[:,0] > .5)[None,:]
    d = np.where(fallback, cdist(x[:,381:],x[:,381:]), d)
    np.fill_diagonal(d,0)
    return (d+d.T)/2


def project_hybrid(records, key):
    if not records: return [0.,0.]
    d = distances([r['representations'][key]['vector'] for r in records], 'shape_hybrid95')
    # Classical MDS of actual comparison distances; not PCA of packed vectors.
    b = -.5*d*d; b -= b.mean(0)[None,:] + b.mean(1)[:,None] - b.mean()
    values, vectors = np.linalg.eigh(b)
    order = np.argsort(values)[::-1][:2]; positive = np.maximum(values[order],0)
    points = vectors[:,order]*np.sqrt(positive)
    points = np.pad(points,((0,0),(0,2-points.shape[1])))
    variance = np.pad(positive / max(np.maximum(values,0).sum(),1e-12),(0,2-len(positive))).tolist()
    groups = (AgglomerativeClustering(n_clusters=min(6,len(records)),metric='precomputed',linkage='average').fit_predict(d)
              if len(records)>1 else np.zeros(1,int))
    for r,p,g in zip(records,points,groups):r['representations'][key].update(point=p.tolist(),cluster=int(g))
    return variance
