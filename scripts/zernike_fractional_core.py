"""Classical and fractional 3D Zernike invariants, independent implementation.

Ashton & Sit-Seth (2026), doi:10.3390/appliedmath6100174, Eqs. 1,4,15,17.
The radial basis is evaluated as shifted Jacobi polynomials, avoiding unstable
monomial cancellation. This satisfies Eq.4 and the Section4 recurrences. The
printed Eq.9's unit-step Pochhammer is inconsistent with those equations; see
docs/FRACTIONAL_ZERNIKE.md. No author software is copied or required.
"""
from functools import lru_cache
import numpy as np
from scipy.special import eval_jacobi, sph_harm_y, expit


def indices(order):
    if not isinstance(order, (int, np.integer)) or not 0 <= order <= 40:
        raise ValueError('Order must be an integer between 0 and 40')
    return [(n, l) for n in range(order+1) for l in range(n+1) if (n-l) % 2 == 0]


def radial(n, l, alpha, r):
    """3∫R_nl R_n'l r²dr=δ_nn'; beta=l+3/(2alpha)-1."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Fractional exponent must be finite and positive')
    if n < l or l < 0 or (n-l) % 2:
        raise ValueError('Require 0 <= l <= n with even n-l')
    r = np.asarray(r, dtype=float)
    if not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
        raise ValueError('Radii must be finite and inside the unit ball')
    k = (n-l)//2
    return (np.sqrt((2*alpha*n+3)/3)*r**(alpha*l)*
            eval_jacobi(k, 0, l+3/(2*alpha)-1, 2*r**(2*alpha)-1))


@lru_cache(maxsize=3)
def quadrature(order=20, n_radial=96, n_theta=32, n_phi=64):
    if n_theta < order+1 or n_phi < 2*order+1 or n_radial < order+1:
        raise ValueError('Quadrature undersamples requested harmonic order')
    u, wu = np.polynomial.legendre.leggauss(n_theta)
    theta, phi = np.meshgrid(np.arccos(u), np.arange(n_phi)*2*np.pi/n_phi, indexing='ij')
    theta, phi = theta.ravel(), phi.ravel()
    directions = np.c_[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    weight = np.repeat(wu, n_phi)*2*np.pi/n_phi
    # Paper uses Y with angular mean-square norm1; scipy uses integral norm1.
    transform = np.stack([sph_harm_y(l,m,theta,phi).conj()*weight/np.sqrt(4*np.pi)
                          for l in range(order+1) for m in range(-l,l+1)], axis=1)
    x, w = np.polynomial.legendre.leggauss(n_radial)
    r, wr = (x+1)/2, w/2
    points = (r[:,None,None]*directions[None]).reshape(-1,3)
    return r, wr, points, transform


def field_descriptors(fields, alphas=(1., .38, .14, .20), order=20, sampling=(96,32,64), normalization='l2'):
    """Integrate arbitrary real fields; output121 norms at order20, L2 normalized.

    Fields have shape [radial, polar*azimuthal]. Only scalar fields are integrated;
    no radial outer-intersection assumption, pose alignment, or learned weights.
    """
    r, wr, _, angular = quadrature(order, *sampling)
    result = {}
    angular_coeff = {}
    for name, values in fields.items():
        values = np.asarray(values, float)
        if values.shape != (len(r), len(angular)) or not np.isfinite(values).all():
            raise ValueError('Invalid sampled scalar field')
        angular_coeff[name] = values@angular
    for alpha in alphas:
        vectors = {name: [] for name in fields}
        for n, l in indices(order):
            weights = 3*radial(n,l,alpha,r)*wr*r*r
            for name, coefficients in angular_coeff.items():
                moment = weights@coefficients[:, l*l:(l+1)**2]
                vectors[name].append(np.linalg.norm(moment))
        for name, values in vectors.items():
            v = np.asarray(values)
            if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12:
                raise ValueError('Empty/nonfinite descriptor')
            if normalization not in ('l2','mass'):
                raise ValueError('Choose l2 or mass normalization')
            result[f'{name}_a{alpha:g}'] = v/(np.linalg.norm(v) if normalization=='l2' else max(v[0],1e-12))
    return result


def normalized_mesh(v, f):
    import trimesh
    from spatial_benchmark_core import surface_frame
    v, f = np.asarray(v,float), np.asarray(f,np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or len(f) == 0:
        raise ValueError('Invalid surface')
    mesh = trimesh.Trimesh(v,f,process=False)
    closed = bool(mesh.is_watertight and mesh.is_winding_consistent and abs(mesh.volume)>1e-12)
    center = mesh.center_mass if closed else surface_frame(v,f)[0]
    radius = np.linalg.norm(v-center,axis=1).max()/.9
    if radius < 1e-12: raise ValueError('Degenerate surface')
    return (v-center)/radius, f, dict(center=center.tolist(), radius=float(radius), closed=closed)


def mesh_descriptors(v, f, alphas=(1., .38, .14, .20), order=20, sampling=(96,32,64), resolution=64, normalization='l2'):
    """Volume control plus paper-style binary surface band, same normalized mesh.

    Spherical product quadrature replaces the paper's Cartesian Riemann sum.
    Surface band has half-width sqrt(3)/resolution (half voxel diagonal at
    Cartesian spacing2/resolution). Volume matches existing ZMPY soft occupancy.
    Finite quadrature introduces rotation error; validate with rotated meshes.
    """
    import igl
    v, f, metadata = normalized_mesh(v,f)
    r, _, points, angular = quadrature(order,*sampling)
    # Bound temporary signed-distance buffers independently of source image size.
    sdf = np.concatenate([igl.signed_distance(p,v,f,igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)[0]
                          for p in np.array_split(points, max(1, int(np.ceil(len(points)/32768))))])
    shape = (len(r),len(angular))
    fields = dict(volume=expit(-sdf*resolution).reshape(shape),
                  surface=(np.abs(sdf)<=np.sqrt(3)/resolution).astype(float).reshape(shape))
    return field_descriptors(fields,alphas,order,sampling,normalization), metadata
