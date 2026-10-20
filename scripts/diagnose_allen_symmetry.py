"""Prespecified structural diagnostic of globally pooled vector codes.

Two exactly D2-symmetric point clouds have no preferred 3D vector direction.
This is an engineered diagnostic, not an estimate of real-cell performance.
"""
import json
import numpy as np
import torch
from allen_reference_core import AllenModel,OUT,chunked_encode


def main():
    torch.set_num_threads(2);m=AllenModel().eval();rng=np.random.default_rng(20261115)
    base=rng.normal(size=(2048,3));base/=np.linalg.norm(base,axis=1,keepdims=True)
    signs=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    latents=[]
    with torch.no_grad():
        for axes in [[1,.9,.8],[1,.7,.5]]:
            p=np.concatenate([base*np.array(axes)*s for s in signs]).astype(np.float32)
            p/=np.ptp(p,axis=0).max()
            z=chunked_encode(m.encoder,torch.tensor(p[None]),chunk=512);latents.append(z.numpy()[0])
    z=np.stack(latents);q=torch.tensor(rng.uniform(-.6,.6,(1,4096,3)).astype(np.float32))
    with torch.no_grad():fields=np.stack([m.decode(torch.tensor(code[None]),q).numpy()[0] for code in z])
    result=dict(engineered_shapes='Exactly D2-symmetric ellipsoid point clouds with aspect ratios1:.9:.8 and1:.7:.5;8192 points each.',
        latent_norms=np.linalg.norm(z,axis=(1,2)).tolist(),invariant_norm_distance=float(np.linalg.norm(np.linalg.norm(z[0],axis=-1)-np.linalg.norm(z[1],axis=-1))),
        maximum_sdf_difference=float(np.max(np.abs(fields[0]-fields[1]))),
        scope='A globally pooled SO(3)-equivariant vector code must vanish on exact D2-invariant inputs in exact arithmetic. Float32 rounding/kNN tie effects can break exact numerical symmetry. This does not estimate natural-cell errors or invalidate the paper’s morphology-specific applications.')
    (OUT/'symmetry_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
