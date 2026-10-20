"""Supplementary, post-primary-test covariance-degeneracy diagnostic; no fitting."""
import json
import numpy as np
import torch
import trimesh
from prepare_spatial_benchmark import OUT
from pointnet_shape_core import save_json,sha256
from spatial_benchmark_core import surface_frame,orbit_grids,orbit_distance
from evaluate_spatial_benchmark import Encoder


def main():
    if not (OUT/'test/metrics.json').exists():raise RuntimeError('Complete the frozen primary test first')
    lock=json.loads((OUT/'freeze.json').read_text());kind=lock['selected'];torch.set_num_threads(4)
    for name,digest in lock['artifacts'].items():
        if sha256(OUT/name)!=digest:raise ValueError('Frozen artifact changed')
    encoder=Encoder(kind,True,torch.device('mps'));template=trimesh.creation.icosphere(subdivisions=4)
    direction=np.asarray(template.vertices);faces=np.asarray(template.faces,dtype=np.int64);base=[];rotated=[];rows=[]
    for index,amplitude in enumerate([.02,.03,.04,.05,.06,.07,.08,.10]):
        radius=1+amplitude*((direction**4).sum(1)-.6)
        vertices=direction*radius[:,None];vertices/=np.linalg.norm(vertices,axis=1).max()
        mesh=trimesh.Trimesh(vertices,faces,process=False);frame=surface_frame(vertices,faces)
        base.append(encoder.encode(orbit_grids(vertices,faces,frame)))
        rng=np.random.default_rng(20261031+index);trials=[]
        for trial in range(8):
            r=np.linalg.qr(rng.normal(size=(3,3)))[0]
            if np.linalg.det(r)<0:r[:,-1]*=-1
            trials.append(encoder.encode(orbit_grids(vertices@r.T,faces)))
        rotated.append(trials)
        rows.append(dict(amplitude=amplitude,axis_gap=frame[3],sphericity=float(np.pi**(1/3)*(6*abs(mesh.volume))**(2/3)/mesh.area)))
    base=np.asarray(base);rotated=np.asarray(rotated);rates=[];correct=np.zeros(len(base))
    for trial in range(8):
        hit=orbit_distance(rotated[:,trial],base).argmin(1)==np.arange(len(base));rates.append(float(hit.mean()));correct+=hit
    for index,row in enumerate(rows):row['rotation_identity']=float(correct[index]/8)
    save_json(OUT/'synthetic_symmetry.json',dict(scope='Supplementary diagnostic added after the frozen primary test; no model selection or tuning.',
        shapes='Eight near-spherical six-lobed radial meshes with exactly/nearly isotropic surface covariance; normalized to unit radius.',
        gallery_cells=8,rotation_trials=8,rotation_identity=float(np.mean(rates)),per_trial=rates,records=rows,
        limitation='Engineered adversarial covariance degeneracy, not real biological specimens. Identity rates use an eight-shape gallery and cannot be directly compared to the 135-cell test.',
        frozen_model_unchanged=True))
    print(json.dumps(dict(rotation_identity=float(np.mean(rates)),records=rows)),flush=True)


if __name__=='__main__':main()
