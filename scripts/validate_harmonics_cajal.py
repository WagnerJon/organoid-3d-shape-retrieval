"""Verify published harmonic/GW associations and fixed training transforms."""
import json
import numpy as np
import pandas as pd
from pointnet_shape_core import ROOT,sha256
from evaluate_spatial_benchmark import residual_apply
from harmonics_cajal_core import pairwise_subset
OUT=ROOT/'results/harmonics_cajal'


def validate(data=None):
    if data is None:data=json.loads((ROOT/'results/explorer_dataset/dataset.json').read_text())
    vectors=np.load(OUT/'atlas_vectors.npz');source=np.load(OUT/'surfaces.npz');d=np.load(OUT/'cajal_qgw.npy');frame=pd.read_csv(OUT/'manifest.csv')
    ids=source['ids'];lookup={c['id']:c for c in data['cells']};assert list(ids)==frame.cell_id.tolist()
    assert sha256(OUT/'surfaces.npz')==data['harmonicsCajal']['source_sha256']
    assert sha256(OUT/'cajal_qgw.npy')==data['harmonicsCajal']['distance_sha256']
    fit=dict(np.load(OUT/'cajal_profile_adjustment.npz'));positions={cid:i for i,cid in enumerate(ids)};train=[positions[c] for c in fit.pop('landmarks')]
    expected=dict(spherical_harmonics=source['harmonics'],
        spherical_harmonics_adjusted=residual_apply(frame,source['harmonics'],dict(np.load(OUT/'harmonics_adjustment.npz'))),
        cajal_qgw=np.c_[np.arange(len(ids)),d],cajal_profile_adjusted=residual_apply(frame,d[:,train],fit))
    for key,values in expected.items():
        actual=np.array([lookup[c]['representations'][key]['vector'] for c in ids]);np.testing.assert_allclose(actual,values,atol=1e-12)
    validation_ids=data['harmonicsCajal']['validation_ids'];validation_d=np.load(OUT/'validation_gw.npy')
    actual=np.array([lookup[c]['representations']['cajal_gw_validation']['vector'] for c in validation_ids])
    np.testing.assert_allclose(actual,np.c_[np.arange(len(validation_ids)),validation_d],atol=1e-12)
    pairwise_subset(actual)
    assert all('cajal_gw_validation' not in c['representations'] for c in data['cells'] if c['id'] not in validation_ids)
    pairwise_subset(expected['cajal_qgw'])
    assert all('cajal_qgw' not in r['representations'] and 'cajal_gw_validation' not in r['representations'] for r in data['organoids'])
    print('PASS: spherical harmonics and CAJAL vectors, global identities, distance symmetry, training-only adjustments and cell-only GW scope.')


if __name__=='__main__':validate()
