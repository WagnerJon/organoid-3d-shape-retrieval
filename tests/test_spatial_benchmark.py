import sys
from pathlib import Path
import unittest
import numpy as np
import trimesh
import pandas as pd
from tempfile import TemporaryDirectory
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from spatial_benchmark_core import surface_frame,SIGNS,orbit_distance,summary_features,geometry_cost
from pointnet_shape_core import rotations


class PoseChecks(unittest.TestCase):
    def test_surface_frame_is_rotation_equivariant_up_to_proper_signs(self):
        mesh=trimesh.creation.icosphere(subdivisions=2);v=np.asarray(mesh.vertices)*[1,.7,.4];f=np.asarray(mesh.faces)
        c,a,e,g=surface_frame(v,f);r=rotations(1,np.random.default_rng(1))[0].astype(float)
        # QR removes float32 round-off in the sampled rotation for this strict test.
        r=np.linalg.qr(r)[0];d,b,ee,gg=surface_frame(v@r.T+3,f)
        np.testing.assert_allclose(e,ee,atol=1e-12)
        self.assertLess(min(np.linalg.norm(b-r@a*s) for s in SIGNS),1e-10)
        np.testing.assert_allclose(d,c@r.T+3,atol=1e-12)

    def test_symmetric_geometry_is_flagged_as_ambiguous(self):
        mesh=trimesh.creation.icosphere(subdivisions=2)
        self.assertLess(surface_frame(np.asarray(mesh.vertices),np.asarray(mesh.faces))[3],1e-10)

    def test_orbit_metric_and_residual_features_ignore_sign_frame_order(self):
        z=np.random.default_rng(2).normal(size=(6,4,12));p=z[:,[2,0,3,1]]
        np.testing.assert_allclose(orbit_distance(z,z),orbit_distance(p,z),atol=1e-12)
        np.testing.assert_allclose(summary_features(z),summary_features(p),atol=1e-12)

    def test_geometry_proxy_recognizes_proper_rotation(self):
        p=np.random.default_rng(3).normal(size=(128,3))*[1,.6,.3]
        r=np.linalg.qr(np.random.default_rng(4).normal(size=(3,3)))[0]
        self.assertLess(geometry_cost(p,p@r.T),1e-12)
        self.assertGreater(geometry_cost(p,p*[1,1,-1]),1e-4)

    def test_residualization_removes_predictable_controls_and_retains_other_variation(self):
        from evaluate_spatial_benchmark import residual_fit,residual_apply
        rng=np.random.default_rng(8);x=rng.normal(size=(500,3))
        frame=pd.DataFrame(dict(volume_um3=np.exp(x[:,0]),sphericity=x[:,1],elongation=x[:,2]))
        features=np.column_stack([2*x[:,0]-3*x[:,1]+x[:,2]**2,rng.normal(size=len(x))])
        residual=residual_apply(frame,features,residual_fit(frame,features))
        self.assertLess(residual[:,0].std(),.01*features[:,0].std())
        self.assertGreater(residual[:,1].std(),.9*features[:,1].std())

    def test_test_stage_requires_freeze_before_opening_geometry(self):
        from evaluate_spatial_benchmark import evaluate
        with TemporaryDirectory() as folder,patch('evaluate_spatial_benchmark.OUT',Path(folder)):
            with self.assertRaisesRegex(RuntimeError,'sealed'):evaluate('test')


if __name__=='__main__':unittest.main()
