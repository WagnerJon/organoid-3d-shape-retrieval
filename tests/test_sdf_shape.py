import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sdf_shape_core import SDFModel,sdf_loss,extract_surface
from pointnet_shape_core import rotations


class SDFChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(3)

    def test_rotation_equivariance_and_invariant_features(self):
        m=SDFModel('vn').eval();p=torch.randn(2,48,3);q=torch.randn(2,31,3)
        r=torch.tensor(rotations(2,np.random.default_rng(1)))
        with torch.no_grad():
            z,s=m(p,q);zr,sr=m(p@r.transpose(1,2),q@r.transpose(1,2))
        torch.testing.assert_close(zr,z@r.transpose(1,2),rtol=1e-4,atol=1e-6)
        torch.testing.assert_close(s,sr,rtol=1e-4,atol=1e-6)
        torch.testing.assert_close(m.embedding(z),m.embedding(zr),rtol=1e-4,atol=1e-6)
        self.assertEqual(m.embedding(z).shape,(2,128))

    def test_point_permutation(self):
        m=SDFModel('vn').eval();p=torch.randn(1,51,3)
        with torch.no_grad():
            a=m.encode(p);b=m.encode(p[:,torch.randperm(51)])
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)

    def test_saturated_prediction_has_gradient(self):
        p=torch.tensor([2.,-2.],requires_grad=True);sdf_loss(p,torch.zeros(2)).backward()
        self.assertTrue(torch.all(p.grad.abs()>0))

    def test_extract_zero_level_in_correct_coordinates(self):
        class Sphere:
            def decode(self,z,q):return q.norm(dim=-1)-.6
        surface,field=extract_surface(Sphere(),torch.zeros(1),resolution=32)
        vertices,faces=surface
        self.assertLess(float(np.abs(np.linalg.norm(vertices,axis=1)-.6).max()),.003)
        self.assertLess(float(field[16,16,16]),0)

    def test_original_mesh_distance_sign(self):
        import trimesh
        sphere=trimesh.creation.icosphere(subdivisions=2,radius=.6)
        signed=-trimesh.proximity.signed_distance(sphere,np.array([[0,0,0],[1.,0,0]]))
        self.assertLess(signed[0],0);self.assertGreater(signed[1],0)

    def test_decoder_can_fit_non_spherical_implicit_surface(self):
        # Isolates the decoder's representational capacity from encoder learning.
        m=SDFModel('vn');z=torch.randn(1,128,3)*.2
        q=torch.rand(1,128,3)*2-1
        target=(q/torch.tensor([.7,.45,.6])).norm(dim=-1)-1
        target=target.clamp(-.15,.15)
        optimizer=torch.optim.Adam(m.decoder.parameters(),lr=.004)
        start=float(sdf_loss(m.decode(z,q),target).detach())
        for _ in range(80):
            optimizer.zero_grad();loss=sdf_loss(m.decode(z,q),target);loss.backward();optimizer.step()
        self.assertLess(float(sdf_loss(m.decode(z,q),target).detach()),start*.5)


if __name__=='__main__':unittest.main()
