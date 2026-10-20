import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from allen_reference_core import AllenModel,chunked_encode,latent_rotation_distances
import numpy as np

class OfficialModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.model=AllenModel().double().eval()
    def test_strict_official_checkpoint_parameters(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),14306163)
    def test_chunked_forward_and_input_gradient(self):
        torch.manual_seed(11);p=torch.randn(1,48,3,dtype=torch.float64)*.2;p.requires_grad_()
        a=self.model.encode(p);g=torch.autograd.grad(a.square().sum(),p)[0]
        b=chunked_encode(self.model.encoder,p,chunk=13,checkpoint=True);h=torch.autograd.grad(b.square().sum(),p)[0]
        torch.testing.assert_close(a,b,rtol=2e-10,atol=1e-11)
        torch.testing.assert_close(g,h,rtol=2e-8,atol=1e-9)
    def test_cached_pre_head_features_match_released_latent(self):
        torch.manual_seed(13);p=torch.randn(1,48,3,dtype=torch.float64)*.2
        with torch.no_grad():
            a=self.model.encode(p)
            features=chunked_encode(self.model.encoder,p,chunk=13,return_features=True)
            b=self.model.encoder.final_conv[-1](features)
        torch.testing.assert_close(a,b,rtol=2e-10,atol=1e-11)
    def test_rotation_equivariance_and_decoder_invariance(self):
        torch.manual_seed(12);p=torch.randn(1,48,3,dtype=torch.float64)*.2;q=torch.randn(1,17,3,dtype=torch.float64)*.2
        r,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        if torch.linalg.det(r)<0:r[:,0]*=-1
        with torch.no_grad():
            z=self.model.encode(p);rotated=self.model.encode(p@r)
            torch.testing.assert_close(rotated,z@r,rtol=1e-8,atol=1e-9)
            torch.testing.assert_close(self.model.decode(z,q),self.model.decode(rotated,q@r),rtol=1e-8,atol=1e-9)

class ReadoutTests(unittest.TestCase):
    def test_full_code_distance_preserves_angles_and_removes_rotation(self):
        a=np.eye(3);r,_=np.linalg.qr(np.random.default_rng(5).normal(size=(3,3)))
        if np.linalg.det(r)<0:r[:,0]*=-1
        b=np.array([[1.,0,0]]*3)
        d=latent_rotation_distances(np.stack([a,a@r,b]))
        self.assertLess(d[0,1],1e-7)
        self.assertGreater(d[0,2],1)
        np.testing.assert_allclose(np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1))
        np.testing.assert_allclose(d,d.T)

if __name__=='__main__':unittest.main()
