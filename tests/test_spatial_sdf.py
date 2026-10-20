import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from spatial_sdf import SpatialSDF


class SpatialChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(26)

    def test_symmetric_inputs_have_nonzero_codes_and_finite_gradients(self):
        a=torch.linspace(-1.15,1.15,32);q=torch.stack(torch.meshgrid(a,a,a,indexing='ij'),-1)
        grids=torch.stack([q.norm(dim=-1)-.6,(q/torch.tensor([1.,.7,.4])).norm(dim=-1)-.6])[:,None]
        model=SpatialSDF();z=model.encode(grids)
        self.assertEqual(z.shape,(2,128));self.assertGreater(float(z.norm().detach()),.01)
        self.assertGreater(float((z[0]-z[1]).norm().detach()),.001)
        loss=model.decode(z,torch.randn(2,50,3)).square().mean();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_decoding_depends_on_geometry_code(self):
        model=SpatialSDF();q=torch.randn(1,32,3)
        self.assertGreater(float((model.decode(torch.zeros(1,128),q)-model.decode(torch.ones(1,128),q)).abs().mean().detach()),1e-4)

    def test_grid_decoder_preserves_xyz_axes(self):
        from evaluate_spatial_sdf import GridDecoder
        a=torch.linspace(-1.15,1.15,32);x,y,z=torch.meshgrid(a,a,a,indexing='ij')
        field=x+2*y+3*z;q=torch.tensor([[[.1,-.2,.3],[-.5,.2,.7]]])
        result=GridDecoder().decode(field[None],q)
        torch.testing.assert_close(result,q@torch.tensor([1.,2.,3.]),rtol=1e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
