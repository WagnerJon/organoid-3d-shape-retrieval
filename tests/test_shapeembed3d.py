import unittest,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shapeembed3d_core import ShapeEmbed3D,edm,normalize_edm,gw_edm_loss,distance_coupling


class ShapeEmbed3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.manual_seed(11)

    def test_similarity_and_reflection_invariance(self):
        p=torch.randn(2,24,3,dtype=torch.float64);q=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64)).Q
        a,_=normalize_edm(edm(p));b,_=normalize_edm(edm(7*p@q+torch.tensor([2.,9.,-5.])))
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)

    def test_encoder_permutation_and_gradients(self):
        m=ShapeEmbed3D(n=24).double().eval();d=edm(torch.randn(2,24,3,dtype=torch.float64)).requires_grad_();perm=torch.randperm(24)
        z=m.encode(d)[0];other=m.encode(d[:,perm][:,:,perm])[0];torch.testing.assert_close(z,other,atol=1e-12,rtol=1e-10)
        g1=torch.autograd.grad(z.square().sum(),d,retain_graph=True)[0];g2=torch.autograd.grad(other.square().sum(),d)[0];torch.testing.assert_close(g1,g2,atol=1e-12,rtol=1e-9)

    def test_decoded_edm_is_three_dimensional(self):
        m=ShapeEmbed3D(n=24).double().eval();points,d=m.decode(torch.randn(2,128,dtype=torch.float64))
        torch.testing.assert_close(d,d.transpose(-2,-1));torch.testing.assert_close(d.diagonal(dim1=-2,dim2=-1),torch.zeros(2,24,dtype=torch.float64))
        h=torch.eye(24,dtype=torch.float64)-torch.ones(24,24,dtype=torch.float64)/24;gram=-.5*h@d.square()@h;values=torch.linalg.eigvalsh(gram)
        self.assertGreaterEqual(float(values.min().detach()),-1e-10);self.assertTrue(bool((values[:,:-3].abs()<1e-10).all()))

    def test_uniform_transport_feasibility(self):
        for n in [24,128]:
            a,_=normalize_edm(edm(torch.randn(2,n,3)));b,_=normalize_edm(edm(torch.randn(2,n,3)*torch.tensor([3.,1.,.3])))
            t=distance_coupling(a,b);torch.testing.assert_close(t.sum(-1),torch.full((2,n),1/n),atol=2e-7,rtol=2e-5);torch.testing.assert_close(t.sum(-2),torch.full((2,n),1/n),atol=2e-7,rtol=2e-5)

    def test_loss_oracle_and_gradient(self):
        n=12;pred=edm(torch.randn(1,n,3,dtype=torch.float64)).requires_grad_();truth=edm(torch.randn(1,n,3,dtype=torch.float64));perm=torch.randperm(n)
        b=truth[:,perm][:,:,perm];t=torch.zeros(1,n,n,dtype=torch.float64);t[:,perm,torch.arange(n)]=1/n
        value,_=gw_edm_loss(pred,b,t);expected=(pred-truth).square().mean();torch.testing.assert_close(value.mean(),expected)
        torch.testing.assert_close(torch.autograd.grad(value.sum(),pred,retain_graph=True)[0],torch.autograd.grad(expected,pred)[0])

    def test_matching_permutation_and_shape_sensitivity(self):
        p=torch.randn(1,24,3,dtype=torch.float64);a,_=normalize_edm(edm(p));b,_=normalize_edm(edm(p*torch.tensor([2.,1.,.4])));perm=torch.randperm(24)
        v,t=gw_edm_loss(a,b);v2,t2=gw_edm_loss(a[:,perm][:,:,perm],b)
        torch.testing.assert_close(v,v2,atol=1e-10,rtol=1e-10);torch.testing.assert_close(t[:,perm],t2,atol=1e-10,rtol=1e-8)
        self.assertLess(float(gw_edm_loss(a,a)[0]),float(v))
        self.assertLess(float(gw_edm_loss(a,a)[0]),1e-12)


if __name__=='__main__':unittest.main()
