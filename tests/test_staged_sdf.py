import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from staged_sdf_core import ReferenceVectorEncoder,VectorAdam,code_loss
from pointnet_shape_core import rotations


class StagedChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(21)

    def test_reference_encoder_rotates_and_preserves_permutations(self):
        model=ReferenceVectorEncoder();p=torch.randn(2,64,3)
        neighbors=torch.tensor(np.stack([cKDTree(x.numpy()).query(x.numpy(),k=20)[1] for x in p]))
        rotation=torch.tensor(rotations(2,np.random.default_rng(1)))
        with torch.no_grad():
            z=model(p,neighbors);zr=model(p@rotation.transpose(1,2),neighbors)
        torch.testing.assert_close(zr,z@rotation.transpose(1,2),rtol=2e-4,atol=3e-6)
        permutation=torch.randperm(64);pp=p[:,permutation]
        new=torch.tensor(np.stack([cKDTree(x.numpy()).query(x.numpy(),k=20)[1] for x in pp]))
        with torch.no_grad():zp=model(pp,new)
        torch.testing.assert_close(z,zp,rtol=2e-4,atol=3e-6)
        self.assertEqual(z.shape,(2,128,3))

    def test_vector_optimizer_commutes_with_rotation(self):
        a=torch.nn.Parameter(torch.randn(2,12,3));r=torch.tensor(rotations(2,np.random.default_rng(2)))
        b=torch.nn.Parameter(a.detach()@r.transpose(1,2));oa=VectorAdam(a);ob=VectorAdam(b)
        for _ in range(6):
            g=torch.randn_like(a);a.grad=g;b.grad=g@r.transpose(1,2);oa.step();ob.step()
        torch.testing.assert_close(b,a@r.transpose(1,2),rtol=1e-5,atol=1e-6)

    def test_code_supervision_has_finite_nonzero_encoder_gradient(self):
        model=ReferenceVectorEncoder();p=torch.randn(2,32,3)
        graph=torch.tensor(np.stack([cKDTree(x.numpy()).query(x.numpy(),k=20)[1] for x in p]))
        loss=code_loss(model(p,graph),torch.randn(2,128,3)*.3,torch.ones(1,128,1)*.3)
        loss.backward();gradients=[x.grad for x in model.parameters()]
        self.assertTrue(all(x is not None and torch.isfinite(x).all() for x in gradients))
        self.assertGreater(sum(float(x.abs().sum()) for x in gradients),0)

    def test_inference_views_are_disjoint_and_balanced(self):
        from evaluate_staged_sdf import observation_indices
        a,b=observation_indices(0),observation_indices(1)
        self.assertEqual(len(a),1536);self.assertEqual(len(b),1536)
        self.assertEqual(len(set(a)&set(b)),0)
        self.assertEqual(len(set(a)|set(b)),3072)
        for x in [a,b]:
            self.assertEqual(sum(x<768),384)
            self.assertEqual(sum((x>=768)&(x<1536)),384)
            self.assertEqual(sum((x>=1536)&(x<2560)),512)

    def test_fitting_freezes_decoder_and_improves_held_out_sphere_queries(self):
        from evaluate_staged_sdf import fit_codes
        class Sphere(torch.nn.Module):
            def __init__(self):super().__init__();self.offset=torch.nn.Parameter(torch.tensor(.05))
            def decode(self,z,q):return q.norm(dim=-1)-z.norm(dim=-1).mean(1)[:,None]+self.offset
        model=Sphere();q=torch.randn(2,200,3)*.3;t=q.norm(dim=-1)-.6
        initial=torch.ones(2,128,3)*.1
        before=float((model.decode(initial,q)-t.clamp(-.15,.15)).abs().mean().detach())
        fitted,_=fit_codes(model,initial,q,t,80)
        after=float((model.decode(fitted,q)-t.clamp(-.15,.15)).abs().mean().detach())
        self.assertLess(after,before*.6)
        self.assertFalse(model.offset.requires_grad);self.assertEqual(float(model.offset),float(torch.tensor(.05)))

    def test_symmetric_shape_cannot_have_arbitrary_global_vector_pose(self):
        # A point set invariant under 180-degree rotations about X and Y forces
        # every permutation-invariant equivariant global vector to be zero.
        # Free auto-decoder codes need not respect that stabilizer constraint.
        base=np.random.default_rng(22).normal(size=(16,3))
        base=base/np.linalg.norm(base,axis=1,keepdims=True)*np.array([1.,.7,.4])
        signs=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
        p=torch.tensor((base[None]*signs[:,None]).reshape(-1,3),dtype=torch.float32)[None]
        graph=torch.tensor(cKDTree(p[0].numpy()).query(p[0].numpy(),k=20)[1])[None]
        with torch.no_grad():z=ReferenceVectorEncoder()(p,graph)
        self.assertLess(float(z.norm()),5e-5)


if __name__=='__main__':unittest.main()
