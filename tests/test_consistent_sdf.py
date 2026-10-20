import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from consistent_sdf import consistency_loss
from pointnet_shape_core import rotations


class ConsistencyChecks(unittest.TestCase):
    def test_identical_views_agree_and_rotation_preserves_objective(self):
        torch.manual_seed(9);a=torch.randn(4,128,3);b=a+.1*torch.randn_like(a)
        r=torch.tensor(rotations(4,np.random.default_rng(9)))
        before=consistency_loss(a,b);after=consistency_loss(a@r.transpose(1,2),b@r.transpose(1,2))
        for x,y in zip(before,after):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)
        self.assertEqual(float(consistency_loss(a,a)[0]),0)

    def test_discrimination_penalizes_identical_cells(self):
        distinct=torch.zeros(4,128,3)
        for i in range(4):distinct[i,i,0]=1
        collapsed=torch.ones(4,128,3)
        self.assertLess(float(consistency_loss(distinct,distinct)[1]),float(consistency_loss(collapsed,collapsed)[1]))

    def test_zero_codes_do_not_produce_nan_gradients(self):
        a=torch.zeros(4,128,3,requires_grad=True);b=torch.zeros_like(a,requires_grad=True)
        loss=sum(consistency_loss(a,b));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(a.grad).all())


if __name__=='__main__':unittest.main()
