import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from point_mae_reference import model,encode_grouped,masked_prediction,fps_indices,KNN,SquaredChamfer,normalized


class PointMAEReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(3);cls.m=model().eval()

    def test_strict_architecture(self):
        self.assertEqual(sum(p.numel() for p in self.m.parameters()),29006432)
        self.assertEqual(len(self.m.MAE_encoder.blocks.blocks),12)
        self.assertEqual(len(self.m.MAE_decoder.blocks),4)

    def test_geometric_adapters(self):
        p=torch.randn(2,64,3,dtype=torch.float64);idx=fps_indices(p,8).numpy()
        for b in range(2):
            chosen=[0]
            for _ in range(7):
                d=np.square(p[b].numpy()[:,None]-p[b,chosen].numpy()[None]).sum(-1).min(-1);chosen.append(int(d.argmax()))
            np.testing.assert_array_equal(idx[b],chosen)
        distance,neighbors=KNN(7)(p,p[:,:8]);brute=((p[:,:8,None]-p[:,None])**2).sum(-1).sqrt()
        torch.testing.assert_close(distance,brute.sort(-1).values[:,:,:7])
        torch.testing.assert_close(brute.gather(-1,neighbors),distance)

    def test_chamfer_definition_and_gradients(self):
        a=torch.randn(2,5,3,dtype=torch.float64,requires_grad=True);b=torch.randn(2,7,3,dtype=torch.float64)
        d=torch.cdist(a,b).square();expected=d.min(-1).values.mean()+d.min(-2).values.mean();actual=SquaredChamfer()(a,b)
        torch.testing.assert_close(actual,expected);torch.testing.assert_close(torch.autograd.grad(actual,a,retain_graph=True)[0],torch.autograd.grad(expected,a)[0])

    def test_original_neural_forward(self):
        p=torch.randn(2,128,3)
        with torch.no_grad():
            groups,centers=self.m.group_divider(p);tokens,_=self.m.MAE_encoder(groups,centers,noaug=True)
            expected=torch.cat([tokens.mean(1),tokens.max(1).values],-1)
            torch.testing.assert_close(encode_grouped(self.m,groups,centers),expected)
            np.random.seed(19);original=self.m(p)
            np.random.seed(19);prediction,target,*_=masked_prediction(self.m,p)
            torch.testing.assert_close(self.m.loss_func(prediction,target),original)

    def test_normalization_rotation(self):
        p=np.random.default_rng(2).normal(size=(128,3));q,_=np.linalg.qr(np.random.default_rng(9).normal(size=(3,3)))
        np.testing.assert_allclose(normalized(p@q),normalized(p)@q,atol=2e-7)


if __name__=='__main__':unittest.main()
