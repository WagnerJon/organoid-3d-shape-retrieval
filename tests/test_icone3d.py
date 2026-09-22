import sys, unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from icone3d_core import losses


class IConELossTests(unittest.TestCase):
    def test_three_terms_and_gradients(self):
        z1 = torch.randn(3, 8, requires_grad=True)
        z2 = torch.randn(3, 8, requires_grad=True)
        table = torch.randn(5, 8, requires_grad=True)
        total, parts = losses(z1, z2, table[:3], table)
        self.assertEqual(set(parts), {"view_instance", "view_view", "diversity"})
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(table.grad).all())

    def test_batch_one_and_orthogonal_diversity(self):
        table = torch.eye(4, requires_grad=True)
        view = table[:1].clone()
        total, parts = losses(view, view, table[:1], table)
        self.assertAlmostEqual(float(parts["view_view"]), 0, places=6)
        self.assertAlmostEqual(float(parts["view_instance"]), 0, places=6)
        self.assertAlmostEqual(float(parts["diversity"]), 0, places=6)
        total.backward()
        self.assertIsNotNone(table.grad)
