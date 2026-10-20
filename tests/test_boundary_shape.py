import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_boundary_shape import geometry, sdf_target
from train_boundary_shape import ShapeEncoder, geometry_rank_loss


class BoundaryShapeTests(unittest.TestCase):
    def test_instance_interface_is_a_distance_boundary(self):
        labels = np.zeros((12, 12, 12), dtype=np.uint8)
        labels[2:10, 2:6, 2:10] = 1
        labels[2:10, 6:10, 2:10] = 2
        sdf = sdf_target(labels, 1., 6.)
        self.assertEqual(float(sdf[5, 5, 5]), 0.)
        self.assertGreater(float(sdf[5, 3, 5]), 0.)
        self.assertLess(float(sdf[0, 0, 0]), 0.)
        descriptor, count = geometry(labels, 1.)
        self.assertEqual(count, 2)
        self.assertEqual(len(descriptor), 5)

    def test_encoder_and_geometry_gradient(self):
        model = ShapeEncoder(base=8, embedding_dim=16)
        raw = torch.rand(3, 1, 32, 32, 32)
        embedding, dense = model(raw)
        self.assertEqual(tuple(embedding.shape), (3, 16))
        self.assertEqual(tuple(dense.shape), (3, 3, 32, 32, 32))
        descriptors = torch.tensor([[.4, 1., 1.], [.7, 1., 1.], [1.6, 1., 1.]])
        loss = geometry_rank_loss(embedding, descriptors, np.array([3, 3, 3])) + dense.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.stem.body[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
