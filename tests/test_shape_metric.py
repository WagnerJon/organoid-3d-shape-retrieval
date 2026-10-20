import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_shape_metric import shape_metric_loss
from shape_metric_common import matched_retrieval


class ShapeMetricTests(unittest.TestCase):
    def test_metric_loss_reaches_encoder(self):
        embedding = torch.randn(4, 8, requires_grad=True)
        target = torch.tensor([[0., 0.], [0.1, 0.], [1., 0.], [1.2, 0.]])
        cfg = {"metric_teacher_temperature": 0.3, "metric_student_temperature": 0.15}
        loss = shape_metric_loss(embedding, target, np.array([2] * 4), cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(embedding.grad.abs().sum()), 0)

    def test_matching_filters_count_volume_and_stack(self):
        frame = pd.DataFrame({"index": range(4), "level": ["single"] * 4,
            "sample": ["A", "B", "C", "D"], "similarity_group": range(4),
            "member_count": [1, 1, 1, 2], "volume_um3": [100., 100., 120., 100.]})
        vectors = np.array([[1., 0.], [.9, .1], [0., 1.], [-1., 0.]])
        targets = [np.array([x], np.float32) for x in [0., .1, 2., 5.]]
        result = matched_retrieval(frame, {"model": vectors},
            {"shape": targets}, ["model", "volume_only"], min_candidates=2)
        self.assertEqual(set(result["query"]), {0, 1, 2})
        self.assertTrue((result["candidates"] == 2).all())


if __name__ == "__main__":
    unittest.main()
