import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).resolve().parents[1] / "scripts/evaluate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MatchingTest(unittest.TestCase):
    def test_arbitrary_label_ids(self):
        ref = np.array([0, 1, 1, 2, 2])
        pred = np.array([0, 90, 90, 4, 4])
        self.assertEqual(module.evaluate(ref, pred)["f1"], 1)

    def test_merge_is_not_two_matches(self):
        ref = np.array([0, 1, 1, 2, 2])
        pred = np.array([0, 7, 7, 7, 7])
        result = module.evaluate(ref, pred)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)

    def test_empty_prediction(self):
        result = module.evaluate(np.array([0, 1]), np.array([0, 0]))
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["f1"], 0)
