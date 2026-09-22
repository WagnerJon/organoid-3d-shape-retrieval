import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_isolated_raw_simclr3d import isolate


class IsolatedRawTests(unittest.TestCase):
    def test_only_target_intensity_survives(self):
        labels = np.zeros((20,20,20), np.uint16)
        labels[3:9,4:10,5:11] = 1
        labels[10:17,10:17,10:17] = 2
        raw = np.ones(labels.shape, np.float32)
        out, mask, _ = isolate(raw, labels, 1, size=32)
        self.assertEqual(out.shape, (32,32,32))
        self.assertTrue(np.all(out[~mask] == 0))
        # Linear resizing softens the boundary, while preserving interior intensity.
        self.assertGreater(out[mask].mean(), .85)
        self.assertLess(mask.sum(), out.size)

    def test_target_geometry_changes_output(self):
        labels = np.zeros((24,24,24), np.uint16)
        labels[4:20,8:14,9:15] = 1
        labels[8:14,4:20,9:15] = 2
        raw = np.ones(labels.shape, np.float32)
        one, _, _ = isolate(raw, labels, 1, size=32)
        two, _, _ = isolate(raw, labels, 2, size=32)
        self.assertFalse(np.array_equal(one, two))
