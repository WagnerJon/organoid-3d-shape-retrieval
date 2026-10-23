import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import ndimage
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from generate_atlas_demo import generate, FAMILIES, SHAPE, field
from prepare_pointnet_shape import read_mesh


class DemoGeometryTests(unittest.TestCase):
    def test_distinct_shapes_include_a_real_torus_hole(self):
        origin = np.zeros((1, 3))
        for family in FAMILIES:
            value = float(field(family, origin, 0)[0])
            self.assertGreater(value, 0) if family == 'Donut' else self.assertLess(value, 0)
        self.assertLess(field('Donut', np.array([[10.5, 0, 0]]), 0)[0], 0)

    def test_generated_dataset_is_self_contained_and_measured(self):
        with tempfile.TemporaryDirectory(prefix='atlas-demo-geometry-') as temporary:
            folder = Path(temporary)
            data = generate(folder, variants=2)
            self.assertTrue(data['syntheticDemo'])
            self.assertEqual(len(data['cells']), 12)
            self.assertEqual({c['name'].split()[0] for c in data['cells']}, set(FAMILIES))
            for sample in data['samples']:
                image, labels = folder / sample['sourceImage'], folder / sample['sourceLabels']
                self.assertFalse(Path(sample['sourceImage']).is_absolute())
                raw, segmentation = tifffile.memmap(image), tifffile.imread(labels)
                self.assertEqual(raw.shape, SHAPE)
                self.assertEqual(set(np.unique(segmentation)), set(range(7)))
                for label in range(1, 7):
                    self.assertEqual(ndimage.label(segmentation == label)[1], 1)
                self.assertEqual((folder / sample['id'] / 'raw.u8').stat().st_size, np.prod(SHAPE))
            for cell in data['cells']:
                self.assertTrue(cell['inMainOrganoid'])
                self.assertTrue(cell['membershipOverride'])
                self.assertTrue(np.isfinite(list(cell['metrics'].values())).all())
                v, f = read_mesh(folder / cell['sample'] / f"{cell['label']}.mesh")
                self.assertGreater(len(v), 30)
                self.assertGreater(len(f), 30)
                self.assertEqual(cell['metrics']['demo_euler_characteristic'], 0 if cell['name'].startswith('Donut') else 2)
            self.assertIn('not biological', data['provenance'])
            with self.assertRaises(FileExistsError):
                generate(folder)


if __name__ == '__main__':
    unittest.main()
