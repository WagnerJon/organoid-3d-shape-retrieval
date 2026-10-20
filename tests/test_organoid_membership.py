import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from organoid_membership import classify, rebuild, update


class MembershipTests(unittest.TestCase):
    def test_touching_cells_and_distant_sparse_id(self):
        labels = np.zeros((12, 12, 12), dtype=np.uint32)
        labels[1:5, 1:5, 1:3] = 10
        labels[1:5, 1:5, 3:5] = 30
        labels[9:11, 9:11, 9:11] = 9000000
        result = classify(labels)
        self.assertTrue(result[10]['inMainOrganoid'])
        self.assertTrue(result[30]['inMainOrganoid'])
        self.assertFalse(result[9000000]['inMainOrganoid'])
        self.assertEqual(result[9000000]['mainOrganoidFraction'], 0)
        self.assertEqual(classify(np.zeros((2, 2, 2), dtype=np.uint16)), {})

    def test_majority_not_single_stray_voxel(self):
        labels = np.zeros((12, 12, 12), dtype=np.uint16)
        labels[1:5, 1:5, 1:5] = 1
        labels[1, 1, 5] = 2
        labels[9:11, 9:11, 9:11] = 2
        self.assertFalse(classify(labels)[2]['inMainOrganoid'])
        with self.assertRaises(ValueError):
            classify(np.zeros((2, 2)))

    def fixture(self):
        cells = []
        for label, volume, include, qc in [(1, 2., True, True), (2, 4., True, True), (3, 900., False, True), (4, 6., True, False)]:
            cells.append(dict(id=f'S:L{label}', sample='S', label=label, qc=qc, flags='', split='test',
                metrics={'volume_um3': volume, 'shape': volume / 10}, inMainOrganoid=include, automaticMainOrganoid=include,
                representations={'traditional': {'vector': [0., 0.], 'point': [0., 0.], 'cluster': 0},
                'learned': {'vector': [volume, 1.], 'point': [0., 0.], 'cluster': 0}}))
        return dict(cells=cells, organoids=[dict(id='S', sample='S', metrics={}, representations={})], samples=[],
                    traditionalFeatures=['volume_um3', 'shape'], provenance='Original surfaces.',
                    methods=[dict(id='traditional', metric='euclidean'), dict(id='learned', metric='cosine')])

    def test_fits_and_stack_aggregates_do_not_use_excluded_cells(self):
        data = self.fixture()
        result = rebuild(copy.deepcopy(data))
        np.testing.assert_allclose(result['normalization']['mean'], [4., .4])
        stack = result['organoids'][0]
        self.assertEqual(stack['cellCount'], 3)
        self.assertEqual(stack['metrics']['volume_um3'], 12.)
        self.assertAlmostEqual(stack['metrics']['shape'], .3)
        np.testing.assert_allclose(stack['representations']['learned']['vector'], np.array([3., 1.]) / np.sqrt(10))
        # Changing only an excluded cell must not alter any included fit or stack statistic.
        data['cells'][2]['metrics']['volume_um3'] = 1000000.
        data['cells'][2]['representations']['learned']['vector'] = [1e6, -1e6]
        second = rebuild(data)
        self.assertEqual(result['organoids'], second['organoids'])
        for a, b in zip(result['cells'], second['cells']):
            if a['inMainOrganoid']:
                self.assertEqual(a['representations'], b['representations'])
        self.assertEqual(result['membershipAnalysis']['excluded'], 1)

    def test_saved_overrides_restore_automatic_and_empty_stack(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            data = self.fixture()
            path = folder / 'dataset.json'
            path.write_text(json.dumps(data))
            excluded = update(folder, {c['id']: False for c in data['cells']})
            self.assertEqual(excluded['organoids'], [])
            self.assertEqual(len(excluded['cells']), 4)
            restored = update(folder, {c['id']: None for c in data['cells']})
            self.assertEqual(restored['organoids'][0]['cellCount'], 3)
            self.assertIsNone(restored['cells'][0]['membershipOverride'])
            self.assertEqual(restored['membershipAnalysis']['included'], 3)

    def test_native_labels_classified_and_missing_source_fails_atomically(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            labels = np.zeros((8, 8, 8), dtype=np.uint16)
            labels[1:4, 1:4, 1:4] = 1
            labels[7, 7, 7] = 3
            source = folder / 'labels.tif'
            tifffile.imwrite(source, labels)
            data = self.fixture()
            data['cells'] = [c for c in data['cells'] if c['label'] in (1, 3)]
            for c in data['cells']:
                del c['automaticMainOrganoid']
            data['samples'] = [dict(id='S', sourceLabels=str(source))]
            path = folder / 'dataset.json'
            path.write_text(json.dumps(data))
            output = update(folder)
            self.assertTrue(output['cells'][0]['inMainOrganoid'])
            self.assertFalse(output['cells'][1]['inMainOrganoid'])
            del output['cells'][0]['automaticMainOrganoid']
            output['samples'][0]['sourceLabels'] = str(folder / 'missing.tif')
            original = json.dumps(output)
            path.write_text(original)
            with self.assertRaises(FileNotFoundError):
                update(folder)
            self.assertEqual(path.read_text(), original)

class WorkbenchMembershipTests(unittest.TestCase):
    def test_segmentation_comparison_excludes_satellites_and_preserves_native_masks(self):
        from segmentation_explorer import publish, compare, agreement, annotate_history
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = dict(nativeShape=[14, 14, 14], shape=[14, 14, 14], spacing=[1., 1., 1.], fingerprint='same-image')
            a = np.zeros((14, 14, 14), dtype=np.uint32)
            a[1:5, 1:5, 1:3] = 1
            a[1:5, 1:5, 3:5] = 2
            b = a.copy()
            a[10:12, 10:12, 10:12] = 70000
            b[11:13, 9:11, 10:12] = 90000
            runs = []
            for key, labels in [('a', a), ('b', b)]:
                folder = root / key
                folder.mkdir()
                runs.append(publish(labels, source, folder, 'cellpose', {}, 0., 'Test'))
            originals = [(root / key / 'labels_native.tif').read_bytes() for key in ('a', 'b')]
            self.assertLess(agreement(a, b)['dice'], 1.)
            curated = compare(*runs, source)
            self.assertEqual(curated['dice'], 1.)
            self.assertEqual(curated['countA'], 2)
            self.assertEqual(curated['excludedA'], [70000])
            self.assertEqual(curated['excludedB'], [90000])
            self.assertFalse(runs[0]['cells'][-1]['inMainOrganoid'])
            # An older saved run is upgraded on raw-stack preparation, without changing labels.
            for cell in runs[0]['cells']:
                del cell['inMainOrganoid']
            (root / 'a' / 'result.json').write_text(json.dumps(runs[0]))
            annotate_history(root, source)
            upgraded = json.loads((root / 'a' / 'result.json').read_text())
            self.assertFalse(upgraded['cells'][-1]['inMainOrganoid'])
            self.assertEqual(originals, [(root / key / 'labels_native.tif').read_bytes() for key in ('a', 'b')])


if __name__ == '__main__':
    unittest.main()
