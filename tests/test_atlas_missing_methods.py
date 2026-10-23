import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import atlas_missing_methods as methods


class MissingMethodsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='atlas-method-completion-')
        self.folder = Path(self.temporary.name)
        (self.folder / 'arbitrary_sample_name').mkdir()
        cells = []
        for i in range(3):
            mesh = trimesh.creation.icosphere(subdivisions=1, radius=2)
            mesh.vertices[:, 0] *= 1 + i * .25
            path = self.folder / 'arbitrary_sample_name' / f'{i + 1}.mesh'
            path.write_bytes(struct.pack('<II', len(mesh.vertices), len(mesh.faces))
                             + np.asarray(mesh.vertices, '<f4').tobytes()
                             + np.asarray(mesh.vertex_normals, '<f4').tobytes()
                             + np.asarray(mesh.faces, '<u4').tobytes())
            metrics = dict(volume_um3=float(mesh.volume), surface_area_um2=float(mesh.area),
                           major_axis_um=4 + i, middle_axis_um=4., minor_axis_um=4.,
                           sphericity=.9 - i * .1, elongation=1 + i * .25, flatness=1., mesh_solidity=1.)
            cells.append(dict(id=f'unusual:{i + 1}', sample='arbitrary_sample_name', label=i + 1,
                              qc=True, split='new', flags='', metrics=metrics, inMainOrganoid=i != 2,
                              representations={'traditional': dict(vector=[float(i)], point=[0., 0.], cluster=0)}))
        self.data = dict(version=1, title='Arbitrary dataset', provenance='Synthetic test geometry.',
                         traditionalFeatures=['volume_um3'], methods=[dict(id='traditional', name='Traditional', dimensions=1,
                         metric='euclidean', variance=[0., 0.], stackVariance=[0., 0.])], cells=cells,
                         samples=[dict(id='arbitrary_sample_name', shape=[20, 20, 20], spacing=[1., 1., 1.], extent=[20., 20., 20.],
                                       rawAvailable=False, sourceImage='unavailable.tif', sourceLabels='unavailable-labels.tif', meshCount=3)],
                         organoids=[dict(id='arbitrary_sample_name', sample='arbitrary_sample_name', qc=True, split='new', flags='',
                                         label=0, cellCount=2, inMainOrganoid=True, metrics={}, representations={})])
        methods.save(self.folder / 'dataset.json', self.data)

    def tearDown(self):
        self.temporary.cleanup()

    def test_plan_detects_partial_coverage_and_explains_missing_inputs(self):
        plan = methods.inspect(self.folder)
        self.assertEqual(plan['cells'], 3)
        self.assertEqual(sum(len(g['methods']) for g in plan['groups']), 35)
        self.assertNotIn('cajal_gw_validation', [k for g in plan['groups'] for k in g['methods']])
        image = next(g for g in plan['groups'] if g['id'] == 'image')
        self.assertFalse(image['available'])
        self.assertIn('original raw image', image['reason'])
    def test_missing_weights_and_meshes_are_reported(self):
        with patch.object(methods, 'requirements', return_value=[methods.ROOT / 'missing-model.pt']):
            self.assertTrue(all(not g['available'] and 'Missing model' in g['reason'] for g in methods.inspect(self.folder)['groups']))
        (self.folder / 'arbitrary_sample_name/1.mesh').unlink()
        self.assertTrue(all(not g['available'] for g in methods.inspect(self.folder)['groups']))

    def test_real_harmonic_inference_publication_and_resume(self):
        # Preserve one existing vector to exercise partial, rather than absent,
        # coverage. Excluded cells get descriptors but never enter stack means.
        spec = copy.deepcopy(next(m for m in methods.catalog() if m['id'] == 'spherical_harmonics'))
        original = np.full(spec['dimensions'], .123).tolist()
        self.data['methods'].append(spec)
        self.data['cells'][0]['representations'][spec['id']] = dict(vector=original, point=[0., 0.], cluster=0)
        methods.save(self.folder / 'dataset.json', self.data)
        mesh_hashes = [methods.digest(methods.mesh_path(self.folder, c)) for c in self.data['cells']]
        result = methods.run(self.folder, self.folder / 'report.json', only=['harmonics'], device='cpu')
        self.assertFalse(result['failed'])
        self.assertEqual(set(result['completed']), {'spherical_harmonics', 'spherical_harmonics_adjusted'})
        updated = json.loads((self.folder / 'dataset.json').read_text())
        self.assertEqual(updated['cells'][0]['representations']['spherical_harmonics']['vector'], original)
        for cell, old in zip(updated['cells'], self.data['cells']):
            self.assertEqual(cell['metrics'], old['metrics'])
            self.assertEqual(cell['inMainOrganoid'], old['inMainOrganoid'])
            self.assertEqual(cell['representations']['traditional'], old['representations']['traditional'])
            self.assertIn('spherical_harmonics_adjusted', cell['representations'])
        expected_mean = np.mean([c['representations']['spherical_harmonics']['vector'] for c in updated['cells'][:2]], axis=0)
        np.testing.assert_allclose(updated['organoids'][0]['representations']['spherical_harmonics']['vector'], expected_mean)
        self.assertEqual(mesh_hashes, [methods.digest(methods.mesh_path(self.folder, c)) for c in updated['cells']])
        self.assertTrue(list((self.folder / '.atlas-methods').glob('dataset.before-*.json')))
        before = (self.folder / 'dataset.json').read_bytes()
        again = methods.run(self.folder, self.folder / 'again.json', only=['harmonics'], device='cpu')
        self.assertEqual(again['completed'], [])
        self.assertEqual(before, (self.folder / 'dataset.json').read_bytes())
    def test_concurrent_dataset_edit_prevents_publication(self):
        work = methods.prepare_workspace(self.folder, self.data, ['harmonics'])
        before = methods.digest(self.folder / 'dataset.json')
        self.data['title'] = 'User changed title'
        methods.save(self.folder / 'dataset.json', self.data)
        with self.assertRaisesRegex(RuntimeError, 'Dataset changed'):
            methods.publish(self.folder, work, 'harmonics', ['spherical_harmonics'], before)
        self.assertEqual(json.loads((self.folder / 'dataset.json').read_text())['title'], 'User changed title')

    def test_resume_keeps_cache_when_the_remaining_families_change(self):
        first = methods.prepare_workspace(self.folder, self.data, ['spatial', 'harmonics'])
        marker = first / 'representations/completed-cell-marker'
        marker.write_text('retained intermediate result')
        second = methods.prepare_workspace(self.folder, self.data, ['harmonics'])
        self.assertEqual(first, second)
        self.assertEqual(marker.read_text(), 'retained intermediate result')


if __name__ == '__main__':
    unittest.main()
