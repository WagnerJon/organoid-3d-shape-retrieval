"""Scientific and identity contracts for interactive projections / discovery."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import unittest
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from atlas_interactive import analyze, comparison_distances, discover, project_three


class InteractiveShapeTests(unittest.TestCase):
    def test_three_pcs_preserve_existing_two_and_degenerate_cases(self):
        x = np.random.default_rng(42).normal(size=(40, 8))
        result = project_three(x, 'euclidean')
        expected = PCA(n_components=2, svd_solver='full').fit(x)
        np.testing.assert_allclose(np.array(result['points'])[:, :2], expected.transform(x), atol=1e-12)
        assert np.array(result['points']).shape == (40, 3)
        np.testing.assert_allclose(result['variance'][:2], expected.explained_variance_ratio_)
        for x in [np.zeros((1, 8)), np.zeros((5, 8)), np.array([[1.], [2.]])]:
            result = project_three(x, 'euclidean')
            assert np.array(result['points']).shape == (len(x), 3)
            assert np.isfinite(result['variance']).all()


    def test_distance_map_preserves_global_subset_identities(self):
        x = np.random.default_rng(12).normal(size=(12, 3)); d = cdist(x, x)
        ids = [8, 2, 11, 5]
        packed = np.c_[np.arange(12), d][ids]
        np.testing.assert_allclose(comparison_distances(packed, 'precomputed'), d[np.ix_(ids, ids)])
        result = project_three(packed, 'precomputed')
        assert result['projection'] == 'mds'
        np.testing.assert_allclose(cdist(result['points'], result['points']), d[np.ix_(ids, ids)], atol=1e-12)


    def test_cosine_metric_matches_full_vector_ranking_and_rejects_zero(self):
        x = np.array([[1., 0., 0.], [5., 1., 0.], [0., 1., 0.]])
        d = comparison_distances(x, 'cosine')
        np.testing.assert_allclose(d, cdist(x, x, metric='cosine'), atol=1e-12)
        assert d[0, 1] < d[0, 2]
        with self.assertRaisesRegex(ValueError, 'non-zero'):
            comparison_distances(np.zeros((3, 4)), 'cosine')


    def test_automatic_groups_use_full_distance_and_actual_medoids(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0, .02, (30, 8)); b = rng.normal(0, .02, (30, 8))
        b[:, -1] += 10  # Separation outside the first two input coordinates.
        d = cdist(np.r_[a, b], np.r_[a, b]); ids = [f'cell{i}' for i in range(60)]
        result = discover(d, ids, 8, 5)
        assert len(result['groups']) == 2
        assert set(result['labels'][:30]) != set(result['labels'][30:])
        for group in result['groups']:
            members = [ids.index(cid) for cid in group['members']]
            expected = members[np.argmin(d[np.ix_(members, members)].mean(1))]
            assert group['representative'] == ids[expected]
            distances = [m['distance'] for m in group['nearest']]
            assert distances == sorted(distances)
            assert all(m['id'] != group['representative'] for m in group['nearest'])
        assert discover(d[:3, :3], ids[:3], 10, 5)['unassigned'] == 3


    def test_cohort_excludes_detached_and_missing_codes_without_relabeling_sources(self):
        cells = [dict(id=f'c{i}', inMainOrganoid=i != 1, representations={'m': dict(vector=[float(i), i*i])}) for i in range(5)]
        cells[-1]['representations'] = {}
        data = dict(methods=[dict(id='m', metric='euclidean', scope='cells')], cells=cells, organoids=[])
        result = analyze(data, dict(action='projection', method='m', ids=['c0', 'c1', 'c3', 'c4']))
        assert result['ids'] == ['c0', 'c3']
        assert cells[1]['inMainOrganoid'] is False and 'point' not in cells[0]['representations']['m']
        with self.assertRaisesRegex(ValueError, 'cell-level'):
            analyze(data, dict(action='projection', method='m', level='Organoids'))

    def test_broad_parent_is_opt_in_and_request_controls_reach_worker(self):
        # Even unstructured data can have exploratory density branches: their
        # existence alone must never be interpreted as biological validation.
        x = np.random.RandomState(0).rand(150, 2)
        d = cdist(x, x); ids = [str(i) for i in range(len(x))]
        broad = discover(d, ids, allow_single=True)
        balanced = discover(d, ids)
        assert len(broad['groups']) == 1
        assert len(balanced['groups']) > 1
        assert balanced['allowSingle'] is False
        data = dict(methods=[dict(id='m', metric='euclidean')],
                    cells=[dict(id=cid, representations={'m': dict(vector=v.tolist())})
                           for cid, v in zip(ids, x)])
        fine = analyze(data, dict(action='groups', method='m', selection='leaf', allowSingle=False))
        expected = discover(d, ids, selection='leaf')
        assert fine['labels'] == expected['labels']
        assert fine['selection'] == 'leaf' and fine['allowSingle'] is False
        assert fine['unassigned'] + sum(len(g['members']) for g in fine['groups']) == len(ids)
        with self.assertRaisesRegex(ValueError, 'Unknown group selection'):
            discover(d, ids, selection='invalid')
        # A constant cloud should not be forced into multiple groups.
        assert discover(np.zeros((30, 30)), ids[:30])['groups'] == []


    def test_hybrid_distance_never_treats_packed_sign_frames_as_plain_vectors(self):
        x = np.zeros((3, 476)); x[:, 0] = 1
        x[1, 1:381] = 500  # Ambiguous frames must not affect fallback distance.
        x[2, 381:] = 2
        d = comparison_distances(x, 'shape_hybrid95')
        assert d[0, 1] == 0
        np.testing.assert_allclose(d[0, 2], np.sqrt(95 * 4))


if __name__ == "__main__":
    unittest.main()
