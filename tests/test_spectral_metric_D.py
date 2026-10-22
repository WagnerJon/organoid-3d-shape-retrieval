import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from spectral_metric_core import *
from scipy.optimize import check_grad


class SpectralMetricTests(unittest.TestCase):
    def test_full_analytic_gradient(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(23, DIM)); y = rng.normal(size=23)
        pw = np.ones(23) / 23; e = rng.uniform(0, .01, DIM)
        theta = rng.uniform(.2, 1.2, DIM + 1)
        args = (x, y, pw, e, .01, .2, .1)
        error = check_grad(lambda t: objective(t, *args)[0],
                           lambda t: objective(t, *args)[1], theta)
        self.assertLess(error, 2e-4)

    def test_metric_learns_informative_coordinate_and_stays_nonnegative(self):
        rng = np.random.default_rng(33); n = 60
        x = np.zeros((n, DIM)); x[:, :2] = rng.normal(size=(n, 2))
        f = pd.DataFrame(dict(similarity_group=np.repeat(np.arange(6), 10)))
        pairs = [dict(i=i, j=j, weight=1.) for i in range(n) for j in range(i+1, n)
                 if f.iloc[i].similarity_group != f.iloc[j].similarity_group]
        labels = {(p['i'], p['j']): float((x[p['i'], 0]-x[p['j'], 0])**2) for p in pairs}
        result = fit_metric(x, x, f, np.arange(n), pairs, labels,
                            dict(ridge=.001, smooth=0., noise=0.))
        self.assertTrue((result['weights'] >= 0).all())
        self.assertGreater(result['weights'][0], 10 * max(result['weights'][1], 1e-7))
        self.assertLessEqual(result['weights'].max(), 20.)
        bad_pairs = pairs + [dict(i=0, j=n, weight=1.)]
        with self.assertRaises(AssertionError):
            fit_metric(x, x, f, np.arange(n), bad_pairs, labels,
                       dict(ridge=.01, smooth=0., noise=0.))

    def test_grouped_cohorts_and_pair_endpoints(self):
        frame = pd.read_csv(OUT / 'manifest.csv')
        plans = json.loads((OUT / 'folds.json').read_text())
        pairs = json.loads((OUT / 'pair_plans.json').read_text()); observed = []
        for plan, pair_plan in zip(plans, pairs):
            observed.extend(plan['test'])
            sets = [plan[k] for k in ['train', 'validation', 'test']]
            for i in range(3):
                for j in range(i + 1, 3):
                    for col in ['sample', 'similarity_group']:
                        self.assertFalse(set(frame.iloc[sets[i]][col]) & set(frame.iloc[sets[j]][col]))
            for key, ids in [('inner_pairs', plan['train']), ('refit_pairs', plan['development'])]:
                allowed = set(ids)
                self.assertTrue(all(p['i'] in allowed and p['j'] in allowed for p in pair_plan[key]))
            self.assertGreaterEqual(len(pair_plan['validation_pools']), 100)
        self.assertEqual(sorted(observed), list(range(len(frame))))

    def test_distance_has_no_pose_or_intercept_and_respects_scale(self):
        rng = np.random.default_rng(23); x = rng.normal(size=(12, DIM))
        w = rng.uniform(0, 2, DIM); d = distance(x, np.arange(12), np.arange(12), w)
        np.testing.assert_allclose(d, d.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(d), 0, atol=1e-12)
        np.testing.assert_allclose(distance(x, [1], np.arange(12), w * 3), d[1:2] * 3)
        self.assertEqual(d[1].argmin(), 1)

    def test_heldout_features_noise_and_labels_cannot_affect_fit(self):
        rng = np.random.default_rng(88); n = 52
        x = np.zeros((n, DIM)); x[:, :4] = rng.normal(size=(n, 4))
        noisy = x + rng.normal(0, .01, x.shape)
        f = pd.DataFrame(dict(similarity_group=np.arange(n) // 8))
        fit_ids = np.arange(40)
        pairs = [dict(i=i, j=j, weight=1.) for i in fit_ids for j in fit_ids if i < j and i//8 != j//8]
        labels = {(p['i'], p['j']): float(np.square(x[p['i'], :2]-x[p['j'], :2]).sum()) for p in pairs}
        params = dict(ridge=.01, smooth=.01, noise=.1)
        a = fit_metric(x, noisy, f, fit_ids, pairs, labels, params)
        x[40:] += 1000; noisy[40:] -= 9000
        labels[(40, 45)] = 1e12
        b = fit_metric(x, noisy, f, fit_ids, pairs, labels, params)
        np.testing.assert_array_equal(a['weights'], b['weights'])
        np.testing.assert_array_equal(a['noise_energy'], b['noise_energy'])


if __name__ == '__main__': unittest.main()
