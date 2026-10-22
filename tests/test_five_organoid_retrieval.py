import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from five_organoid_retrieval_core import *
from scipy.spatial.transform import Rotation
from run_five_organoid_retrieval import query_distance


class RetrievalTests(unittest.TestCase):
    def test_fixed_dimension_zeros_and_shape_discrimination(self):
        sphere=trimesh.creation.icosphere(subdivisions=2)
        v=sphere.vertices.copy();f=sphere.faces
        a=zernike(v,f); b=zernike(v*np.array([1.8,1.,.7]),f)
        self.assertEqual(a.shape,(121,));self.assertTrue(np.isfinite(a).all())
        self.assertAlmostEqual(np.linalg.norm(a),1.,places=12)
        self.assertGreater(np.linalg.norm(a-b),.05)

    def test_pose_scale_and_translation_stability(self):
        sphere=trimesh.creation.icosphere(subdivisions=2)
        v=sphere.vertices*np.array([1.8,1.,.7]);f=sphere.faces
        a=zernike(v,f);r=Rotation.random(random_state=42).as_matrix()
        b=zernike((v@r.T)*13+np.array([30.,-21.,18.]),f)
        self.assertLess(np.linalg.norm(a-b),.015)

    def test_hybrid_query_matches_existing_atlas_metric(self):
        from atlas_shape_space import distances
        rng=np.random.default_rng(21);x=rng.normal(size=(8,476));x[:,0]=[0,0,1,0,1,1,0,0]
        actual=query_distance('spatial_robust',x[0],x[1:])
        reference=distances(x,'shape_hybrid95')[0,1:]
        np.testing.assert_allclose(actual,reference,atol=1e-12)

    def test_queries_and_probes_preserve_qc_and_cross_organoid_boundaries(self):
        f=pd.read_csv(OUT/'manifest.csv');p=json.loads((OUT/'plan.json').read_text())
        self.assertEqual(int(f.included.sum()),747)
        self.assertEqual(len(p['queries']),100);self.assertEqual(len(set(p['queries'])),100)
        self.assertTrue(f.iloc[p['queries']].included.all())
        self.assertTrue(f.iloc[p['probes']].included.all())
        for i,refs in p['random_reference_pairs'].items():
            self.assertTrue(f.iloc[refs].included.all())
            self.assertFalse((f.iloc[refs]['sample']==f.iloc[int(i)]['sample']).any())


if __name__=='__main__':unittest.main()
