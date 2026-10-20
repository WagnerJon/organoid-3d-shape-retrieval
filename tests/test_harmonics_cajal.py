import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import trimesh
from scipy.spatial.distance import cdist
from harmonics_cajal_core import harmonics,surface_points,pairwise_subset,angular_basis

class HarmonicTests(unittest.TestCase):
    def test_angular_quadrature(self):
        d,t,bands=angular_basis()
        self.assertEqual(d.shape,(2048,3));self.assertEqual(t.shape,(2048,169))
        np.testing.assert_allclose(np.linalg.norm(d,axis=1),1,atol=1e-14)
        self.assertEqual(len(bands),13)
        # Constant spherical function has no energy outside degree zero.
        c=np.ones(2048)@t
        self.assertAlmostEqual(abs(c[0]),np.sqrt(4*np.pi),places=12)
        np.testing.assert_allclose(c[1:],0,atol=1e-13)

    def test_scale_translation_and_rotation(self):
        mesh=trimesh.creation.icosphere(subdivisions=2);v=mesh.vertices*np.array([1,.8,.6])
        q,_=np.linalg.qr(np.random.default_rng(3).normal(size=(3,3)))
        a=harmonics(v,mesh.faces);b=harmonics(2*v@q+[4,2,1],mesh.faces)
        self.assertEqual(a.shape,(416,));self.assertLess(np.linalg.norm(a-b),.015)
        points=surface_points(v,mesh.faces,128,9);other=surface_points(2*v@q+[4,2,1],mesh.faces,128,9)
        np.testing.assert_allclose(cdist(points,points),cdist(other,other),atol=2e-13)

    def test_pairwise_subset_keeps_global_ids(self):
        d=np.array([[0,1,2],[1,0,3],[2,3,0.]])
        x=np.c_[np.arange(3),d]
        np.testing.assert_equal(pairwise_subset(x[[2,0]]),[[0,2],[2,0]])
        with self.assertRaises(ValueError):pairwise_subset([[4,0,1],[0,1,0]])

if __name__=='__main__':unittest.main()

class PairwiseMembershipTests(unittest.TestCase):
    def test_rebuild_does_not_average_pairwise_rows(self):
        from organoid_membership import rebuild
        d=np.array([[0.,1,2],[1,0,3],[2,3,0]])
        cells=[dict(id=str(i),sample='S',qc=True,inMainOrganoid=i!=1,metrics={'volume_um3':float(i+1)},
                    representations={'traditional':dict(vector=[0.],point=[0,0],cluster=0),
                                     'gw':dict(vector=np.r_[i,d[i]].tolist(),point=[999,999],cluster=0)}) for i in range(3)]
        data=dict(cells=cells,organoids=[dict(id='S',sample='S')],traditionalFeatures=['volume_um3'],
                  methods=[dict(id='traditional',metric='euclidean'),dict(id='gw',metric='precomputed')],provenance='Test')
        rebuild(data)
        self.assertNotIn('gw',data['organoids'][0]['representations'])
        self.assertEqual(cells[1]['representations']['gw']['point'],[999,999])
        self.assertEqual(data['organoids'][0]['metrics']['volume_um3'],4.)

class OfficialCajalTests(unittest.TestCase):
    def test_rigid_transform_preserves_comparison(self):
        try:from cajal.qgw import quantized_icdm,quantized_gw
        except ImportError:self.skipTest('Official CAJAL environment required')
        mesh=trimesh.creation.icosphere(subdivisions=2);p=surface_points(mesh.vertices*np.array([1,.8,.6]),mesh.faces,128,8)
        q=surface_points(mesh.vertices*np.array([1,.7,.5]),mesh.faces,128,9)
        rotation,_=np.linalg.qr(np.random.default_rng(7).normal(size=(3,3)));u=np.full(128,1/128)
        make=lambda x:quantized_icdm(cdist(x,x),u,32)
        a=make(p);b=make(q);c=make(p@rotation+[2,3,1])
        _,before=quantized_gw(a,b);_,after=quantized_gw(c,b)
        self.assertAlmostEqual(before,after,places=9)
