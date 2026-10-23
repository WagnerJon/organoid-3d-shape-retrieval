import sys, unittest
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi, eval_jacobi
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from zernike_fractional_core import radial, indices, quadrature, field_descriptors, mesh_descriptors


class FractionalZernikeTests(unittest.TestCase):
    def test_radial_orthogonality_and_classical_limit(self):
        for alpha in (1.,.38,.20,.14):
            for l in (0,1,6,20):
                beta=l+3/(2*alpha)-1
                x,w=roots_jacobi(32,0,beta);t=(x+1)/2
                # Factor out r^(alpha*l) analytically; exact polynomial integral.
                basis=np.array([np.sqrt((2*alpha*(l+2*k)+3)/3)*eval_jacobi(k,0,beta,x) for k in range(11)])
                gram=(basis*(w*3/(2*alpha)/2**(beta+1)))@basis.T
                np.testing.assert_allclose(gram,np.eye(11),atol=2e-12)
        r=np.linspace(0,1,101)
        np.testing.assert_allclose(radial(2,0,1.,r), np.sqrt(7/3)*(2.5*r*r-1.5),atol=1e-14)
        np.testing.assert_allclose(radial(4,0,1.,r), np.sqrt(11/3)*(1.875-8.75*r*r+7.875*r**4),atol=1e-14)
        self.assertEqual(len(indices(20)),121)

    def test_agrees_with_paper_section_four_recurrence(self):
        r=np.linspace(.05,.95,23)
        for a in (1.,.38,.20,.14):
            for l in (0,1,4):
                start=np.sqrt((2*a*l+3)/3)*r**(a*l)
                for k in range(5):
                    if k:
                        start=-np.sqrt((2*a*l+4*a*k+3)/(2*a*l+4*a*(k-1)+3))*(2*a*l+2*a*(k-1)+3)/(2*a*k)*start
                    term=start.copy();total=term.copy()
                    for nu in range(1,k+1):
                        term*=-(k-nu+1)*(2*a*l+2*a*(nu+k-1)+3)/(nu*(2*a*l+2*a*(nu-1)+3))*r**(2*a)
                        total+=term
                    np.testing.assert_allclose(radial(l+2*k,l,a,r),total,atol=2e-10)

    def test_constant_ball_and_invalid_input(self):
        r,_,_,angular=quadrature(6,48,16,32)
        codes=field_descriptors({'ball':np.ones((len(r),len(angular)))},(1.,),6,(48,16,32))
        v=codes['ball_a1'];self.assertAlmostEqual(v[0],1.,places=12)
        self.assertLess(np.linalg.norm(v[1:]),1e-12)
        for a in (0.,-1.,np.nan):
            with self.assertRaises(ValueError):radial(2,0,a,np.array([.5]))

    def test_mesh_pose_scale_translation_and_shape_discrimination(self):
        import trimesh
        m=trimesh.creation.icosphere(subdivisions=2)
        v=m.vertices*np.array([1.8,1.,.7]);f=m.faces
        a,_=mesh_descriptors(v,f,alphas=(1.,.38))
        b,_=mesh_descriptors(v@Rotation.random(random_state=42).as_matrix().T*7+[3.,-8.,4.],f,alphas=(1.,.38))
        c,_=mesh_descriptors(m.vertices,f,alphas=(1.,.38))
        for key in a:
            self.assertEqual(a[key].shape,(121,))
            self.assertLess(np.linalg.norm(a[key]-b[key]),.04)
            self.assertGreater(np.linalg.norm(a[key]-c[key]),.08)

    def test_minmax_excludes_query_organoid_and_does_not_clip(self):
        import pandas as pd
        from run_fractional_zernike import scaled
        frame=pd.DataFrame(dict(included=[True,True,True,False],sample=['held','train','train','train']))
        x=np.array([[100.,1.],[2.,1.],[4.,1.],[1000.,1.]])
        z=scaled(x,'volume_a1_minmax',frame,'held')
        np.testing.assert_allclose(z,[[49,0],[0,0],[1,0],[499,0]])
        changed=x.copy();changed[0,0]=200
        np.testing.assert_allclose(scaled(changed,'volume_a1_minmax',frame,'held')[1:],z[1:])


if __name__=='__main__':unittest.main()
