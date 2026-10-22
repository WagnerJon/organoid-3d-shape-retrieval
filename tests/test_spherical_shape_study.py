import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from spherical_shape_study_core import (fib_basis,basis,coefficients,power,bispectrum,random_radial,map_sphere,balance_sphere,ray_samples)

class SphericalStudyChecks(unittest.TestCase):
    def test_rotations_preserve_invariants_not_raw_coefficients(self):
        d,Y=fib_basis(8192,8);a=random_radial(71,8)
        for seed in [1,2,3]:
            R=Rotation.random(random_state=seed).as_matrix()
            b=coefficients((basis(d@R,8)@a).real,Y)
            self.assertGreater(np.linalg.norm(a-b),.1)
            self.assertLess(np.linalg.norm(power(a,8)-power(b,8)),1e-4)
            self.assertLess(np.linalg.norm(bispectrum(a)-bispectrum(b)),1e-6)
    def test_same_power_does_not_imply_same_bispectrum(self):
        a,b=random_radial(1),random_radial(2)
        for l in range(1,9):
            s=slice(l*l,(l+1)**2);b[s]*=np.linalg.norm(a[s])/max(np.linalg.norm(b[s]),1e-15)
        np.testing.assert_allclose(power(a,8),power(b,8),atol=1e-14)
        self.assertGreater(np.linalg.norm(bispectrum(a)-bispectrum(b)),1e-6)
    def test_mapper_rejects_torus_without_repair(self):
        m=trimesh.creation.torus(major_radius=1,minor_radius=.2)
        v=m.vertices.copy();mapped,q=map_sphere(v,m.faces)
        self.assertIsNone(mapped);self.assertFalse(q['valid'])
        np.testing.assert_array_equal(v,m.vertices)
    def test_mapping_and_attribute_interpolation_on_sphere(self):
        m=trimesh.creation.icosphere(subdivisions=2);original=m.vertices.copy()
        mapped,q=map_sphere(m.vertices,m.faces);self.assertTrue(q['valid'])
        mapped,b=balance_sphere(mapped,m.vertices,m.faces);self.assertTrue(b['valid'])
        d,Y=fib_basis(1024,4);r,xyz,q=ray_samples(mapped,m.faces,d,attribute=m.vertices)
        self.assertEqual(q['coverage'],1.)
        self.assertLess(np.max(np.linalg.norm(xyz-d,axis=1)),.03)
        np.testing.assert_array_equal(m.vertices,original)
if __name__=='__main__':unittest.main()
