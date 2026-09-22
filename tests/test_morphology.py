import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('morphology',Path(__file__).resolve().parents[1]/'scripts/analyze_morphology.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MorphologyTest(unittest.TestCase):
    def test_sphere_and_physical_scaling(self):
        z,y,x=np.indices((25,25,25))-12
        a=((x*x+y*y+z*z)<=64).astype(np.uint16)
        one=module.properties(a,[1,1,1])[0]
        two=module.properties(a,[2,2,2])[0]
        self.assertAlmostEqual(one['aspect_ratio'],1,places=10)
        self.assertAlmostEqual(two['volume_um3']/one['volume_um3'],8)
        self.assertAlmostEqual(two['surface_area_um2']/one['surface_area_um2'],4)
        self.assertAlmostEqual(two['equivalent_diameter_um']/one['equivalent_diameter_um'],2)
        self.assertAlmostEqual(two['sphericity'],one['sphericity'])
        self.assertEqual(one['components_26'],1)

    def test_anisotropic_axis_lengths(self):
        a=np.zeros((22,22,22),np.uint16);a[1:21,1:21,1:21]=1
        row=module.properties(a,[3,2,1])[0]
        self.assertAlmostEqual(row['elongation'],1.5,places=10)
        self.assertAlmostEqual(row['flatness'],2,places=10)
