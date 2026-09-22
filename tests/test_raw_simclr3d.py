import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_raw_simclr3d import raw_crop
from simclr3d_core import augment_raw
class RawInputTests(unittest.TestCase):
    def test_coordinates_and_continuous_intensity(self):
        z,y,x=np.indices((25,25,25));im=(z+2*y+3*x).astype(np.float32)/150
        crop,out=raw_crop(im,[12,12,12],[1,1,1],16,16)
        self.assertEqual(crop.shape,(16,16,16));self.assertEqual(out,0)
        self.assertAlmostEqual(float(crop.mean()),72/150,places=5)
        self.assertGreater(len(np.unique(crop)),2)
    def test_boundary_recorded(self):
        crop,out=raw_crop(np.ones((20,20,20)),[0,0,0],[1,1,1],16,16)
        self.assertGreater(out,.5);self.assertTrue(np.isfinite(crop).all())
    def test_raw_augmentation_not_binarized(self):
        torch.set_num_threads(2);x=torch.linspace(0,1,32**3).reshape(1,1,32,32,32);g=torch.Generator().manual_seed(3)
        cfg=dict(augmentation_scale=[.95,1.05],augmentation_translation=.04,intensity_gain=[.8,1.2],intensity_gamma=[.8,1.2],intensity_noise_std=.02)
        a=augment_raw(x,g,cfg);b=augment_raw(x,g,cfg)
        self.assertGreater(a.unique().numel(),100);self.assertFalse(torch.equal(a,b));self.assertTrue(torch.isfinite(a).all())
