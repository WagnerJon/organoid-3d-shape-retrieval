import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from spectral64_core import *
from train_spectral64 import Model

class Spectral64Tests(unittest.TestCase):
    def test_packing_preserves_real_fields_and_energy(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(32,169));c=unpack(x)
        np.testing.assert_allclose(pack(c),x,atol=1e-14)
        np.testing.assert_allclose(np.square(x).sum(),np.square(abs(c)).sum(),rtol=1e-14)
        d,_,_=angular_basis();Y=basis(d,12)
        signal=c@Y.T
        self.assertLess(abs(signal.imag).max(),1e-12)
    def test_power_discards_mirror_information(self):
        v,f=synthetic_mesh(44);a,_=canonical(v,f)
        vv=v.copy();vv[:,0]*=-1;ff=f[:,[0,2,1]];b,_=canonical(vv,ff)
        ca,cb=field_coefficients(a,f),field_coefficients(b,ff)
        self.assertGreater(np.linalg.norm(ca-cb),.01)
        self.assertLess(np.linalg.norm(powers(ca)-powers(cb)),.001)
    def test_known_pose_decoder_action(self):
        from rotate_spectral64 import action,real_basis
        from scipy.spatial.transform import Rotation
        R=Rotation.random(random_state=42).as_matrix();D=action(R)
        d,T,_=angular_basis();weight=T[:,0].real*np.sqrt(4*np.pi)
        x=np.random.default_rng(12).normal(size=169)
        projected=(real_basis(d).T*weight)@(real_basis(d@R)@x)
        np.testing.assert_allclose(projected,D@x,atol=2e-6)
        np.testing.assert_allclose(D@D.T,np.eye(169),atol=2e-6)
    def test_bottleneck_and_no_cross_split_organoids(self):
        model=Model();self.assertEqual(model.mean.out_features,64)
        import pandas as pd
        f=pd.read_csv(OUT/'manifest.csv')
        self.assertEqual(f.groupby('sample').split.nunique().max(),1)
        self.assertEqual(f.groupby('similarity_group').split.nunique().max(),1)
        self.assertEqual((f.split=='test').sum(),135)
if __name__=='__main__':unittest.main()
