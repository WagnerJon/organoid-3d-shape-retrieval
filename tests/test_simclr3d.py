import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_simclr3d import standardize
from simclr3d_core import augment,nt_xent,ResNet3DSimCLR

import unittest

class SimCLRTests(unittest.TestCase):
    def test_isotropic_physical_shape_and_binary(self):
        mask=np.zeros((12,24,24),np.uint8);mask[2:10,4:20,4:20]=1
        out,info=standardize(mask,[2,1,1]);assert out.shape==(64,64,64)
        assert set(np.unique(out))=={0,1}
        coords=np.argwhere(out);assert np.ptp(coords,axis=0).max()-np.ptp(coords,axis=0).min()<=1
        assert mask.sum()==8*16*16 and info['isotropic_shape_zyx']==[16,16,16]

    def test_contrastive_loss_and_gradients(self):
        z=torch.eye(8,requires_grad=True)
        good=nt_xent(z,z);bad=nt_xent(z,z.roll(1,0))
        assert good<bad
        good.backward();assert torch.isfinite(z.grad).all()
        collapsed=nt_xent(torch.ones(8,4),torch.ones(8,4))
        assert abs(float(collapsed)-np.log(15))<1e-5

    def test_independent_augmentations_preserve_support(self):
        x=torch.zeros(2,1,64,64,64);x[:,:,22:42,27:37,29:35]=1
        g=torch.Generator().manual_seed(11);a=augment(x,g);b=augment(x,g)
        assert not torch.equal(a,b) and a.sum()>0 and b.sum()>0
        assert a[:,:,:2].sum()==0 and a[:,:,-2:].sum()==0
        assert set(a.unique().tolist())=={0,1}

    def test_encoder_shape_and_backward(self):
        torch.set_num_threads(2);m=ResNet3DSimCLR();h,z=m(torch.randn(2,1,32,32,32))
        assert h.shape==(2,128) and z.shape==(2,64)
        nt_xent(z,z).backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
