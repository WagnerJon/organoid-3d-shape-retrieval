import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import atlas_cellpose_models as models
from atlas_cellpose_3d import predict


class ClassicCellposeTests(unittest.TestCase):
    def test_classic_ids_use_correct_legacy_model_and_weight_filename(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(models.os.environ,{'CELLPOSE_LOCAL_MODELS_PATH':tmp}):
            seen=[]
            for key in models.LEGACY_MODELS:models.weights_path(key).write_bytes(key.encode())
            def factory(**kwargs):
                seen.append(kwargs)
                return SimpleNamespace(pretrained_model=str(models.weights_path(kwargs['model_type'])),device='cpu')
            with patch.object(models,'legacy_backend',return_value=SimpleNamespace(CellposeModel=factory)):
                for key in models.LEGACY_MODELS:
                    model=models.make_model(pretrained_model=key,gpu=False)
                    self.assertEqual(model.atlas_runtime['version'],'3.1.1.3')
                    self.assertEqual(model.atlas_runtime['checkpointSHA256'],models.digest(models.weights_path(key)))
            self.assertEqual([s['model_type'] for s in seen],list(models.LEGACY_MODELS))
            self.assertEqual(models.weights_path('cyto3').name,'cyto3')
            with patch.object(models,'legacy_backend',return_value=SimpleNamespace(CellposeModel=lambda **kwargs:SimpleNamespace(pretrained_model='/wrong/cyto3'))),self.assertRaisesRegex(ValueError,'unexpected weights'):
                models.make_model(pretrained_model='cyto3')
            with self.assertRaisesRegex(ValueError,'Unsupported'):models.make_model(pretrained_model='unknown')

    def test_legacy_orthoview_axes_and_gray_channels(self):
        volume=np.arange(5*6*7,dtype=np.float32).reshape(5,6,7)
        class Model:
            def eval(self,images,**kwargs):
                self_test.assertEqual(kwargs['channels'],[0,0]);self_test.assertFalse(kwargs['compute_masks'])
                image=images[0]
                return None,[[None,np.stack([image+1,image+2]),image+3]],np.zeros((1,256))
        self_test=self
        with patch.object(models,'make_model',return_value=Model()):
            for view in ('xy','xz','yz'):
                _,prob,flow,_=models.legacy_view(volume,view,'cyto3',False,30)
                np.testing.assert_array_equal(prob,volume+3)
                np.testing.assert_array_equal(flow,np.stack([volume+1,volume+2]))

    def test_native_three_dimensional_model_uses_router(self):
        volume=np.ones((5,6,7),np.float32)
        class Model:
            device='cpu';atlas_runtime={'engine':'Cellpose3 CNN','version':'3.1.1.3'}
            def eval(self,value,**kwargs):
                self_test.assertTrue(kwargs['do_3D']);self_test.assertEqual(kwargs['anisotropy'],1.)
                return np.ones(value.shape[:3],np.uint16),[],None
        self_test=self
        with patch.object(models,'make_model',return_value=Model()) as factory:
            labels,metadata=predict(volume,dict(model='cyto3',gpu=False,diameter=30))
        factory.assert_called_once_with(pretrained_model='cyto3',gpu=False)
        self.assertEqual(labels.shape,volume.shape)
        self.assertEqual(metadata['modelRuntime']['engine'],'Cellpose3 CNN')


if __name__=='__main__':unittest.main()
