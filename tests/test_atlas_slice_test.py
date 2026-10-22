import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_atlas_dataset import inspect
from atlas_slice_test import read_slice, preprocess, run


class SliceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='atlas-slice-test-')
        self.root = Path(self.temp.name)
        self.raw = np.arange(12*40*48, dtype=np.uint16).reshape(12,40,48)
        self.path = self.root/'raw.ome.tif'
        tifffile.imwrite(self.path, self.raw, compression='zlib', ome=True, photometric='minisblack',
                         metadata=dict(axes='ZYX',PhysicalSizeZ=2.,PhysicalSizeY=.5,PhysicalSizeX=.5))
        self.entry = inspect(self.path)
        self.settings = dict(model='cpdino-vitb',scale=.5,diameter=21.,gpu=False,background=True,flowThreshold=.75)
        self.request = dict(input=self.entry,plane='xy',index=6,settings=self.settings)

    def tearDown(self): self.temp.cleanup()

    def test_reads_exact_orthogonal_planes_without_full_stack_read(self):
        with patch('tifffile.imread', side_effect=AssertionError('Full volume read forbidden')):
            for plane, axis in [('xy',0),('xz',1),('yz',2)]:
                actual, _ = read_slice(self.entry,plane,3)
                np.testing.assert_array_equal(actual,np.take(self.raw,3,axis=axis))
        with self.assertRaisesRegex(ValueError,'outside'):
            read_slice(self.entry,'xy',12)
        entry = copy.deepcopy(self.entry); entry['sourceStat'][0]+=1
        with self.assertRaisesRegex(ValueError,'changed'):
            read_slice(entry,'xy',6)

    def test_preview_never_loads_cellpose_or_writes_dataset(self):
        output = self.root/'raw-preview'
        with patch.dict('sys.modules', {'cellpose.models':None}):
            result = run(self.request,output)
        self.assertIsNone(result['cells'])
        self.assertTrue((output/'raw.png').exists())
        self.assertFalse((output/'labels.tif').exists())
        self.assertFalse(list(self.root.rglob('dataset.json')))

    def test_settings_dispatch_masks_overlay_and_source_preservation(self):
        before = self.path.read_bytes()
        observed = {}
        class Model:
            device = 'cpu'
            def __init__(self, **kwargs): observed.update(kwargs)
            def eval(self,image,**kwargs):
                observed.update(kwargs); observed['shape']=image.shape
                masks = np.zeros(image.shape,np.uint16)
                masks[2:6,2:6]=1; masks[8:13,8:13]=2
                return masks, [], None
        output = self.root/'test'
        result = run(self.request,output,segment=True,model_factory=Model)
        self.assertEqual(observed['pretrained_model'],'cpdino-vitb')
        self.assertFalse(observed['gpu']);self.assertEqual(observed['diameter'],21.)
        self.assertEqual(observed['flow_threshold'],.75)
        self.assertFalse(observed['do_3D']); self.assertTrue(observed['compute_masks'])
        self.assertEqual(observed['shape'],(20,24))
        self.assertEqual(result['cells'],2)
        self.assertEqual(result['settings'],self.settings)
        masks = tifffile.imread(output/'labels.tif')
        self.assertEqual(masks.shape,(40,48));self.assertEqual(set(np.unique(masks)),{0,1,2})
        with Image.open(output/'labels.png') as image:
            alpha = np.asarray(image)[:,:,3]
            self.assertTrue((alpha==0).any());self.assertTrue((alpha>0).any())
        self.assertEqual(self.path.read_bytes(),before)
        self.assertIn('no cross-slice',result['note'])
        with self.assertRaises(FileExistsError): run(self.request,output)

    def test_flow_threshold_default_zero_and_invalid_values(self):
        observed = []
        class Model:
            device = 'cpu'
            def __init__(self, **kwargs): pass
            def eval(self, image, **kwargs):
                observed.append(kwargs['flow_threshold'])
                return np.zeros(image.shape, np.uint16), [], None
        legacy = copy.deepcopy(self.request); del legacy['settings']['flowThreshold']
        run(legacy, self.root/'legacy', segment=True, model_factory=Model)
        disabled = copy.deepcopy(self.request); disabled['settings']['flowThreshold'] = 0.
        run(disabled, self.root/'disabled', segment=True, model_factory=Model)
        self.assertEqual(observed, [.4, 0.])
        for i, value in enumerate([-1., float('nan'), float('inf')]):
            invalid = copy.deepcopy(self.request); invalid['settings']['flowThreshold'] = value
            with self.assertRaisesRegex(ValueError, 'Flow threshold'):
                run(invalid, self.root/f'invalid-{i}', segment=True, model_factory=Model)
        self.assertEqual(observed, [.4, 0.])

    def test_anisotropy_invalid_spacing_and_pixel_limits(self):
        image, dimensions = read_slice(self.entry,'xz',10)
        self.assertEqual(preprocess(image,self.entry['spacing'],dimensions,self.settings).shape,(24,24))
        with self.assertRaisesRegex(ValueError,'spacing'):
            preprocess(image,[0,1,1],dimensions,self.settings)
        with self.assertRaisesRegex(ValueError,'too large'):
            preprocess(image,[1e6,1,1],dimensions,self.settings)
        settings = dict(self.settings,scale=0)
        with self.assertRaisesRegex(ValueError,'Scale'):
            preprocess(image,self.entry['spacing'],dimensions,settings)

if __name__=='__main__': unittest.main()
