"""Memory rejection before pixel reads, global interpolation and normalization parity."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tifffile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_standalone_cellpose as pipeline
from atlas_preprocessing import memory_plan,check_memory,positive_median,resize_tiff,preprocess

class PreprocessingTests(unittest.TestCase):
    def test_reported_big_stack_is_rejected_before_read_or_output(self):
        shape=[578,1920,1920];spacing=[.75,.22872330084318845,.22872330084318845]
        plan=memory_plan(shape,spacing,.85,16*1024**3)
        self.assertEqual(plan['processedShape'],[1611,1632,1632]);self.assertFalse(plan['fits'])
        self.assertAlmostEqual(plan['singleArrayGiB'],15.9843864441)
        self.assertEqual(plan['suggestedScale'],.17)
        self.assertTrue(memory_plan(shape,spacing,.17,16*1024**3)['fits'])
        with self.assertRaisesRegex(ValueError,'Try processing scale 0.17'):
            check_memory(shape,spacing,.85,16*1024**3)
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            args=SimpleNamespace(input=Path(tmp)/'raw.tif',output=output,scale=.85,diameter=30,model='cpsam_v2',cpu=False,no_background=False,spacing=spacing)
            with patch.object(pipeline,'calibration',return_value=dict(nativeShape=shape,spacingZYXum=spacing,dtype='uint16')),patch.object(pipeline,'source_stat',return_value=[1,2]),patch.object(pipeline,'digest',side_effect=AssertionError('Must not read image')):
                with self.assertRaisesRegex(ValueError,'one float32 image alone'):pipeline.setup(args)
            self.assertFalse(output.exists())

    def test_streamed_global_resize_matches_torch_and_exact_positive_median(self):
        from segment3D import gpu,filters
        rng=np.random.default_rng(903)
        for dtype in ['uint8','uint16']:
            for zoom in [[1.3,.73,.81],[.65,1.2,.9]]:
                raw=rng.integers(0,100,(17,21,23),dtype=dtype);raw[raw<20]=0
                with tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);source=root/'raw.tif';tifffile.imwrite(source,raw,compression='zlib',photometric='minisblack')
                    with tifffile.TiffFile(source) as tif:
                        fill=positive_median(tif)
                        self.assertEqual(fill,int(np.median(raw[raw>0])))
                        actual=resize_tiff(tif,zoom,root/'resized.npy',fill)
                    expected=gpu.zoom_3d_pytorch(filters.fill_array(raw,method='median'),zoom)
                    np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=2e-6)

    def test_background_normalization_and_slab_median_match_reference(self):
        from segment3D import parameters,usegment3d
        from scipy import ndimage
        rng=np.random.default_rng(19);raw=rng.integers(0,4096,(32,41,43),dtype=np.uint16);raw[raw<500]=0
        for background in [False,True]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=root/'raw.tif';tifffile.imwrite(source,raw,photometric='minisblack')
                params=parameters.get_preprocess_params();params.update(factor=.82,voxel_res=[1.4,1.,1.],bg_ds=8,do_bg_correction=background)
                # Plane-streaming must never call tifffile's whole-stack read.
                with patch('tifffile.imread',side_effect=AssertionError('Must stream TIFF planes')):
                    details=preprocess(source,root,params,progress=None)
                actual=np.load(root/'processed.npy')
                expected=ndimage.median_filter(np.squeeze(usegment3d.preprocess_imgs(raw,params)),size=3)
                np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=3e-5)
                self.assertEqual(details['processedShape'],list(expected.shape))
                self.assertFalse((root/'preprocess-resized.tmp.npy').exists())
                self.assertFalse((root/'processed.tmp.npy').exists())

    def test_all_zero_stack_has_valid_streaming_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'zeros.tif';tifffile.imwrite(source,np.zeros((8,12,14),dtype='uint16'),photometric='minisblack')
            with tifffile.TiffFile(source) as tif:self.assertEqual(positive_median(tif),0)

if __name__=='__main__':unittest.main()
