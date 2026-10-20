import json
from unittest.mock import patch
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import tifffile
from segmentation_explorer import agreement, prepare, publish, compare, verify_source, PansegProgress

class SegmentationTests(unittest.TestCase):
    def test_panseg_progress_counts_completed_patches(self):
        with patch('segmentation_explorer.progress') as report:
            tracker=PansegProgress();tracker.total=4
            tracker.progress += 1
            self.assertAlmostEqual(report.call_args.args[1],.1)
            tracker.progress += 1
            self.assertAlmostEqual(report.call_args.args[1],.25)
            self.assertIn('1/4 patches',report.call_args.args[0])
            tracker.progress += 1
            tracker.progress += 1
            self.assertAlmostEqual(report.call_args.args[1],.55)

    def test_permuted_sparse_ids(self):
        a=np.zeros((4,4,4),dtype=np.uint32);a[:2]=900000;a[2:]=15
        b=np.where(a==15,6000000,7).astype(np.uint32)
        result=agreement(a,b)
        self.assertEqual(result['matched'],2)
        self.assertEqual(result['dice'],1)
        self.assertEqual(result['meanIoU'],1)
        self.assertEqual({(p['a'],p['b']) for p in result['pairs']},{(15,6000000),(900000,7)})

    def test_split_not_perfect_instance_agreement(self):
        a=np.ones((4,4,4),dtype=np.uint32);b=a.copy();b[2:]=2
        result=agreement(a,b)
        self.assertEqual(result['dice'],1)
        self.assertEqual(result['matched'],1)
        self.assertEqual(result['meanIoU'],.5)

    def test_empty_and_disjoint(self):
        a=np.zeros((3,3,3),dtype=np.uint32)
        self.assertEqual(agreement(a,a)['dice'],1)
        self.assertEqual(agreement(a,a)['matched'],0)
        b=a.copy();b[0]=1;c=a.copy();c[2]=4
        self.assertEqual(agreement(b,c)['dice'],0)
        self.assertEqual(agreement(b,c)['matched'],0)

    def test_reject_rgb_and_2d(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            tifffile.imwrite(p/'color.tif',np.ones((32,32,3),dtype=np.uint8),photometric='rgb')
            with self.assertRaises(ValueError):prepare(p/'color.tif',[1.,1.,1.],p/'preview')
            tifffile.imwrite(p/'flat.tif',np.ones((32,32),dtype=np.uint8))
            with self.assertRaises(ValueError):prepare(p/'flat.tif',[1.,1.,1.],p/'preview')

    def test_grid_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);raw=np.arange(64,dtype=np.uint16).reshape(4,4,4)
            tifffile.imwrite(p/'raw.tif',raw,photometric='minisblack')
            source=prepare(p/'raw.tif',[1.,.2,.2],p/'source')
            a=np.zeros_like(raw,dtype=np.uint32);a[1:3,1:3,1:3]=999999
            b=np.where(a>0,4,0).astype(np.uint32)
            for name in ['a','b']:(p/name).mkdir()
            ar=publish(a,source,p/'a','cellpose',{},1.,'test')
            br=publish(b,source,p/'b','panseg',{},1.,'test')
            self.assertEqual(ar['cells'][0]['label'],999999)
            self.assertAlmostEqual(ar['cells'][0]['volume'],.32)
            self.assertEqual(len((p/'a/labels.u32').read_bytes()),4*4*4*4)
            self.assertEqual(compare(ar,br,source)['matched'],1)
            wrong=dict(br,source='different image')
            with self.assertRaises(ValueError):compare(ar,wrong,source)
            wrong=dict(br,spacing=[2.,.2,.2])
            with self.assertRaises(ValueError):compare(ar,wrong,source)
            tifffile.imwrite(p/'b/labels_native.tif',a,photometric='minisblack')
            with self.assertRaises(ValueError):compare(ar,br,source)
            tifffile.imwrite(p/'raw.tif',raw+1,photometric='minisblack')
            with self.assertRaises(ValueError):verify_source(source)
            with self.assertRaises(ValueError):prepare(p/'raw.tif',[0,.2,.2],p/'bad')

if __name__=='__main__':unittest.main()
