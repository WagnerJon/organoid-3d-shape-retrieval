import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import tifffile
from segmentation_explorer import fingerprint
from panseg_explorer import DEFAULTS, normalized_settings, prediction_key, cached_boundary, segment_boundaries

class PansegWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.source=dict(fingerprint='raw-hash',spacing=[1.,.2,.2],nativeShape=[16,16,16])

    def test_segmentation_changes_reuse_prediction_but_inference_changes_do_not(self):
        key=prediction_key(self.source,{},fingerprint)
        for changes in [dict(threshold=.7,sigmaSeeds=3,clustering='gasp',beta=.3),dict(stacked=True,foreground=.2,postMinSize=400)]:
            self.assertEqual(key,prediction_key(self.source,changes,fingerprint))
        for changes in [dict(model='another-model'),dict(rescaleZ=.5),dict(gaussianSigma=1),dict(boundaryChannel=1),dict(haloMode='manual'),dict(device='cpu')]:
            self.assertNotEqual(key,prediction_key(self.source,changes,fingerprint))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'model.yaml';p.write_text('model: first')
            s=dict(modelSource='custom',configPath=str(p))
            before=prediction_key(self.source,s,fingerprint);p.write_text('model: modified')
            self.assertNotEqual(before,prediction_key(self.source,s,fingerprint))

    def test_reuse_requires_source_settings_grid_and_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);boundary=np.ones((16,16,16),dtype=np.float32)*.3
            tifffile.imwrite(p/'boundaries.tif',boundary,photometric='minisblack')
            result=dict(source='raw-hash',spacing=self.source['spacing'],folder=str(p),prediction=dict(
                key=prediction_key(self.source,{},fingerprint),hash=fingerprint(p/'boundaries.tif'),shape=[16,16,16],spacing=self.source['spacing']))
            (p/'result.json').write_text(json.dumps(result))
            loaded,_=cached_boundary(p/'result.json',self.source,dict(beta=.8),fingerprint)
            np.testing.assert_array_equal(loaded,boundary)
            with self.assertRaises(ValueError):cached_boundary(p/'result.json',self.source,dict(gaussianSigma=.2),fingerprint)
            with self.assertRaises(ValueError):cached_boundary(p/'result.json',dict(self.source,fingerprint='other'),{},fingerprint)
            result['prediction']['shape']=[8,16,16];(p/'result.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):cached_boundary(p/'result.json',self.source,{},fingerprint)
            result['prediction']['shape']=[16,16,16];(p/'result.json').write_text(json.dumps(result))
            tifffile.imwrite(p/'boundaries.tif',boundary+.1,photometric='minisblack')
            with self.assertRaises(ValueError):cached_boundary(p/'result.json',self.source,{},fingerprint)

    def test_controls_reach_upstream_and_results_return_to_native_grid(self):
        boundary=np.ones((8,8,8),dtype=np.float32)*.2
        for mode in ['watershed','multicut','gasp','mutex_ws']:
            ws=Mock(return_value=np.ones(boundary.shape,dtype=np.uint32))
            cluster=Mock(return_value=np.zeros(boundary.shape,dtype=np.uint32))
            api=dict(watershed=ws,multicut=cluster,gasp=cluster,mutex_ws=cluster)
            settings=dict(clustering=mode,threshold=.6,minSize=42,sigmaSeeds=1.4,sigmaWeights=2.6,alpha=.3,
                          stacked=True,nonmax=True,blockwise=True,threads=2,beta=.7,postMinSize=123,linkage='sum')
            labels=segment_boundaries(boundary,self.source,settings,[2,.4,.4],Mock(),api=api)
            self.assertEqual(labels.shape,(16,16,16))
            self.assertTrue((labels==1).all()) # graph cell zero is preserved as instance one
            kwargs=ws.call_args.kwargs
            self.assertEqual(kwargs['sigma_seeds'],1.4);self.assertEqual(kwargs['min_size'],42)
            self.assertEqual(kwargs['pixel_pitch'],(1.,1.));self.assertTrue(kwargs['stacked']);self.assertTrue(kwargs['apply_nonmax_suppression'])
            if mode=='watershed':cluster.assert_not_called()
            else:
                self.assertEqual(cluster.call_args.kwargs['beta'],.7)
                self.assertEqual(cluster.call_args.kwargs['post_minsize'],123)
                if mode=='gasp':self.assertEqual(cluster.call_args.kwargs['gasp_linkage_criteria'],'sum')

    def test_calibration_dimensions_follow_watershed_mode(self):
        boundary=np.zeros((8,8,8),dtype=np.float32)
        ws=Mock(return_value=np.ones(boundary.shape,dtype=np.uint32))
        api=dict(watershed=ws)
        for stacked, expected in [(False,(5.,1.,2.)),(True,(1.,2.))]:
            segment_boundaries(boundary,self.source,dict(clustering='watershed',stacked=stacked),[2.,.4,.8],Mock(),api=api)
            self.assertEqual(ws.call_args.kwargs['pixel_pitch'],expected)
        segment_boundaries(boundary,self.source,dict(clustering='watershed',anisotropy=False),[2.,.4,.8],Mock(),api=api)
        self.assertIsNone(ws.call_args.kwargs['pixel_pitch'])

    def test_invalid_settings_and_old_run_migration(self):
        for settings in [dict(rescaleX=float('nan')),dict(sigmaSeeds=-1),dict(minSize=1.5),dict(threads=0),dict(beta=1),dict(threshold=0),dict(clustering='invented'),dict(gpu='true'),dict(modelSource='bioimageio',modelID='')]:
            with self.assertRaises(ValueError):normalized_settings(settings)
        self.assertEqual(normalized_settings(dict(multicut=False))['clustering'],'watershed')
        self.assertEqual(normalized_settings(dict(multicut=True,minSize=77))['postMinSize'],77)

if __name__=='__main__':unittest.main()
