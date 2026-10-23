import json
import logging
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from atlas_cellpose_3d import settings, stage_sequence, predict
from prepare_atlas_dataset import inspect, validate, prepare_stack
from run_standalone_cellpose import setup, worker, save


class Native3DTests(unittest.TestCase):
    def test_native_dispatch_preserves_grid_and_does_not_apply_anisotropy_twice(self):
        image=np.ones((8,20,24),np.float32);seen={};events=[]
        class Model:
            device='mps:0'
            def __init__(self,**kwargs):seen.update(kwargs)
            def eval(self,value,**kwargs):
                seen.update(kwargs);seen['shape']=value.shape
                for view in ['YX','ZY','ZX']:logging.getLogger('cellpose.core').info('running %s: 8 planes of size (20, 24)',view)
                logging.getLogger('cellpose.models').warning('MPS does not support 3D post-processing, using CPU')
                value[:]=0
                return np.ones(value.shape[:3],np.uint16),[],None
        controls=dict(segmentationMode='cellpose3d',model='cpdino-vitb',gpu=True,diameter=25,flow3DSmooth=1.5,cellprobThreshold=-1.,minSize3D=40,flowThreshold=.9)
        labels,info=predict(image,controls,lambda *args:events.append(args),Model)
        self.assertEqual(seen['shape'],(8,20,24,1));self.assertTrue(seen['do_3D']);self.assertEqual(seen['anisotropy'],1.)
        self.assertEqual(seen['pretrained_model'],'cpdino-vitb');self.assertEqual(seen['flow3D_smooth'],1.5)
        self.assertEqual(seen['min_size'],40);self.assertEqual(seen['cellprob_threshold'],-1.)
        self.assertNotIn('flow_threshold',seen)
        self.assertEqual(labels.shape,image.shape);self.assertEqual(labels.dtype,np.uint32)
        self.assertTrue(image.all());self.assertEqual(info['dynamicsDevice'],'cpu');self.assertEqual(events[-1][1],1.)
        self.assertEqual([fraction for _,fraction in events],[.05,.15,.2,.4,.6,.8,1.])

    def test_defaults_validation_and_stage_plan(self):
        self.assertEqual(settings({})['segmentationMode'],'usegment3d')
        self.assertEqual(len(stage_sequence({})),6)
        self.assertEqual(stage_sequence(dict(segmentationMode='cellpose3d')),['preprocess','aggregate','export'])
        for invalid in [dict(segmentationMode='unknown'),dict(flow3DSmooth=-1),dict(flow3DSmooth=float('nan')),dict(cellprobThreshold=float('inf')),dict(minSize3D=1.2)]:
            with self.assertRaises(ValueError):settings(invalid)

    def test_native_stage_writes_labels_and_resume_rejects_mode_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw.tif'
            tifffile.imwrite(raw,np.ones((8,20,24),np.uint16),metadata=dict(axes='ZYX'),photometric='minisblack')
            args=Namespace(input=raw,output=root/'run',spacing=[2,1,1],scale=.5,diameter=25,model='cpsam_v2',cpu=True,no_background=True,segmentation_mode='cellpose3d',flow_3d_smooth=1.,min_size_3d=20,cellprob_threshold=-.5)
            cfg=setup(args);self.assertEqual(cfg['segmentationMode'],'cellpose3d');setup(args)
            args.segmentation_mode='usegment3d'
            with self.assertRaisesRegex(AssertionError,'different segmentation mode'):setup(args)
            args.segmentation_mode='cellpose3d';args.flow_3d_smooth=2.
            with self.assertRaisesRegex(AssertionError,'different settings'):setup(args)
            np.save(args.output/'processed.npy',np.ones((8,10,12),np.float32))
            fake=np.ones((8,10,12),np.uint32)
            with patch('atlas_cellpose_3d.predict',return_value=(fake,dict(cells=1,segmentationMode='cellpose3d'))) as call:
                worker(args.output,'aggregate')
            self.assertEqual(call.call_args.args[1]['diameter'],25)
            np.testing.assert_array_equal(np.load(args.output/'labels_processed.npy'),fake)
            self.assertTrue((args.output/'aggregate.completed.json').exists())
            self.assertFalse((args.output/'xy.completed.json').exists())

    def test_preparation_routes_full_stacks_and_rejects_plane_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw.tif'
            tifffile.imwrite(raw,np.ones((8,20,24),np.uint16),metadata=dict(axes='ZYX'),photometric='minisblack')
            entry=inspect(raw);controls=dict(model='cpsam_v2',gpu=False,background=False,scale=.5,diameter=25,segmentationMode='cellpose3d',flow3DSmooth=1.,cellprobThreshold=-.5,minSize3D=20)
            request=dict(title='Native',inputs=[entry],settings=controls)
            validate(request)
            for extra in [dict(execution=dict(splitStacks=True)),dict(linkingMode='streamed')]:
                with self.assertRaisesRegex(ValueError,'complete stacks'):validate(dict(request,**extra))
            run=root/'run';run.mkdir();save(run/'qc_summary.json',{})
            with patch('prepare_atlas_dataset.call') as call:
                prepare_stack(entry,controls,run,root/'logs',None,export_atlas=False)
            args=call.call_args.args[0]
            self.assertEqual(args[args.index('--segmentation-mode')+1],'cellpose3d')
            self.assertEqual(args[args.index('--min-size-3d')+1],20)


if __name__=='__main__':unittest.main()
