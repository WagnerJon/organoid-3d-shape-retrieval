"""Initial-stack routing, calibration, saved settings and restart guarantees."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import atlas_focus_preparation as focus
import focus3d_explorer as adapter
import prepare_atlas_dataset as preparation


class FocusPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='atlas-initial-focus-');self.root=Path(self.temporary.name)
        z,y,x=np.indices((20,36,40));self.labels=np.zeros(z.shape,np.uint32)
        self.labels[(z-10)**2+(y-14)**2+(x-14)**2<36]=17
        self.labels[(z-10)**2+(y-14)**2+(x-25)**2<36]=42
        self.raw=self.root/'raw.ome.tif'
        tifffile.imwrite(self.raw,self.labels.astype(np.uint16)*100,ome=True,photometric='minisblack',
            metadata=dict(axes='ZYX',PhysicalSizeZ=2.,PhysicalSizeY=.5,PhysicalSizeX=.5))
        self.entry=preparation.inspect(self.raw)
        self.checkpoint=self.root/'test.pth';self.checkpoint.write_bytes(b'not a real checkpoint; mocked model inference')
        self.config=self.root/'test.yaml';self.config.write_text('MODEL: {}')
        self.settings=dict(checkpoint=str(self.checkpoint),config=str(self.config),device='cpu',cell_radius=24.,
                           score_threshold=.2,mask_threshold=.6,min_size=3,batch_size=1,stride_z=16)
        self.request=dict(title='FOCUS preparation test',inputs=[self.entry],segmentationMethod='focus3d',
            focusSettings=self.settings,settings=dict(model='cpsam_v2',scale=.35,diameter=30.,gpu=False,background=True),
            harmonics=False,excludeDetached=True)

    def tearDown(self):self.temporary.cleanup()

    def test_focus_validation_is_independent_of_cellpose_and_rejects_workers(self):
        request=copy.deepcopy(self.request);request['settings']=dict(model='unused',scale=-1,diameter=-1)
        preparation.validate(request)
        for execution in (dict(splitStacks=True),dict(workers=[{'id':'other'}]),dict(useLocal=False)):
            with self.subTest(execution=execution),self.assertRaisesRegex(ValueError,'this Mac'):
                preparation.validate(dict(self.request,execution=execution))
        with self.assertRaisesRegex(ValueError,'own stitching'):
            preparation.validate(dict(self.request,linkingMode='streamed'))
        invalid=copy.deepcopy(self.request);invalid['focusSettings']['stride_z']=32
        with self.assertRaisesRegex(ValueError,'stride_z'):preparation.validate(invalid)
        missing=copy.deepcopy(self.request);missing['focusSettings']['checkpoint']=''
        with self.assertRaisesRegex(ValueError,'checkpoint'):preparation.validate(missing)

    def test_memory_plan_matches_upstream_radius_and_z_rules(self):
        a=focus.memory_plan([32,96,96],[2,.5,.5],dict(cell_radius=30),16*1024**3)
        self.assertEqual(a['processedShape'],[32,48,48]);self.assertTrue(a['fits'])
        b=focus.memory_plan([32,96,96],[.5,.5,.5],dict(cell_radius=30),16*1024**3)
        self.assertEqual(b['processedShape'],[16,48,48])
        self.assertFalse(focus.memory_plan([1024,2048,2048],[2,1,1],dict(cell_radius=15),16*1024**3)['fits'])

    def test_initial_focus_build_preserves_native_calibration_and_selected_parameters(self):
        calls=[]
        def infer(**kwargs):
            calls.append(kwargs);kwargs['progress_callback'](50,'Patch inference')
            return dict(instance_map=self.labels,log_info={},num_infer_patches=1,num_skipped_background_patches=0)
        real_call=preparation.call
        def dispatch(arguments,log,progress=None):
            if 'atlas_focus_preparation.py' in str(arguments[2]):
                request=json.loads(Path(arguments[arguments.index('--request')+1]).read_text())
                return focus.run(request['input'],request['settings'],Path(arguments[arguments.index('--output')+1]))
            return real_call(arguments,log,progress)
        def guarded(arguments,folder,name):
            if name=='focus3d':focus.infer(self.entry,self.settings,folder)
            elif name=='export':focus.export(folder,json.loads((folder/'run.json').read_text()))
            else:self.fail('Unexpected inference stage')
            return .1
        with patch.object(adapter,'load_backend',return_value=SimpleNamespace(infer_volume=infer)), \
             patch.object(preparation,'call',side_effect=dispatch),patch.object(focus,'guarded',side_effect=guarded):
            output=self.root/'analysis';atlas=preparation.build(self.request,output)
            data=json.loads((atlas/'dataset.json').read_text());run=output/'stacks/Stack_001'
            np.testing.assert_array_equal(np.load(run/'labels_processed.npy'),self.labels)
            grid=json.loads((run/'processing.json').read_text())
            self.assertEqual(grid['effectiveSpacingZYXum'],[2.,.5,.5]);self.assertEqual(grid['rawGridOffsetZYXum'],[0,0,0])
            self.assertEqual(calls[0]['cell_radius'],24);self.assertEqual(calls[0]['z_ratio'],4)
            self.assertEqual(calls[0]['score_thresh'],.2);self.assertEqual(calls[0]['mask_thresh'],.6)
            self.assertEqual(calls[0]['stride'],(16,64,64))
            self.assertEqual({c['label'] for c in data['cells']},{17,42})
            for cell in data['cells']:
                self.assertAlmostEqual(cell['metrics']['volume_um3'],float((self.labels==cell['label']).sum())*.5)
            self.assertEqual(data['samples'][0]['labelGrid']['segmentationMode'],'focus3d')
            self.assertEqual(data['samples'][0]['segmentation']['settings'],self.settings)
            before=(atlas/'dataset.json').read_bytes();preparation.build(self.request,output)
            self.assertEqual(before,(atlas/'dataset.json').read_bytes());self.assertEqual(len(calls),1)
            changed=copy.deepcopy(self.request);changed['focusSettings']['score_threshold']=.3
            with self.assertRaisesRegex(ValueError,'different preparation settings'):preparation.build(changed,output)
            self.checkpoint.write_bytes(b'replaced weights')
            with self.assertRaisesRegex(ValueError,'weights or settings changed'):focus.run(self.entry,self.settings,run)

    def test_empty_or_wrong_grid_predictions_are_not_completed(self):
        for labels in (np.zeros(self.labels.shape,np.uint32),np.zeros((2,2,2),np.uint32)):
            backend=SimpleNamespace(infer_volume=lambda **kwargs:dict(instance_map=labels,log_info={},num_infer_patches=1,num_skipped_background_patches=0))
            with patch.object(adapter,'load_backend',return_value=backend),self.assertRaises(ValueError):
                focus.infer(self.entry,self.settings,self.root)
            self.assertFalse((self.root/'aggregate.completed.json').exists())
            self.assertFalse((self.root/'labels_processed.npy').exists())


if __name__=='__main__':unittest.main()
