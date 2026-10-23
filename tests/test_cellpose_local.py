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
import atlas_cellpose_models as models
from atlas_cellpose_3d import predict
from atlas_slice_test import run as slice_run
from prepare_atlas_dataset import inspect, prepare_stack
from run_standalone_cellpose import setup


class LocalCellposeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.checkpoint=self.root/'custom model';self.checkpoint.write_bytes(b'checkpoint fixture')
        self.settings=dict(model='local',localModelPath=str(self.checkpoint),localModelRuntime='cellpose3',gpu=False,diameter=30.,scale=.35,background=False)
        self.raw=self.root/'raw.ome.tif';tifffile.imwrite(self.raw,np.ones((8,12,16),np.uint16),photometric='minisblack',metadata={'axes':'ZYX'})
        self.entry=inspect(self.raw);self.entry['spacing']=[1,1,1]
    def tearDown(self):self.temp.cleanup()

    def test_local_classic_and_modern_checkpoints_use_explicit_paths(self):
        seen=[]
        def factory(**kwargs):
            seen.append(kwargs);return SimpleNamespace(pretrained_model=kwargs['pretrained_model'],device='cpu')
        modern=SimpleNamespace(CellposeModel=factory)
        for runtime in ['cellpose3','cellpose4']:
            values=dict(self.settings,localModelRuntime=runtime)
            with patch.object(models,'legacy_backend',return_value=modern),patch.dict(sys.modules,{'cellpose.models':modern}):
                actual=models.make_model(**models.factory_arguments(values))
            self.assertEqual(seen[-1],dict(pretrained_model=str(self.checkpoint),gpu=False))
            self.assertEqual(actual.atlas_runtime['checkpointSHA256'],models.digest(self.checkpoint))
            self.assertEqual(actual.atlas_runtime['engine'],'Cellpose3 CNN' if runtime=='cellpose3' else 'Cellpose4 transformer')
        self.assertTrue(models.explicit_inference(self.settings))

    def test_missing_mismatched_and_changed_checkpoints_fail_instead_of_falling_back(self):
        with self.assertRaisesRegex(ValueError,'missing or empty'):models.local_fields(dict(self.settings,localModelPath=str(self.root/'missing')))
        with self.assertRaisesRegex(ValueError,'Choose Cellpose'):models.local_fields(dict(self.settings,localModelRuntime='unknown'))
        bound=models.local_fields(self.settings,freeze=True)
        self.checkpoint.write_bytes(b'new weights')
        with self.assertRaisesRegex(ValueError,'changed'):models.factory_arguments(dict(self.settings,**bound))
        with patch.object(models,'legacy_backend',return_value=SimpleNamespace(CellposeModel=lambda **kwargs:SimpleNamespace(pretrained_model='/wrong/default'))):
            with self.assertRaisesRegex(ValueError,'refusing local-model fallback'):models.make_model(**models.factory_arguments(self.settings))

    def test_slice_and_native_3d_receive_bound_model_choice(self):
        seen=[]
        class Model:
            device='cpu';atlas_runtime={'engine':'test local'}
            def __init__(self,**kwargs):seen.append(kwargs)
            def eval(self,image,**kwargs):return np.ones(image.shape[:3] if kwargs.get('do_3D') else image.shape,np.uint16),[],None
        result=slice_run(dict(input=self.entry,plane='xy',index=3,settings=dict(self.settings)),self.root/'slice',segment=True,model_factory=Model)
        labels,meta=predict(np.ones((5,6,7),np.float32),self.settings,model_factory=Model)
        for args in seen:
            self.assertEqual(args['pretrained_model'],'local');self.assertEqual(args['local_settings']['localModelPath'],str(self.checkpoint))
            self.assertEqual(args['local_settings']['localModelSHA256'],models.digest(self.checkpoint))
        self.assertEqual(result['modelRuntime']['checkpoint'],str(self.checkpoint));self.assertEqual(labels.shape,(5,6,7))

    def test_preparation_cli_and_resume_pin_checkpoint_contents(self):
        run=self.root/'run';run.mkdir();(run/'complete.json').write_text('{}');(run/'qc_summary.json').write_text('{}')
        # Use a fresh run with fake segmentation; subsequent QC/export are already present.
        (run/'complete.json').unlink();calls=[]
        with patch('prepare_atlas_dataset.call',side_effect=lambda args,*rest:calls.append(args)):
            prepare_stack(self.entry,self.settings,run,self.root/'logs',None,export_atlas=False)
        args=calls[0];self.assertIn('--local-model-path',args);self.assertIn(str(self.checkpoint),args)
        self.assertIn(models.digest(self.checkpoint),args)
        options=SimpleNamespace(input=self.raw,output=self.root/'standalone',spacing=[1,1,1],scale=.35,diameter=30,model='local',cpu=True,no_background=True,local_model_path=str(self.checkpoint),local_model_runtime='cellpose3')
        config=setup(options);self.assertEqual(config['localModelSHA256'],models.digest(self.checkpoint));self.assertEqual(setup(options),config)
        self.checkpoint.write_bytes(b'different checkpoint')
        with self.assertRaisesRegex(AssertionError,'different settings'):setup(options)

    def test_worker_requires_owned_checkpoint_and_remaps_client_path(self):
        import atlas_worker as worker
        state=worker.WorkerState(self.root/'worker');owner='fixture-owner'
        raw_hash=models.digest(self.raw);model_hash=models.digest(self.checkpoint)
        for source,key in [(self.raw,raw_hash),(self.checkpoint,model_hash)]:
            (state.folder/'blobs'/key).write_bytes(source.read_bytes());state.blob_owners[key]=[owner]
        settings=dict(self.settings,localModelPath='/not/a/path/on/worker',localModelSHA256=model_hash)
        spec=dict(kind='inference-batch',rawHash=raw_hash,spacing=[1,1,1],settings=settings)
        payload=dict(spec=spec,compatibility=state.compatibility)
        with self.assertRaisesRegex(ValueError,'checkpoint first'):state.submit(payload,owner)
        spec['modelHash']=model_hash;settings['localModelSHA256']='0'*64
        with self.assertRaisesRegex(ValueError,'identity mismatch'):state.submit(payload,owner)
        settings['localModelSHA256']=model_hash;state.blob_owners[model_hash]=['other-owner']
        with self.assertRaisesRegex(ValueError,'verified input'):state.submit(payload,owner)
        state.blob_owners[model_hash]=[owner]
        with patch('threading.Thread.start'):
            job=state.submit(payload,owner)
        with patch('atlas_inference_batches.infer') as infer:
            worker.run_job(state.folder,job['id'])
        received=infer.call_args.args[1]
        self.assertEqual(received['localModelPath'],str(state.folder/'blobs'/model_hash));self.assertEqual(received['localModelSHA256'],model_hash)

if __name__=='__main__':unittest.main()
