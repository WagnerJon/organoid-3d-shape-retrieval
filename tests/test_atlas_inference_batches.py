"""Noncubic, directional fixtures catch flow axis swaps and missing batch boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import atlas_inference_batches as batches
from atlas_worker import atomic
from prepare_atlas_dataset import validate

class BatchTests(unittest.TestCase):
    def fixture(self, root):
        shape=(5,7,9); (root/'predictions').mkdir()
        atomic(root/'processing.json',dict(processedShape=shape))
        z,y,x=np.indices(shape); reference=(100*z+10*y+x).astype('float32'); jobs=[]
        for view,order in batches.ORDERS.items():
            p=reference.transpose(order)
            for start in range(0,len(p),2):
                stop=min(start+2,len(p)); folder=root/f'{view}_{start}';folder.mkdir()
                np.save(folder/'prob.npy',p[start:stop]);np.save(folder/'flow.npy',np.stack((p[start:stop]+1000,p[start:stop]+2000)))
                jobs.append(dict(view=view,start=start,stop=stop,result=str(folder)))
        return reference,jobs

    def test_directional_assembly_single_plane_tail_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref,jobs=self.fixture(root);batches.assemble(root,jobs)
            for view in batches.ORDERS:
                np.testing.assert_array_equal(np.load(root/'predictions'/f'{view}_prob.npy'),ref)
                np.testing.assert_array_equal(np.load(root/'predictions'/f'{view}_flow.npy'),np.stack((ref+1000,ref+2000)))
            # Completed views may be reused without any batch files.
            batches.assemble(root,[])

    def test_missing_and_overlapping_batches_are_not_committed(self):
        for duplicate in [False,True]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);_,jobs=self.fixture(root)
                jobs=jobs+[jobs[0]] if duplicate else jobs[1:]
                with self.assertRaises(ValueError):batches.assemble(root,jobs)
                self.assertFalse((root/'xy.completed.json').exists())

    def test_nonfinite_batch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,jobs=self.fixture(root)
            p=Path(jobs[0]['result'])/'prob.npy';a=np.load(p);a[0,0,0]=np.nan;np.save(p,a)
            with self.assertRaises(ValueError):batches.assemble(root,jobs)
            self.assertFalse((root/'xy.completed.json').exists())

    def test_single_stack_dispatch_uses_global_preprocessing_and_full_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'logs').mkdir()
            run=root/'stacks/Stack_001'
            reference=np.arange(5*7*9,dtype='float32').reshape(5,7,9)
            def prepare_once(*args,**kwargs):
                (run/'predictions').mkdir(parents=True)
                np.save(run/'processed.npy',reference)
                atomic(run/'processing.json',dict(processedShape=list(reference.shape)))
            def dispatch(request, output):
                self.assertTrue(request['_inferenceBatches'])
                self.assertEqual(request['execution']['workers'],[{'id':'fixture'}])
                for job in request['inputs']:
                    planes=np.load(job['path']);result=Path(job['result']);result.mkdir()
                    np.save(result/'prob.npy',planes);np.save(result/'flow.npy',np.stack((planes+1000,planes+2000)))
                    atomic(result/'batch.completed.json',dict(shape=list(planes.shape)))
            request=dict(settings={},execution=dict(splitStacks=True,planesPerBatch=2,useLocal=False,workers=[{'id':'fixture'}]))
            with patch('prepare_atlas_dataset.call',side_effect=prepare_once) as prep, patch('atlas_worker_client.distribute',side_effect=dispatch) as scheduler:
                batches.distribute_stack(request,root,0,dict(path='raw.tif',spacing=[1,1,1]))
                self.assertEqual(prep.call_count,1);self.assertEqual(scheduler.call_count,1)
                jobs=scheduler.call_args.args[0]['inputs']
                self.assertEqual(len(jobs),12)
                for job in jobs:self.assertFalse(Path(job['path']).exists())
                for view in batches.ORDERS:
                    np.testing.assert_array_equal(np.load(run/'predictions'/f'{view}_prob.npy'),reference)
                    self.assertTrue((run/f'{view}.completed.json').exists())
                self.assertFalse((run/'aggregate.completed.json').exists())

    def test_imported_and_completed_stacks_do_not_resegment(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('prepare_atlas_dataset.call', side_effect=AssertionError('Must skip')):
                batches.distribute_stack({},tmp,0,dict(labels='existing.tif'))
                run=Path(tmp)/'stacks/Stack_001';run.mkdir(parents=True);atomic(run/'complete.json',{})
                batches.distribute_stack({},tmp,0,{})

if __name__=='__main__':unittest.main()
