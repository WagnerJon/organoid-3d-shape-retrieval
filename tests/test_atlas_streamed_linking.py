"""Production streaming parity, bounded inference, interruption and export controls."""
from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_standalone_cellpose as standalone
import atlas_streaming_core as core
import atlas_streamed_linking as linking


def fixture(root):
    from segment3D import parameters
    shape=(37,29,31);z,y,x=np.indices(shape);centers=np.array([(18,14,10),(18,14,22)])
    distances=np.stack([sum((a-c)**2 for a,c in zip((z,y,x),center)) for center in centers]);owner=distances.argmin(0)
    field=np.stack([np.take(centers[:,a],owner)-coordinate for a,coordinate in enumerate((z,y,x))]).astype('float32')
    probability=np.clip(3-4*distances.min(0)/64,-5,3).astype('float32')
    predictions=root/'predictions';predictions.mkdir()
    flows=[]
    for view,axes in zip(['xy','xz','yz'],[(1,2),(0,2),(0,1)]):
        flow=field[list(axes)].copy();flows.append(flow)
        np.save(predictions/f'{view}_prob.npy',probability);np.save(predictions/f'{view}_flow.npy',flow)
    params=parameters.get_2D_to_3D_aggregation_params();params['gradient_descent'].update(n_iter=30,gradient_decay=.05,do_mp=False)
    params['postprocess_binary']['remove_small_objects']=10
    return params,[probability.copy() for _ in range(3)],flows


def same_partition(a,b):
    from scipy.optimize import linear_sum_assignment
    n,m=int(a.max())+1,int(b.max())+1
    table=np.bincount((a.astype('int64')*m+b).ravel(),minlength=n*m).reshape(n,m)
    rows,columns=linear_sum_assignment(-table)
    return table[rows,columns].sum()==a.size and n==m


class StreamedAtlasTests(unittest.TestCase):
    def test_production_fusion_and_labels_match_upstream_across_remap_boundaries(self):
        from segment3D import usegment3d
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);params,probabilities,flows=fixture(root)
            expected,(probability,field)=usegment3d.aggregate_2D_to_3D_segmentation_direct_method(probabilities,flows,params)
            scratch=root/'reference';scratch.mkdir()
            core.fuse_predictions(root/'predictions',scratch,params)
            np.testing.assert_array_equal(np.load(scratch/'probability.npy'),probability)
            np.testing.assert_array_equal(np.load(scratch/'gradient.npy'),field.transpose(3,0,1,2))
            result=linking.aggregate(root,params)
            self.assertTrue(same_partition(expected,np.load(root/'labels_processed.npy')))
            self.assertEqual(result['cells'],2)
            self.assertEqual(list((root/'streamed_linking').glob('*.npy')),[])
            self.assertTrue((root/'predictions/xy_prob.npy').exists())
            with patch.object(core,'fuse_predictions',side_effect=AssertionError('must reuse')):
                self.assertEqual(linking.aggregate(root,params),result)
            params['gradient_descent']['n_iter']+=1
            with self.assertRaisesRegex(ValueError,'changed'):linking.aggregate(root,params)

    def test_interrupt_resume_reuses_fusion_and_completed_trajectory_chunks(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);params,_,_=fixture(root)
            def stop(label,fraction):
                if label=='Following fine-grid cell flows' and fraction>.4:raise InterruptedError('stop')
            with self.assertRaises(InterruptedError):linking.aggregate(root,params,progress=stop)
            state=json.loads((root/'streamed_linking/state.json').read_text())
            self.assertTrue(state['fusionDone']);self.assertEqual(state['trajectoryNextZ'],8)
            saved_fusion=core.fuse_predictions
            with patch.object(core,'fuse_predictions',side_effect=AssertionError('must reuse')):
                result=linking.aggregate(root,params)
            resumed=np.load(root/'labels_processed.npy').copy()
            reference=root/'other';reference.mkdir();fixture(reference)
            linking.aggregate(reference,params)
            np.testing.assert_array_equal(resumed,np.load(reference/'labels_processed.npy'))
            self.assertEqual(result['cells'],2)

    def test_streamed_cellpose_preserves_view_axes_and_resumes_at_plane_checkpoint(self):
        for view,order in [('xy',(0,1,2)),('xz',(1,0,2)),('yz',(2,0,1))]:
            with self.subTest(view=view),tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
                root=Path(tmp);(root/'predictions').mkdir();image=np.arange(11*12*13,dtype='float32').reshape(11,12,13);np.save(root/'processed.npy',image)
                calls=[]
                class Model:
                    def __init__(self,**kwargs):pass
                    def eval(self,images,**kwargs):
                        self_image=images[0];calls.append(self_image.copy())
                        if len(calls)==10:raise InterruptedError('stop')
                        return None,[[None,np.stack([self_image+1,self_image+2]),self_image+3]],None
                settings=dict(model='cpsam_v2',gpu=False,diameter=30)
                with self.assertRaises(InterruptedError):linking.infer_view(root,view,settings,Model)
                self.assertEqual(json.loads((root/'predictions'/f'{view}.streamed.json').read_text())['nextPlane'],8)
                calls.clear();linking.infer_view(root,view,settings,Model)
                self.assertEqual(len(calls),image.shape[order[0]]-8)
                np.testing.assert_array_equal(np.load(root/'predictions'/f'{view}_prob.npy'),image+3)
                np.testing.assert_array_equal(np.load(root/'predictions'/f'{view}_flow.npy'),np.stack([image+1,image+2]))
                # A completed view must not instantiate a model again.
                linking.infer_view(root,view,settings,lambda **kwargs: self.fail('completed model'))

    def test_disk_guard_and_scratch_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);params,_,_=fixture(root)
            with patch.object(linking.shutil,'disk_usage',return_value=type('Space',(),{'free':0})()):
                with self.assertRaisesRegex(ValueError,'disk space'):linking.aggregate(root,params)
            self.assertFalse((root/'labels_processed.npy').exists())
            self.assertFalse((root/'streamed_linking/gradient.npy').exists())
            self.assertEqual(linking.validate_mode(None),'standard')
            with self.assertRaises(ValueError):linking.validate_mode('coarse')
            (root/'streamed_linking/state.json').unlink();(root/'streamed_linking/notes.txt').write_text('keep')
            with self.assertRaisesRegex(ValueError,'unrecognized'):linking.aggregate(root,params)
            self.assertEqual((root/'streamed_linking/notes.txt').read_text(),'keep')

    def test_memory_budget_and_actual_native_grid_mapping(self):
        from atlas_preprocessing import memory_plan
        shape=[578,1920,1920];spacing=[.75,.22872330084318845,.22872330084318845]
        original=memory_plan(shape,spacing,.85,16*2**30)
        streamed=memory_plan(shape,spacing,.85,16*2**30,'streamed')
        self.assertEqual(original['suggestedScale'],.17)
        self.assertFalse(streamed['fits']);self.assertGreater(streamed['suggestedScale'],original['suggestedScale'])
        labels=np.arange(4*5*6,dtype='uint32').reshape(4,5,6);native=[7,9,11];zoom=[.63,.62,.61]
        for z in range(native[0]):
            actual=standalone.native_plane(labels,native,z,zoom)
            for y,x in [(0,0),(4,5),(8,10)]:
                ijk=np.minimum(np.array(labels.shape)-1,((np.array([z,y,x])+.5)*zoom).astype(int))
                self.assertEqual(actual[y,x],labels[tuple(ijk)])

if __name__=='__main__':unittest.main()
