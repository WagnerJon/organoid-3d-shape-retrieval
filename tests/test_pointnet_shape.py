import copy
import json
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pointnet_shape_core import sample_surface,normalize_cloud,hierarchy,batch_hierarchy,PointCloudAutoencoder,chamfer,rotations,assert_split,load_data
from evaluate_pointnet_shape import gate


def config():
    value=json.loads((Path(__file__).resolve().parents[1]/'config/pointnet_shape.json').read_text())
    value.update(points=64,centers=[16,4],neighbors=8)
    return value


class PointNetShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_area_sampling_and_physical_normalization(self):
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[10,0,0],[12,0,0],[10,2,0]],float)
        points=sample_surface(vertices,np.array([[0,1,2],[3,4,5]]),10000,np.random.default_rng(7))
        self.assertAlmostEqual(float((points[:,0]>5).mean()),.8,delta=.02)
        target=points[::2];normal,truth,center,scale=normalize_cloud(points,target)
        np.testing.assert_allclose(normal.mean(0),0,atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(normal,axis=1).max()),1.,places=6)
        np.testing.assert_allclose(truth*scale+center,target,atol=2e-6)
        shifted,_,_,_=normalize_cloud(points*3+10,target*3+10)
        np.testing.assert_allclose(normal,shifted,atol=2e-6)
        with self.assertRaises(ValueError):sample_surface(np.zeros((3,3)),np.array([[0,1,2]]),10,np.random.default_rng(1))

    def test_hierarchy_and_permutation_invariance(self):
        torch.manual_seed(1);rng=np.random.default_rng(2)
        points=torch.from_numpy(rng.normal(size=(2,64,3)).astype(np.float32))
        model=PointCloudAutoencoder(config()).eval()
        with torch.no_grad():
            latent,reconstruction=model(points)
            permutation=torch.randperm(64);other,_=model(points[:,permutation])
        self.assertEqual(tuple(latent.shape),(2,128));self.assertEqual(tuple(reconstruction.shape),(2,64,3))
        torch.testing.assert_close(latent,other,atol=1e-5,rtol=1e-5)
        groups=hierarchy(points[0].numpy(),[16,4],8)
        self.assertEqual(groups['neighbors1'].shape,(16,8));self.assertEqual(len(set(groups['centers1'])),16)

    def test_chamfer_matches_dense_reference_and_gradients(self):
        torch.manual_seed(3)
        a=torch.randn(2,17,3,requires_grad=True);b=torch.randn(2,13,3)
        distance=(a[:,:,None]-b[:,None]).square().sum(-1)
        expected=distance.min(2).values.mean(1)+distance.min(1).values.mean(1)
        actual=chamfer(a,b,5)
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        actual.mean().backward();self.assertTrue(torch.isfinite(a.grad).all());self.assertGreater(float(a.grad.norm()),0)
        self.assertLess(float(chamfer(b,b,4).max()),2e-6)

    def test_cpu_training_reduces_reconstruction_loss(self):
        torch.manual_seed(4);rng=np.random.default_rng(5)
        points=rng.normal(size=(3,64,3)).astype(np.float32)
        points/=np.linalg.norm(points,axis=-1,keepdims=True)
        points*=np.array([1,.7,.4],np.float32)
        points=torch.from_numpy(points)
        cfg=config();model=PointCloudAutoencoder(cfg)
        groups=batch_hierarchy(points,cfg['centers'],cfg['neighbors'],'cpu')
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        initial=float(chamfer(model(points,groups)[1],points,32).mean().detach())
        for _ in range(15):
            loss=chamfer(model(points,groups)[1],points,32).mean();optimizer.zero_grad();loss.backward();optimizer.step()
        final=float(chamfer(model(points,groups)[1],points,32).mean().detach())
        self.assertLess(final,initial*.5)

    def test_only_proper_rotations_and_grouped_splits(self):
        rotation=rotations(30,np.random.default_rng(3))
        np.testing.assert_allclose(rotation@rotation.transpose(0,2,1),np.broadcast_to(np.eye(3),(30,3,3)),atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(rotation),1,atol=1e-6)
        frame=pd.DataFrame(dict(cell_id=['a','b','c'],sample=['A','B','C'],similarity_group=[0,1,2],split=['train','validation','test'],training_eligible=[True]*3))
        assert_split(frame)
        frame.loc[1,'similarity_group']=0
        with self.assertRaisesRegex(ValueError,'crosses splits'):assert_split(frame)

    def test_changed_manifest_is_rejected_before_loading_geometry(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            (folder/'manifest.csv').write_text('cell_id,split\na,train\n')
            (folder/'dataset.json').write_text(json.dumps(dict(manifest_sha256='wrong')))
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                load_data(folder)

    def test_smoke_predictions_cannot_be_published_to_atlas(self):
        from export_pointnet_shape import export
        with tempfile.TemporaryDirectory() as name:
            cfg=config();cfg['output_dir']=name
            folder=Path(name)/'baseline';folder.mkdir()
            (folder/'status.json').write_text(json.dumps(dict(smoke=True,status='trained')))
            with self.assertRaisesRegex(ValueError,'non-smoke'):
                export(cfg,'cpu')

    def test_stage_two_cannot_be_enabled_by_smoke_or_pretty_reconstruction_alone(self):
        cfg=config();metrics=dict(cells=100,groups=5,untrained_improvement=.8,template_improvement=.2,resampling_self_top1=.95,effective_rank=20,
            conditional_retrieval=dict(groups=5,group_bootstrap_95_ci=[.01,.1]))
        status=dict(smoke=False,completed_epochs=20)
        self.assertTrue(gate(metrics,status,cfg)['eligible'])
        status['smoke']=True;self.assertFalse(gate(metrics,status,cfg)['eligible'])
        status['smoke']=False;metrics['conditional_retrieval']['group_bootstrap_95_ci']=[-.01,.1]
        self.assertFalse(gate(metrics,status,cfg)['eligible'])


if __name__=='__main__':unittest.main()
