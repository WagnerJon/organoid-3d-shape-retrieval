import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

import focus3d_explorer as focus
import segmentation_explorer as seg


class FocusAdapterTests(unittest.TestCase):
    def test_device_resolution(self):
        with patch('torch.cuda.is_available',return_value=False), patch('torch.backends.mps.is_available',return_value=True):
            self.assertEqual(focus.resolve_device('auto'),'mps')
            self.assertEqual(focus.resolve_device('mps'),'mps')
            self.assertEqual(focus.resolve_device('cpu'),'cpu')
            with self.assertRaisesRegex(ValueError,'CUDA'):
                focus.resolve_device('cuda:0')
        with patch('torch.cuda.is_available',return_value=False), patch('torch.backends.mps.is_available',return_value=False):
            self.assertEqual(focus.resolve_device('auto'),'cpu')
            with self.assertRaisesRegex(ValueError,'MPS'):
                focus.resolve_device('mps')
        with patch('torch.cuda.is_available',return_value=True):
            self.assertEqual(focus.resolve_device('auto'),'cuda:0')

    @unittest.skipUnless(focus.DEFAULT_CONFIG.is_file() and importlib.util.find_spec('zarr'), 'FOCUS-3D environment not installed')
    def test_upstream_resampling_restores_exact_native_grid(self):
        backend = focus.load_backend()
        raw = np.arange(10*24*28,dtype=np.float32).reshape(10,24,28)
        resampled, info = backend.resample_volume_for_inference(raw, cell_radius=30, z_ratio=2, reference_cell_radius=15, order=1)
        self.assertTrue(info['enabled'])
        self.assertNotEqual(resampled.shape,raw.shape)
        labels = np.zeros(resampled.shape,dtype=np.uint32); labels[1:-1,1:-1,1:-1] = 900001
        restored = backend.resize_volume_to_shape(labels,raw.shape,order=0)
        self.assertEqual(restored.shape,raw.shape)
        self.assertEqual(set(np.unique(restored)),{0,900001})

    def test_checkpoint_missing_learned_parameters_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pth'
            backend = SimpleNamespace(build_maskformer_model_from_cfg=lambda cfg:torch.nn.Linear(2,1),
                                      _extract_model_state_dict=lambda checkpoint:checkpoint['model'],
                                      _clean_state_dict_keys=lambda state:state)
            cfg = SimpleNamespace(MODEL=SimpleNamespace(WEIGHTS=str(path),DEVICE='cpu'))
            torch.save({'model':{'bias':torch.zeros(1)}},path)
            with self.assertRaisesRegex(ValueError,'missing learned parameters'):
                focus.checked_predictor(backend,cfg)
            torch.save({'model':{'bias':torch.zeros(1),'weight':torch.ones(1,2)}},path)
            model = focus.checked_predictor(backend,cfg)
            self.assertFalse(model.training)
            self.assertEqual(model(torch.ones(1,2)).item(),2)

    def test_calibration_and_options(self):
        options = focus.inference_options({'cell_radius':30, 'max_size':500}, [1, .2, .2])
        self.assertEqual(options['z_ratio'], 5)
        self.assertEqual(options['cell_radius'], 30)
        self.assertTrue(options['restore_resampled_mask'])
        self.assertEqual(options['patch_size'], (32, 96, 96))
        self.assertIsNone(options['background_threshold'])
        self.assertEqual(options['data_loader_num_workers'], 0)
        self.assertEqual(focus.inference_options({'automatic_background':False, 'background_threshold':42}, [1,1,1])['background_threshold'], 42)
        for settings, spacing in [({}, [1,.2,.3]), ({'lower_percentile':99,'upper_percentile':1}, [1,1,1]),
                                  ({'stride_z':32}, [1,1,1]), ({'stride_y':0}, [1,1,1]),
                                  ({'min_size':100,'max_size':50}, [1,1,1]), ({'device':'unknown'}, [1,1,1]),
                                  ({'cell_radius':float('nan')}, [1,1,1]), ({'batch_size':1.5}, [1,1,1])]:
            with self.subTest(settings=settings,spacing=spacing), self.assertRaises(ValueError):
                focus.inference_options(settings,spacing)

    def fixture(self, root):
        raw = np.arange(4*8*8,dtype=np.uint16).reshape(4,8,8)
        raw_path = root/'raw.tif'
        tifffile.imwrite(raw_path,raw,metadata={'axes':'ZYX'},photometric='minisblack')
        source = seg.prepare(raw_path,[1,.2,.2],root/'source')
        weights = root/'checkpoint.pth'; weights.write_bytes(b'fixture, not trained weights')
        config = root/'config.yaml'; config.write_text('MODEL: {}')
        settings = dict(checkpoint=str(weights),config=str(config),cell_radius=20)
        return raw,source,settings

    def test_worker_dispatch_publication_and_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw,source,settings = self.fixture(root)
            labels = np.zeros(raw.shape,dtype=np.uint32); labels[:,2:6,2:6] = 98765
            calls = []
            def infer(**kwargs):
                calls.append(kwargs)
                kwargs['progress_callback'](35,'Inference patches')
                kwargs['progress_callback'](-1,'Stitching patches')
                return dict(instance_map=labels,log_info={'resampling':{'enabled':True}},num_infer_patches=2,num_skipped_background_patches=1)
            backend = SimpleNamespace(infer_volume=infer)
            with patch.object(focus,'load_backend',return_value=backend), patch.object(focus.subprocess,'check_output',return_value=focus.FOCUS_COMMIT):
                result = seg.run(source,'focus3d',settings,root/'focus-run')
            self.assertEqual(calls[0]['image_path'],source['path'])
            self.assertEqual(calls[0]['z_ratio'],5)
            self.assertEqual(result['method'],'focus3d')
            self.assertEqual(result['shape'],list(raw.shape))
            self.assertEqual(result['settings']['checkpointSHA256'],seg.fingerprint(settings['checkpoint']))
            self.assertEqual(result['settings']['upstreamCommit'],focus.FOCUS_COMMIT)
            self.assertEqual(result['cells'][0]['label'],98765)
            self.assertAlmostEqual(result['cells'][0]['volume'],64*.04)
            a_folder = root/'cellpose'; a_folder.mkdir()
            a = seg.publish(np.where(labels > 0, 22, 0).astype(np.uint32),source,a_folder,'cellpose',{},0,'fixture')
            agreement = seg.compare(a,result,source)
            self.assertEqual(agreement['dice'],1)
            self.assertEqual(agreement['matched'],1)
            self.assertEqual(agreement['pairs'][0]['b'],98765)
            self.assertEqual(json.loads((root/'focus-run/result.json').read_text())['method'],'focus3d')

    def test_invalid_grid_or_missing_checkpoint_never_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw,source,settings = self.fixture(root)
            backend = SimpleNamespace(infer_volume=lambda **kw:dict(instance_map=np.zeros((2,2,2),dtype=np.uint32)))
            with patch.object(focus,'load_backend',return_value=backend), self.assertRaisesRegex(ValueError,'original raw grid'):
                seg.run(source,'focus3d',settings,root/'wrong-grid')
            self.assertFalse((root/'wrong-grid/result.json').exists())
            with self.assertRaisesRegex(ValueError,'checkpoint'):
                seg.run(source,'focus3d',dict(settings,checkpoint=''),root/'no-checkpoint')
            self.assertFalse((root/'no-checkpoint/result.json').exists())

    def test_status_only_progress_preserves_fraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw,source,settings = self.fixture(root)
            events = []
            def infer(**kwargs):
                for value, message in [(50,'Halfway'),(-1,'Stitching'),(40,'Older progress'),(100,'Done')]:
                    kwargs['progress_callback'](value,message)
                return dict(instance_map=np.zeros(raw.shape,dtype=np.uint32),log_info={},num_infer_patches=1,num_skipped_background_patches=0)
            with patch.object(focus,'load_backend',return_value=SimpleNamespace(infer_volume=infer)), patch.object(focus.subprocess,'check_output',return_value=focus.FOCUS_COMMIT):
                focus.run_focus3d(raw,source,settings,root,lambda stage,fraction:events.append((stage,fraction)),seg.fingerprint)
            fractions = [event[1] for event in events]
            self.assertEqual(fractions,sorted(fractions))
            self.assertAlmostEqual(fractions[1],fractions[2])


if __name__ == '__main__':
    unittest.main()
