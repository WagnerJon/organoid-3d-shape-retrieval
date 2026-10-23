import copy
import json
from pathlib import Path
import sys
import tempfile
from argparse import Namespace
import unittest

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_atlas_dataset import inspect, validate, build


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='atlas-preparation-test-')
        cls.root = Path(cls.temporary.name)
        z,y,x = np.indices((24,44,52))
        labels = np.zeros(z.shape, np.uint16)
        labels[(z-9)**2+(y-18)**2+(x-17)**2 < 49] = 1
        labels[(z-9)**2+(y-18)**2+(x-29)**2 < 49] = 2
        labels[(z-20)**2+(y-39)**2+(x-47)**2 < 5] = 3
        cls.labels = labels
        cls.entries = []
        for index in range(2):
            raw, mask = cls.root/f'raw{index}.ome.tif', cls.root/f'labels{index}.tif'
            tifffile.imwrite(raw, (labels*1000).astype(np.uint16), compression='zlib', ome=True,
                             metadata=dict(axes='ZYX', PhysicalSizeZ=2000, PhysicalSizeZUnit='nm',
                                           PhysicalSizeY=0.5, PhysicalSizeYUnit='µm', PhysicalSizeX=0.5, PhysicalSizeXUnit='µm'))
            tifffile.imwrite(mask, labels, metadata={'axes':'ZYX'}, photometric='minisblack')
            entry = inspect(raw)
            entry.update(labels=str(mask), labelStat=[mask.stat().st_size,mask.stat().st_mtime_ns])
            cls.entries.append(entry)
        cls.request = dict(title='Preparation test', inputs=cls.entries,
                           settings=dict(model='cpsam_v2', scale=.35, diameter=30., gpu=False, background=True),
                           harmonics=True, excludeDetached=True)
        cls.output = cls.root/'prepared'
        cls.atlas = build(cls.request, cls.output)
        cls.data = json.loads((cls.atlas/'dataset.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_metadata_calibration_and_unsupported_axes(self):
        self.assertEqual(self.entries[0]['spacing'], [2.,.5,.5])
        path = self.root/'time.ome.tif'
        tifffile.imwrite(path, np.zeros((2,4,8,8),np.uint8), ome=True, metadata={'axes':'TZYX'})
        with self.assertRaisesRegex(ValueError, 'single-channel'):
            inspect(path)

    def test_missing_calibration_estimates_depth_without_stretching_xy(self):
        path=self.root/'uncalibrated.tif'
        tifffile.imwrite(path,np.zeros((3,4,5),np.uint16),metadata={'axes':'ZYX'},photometric='minisblack')
        metadata=inspect(path)
        self.assertEqual(metadata['spacing'],[5/3,1.,1.])
        self.assertIn('assume cubic stack',metadata['calibrationSource'])
        self.assertIn('Absolute physical scale is unknown',metadata['calibrationSource'])
        self.assertEqual(inspect(self.entries[0]['path'])['spacing'],[2.,.5,.5])

    def test_cubic_estimate_keeps_xy_equal_for_square_rectangular_and_deep_stacks(self):
        from atlas_calibration import assumed_cubic_spacing
        for shape in [(64,512,512),(64,256,512),(64,512,256),(1024,256,512)]:
            z,y,x=shape; spacing=assumed_cubic_spacing(shape)
            self.assertEqual(spacing[1:],[1.,1.])
            self.assertEqual(z*spacing[0],max(y,x))
            self.assertEqual((x*spacing[2])/(y*spacing[1]),x/y)
        for shape in [(0,12,12),(12,12),(12,12,-1)]:
            with self.assertRaises(ValueError): assumed_cubic_spacing(shape)

    def test_joint_dataset_geometry_measurements_and_exclusions(self):
        data = self.data
        self.assertEqual(len(data['cells']),6)
        self.assertEqual(len(data['samples']),2)
        self.assertEqual(len(data['organoids']),2)
        self.assertEqual(sum(c['inMainOrganoid'] for c in data['cells']),4)
        self.assertTrue(all(c['automaticMainOrganoid'] == c['inMainOrganoid'] for c in data['cells']))
        for cell in data['cells']:
            self.assertAlmostEqual(cell['metrics']['volume_um3'], (self.labels==cell['label']).sum()*.5)
            self.assertTrue((self.atlas/cell['sample']/f"{cell['label']}.mesh").exists())
            for rep in cell['representations'].values():
                self.assertTrue(np.isfinite(rep['vector']).all())
                self.assertEqual(len(rep['point']),2)
        for stack in data['organoids']:
            cells = [c for c in data['cells'] if c['sample']==stack['sample'] and c['inMainOrganoid']]
            self.assertEqual(stack['cellCount'],len(cells))
            self.assertAlmostEqual(stack['metrics']['volume_um3'],sum(c['metrics']['volume_um3'] for c in cells))
        self.assertIn('spherical_harmonics',[m['id'] for m in data['methods']])
        self.assertIn('morph_size',[m['id'] for m in data['methods']])
        self.assertIn('morph_shape',[m['id'] for m in data['methods']])
        self.assertEqual(len(data['traditionalFeatures']),9)

    def test_resume_is_idempotent_and_changed_settings_rejected(self):
        before = (self.atlas/'dataset.json').read_bytes()
        self.assertEqual(build(self.request,self.output),self.atlas)
        self.assertEqual((self.atlas/'dataset.json').read_bytes(),before)
        changed = copy.deepcopy(self.request); changed['settings']['diameter']=40
        with self.assertRaisesRegex(ValueError,'different preparation settings'):
            build(changed,self.output)
        occupied = self.root/'occupied'; occupied.mkdir(); (occupied/'important.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError,'other data'):
            build(self.request,occupied)
        self.assertEqual((occupied/'important.txt').read_text(),'keep')

    def test_checkpoint_recovery_and_single_cell_export(self):
        # Re-enter after an interrupted final publication: all per-stack work is reused.
        (self.output/'complete.json').unlink()
        mesh = self.atlas/'Stack_001/1.mesh'
        before = mesh.stat().st_mtime_ns
        (self.atlas/'dataset.json').unlink()
        build(self.request,self.output)
        self.assertEqual(mesh.stat().st_mtime_ns,before)
        self.assertEqual(len(json.loads((self.atlas/'dataset.json').read_text())['cells']),6)
        # One cell cannot define PCA variation, but must still produce a valid dataset.
        from prepare_atlas_dataset import imported, call, ROOT
        from export_standalone_atlas import export
        one = self.root/'one.tif'
        tifffile.imwrite(one,(self.labels==1).astype(np.uint16),metadata={'axes':'ZYX'},photometric='minisblack')
        entry = dict(self.entries[0],labels=str(one))
        run = self.root/'single-run'
        imported(entry,run,self.request['settings'])
        call([sys.executable,ROOT/'scripts/qc_standalone_cellpose.py',run],self.root/'single-qc.log')
        export(run,self.root/'single-atlas','Single')
        data = json.loads((self.root/'single-atlas/dataset.json').read_text())
        self.assertEqual(len(data['cells']),1)
        self.assertEqual(data['cells'][0]['representations']['traditional']['point'],[0.,0.])
        self.assertEqual(data['methods'][0]['variance'],[0.,0.])

    def test_cellpose_configuration_preserves_confirmed_spacing_and_settings(self):
        from run_standalone_cellpose import setup
        args = Namespace(input=Path(self.entries[0]['path']), output=self.root/'cellpose-configuration',
                         spacing=[3.,.7,.6],scale=.25,diameter=45.,model='cpdino-vitb',cpu=True,no_background=True)
        config = setup(args)
        self.assertEqual(config['spacingZYXum'],[3.,.7,.6])
        self.assertEqual(config['model'],'cpdino-vitb')
        self.assertFalse(config['gpu'])
        self.assertFalse(config['backgroundCorrection'])
        self.assertEqual(config['diameterProcessedPx'],45.)
        self.assertEqual(setup(args),config)

    def test_missing_spacing_and_changed_source_rejected(self):
        request = copy.deepcopy(self.request); request['inputs'][0]['spacing']=[0,1,1]
        with self.assertRaisesRegex(ValueError,'spacing'):
            validate(request)
        request = copy.deepcopy(self.request); request['inputs'][0]['sourceStat'][0]+=1
        with self.assertRaisesRegex(ValueError,'changed'):
            validate(request)
        request = copy.deepcopy(self.request); request['inputs'][0]['labelStat'][0]+=1
        with self.assertRaisesRegex(ValueError,'labels changed'):
            validate(request)


if __name__=='__main__': unittest.main()
