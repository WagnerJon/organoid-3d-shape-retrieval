import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy import ndimage as ndi
import tifffile
import torch
from membrane_refinement import refine,blur,options,shape_quality,quality_failures
from segmentation_explorer import prepare,publish,refine_run,fingerprint


def fixture():
    shape = (24,24,24)
    grids = np.indices(shape)
    radius = np.sqrt(sum((grids[i]-12)**2 for i in range(3)))
    truth = radius <= 5
    initial = np.where(radius <= 6,900001,0).astype(np.uint32)
    signed = ndi.distance_transform_edt(truth)-ndi.distance_transform_edt(~truth)
    signed = np.sign(signed)*np.maximum(0,np.abs(signed)-.5)
    inside = torch.sigmoid(torch.tensor(signed,dtype=torch.float32)/.6)
    raw = blur(4*inside*(1-inside),(.5,.5,.5),(1,1,1)).numpy()
    # A known one-voxel radius error deliberately needs a larger trust region
    # than the conservative defaults used for unannotated experimental cells.
    settings = dict(iterations=45,shapeWeight=.05,smoothness=.05,maxDisplacement=2.,controlSpacing=3.,learningRate=.12,psfZ=.5,psfXY=.5,device='cpu',maxVolumeChange=.5,maxChangedFraction=1.)
    return raw,initial,truth,settings

def dice(a,b):
    return 2*np.count_nonzero(a & b)/(np.count_nonzero(a)+np.count_nonzero(b))


class MembraneRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_membrane_fit_improves_synthetic_boundary(self):
        raw,initial,truth,settings = fixture()
        result,opts,stats,rendered,slices = refine(raw,initial,[1,1,1],settings)
        self.assertLess(stats['finalLoss'],stats['initialLoss'])
        self.assertLess(stats['finalMembraneLoss'],stats['initialMembraneLoss'])
        self.assertGreater(dice(result > 0,truth),dice(initial > 0,truth)+.03)
        self.assertEqual(set(np.unique(result)),{0,900001})
        self.assertEqual(result.shape,initial.shape)
        self.assertTrue(np.isfinite(rendered).all())
        self.assertFalse(stats['rejectedCells'])
        self.assertGreater(stats['changedVoxels'],0)

    def test_shape_guards_detect_holes_handles_and_roughness(self):
        _,labels,_,_ = fixture(); original=labels > 0
        hole=original.copy();hole[12,12,12]=False
        tunnel=original.copy();tunnel[:,12,12]=False
        opts=options(dict(maxChangedFraction=2.,maxRoughnessIncrease=1.))
        for candidate in (hole,tunnel):
            failures,after=quality_failures(original,candidate,[1,1,1],opts)
            self.assertIn('new topology defects',failures)
        rough=original.copy();rough[12,12,18:22]=True
        failures,_=quality_failures(original,rough,[1,1,1],options(dict(maxRoughnessIncrease=0.,maxChangedFraction=2.)))
        self.assertIn('excess surface roughness',failures)

    def test_conservative_defaults_keep_topology_on_irregular_anisotropic_cell(self):
        grid=np.indices((16,24,24));mask=((grid[0]-8)**2*4+(grid[1]-12)**2+(grid[2]-12)**2)<49
        mask[7:10,11:14,17:21]=True
        labels=np.where(mask,17,0).astype(np.uint32)
        # A deliberately misleading interior signal must not punch holes.
        raw=ndi.gaussian_filter(np.random.default_rng(4).random(mask.shape),.5).astype(np.float32)
        raw[8,12,12]=10
        output,opts,stats,_,_=refine(raw,labels,[2,1,1],dict(iterations=10,device='cpu'))
        failures,_=quality_failures(mask,output == 17,[2,1,1],opts)
        self.assertEqual(failures,[])
        self.assertLessEqual(np.count_nonzero(mask != (output == 17)),mask.sum()*.1)

    def test_anisotropic_competing_cells_preserve_ids_and_motion_bound(self):
        raw,initial,truth,settings = fixture()
        initial[:6,8:16,8:16] = 7000000
        original = initial.copy()
        result,_,stats,_,_ = refine(raw,initial,[2,1,1],dict(settings,iterations=4,ellipsoidWeight=.02))
        self.assertEqual(set(np.unique(result)),{0,900001,7000000})
        np.testing.assert_array_equal(initial,original)
        for label in (900001,7000000):
            distance = ndi.distance_transform_edt(initial != label,sampling=[2,1,1])
            self.assertLessEqual(distance[result == label].max(),settings['maxDisplacement']+1)
        self.assertTrue(np.isfinite(stats['finalLoss']))

    def test_volume_guard_and_invalid_inputs(self):
        raw,initial,truth,settings = fixture()
        result,_,stats,_,_ = refine(raw,initial,[1,1,1],dict(settings,maxVolumeChange=.01))
        self.assertEqual(stats['rejectedCells'],[900001])
        np.testing.assert_array_equal(result,initial)
        for change in ({'iterations':0},{'psfZ':float('nan')},{'device':'bad'},{'iterations':2.5}):
            with self.assertRaises(ValueError): options(change)
        with self.assertRaisesRegex(ValueError,'no cells'): refine(raw,np.zeros_like(initial),[1,1,1],settings)
        with self.assertRaisesRegex(ValueError,'contrast'): refine(np.zeros_like(raw),initial,[1,1,1],settings)
        with self.assertRaisesRegex(ValueError,'voxel spacing'): refine(raw,initial,[0,1,1],settings)

    def test_saved_refinement_preserves_parent_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw,initial,truth,settings = fixture()
            image = root/'raw.tif'; tifffile.imwrite(image,raw,photometric='minisblack',metadata={'axes':'ZYX'})
            source = prepare(image,[1,1,1],root/'source')
            parent_folder = root/'original'; parent_folder.mkdir()
            parent = publish(initial,source,parent_folder,'cellpose',dict(model='cpsam_v2'),0,'fixture',extras=dict(cellposeModel='cpsam_v2'))
            digest = fingerprint(parent_folder/'labels_native.tif')
            child = refine_run(source,parent,dict(settings,iterations=3),root/'child')
            self.assertEqual(fingerprint(parent_folder/'labels_native.tif'),digest)
            self.assertEqual(child['method'],'refined')
            self.assertEqual(child['refinement']['parentID'],parent['id'])
            self.assertEqual(child['source'],parent['source'])
            self.assertEqual(child['spacing'],parent['spacing'])
            self.assertEqual(json.loads((root/'child/parent.json').read_text())['labelHash'],digest)
            with self.assertRaisesRegex(ValueError,'already refined'):
                refine_run(source,child,settings,root/'again')
            with (parent_folder/'labels_native.tif').open('ab') as handle: handle.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Parent labels changed'):
                refine_run(source,parent,settings,root/'changed')


if __name__ == '__main__': unittest.main()
