import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from atlas_tiff import describe, selection_entry, inspect_entry, materialize
from atlas_slice_test import read_slice
from prepare_atlas_dataset import inspect, inspect_inputs, validate, build


class TIFFSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.path=self.root/'movie.ome.tif'
        self.data=np.arange(2*3*5*12*14,dtype=np.uint16).reshape(2,3,5,12,14)
        tifffile.imwrite(self.path,self.data,ome=True,compression='zlib',photometric='minisblack',
            metadata=dict(axes='TCZYX',PhysicalSizeZ=2.,PhysicalSizeY=.5,PhysicalSizeX=.5,Channel={'Name':['membrane','nuclei','other']}))
        self.image=describe(self.path)['images'][0]
        self.entry=selection_entry(self.image,1,2)
        self.settings=dict(model='cyto3',scale=.35,diameter=30.,gpu=False,background=False)
    def tearDown(self):self.temp.cleanup()
    def request(self,entries):return dict(title='TIFF selection test',inputs=entries,settings=self.settings,harmonics=False,excludeDetached=False)

    def test_catalog_and_explicit_selection_preserve_axis_identity(self):
        catalog=inspect_inputs([self.path]);self.assertEqual(catalog['inputs'],[])
        self.assertEqual(catalog['tiffImages'][0]['channels'],['membrane','nuclei','other'])
        self.assertEqual(self.entry['shape'],[5,12,14]);self.assertEqual(self.entry['spacing'],[2,.5,.5])
        self.assertEqual(inspect_entry(self.entry)['tiffSelection'],self.entry['tiffSelection'])
        with self.assertRaisesRegex(ValueError,'select a TIFF series'):inspect(self.path)
        validate(self.request([self.entry,selection_entry(self.image,0,2)]))
        with self.assertRaisesRegex(ValueError,'more than once'):validate(self.request([self.entry,self.entry]))
        for t,c in [(-1,0),(2,0),(0,3),(True,0)]:
            with self.assertRaisesRegex(ValueError,'outside'):selection_entry(self.image,t,c)

    def test_exact_orthogonal_planes_without_full_volume_decode(self):
        with patch('tifffile.imread',side_effect=AssertionError('Full volume read')),patch.object(tifffile.TiffPageSeries,'asarray',side_effect=AssertionError('Full series decode')):
            for view,axis,index in [('xy',0,3),('xz',1,7),('yz',2,9)]:
                actual,_=read_slice(self.entry,view,index)
                np.testing.assert_array_equal(actual,np.take(self.data[1,2],index,axis=axis))
        changed=copy.deepcopy(self.entry);changed['sourceStat'][0]+=1
        with self.assertRaisesRegex(ValueError,'changed'):read_slice(changed,'xy',0)
        changed=copy.deepcopy(self.entry);changed['tiffSelection']['axes']='TZCYX'
        with self.assertRaisesRegex(ValueError,'changed'):inspect_entry(changed)

    def test_streamed_export_exact_pixels_calibration_provenance_and_resume(self):
        before=self.path.read_bytes();out=self.root/'import'
        entry=copy.deepcopy(self.entry);entry['spacing']=[3.,.7,.7]
        with patch('tifffile.imread',side_effect=AssertionError('Full volume read')),patch.object(tifffile.TiffPageSeries,'asarray',side_effect=AssertionError('Full series decode')):
            selected=materialize(entry,out)
        self.assertNotIn('tiffSelection',selected)
        self.assertEqual(selected['spacing'],[3.,.7,.7])
        np.testing.assert_array_equal(tifffile.imread(selected['path']),self.data[1,2])
        source=json.loads((out/'source.json').read_text());self.assertEqual(source['selection']['time'],1)
        self.assertEqual(source['selection']['channel'],2);self.assertEqual(source['selection']['sourceAxes'],'TCZYX')
        self.assertEqual(self.path.read_bytes(),before)
        timestamp=Path(selected['path']).stat().st_mtime_ns
        self.assertEqual(materialize(entry,out),selected);self.assertEqual(Path(selected['path']).stat().st_mtime_ns,timestamp)
        with self.assertRaisesRegex(ValueError,'another selection'):materialize(selection_entry(self.image,0,2),out)
        with Path(selected['path']).open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'cache changed'):materialize(entry,out)

    def test_imagej_time_series_and_singleton_channels(self):
        path=self.root/'imagej.tif';data=np.arange(3*4*9*11,dtype=np.uint16).reshape(3,4,9,11)
        tifffile.imwrite(path,data,imagej=True,photometric='minisblack',metadata={'axes':'TZYX','unit':'um','spacing':1.5},resolution=(2,2))
        image=describe(path)['images'][0];self.assertEqual(image['timepoints'],3);self.assertEqual(image['spacing'],[1.5,.5,.5])
        entry=selection_entry(image,2,0);actual,_=read_slice(entry,'xy',2);np.testing.assert_array_equal(actual,data[2,2])
        path=self.root/'singleton.tif';tifffile.imwrite(path,data[:1,None],metadata={'axes':'TCZYX'},photometric='minisblack')
        image=describe(path)['images'][0];self.assertEqual(image['shapeZYX'],[4,9,11])
        actual,_=read_slice(selection_entry(image,0,0),'xy',3);np.testing.assert_array_equal(actual,data[0,3])

    def test_series_specific_calibration_and_rgb_sample_channels(self):
        path=self.root/'series.ome.tif'
        with tifffile.TiffWriter(path,ome=True) as w:
            for index in range(2):w.write(np.full((4,8,10),index+2,np.uint16),photometric='minisblack',metadata=dict(axes='ZYX',Name=f'Specimen {index}',PhysicalSizeZ=index+1.,PhysicalSizeY=.2,PhysicalSizeX=.2))
        images=describe(path)['images'];self.assertEqual(len(images),2);self.assertEqual(images[1]['spacing'],[2.,.2,.2])
        actual,_=read_slice(selection_entry(images[1],0,0),'xy',2);np.testing.assert_array_equal(actual,np.full((8,10),3))
        path=self.root/'rgb.tif';rgb=np.arange(4*8*10*3,dtype=np.uint16).reshape(4,8,10,3)
        tifffile.imwrite(path,rgb,photometric='rgb',metadata={'axes':'ZYXS'})
        image=describe(path)['images'][0];self.assertEqual(len(image['channels']),3)
        actual,_=read_slice(selection_entry(image,0,1),'xy',2);np.testing.assert_array_equal(actual,rgb[2,:,:,1])
        path=self.root/'unknown.tif';tifffile.imwrite(path,np.zeros((2,4,8,10),np.uint8),photometric='minisblack',metadata={'axes':'QZYX'})
        with self.assertRaisesRegex(ValueError,'Unsupported TIFF axes'):describe(path)

    def test_full_dataset_uses_selected_frames_not_time_as_depth(self):
        path=self.root/'biological.ome.tif';z,y,x=np.indices((12,20,24));labels=((z-6)**2+(y-10)**2+(x-12)**2<16).astype(np.uint16)
        raw=np.stack([labels*1000,labels*2000])
        tifffile.imwrite(path,raw,ome=True,photometric='minisblack',metadata=dict(axes='TZYX',PhysicalSizeZ=2.,PhysicalSizeY=.5,PhysicalSizeX=.5))
        mask=self.root/'labels.tif';tifffile.imwrite(mask,labels,photometric='minisblack',metadata={'axes':'ZYX'})
        image=describe(path)['images'][0];entries=[selection_entry(image,t,0) for t in range(2)]
        for entry in entries:entry.update(labels=str(mask),labelStat=[mask.stat().st_size,mask.stat().st_mtime_ns])
        numpy_mask=self.root/'labels.npy';np.save(numpy_mask,labels)
        entries[1].update(labels=str(numpy_mask),labelStat=[numpy_mask.stat().st_size,numpy_mask.stat().st_mtime_ns])
        request=self.request(entries);out=self.root/'analysis';atlas=build(request,out)
        dataset=json.loads((atlas/'dataset.json').read_text());self.assertEqual(len(dataset['samples']),2);self.assertEqual(len(dataset['cells']),2)
        self.assertEqual([s['sourceTIFF']['time'] for s in dataset['samples']],[0,1])
        for index in range(2):
            np.testing.assert_array_equal(tifffile.imread(out/'inputs'/f'Stack_{index+1:03d}'/'raw.ome.tif'),raw[index])
        self.assertEqual(build(request,out),atlas)
        (out/'complete.json').unlink();(atlas/'dataset.json').unlink()
        self.assertEqual(build(request,out),atlas)  # Reuse TIFF and NumPy-label caches after interrupted publication.

if __name__=='__main__':unittest.main()
