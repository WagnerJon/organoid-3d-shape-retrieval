import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tifffile
import zarr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from atlas_ome_zarr import discover,inspect_store,selection_entry,read_slice,materialize,inspect_entry
from prepare_atlas_dataset import validate,build

class OMEZarrTests(unittest.TestCase):
    def fixture(self,root,version=3,units='micrometer',permuted=False):
        path=root/'image.ome.zarr';group=zarr.open_group(str(path),mode='w',zarr_format=version)
        z,y,x=np.indices((16,24,28));mask=np.zeros(z.shape,np.uint16)
        mask[(z-8)**2+(y-12)**2+(x-10)**2<25]=1
        mask[(z-8)**2+(y-12)**2+(x-19)**2<16]=2
        data=np.stack([np.stack([mask*100+t,mask*200+t]) for t in range(3)])
        axes=['t','c','z','y','x']
        if permuted:axes=['c','t','x','z','y'];data=data.transpose(1,0,4,2,3)
        scale=dict(t=1,c=1,z=2000 if units=='nanometer' else 2,y=500 if units=='nanometer' else .5,x=500 if units=='nanometer' else .5)
        specs=[]
        for level in range(2):
            d=data if level==0 else np.take(np.take(data,np.arange(0,24,2),axis=axes.index('y')),np.arange(0,28,2),axis=axes.index('x'))
            kwargs=dict(shape=d.shape,dtype=d.dtype,chunks=tuple(1 if a in 'tc' else min(8,d.shape[i]) for i,a in enumerate(axes)))
            if version==3:kwargs['dimension_names']=axes
            group.create_array(str(level),**kwargs)[:]=d
            specs.append(dict(path=str(level),coordinateTransformations=[dict(type='scale',scale=[scale[a]*(2 if level and a in 'yx' else 1) for a in axes]),dict(type='translation',translation=[0]*5)]))
        metadata=dict(multiscales=[dict(axes=[dict(name=a,**(dict(type='space',unit=units) if a in 'zyx' else dict(type='time' if a=='t' else 'channel'))) for a in axes],datasets=specs)],omero=dict(channels=[dict(label='mem9'),dict(label='H2B')]))
        group.attrs.update(dict(ome=metadata,dca=dict(acquisition=dict(timestamps=[100.,107.,130.])))) if version==3 else group.attrs.update(metadata)
        labels=group.create_group('labels');labels.attrs.update(dict(ome=dict(labels=['cell','lumen']))) if version==3 else labels.attrs.update(dict(labels=['cell','lumen']))
        for name in ['cell','lumen']:
            label=labels.create_group(name);datasets=[]
            for level in range(2):
                m=mask if level==0 else mask[:,::2,::2]
                label.create_array(str(level),shape=(3,)+m.shape,dtype=m.dtype,chunks=(1,8,8,8))[:]=np.stack([m,m,m])
                datasets.append(dict(path=str(level),coordinateTransformations=[dict(type='scale',scale=[1,scale['z'],scale['y']*(2 if level else 1),scale['x']*(2 if level else 1)])]))
            md=dict(multiscales=[dict(axes=[dict(name=a,**(dict(unit=units) if a in 'zyx' else {})) for a in 'tzyx'],datasets=datasets)],**{'image-label':{}})
            label.attrs.update(dict(ome=md)) if version==3 else label.attrs.update(md)
        return path,data,mask,axes

    def test_v3_v2_metadata_units_channel_level_and_time_selection(self):
        for version in [2,3]:
            with tempfile.TemporaryDirectory() as tmp:
                path,data,_,_=self.fixture(Path(tmp),version,units='nanometer')
                image=inspect_store(path);self.assertEqual(image['channels'],['mem9','H2B'])
                e=selection_entry(image,2,1,1,'cell');self.assertEqual(e['shape'],[16,12,14]);self.assertEqual(e['spacing'],[2,1,1])
                for plane,index,expected in [('xy',5,data[2,1,5,::2,::2]),('xz',5,data[2,1,:,10,::2]),('yz',5,data[2,1,:,::2,10])]:
                    actual,_=read_slice(e,plane,index);np.testing.assert_array_equal(actual,expected)
                with self.assertRaisesRegex(ValueError,'semantic'):selection_entry(image,0,0,0,'lumen')
                with self.assertRaisesRegex(ValueError,'channel'):selection_entry(image,3,0,0)

    def test_missing_calibration_defaults_to_cubic_depth_at_selected_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,data,_,_=self.fixture(root,units='')
            image=inspect_store(path)
            self.assertIsNone(image['levels'][0]['spacing'])
            entry=selection_entry(image,0,0,0)
            self.assertEqual(entry['spacing'],[28/16,1.,1.])
            self.assertIn('assume cubic stack',entry['calibrationSource'])
            coarse=selection_entry(image,0,0,1)
            self.assertEqual(coarse['spacing'],[14/16,1.,1.])
            converted=materialize(entry,root/'unit-default')
            from prepare_atlas_dataset import inspect
            self.assertEqual(inspect(converted['path'])['spacing'],[28/16,1.,1.])
            np.testing.assert_array_equal(tifffile.imread(converted['path']),data[0,0])

    def test_axis_metadata_controls_slicing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,data,_,axes=self.fixture(Path(tmp),permuted=True);e=selection_entry(inspect_store(path),1,1,0)
            actual,_=read_slice(e,'xy',8)
            np.testing.assert_array_equal(actual,data[1,1,:,8,:].T)

    def test_materialization_is_calibrated_exact_and_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,data,mask,_=self.fixture(root);e=selection_entry(inspect_store(path),2,1,1,'cell')
            output=root/'import';entry=materialize(e,output)
            np.testing.assert_array_equal(tifffile.imread(entry['path']),data[2,1,:,::2,::2])
            np.testing.assert_array_equal(tifffile.imread(entry['labels']),mask[:,::2,::2])
            self.assertEqual(entry['spacing'],[2,1,1]);self.assertEqual(json.loads((output/'source.json').read_text())['source']['omeZarr']['timestamp'],130)
            before=Path(entry['path']).stat().st_mtime_ns;materialize(e,output);self.assertEqual(before,Path(entry['path']).stat().st_mtime_ns)
            with (output/'raw.ome.tif').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):materialize(e,output)

    def test_changed_source_duplicates_and_projection_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,_,_=self.fixture(root);e=selection_entry(inspect_store(path),0,0,0)
            req=dict(title='check',inputs=[e,e],settings=dict(model='cpsam_v2',scale=.2,diameter=30,gpu=False,background=True))
            with self.assertRaisesRegex(ValueError,'more than once'):validate(req)
            req['inputs']=[e,selection_entry(inspect_store(path),1,0,0)];validate(req)
            group=zarr.open_group(str(path),mode='a');group['0'][0,0,0,0,0]=123
            with self.assertRaisesRegex(ValueError,'changed'):inspect_entry(e)

    def test_streams_just_selected_frame_and_keeps_atlas_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,_,_=self.fixture(root)
            image=inspect_store(path);entries=[selection_entry(image,t,0,0,'cell') for t in [0,2]]
            req=dict(title='timelapse',inputs=entries,settings=dict(model='cpsam_v2',scale=.2,diameter=30,gpu=False,background=True),harmonics=False,excludeDetached=False)
            atlas=build(req,root/'prepared');data=json.loads((atlas/'dataset.json').read_text())
            self.assertEqual(len(data['samples']),2);self.assertEqual(len(data['cells']),4)
            self.assertEqual([s['sourceOMEZarr']['time'] for s in data['samples']],[0,2])
            self.assertTrue(all(s['sourceOMEZarr']['channel']==0 for s in data['samples']))
            self.assertTrue(all(Path(s['sourceImage']).is_file() for s in data['samples']))
            build(req,root/'prepared')
            with self.assertRaisesRegex(ValueError,'dedicated results folder'):build(req,path/'output')

if __name__=='__main__':unittest.main()
