import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from atlas_numpy import materialize
from atlas_slice_test import read_slice
from prepare_atlas_dataset import inspect, validate, build


class NumPyImportTests(unittest.TestCase):
    def fixture(self, root, fortran=False, raw_numpy=True, labels_numpy=True):
        z, y, x = np.indices((18, 26, 32))
        labels = np.zeros(z.shape, np.uint16)
        labels[(z-8)**2+(y-12)**2+(x-10)**2 < 25] = 1
        labels[(z-8)**2+(y-12)**2+(x-21)**2 < 16] = 2
        raw = (z*100+y*10+x+labels*300).astype(np.uint16)
        path = root/('raw.npy' if raw_numpy else 'raw.ome.tif')
        mask = root/('labels.npy' if labels_numpy else 'labels.tif')
        if raw_numpy: np.save(path, np.asfortranarray(raw) if fortran else raw)
        else: tifffile.imwrite(path, raw, metadata={'axes':'ZYX'}, photometric='minisblack')
        if labels_numpy: np.save(mask, labels)
        else: tifffile.imwrite(mask, labels, metadata={'axes':'ZYX'}, photometric='minisblack')
        entry = inspect(path)
        entry.update(spacing=[2., .5, .75], labels=str(mask), labelStat=[mask.stat().st_size, mask.stat().st_mtime_ns])
        return entry, raw, labels

    def request(self, entry):
        return dict(title='NumPy cells', inputs=[entry], settings=dict(model='cpsam_v2', scale=.25, diameter=30., gpu=False, background=True),
                    harmonics=False, excludeDetached=False)

    def test_metadata_defaults_spacing_and_rejects_ambiguous_or_pickled_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry, _, _ = self.fixture(root)
            info = inspect(entry['path'])
            self.assertEqual(info['shape'], [18,26,32])
            self.assertEqual(info['axes'], 'ZYX')
            self.assertEqual(info['spacing'], [32/18,1.,1.])
            self.assertIn('assume cubic stack',info['calibrationSource'])
            self.assertIn('assumed Z/Y/X', info['calibrationSource'])
            request = self.request(dict(entry, spacing=None))
            with self.assertRaisesRegex(ValueError, 'positive Z/Y/X spacing'): validate(request)
            for number, value in enumerate([np.zeros((5,8)), np.zeros((1,5,8)), np.zeros((2,3,4,5)),
                                            np.zeros((2,3,4), dtype=complex), np.zeros((2,3,4), dtype=bool),
                                            np.full((2,3,4), {'unsafe':'never unpickle'}, dtype=object)]):
                path = root/f'invalid{number}.npy'; np.save(path, value)
                with self.assertRaises(ValueError): inspect(path)
            path = root/'archive.npy'
            with path.open('wb') as output: np.savez(output, image=np.ones((2,3,4)))
            with self.assertRaisesRegex(ValueError, 'single .npy array'): inspect(path)

    def test_all_planes_are_exact_detached_and_do_not_load_the_volume(self):
        for fortran in [False, True]:
            with tempfile.TemporaryDirectory() as tmp:
                entry, raw, _ = self.fixture(Path(tmp), fortran=fortran)
                original = np.load
                calls = []
                def bounded_load(*args, **kwargs):
                    calls.append(kwargs)
                    self.assertEqual(kwargs.get('mmap_mode'), 'r')
                    self.assertIs(kwargs.get('allow_pickle'), False)
                    return original(*args, **kwargs)
                with patch('numpy.load', bounded_load):
                    for plane, index, expected in [('xy',5,raw[5]), ('xz',6,raw[:,6]), ('yz',7,raw[:,:,7])]:
                        image, dimensions = read_slice(entry, plane, index)
                        np.testing.assert_array_equal(image, expected)
                        self.assertTrue(image.flags.owndata)
                        self.assertEqual(image.ndim, 2)
                        self.assertEqual(tuple(image.shape), tuple(raw.shape[i] for i in dimensions))
                self.assertTrue(calls)
                for plane,index in [('xy',18), ('zz',0), ('xy',True)]:
                    with self.assertRaises(ValueError): read_slice(entry, plane, index)

    def test_materialization_preserves_voxels_calibration_sources_and_resume(self):
        for raw_numpy, labels_numpy in [(True,True),(True,False),(False,True)]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); entry, raw, labels = self.fixture(root, fortran=raw_numpy and labels_numpy, raw_numpy=raw_numpy, labels_numpy=labels_numpy)
                before = {p:p.read_bytes() for p in [Path(entry['path']),Path(entry['labels'])]}
                converted = materialize(entry, root/'import')
                np.testing.assert_array_equal(tifffile.imread(converted['path']), raw)
                np.testing.assert_array_equal(tifffile.imread(converted['labels']), labels)
                self.assertEqual(converted['spacing'], [2.,.5,.75])
                if raw_numpy: self.assertEqual(inspect(converted['path'])['spacing'], [2.,.5,.75])
                metadata = json.loads((root/'import/source.json').read_text())
                self.assertEqual(metadata['source']['path'], entry['path'])
                self.assertIn('no squeezing', metadata['axisInterpretation'])
                self.assertEqual(len(metadata['sourcesSHA256']), int(raw_numpy)+int(labels_numpy))
                timestamps = {p:p.stat().st_mtime_ns for p in (root/'import').iterdir()}
                self.assertEqual(materialize(entry, root/'import'), converted)
                self.assertEqual(timestamps, {p:p.stat().st_mtime_ns for p in (root/'import').iterdir()})
                self.assertTrue(all(p.read_bytes()==value for p,value in before.items()))
                cached = root/'import'/('raw.ome.tif' if raw_numpy else 'labels.ome.tif')
                with cached.open('ab') as handle: handle.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'): materialize(entry, root/'import')

    def test_finite_label_range_source_changes_and_space_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); entry, _, labels = self.fixture(root)
            with patch('atlas_streamed_linking.shutil.disk_usage', return_value=type('Disk', (), {'free':0})()), \
                 patch('run_standalone_cellpose.digest', side_effect=AssertionError('must not read the full image before checking disk')):
                with self.assertRaisesRegex(ValueError, 'NumPy import needs'): materialize(entry, root/'no-space')
            self.assertFalse((root/'no-space').exists())
            materialize(entry, root/'cached')
            source = Path(entry['path']); original_stat = source.stat()
            array = np.load(source, mmap_mode='r+', allow_pickle=False); array[0,0,0] += 1; array.flush(); array._mmap.close()
            os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'source content changed'): materialize(entry, root/'cached')
            entry = inspect(source); entry['spacing']=[2.,.5,.75]
            bad = labels.astype(np.int32); bad[0,0,0]=-1
            np.save(root/'badlabels.npy', bad)
            p=root/'badlabels.npy'; entry.update(labels=str(p),labelStat=[p.stat().st_size,p.stat().st_mtime_ns])
            with self.assertRaisesRegex(ValueError, 'integers in 0–65535'): materialize(entry,root/'negative-labels')
            self.assertFalse((root/'negative-labels/labels.ome.tif.tmp').exists())
            entry.pop('labels'); entry.pop('labelStat')
            p=root/'nonfinite.npy'; raw=np.ones((4,6,8),np.float32);raw[2,1,1]=np.nan;np.save(p,raw)
            entry=inspect(p);entry['spacing']=[1,1,1]
            with self.assertRaisesRegex(ValueError, 'nonfinite'): materialize(entry,root/'nonfinite-cache')
            self.assertFalse((root/'nonfinite-cache/raw.ome.tif.tmp').exists())

    def test_interrupted_import_restarts_owned_cache_and_preserves_other_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); entry,raw,_=self.fixture(root)
            def cancel(label):
                if 'Z 4/' in label: raise InterruptedError('simulate cancel')
            with self.assertRaises(InterruptedError): materialize(entry, root/'cache', cancel)
            self.assertFalse((root/'cache/import.completed.json').exists())
            self.assertFalse((root/'cache/raw.ome.tif.tmp').exists())
            converted=materialize(entry,root/'cache')
            np.testing.assert_array_equal(tifffile.imread(converted['path']),raw)
            changed=copy.deepcopy(entry);changed['spacing']=[3,1,1]
            with self.assertRaisesRegex(ValueError, 'different inputs'): materialize(changed,root/'cache')
            (root/'cache/important.txt').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'unrelated files'): materialize(entry,root/'cache')
            self.assertEqual((root/'cache/important.txt').read_text(),'keep')

    def test_preparation_supplied_labels_exports_physical_cells_and_source_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entry,_,labels=self.fixture(root)
            request=self.request(entry)
            atlas=build(request,root/'prepared')
            data=json.loads((atlas/'dataset.json').read_text())
            self.assertEqual(len(data['cells']),2)
            source=data['samples'][0]['sourceNumPy']
            self.assertEqual(source['path'],entry['path'])
            self.assertEqual(source['labels'],entry['labels'])
            self.assertEqual(source['shape'],[18,26,32])
            self.assertEqual(source['spacingZYXum'],[2.,.5,.75])
            self.assertTrue(Path(data['samples'][0]['sourceImage']).is_file())
            for cell in data['cells']:
                self.assertAlmostEqual(cell['metrics']['volume_um3'], (labels==cell['label']).sum()*2*.5*.75)
                self.assertTrue((atlas/cell['sample']/f"{cell['label']}.mesh").is_file())
            before=(atlas/'dataset.json').read_bytes()
            self.assertEqual(build(request,root/'prepared'),atlas)
            self.assertEqual((atlas/'dataset.json').read_bytes(),before)


if __name__ == '__main__': unittest.main()
