import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import tifffile
from segmentation_explorer import prepare,run,compare


class CellposeModelComparisonTests(unittest.TestCase):
    def test_model_dispatch_and_native_grid_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.arange(6*8*8,dtype=np.uint16).reshape(6,8,8)
            image = root/'raw.tif'; tifffile.imwrite(image,raw,metadata={'axes':'ZYX'},photometric='minisblack')
            source = prepare(image,[1,.2,.2],root/'source')
            labels = np.zeros(raw.shape,dtype=np.uint32); labels[:,2:6,2:6] = 3
            runs = []
            for index,model in enumerate(['cpsam_v2','cpdino','cpdino-vitb']):
                def predict(path,config,folder,progress=None):
                    self.assertEqual(config['model'],model)
                    self.assertEqual(config['spacing_zyx_um'],source['spacing'])
                    return raw,np.where(labels > 0,index+1,0).astype(np.uint32),[1,.2,.2]
                with patch('reconstruct.predict',side_effect=predict):
                    result = run(source,'cellpose',dict(model=model,scale=.15,diameter=30,gpu=False,background=False),root/model,comparison_side='A' if index == 0 else 'B')
                self.assertEqual(result['method'],'cellpose')
                self.assertEqual(result['cellposeModel'],model)
                self.assertEqual(result['comparisonSide'],'A' if index == 0 else 'B')
                self.assertIn(model,result['provenance'])
                self.assertEqual(result['settings']['model'],model)
                runs.append(result)
            for b in runs[1:]:
                stats = compare(runs[0],b,source)
                self.assertEqual(stats['matched'],1)
                self.assertEqual(stats['dice'],1)

    def test_invalid_model_never_runs_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/'raw.tif'; tifffile.imwrite(image,np.zeros((4,5,6),dtype=np.uint16),photometric='minisblack',metadata={'axes':'ZYX'})
            source = prepare(image,[1,1,1],root/'source')
            with patch('reconstruct.predict') as predict, self.assertRaisesRegex(ValueError,'model must'):
                run(source,'cellpose',dict(model='invalid',scale=.15,diameter=30,gpu=False,background=True),root/'bad')
            predict.assert_not_called()
            self.assertFalse((root/'bad/result.json').exists())


if __name__ == '__main__':
    unittest.main()
