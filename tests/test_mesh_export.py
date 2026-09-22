"""Check physical mesh geometry independently of Cellpose predictions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import trimesh

spec = importlib.util.spec_from_file_location("reconstruct", Path(__file__).resolve().parents[1] / "scripts/reconstruct.py")
reconstruct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconstruct)


class MeshExportTest(unittest.TestCase):
    def test_anisotropic_coordinates_and_orientation(self):
        labels = np.zeros((12, 14, 16), dtype=np.uint16)
        labels[2:8, 3:9, 4:10] = 7
        with tempfile.TemporaryDirectory() as directory:
            reconstruct.export(labels, np.array([2., 1., 0.5]), Path(directory))
            mesh = trimesh.load(Path(directory) / "cell_0007.ply")
            self.assertTrue(mesh.is_watertight)
            self.assertGreater(mesh.volume, 0)
            np.testing.assert_allclose(mesh.bounds, [[1.75, 2.5, 3], [4.75, 8.5, 15]])


if __name__ == "__main__":
    unittest.main()
