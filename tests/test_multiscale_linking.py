"""Numerical and seam controls for the offline linking experiment."""
from pathlib import Path
import sys
import tempfile
import unittest
import json
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_standalone_cellpose  # Pinned runtime/cache configuration.
import multiscale_linking_core as core


class StreamingLinkTests(unittest.TestCase):
    def test_global_components_cross_slab_edges_and_connectivity(self):
        import cc3d
        from scipy.optimize import linear_sum_assignment
        rng = np.random.default_rng(41)
        source = rng.random((21, 23, 25)) > .77
        for connectivity in (18, 26):
            with tempfile.TemporaryDirectory() as tmp:
                result = core.components_disk(source, Path(tmp)/'c.npy', connectivity, depth=3)
                expected = cc3d.connected_components(source, connectivity=connectivity)
                n, m = int(expected.max())+1, int(result.max())+1
                table = np.bincount((expected.astype('int64')*m+result).ravel(), minlength=n*m).reshape(n, m)
                rows, cols = linear_sum_assignment(-table)
                self.assertEqual(int(table[rows, cols].sum()), source.size)
                self.assertEqual(n, m)
                minimum = core.components_disk(source, Path(tmp)/'small.npy', connectivity, minimum=11, depth=4)
                counts = np.bincount(expected.ravel())
                np.testing.assert_array_equal(minimum > 0, (expected > 0) & (counts[expected] >= 11))

    def test_halo_filters_and_noninteger_area_resampling(self):
        from skimage.transform import resize_local_mean
        from skimage import morphology
        rng = np.random.default_rng(901); data = rng.random((19, 23, 29)).astype('float32')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = core.gaussian_disk(data, root/'g.npy', [1.3, .5, 1.1], depth=3)
            np.testing.assert_array_equal(actual, ndi.gaussian_filter(data, [1.3, .5, 1.1]))
            for operation in ('closing', 'dilation'):
                result = core.morph_disk(data > .8, root/'m.npy', operation, depth=3)
                np.testing.assert_array_equal(result, getattr(morphology, 'binary_'+operation)(data > .8, morphology.ball(1)))
            for shape in [(10, 12, 15), (13, 16, 20), data.shape]:
                actual = core.area_reduce(data, root/'r.npy', shape, depth=3)
                np.testing.assert_allclose(actual, resize_local_mean(data, shape, preserve_range=True), atol=2e-6)

    def test_fusion_and_batched_dynamics_match_pinned_upstream(self):
        from segment3D import parameters, usegment3d
        from scipy.optimize import linear_sum_assignment
        z, y, x = np.indices((37, 29, 31)); centers = [(18, 14, 10), (18, 14, 22)]
        distances = np.stack([sum((a-c)**2 for a, c in zip((z, y, x), center)) for center in centers])
        owner = np.argmin(distances, axis=0)
        flow = np.stack([np.take(np.array(centers)[:, axis], owner)-coord for axis, coord in enumerate((z, y, x))]).astype('float32')
        probability = np.clip(3.-4.*distances.min(0)/8**2, -5., 3.).astype('float32')
        params = parameters.get_2D_to_3D_aggregation_params()
        params['gradient_descent'].update(n_iter=30, gradient_decay=.05, do_mp=False)
        params['postprocess_binary'].update(remove_small_objects=10)
        components = [(1, 2), (0, 2), (0, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); predictions = root/'predictions'; predictions.mkdir()
            probabilities, flows = [], []
            for view, axes in zip(['xy', 'xz', 'yz'], components):
                p = probability.copy(); f = flow[list(axes)].copy()
                np.save(predictions/f'{view}_prob.npy', p); np.save(predictions/f'{view}_flow.npy', f)
                probabilities.append(p); flows.append(f)
            expected, (p, field) = usegment3d.aggregate_2D_to_3D_segmentation_direct_method(probabilities, flows, params)
            actual = root/'streamed'
            core.fuse_predictions(predictions, actual, params, depth=3)
            np.testing.assert_array_equal(np.load(actual/'probability.npy'), p)
            np.testing.assert_array_equal(np.load(actual/'gradient.npy'), field.transpose(3, 0, 1, 2))
            core.trajectories(actual, params['gradient_descent'], depth=3, batch=71)
            core.cluster_endpoints(actual, sigma=1, depth=3)
            result = np.load(actual/'labels.npy')
            n, m = int(expected.max())+1, int(result.max())+1
            table = np.bincount((expected.astype('int64')*m+result).ravel(), minlength=n*m).reshape(n, m)
            rows, cols = linear_sum_assignment(-table)
            self.assertEqual(int(table[rows, cols].sum()), result.size)

    def test_vector_coordinate_transform_preserves_physical_displacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fine = root/'fine'; fine.mkdir()
            field = np.ones((3, 18, 24, 30), dtype='float32')
            np.save(fine/'gradient.npy', field); np.save(fine/'binary.npy', np.ones(field.shape[1:], bool))
            coarse = root/'coarse'; ratios = core.coarsen(fine, coarse, (9, 8, 15))
            actual = np.load(coarse/'gradient.npy')
            fine_spacing = np.array([.7, .8, .9]); coarse_spacing = fine_spacing/ratios
            np.testing.assert_allclose(actual[:, 2, 3, 4]*coarse_spacing, fine_spacing, rtol=1e-6)

    def test_joint_local_relinking_can_split_a_merge_without_overlaps(self):
        z,y,x=np.indices((41,51,65));centers=np.array([[20,25,20],[20,25,44]])
        distance=np.stack([sum((a-c)**2 for a,c in zip((z,y,x),center)) for center in centers])
        binary=distance.min(0)<8**2;owner=distance.argmin(0)
        field=np.stack([np.take(centers[:,a],owner)-coord for a,coord in enumerate((z,y,x))]).astype('float32')
        field/=np.linalg.norm(field,axis=0)[None]+1e-20
        initial=binary.astype('uint32')
        settings=dict(delta=1.,momenta=.98,n_iter=80,gradient_decay=.05,interp=False,renorm_gradient=False)
        result,next_id,details=core.refine_neighborhood(binary,field,initial,settings,1)
        self.assertEqual(set(np.unique(result)),{0,1,2})
        self.assertEqual(next_id,2)
        np.testing.assert_array_equal(result>0,binary)
        self.assertGreater(details['changedVoxels'],1000)

    def test_grid_audit_uses_actual_sampling_with_floor_rounded_dimensions(self):
        from benchmark_multiscale_linking import reference_info
        spacing=np.array([.75,.22872330084318845,.22872330084318845])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'run.json').write_text(json.dumps(dict(spacingZYXum=spacing.tolist())))
            (root/'processing.json').write_text(json.dumps(dict(processedShape=[120,119,119],effectiveSpacingZYXum=[.5,.5035756708,.5035756708],parameters=dict(factor=float(spacing.min()/.5),voxel_res=(spacing/spacing.min()).tolist()))))
            (root/'aggregation.json').write_text(json.dumps(dict(parameters={})))
            before=(root/'processing.json').read_bytes()
            cfg,info,_=reference_info(root)
            np.testing.assert_allclose(info['effectiveSpacingZYXum'],[.5,.5,.5])
            np.testing.assert_allclose(cfg['physicalOriginZYXum'],(.5-spacing)/2)
            self.assertEqual((root/'processing.json').read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
