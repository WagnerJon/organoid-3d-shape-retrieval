"""Reproducible, self-contained synthetic shape gallery for Organoid Atlas.

Build geometry, then use atlas_missing_methods.py for real frozen inference.
Packaging includes only portable assets and precomputed analyses, never models,
training data, machine paths, inference caches or experimental datasets.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
from skimage import measure
import tifffile
import trimesh

from atlas_missing_methods import save

FAMILIES = ['Star', 'Sphere', 'Ellipsoid', 'Cactus', 'Sausage', 'Donut']
SHAPE = (64, 112, 160)
SPACING = .8
SEED = 20261023
FEATURES = ['volume_um3', 'surface_area_um2', 'sphericity', 'elongation', 'flatness',
            'major_axis_um', 'middle_axis_um', 'minor_axis_um', 'mesh_solidity']


def capsule(points, a, b, radius):
    a, b = np.asarray(a), np.asarray(b)
    direction = b - a
    t = np.clip(np.sum((points - a) * direction, axis=-1) / np.sum(direction ** 2), 0, 1)
    return np.linalg.norm(points - a - t[..., None] * direction, axis=-1) - radius


def field(name, points, variant):
    """Signed implicit fields in XYZ, with controlled changes across variants."""
    x, y, z = points.T
    t = variant / 11
    if name == 'Sphere':
        return np.linalg.norm(points, axis=1) - (10 + 1.6 * np.sin(variant * 1.7))
    if name == 'Ellipsoid':
        axes = np.array([14 + 3 * t, 9 - t, 6 + 1.5 * t])
        return (np.linalg.norm(points / axes, axis=1) - 1) * axes.min()
    if name == 'Donut':
        return np.sqrt((np.hypot(x, y) - (10.5 + 1.2 * t)) ** 2 + z ** 2) - (3.2 + t)
    if name == 'Sausage':
        # A gently bent capsule, with two rounded ends.
        curve = np.array([[u, (1.5 + 2 * t) * (u / 11) ** 2, 0] for u in np.linspace(-11, 11, 12)])
        return np.min([capsule(points, a, b, 4.8 + .7 * t) for a, b in zip(curve[:-1], curve[1:])], axis=0)
    if name == 'Cactus':
        trunk = capsule(points, [0, 0, -12], [0, 0, 12 + t], 4.2)
        left = np.array([[0, 0, -3], [-9 - t, 0, -3], [-9 - t, 0, 7 + 3 * t]])
        right = np.array([[0, 0, -7], [9, 0, -7], [9, 0, 3 + 2 * t]])
        segments = [capsule(points, a, b, 2.8 + .4 * t) for arm in (left, right) for a, b in zip(arm[:-1], arm[1:])]
        return np.minimum(trunk, np.min(segments, axis=0))
    if name == 'Star':
        from matplotlib.path import Path as PolygonPath
        angles = np.arange(10) * np.pi / 5 + np.pi / 2
        radii = np.where(np.arange(10) % 2 == 0, 17 + t, 7 + 1.8 * t)
        polygon = np.c_[np.cos(angles) * radii, np.sin(angles) * radii]
        xy = points[:, :2]
        distances = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            d = b - a
            u = np.clip(np.sum((xy - a) * d, axis=1) / (d @ d), 0, 1)
            distances.append(np.linalg.norm(xy - a - u[:, None] * d, axis=1))
        planar = np.min(distances, axis=0) * np.where(PolygonPath(polygon).contains_points(xy), -1, 1)
        q = np.c_[planar, np.abs(z) - (3.5 + t)]
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0) - .7
    raise ValueError(name)


def write_mesh(path, mesh):
    path.write_bytes(struct.pack('<II', len(mesh.vertices), len(mesh.faces))
                     + np.asarray(mesh.vertices, '<f4').tobytes()
                     + np.asarray(mesh.vertex_normals, '<f4').tobytes()
                     + np.asarray(mesh.faces, '<u4').tobytes())


def generate(folder, variants=12):
    folder = Path(folder).resolve()
    if (folder / 'dataset.json').exists():
        raise FileExistsError('Choose a new folder; existing demo data will not be overwritten')
    folder.mkdir(parents=True, exist_ok=True)
    data = dict(version=1, title='Demo · 3D Shape Gallery', syntheticDemo=True,
                demoVersion=1, demoShapes=FAMILIES, cells=[], organoids=[], samples=[],
                traditionalFeatures=FEATURES, methods=[dict(id='traditional', name='Traditional morphometry', metric='euclidean',
                dimensions=9, variance=[0., 0.], stackVariance=[0., 0.], family='traditional')],
                provenance='Synthetic demonstration, not biological data. Six procedurally generated shape families with size, proportion and orientation variations. '
                'Synthetic membrane-like intensities are derived from the known shape boundaries; labels are generated directly, not segmented by a model. '
                'Physical units are nominal, isotropic 0.8 µm voxels. Volume and principal axes use occupied voxels, surface area and solidity use closed implicit-surface meshes. '
                'Learned representations are frozen inference with existing biological encoders; their behavior on these artificial shapes is not a biological or segmentation benchmark. '
                'Known shape labels are display annotations only and do not train or determine method clusters. All demo objects are deliberately included, despite being spatially separated.')
    centers = np.array([[32, y, x] for y in (32, 80) for x in (32, 80, 128)])
    rng = np.random.default_rng(SEED)
    for variant in range(variants):
        sample = f'Gallery_{variant + 1:02d}'
        target, run = folder / sample, folder / 'sources' / sample
        target.mkdir(); run.mkdir(parents=True)
        labels = np.zeros(SHAPE, np.uint16)
        raw = np.full(SHAPE, 70, np.float32)
        permutation = rng.permutation(6)
        for label, name in enumerate(FAMILIES, 1):
            center = centers[permutation[label - 1]]
            lo, hi = center - 25, center + 26
            bbox = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            grid = np.indices((51, 51, 51)).reshape(3, -1).T + lo
            rotation = Rotation.from_euler('xyz', [.13 * np.sin(variant), .16 * np.cos(variant), variant * .37 + label * .13]).as_matrix()
            local = ((grid[:, ::-1] - center[::-1]) * SPACING) @ rotation
            sdf = field(name, local, variant).reshape((51,) * 3)
            mask = sdf < 0
            if ndimage.label(mask)[1] != 1 or any(mask.take(i, axis=a).any() for a in range(3) for i in (0, -1)):
                raise ValueError(f'Invalid or clipped {name} geometry')
            occupied = labels[bbox]
            if np.any(occupied[mask]):
                raise ValueError('Demo shapes overlap')
            occupied[mask] = label
            membrane = np.exp(-.5 * (sdf / .65) ** 2)
            raw[bbox] += 2100 * membrane + 100 * mask
            vertices, faces, _, _ = measure.marching_cubes(sdf, level=0, spacing=(SPACING,) * 3)
            vertices += lo * SPACING
            mesh = trimesh.Trimesh(vertices[:, ::-1], faces[:, ::-1], process=False)
            mesh.fix_normals(multibody=True)
            if not mesh.is_watertight or mesh.volume <= 0:
                raise ValueError('Invalid closed demo surface: ' + name)
            write_mesh(target / f'{label}.mesh', mesh)
            xyz = (np.argwhere(mask) + lo) * SPACING
            covariance = np.cov(xyz, rowvar=False, bias=True) + np.eye(3) * SPACING ** 2 / 12
            axes = np.sqrt(20 * np.linalg.eigvalsh(covariance)[::-1])
            volume, area = float(mask.sum() * SPACING ** 3), float(mesh.area)
            metrics = dict(volume_um3=volume, surface_area_um2=area, sphericity=float(np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area),
                           major_axis_um=float(axes[0]), middle_axis_um=float(axes[1]), minor_axis_um=float(axes[2]),
                           elongation=float(axes[0] / axes[1]), flatness=float(axes[1] / axes[2]),
                           mesh_solidity=float(mesh.volume / mesh.convex_hull.volume), demo_shape_class=label - 1,
                           demo_variant=variant + 1, demo_euler_characteristic=int(mesh.euler_number))
            metrics.update(zip(['centroid_z_um', 'centroid_y_um', 'centroid_x_um'], xyz.mean(0).tolist()))
            data['cells'].append(dict(id=f'{sample}:L{label}', name=f'{name} {variant + 1:02d}', sample=sample, label=label,
                qc=True, split='synthetic demo', flags=f'Synthetic {name.lower()}; known family {label - 1}; generated ground-truth geometry.',
                inMainOrganoid=True, automaticMainOrganoid=True, membershipOverride=True, mainOrganoidFraction=1.,
                metrics=metrics, representations={'traditional': dict(vector=[0.] * 9, point=[0., 0.], cluster=0)}))
        # Quantized low-amplitude noise keeps the small archive compressible.
        raw += rng.integers(0, 5, size=SHAPE, dtype=np.uint8) * 2
        raw = np.minimum(raw, 65535).astype(np.uint16)
        metadata = dict(axes='ZYX', PhysicalSizeZ=SPACING, PhysicalSizeY=SPACING, PhysicalSizeX=SPACING,
                        PhysicalSizeZUnit='µm', PhysicalSizeYUnit='µm', PhysicalSizeXUnit='µm')
        tifffile.imwrite(run / 'synthetic.ome.tif', raw, ome=True, metadata=metadata, photometric='minisblack')
        tifffile.imwrite(run / 'labels.ome.tif', labels, ome=True, metadata=metadata, photometric='minisblack')
        np.save(run / 'labels_processed.npy', labels.astype(np.uint32))
        save(run / 'run.json', dict(source=f'sources/{sample}/synthetic.ome.tif', nativeShape=SHAPE,
                                   spacingZYXum=[SPACING] * 3, model='procedural geometry; no segmentation', syntheticDemo=True))
        save(run / 'processing.json', dict(processedShape=SHAPE, effectiveSpacingZYXum=[SPACING] * 3,
                                          rawGridOffsetZYXum=[0., 0., 0.]))
        labels.astype('<u2').tofile(target / 'labels.u16')
        np.clip(raw / 2400 * 255, 0, 255).astype('u1').tofile(target / 'raw.u8')
        data['samples'].append(dict(id=sample, name=f'Synthetic gallery {variant + 1:02d}', shape=list(SHAPE),
            spacing=[SPACING] * 3, extent=((np.array(SHAPE) - 1) * SPACING).tolist(), rawAvailable=True,
            sourceImage=f'sources/{sample}/synthetic.ome.tif', sourceLabels=f'sources/{sample}/labels.ome.tif', meshCount=6))
        data['organoids'].append(dict(id=sample, name=f'Shape set {variant + 1:02d}', sample=sample, label=0, qc=True,
            split='synthetic demo', flags='Synthetic collection, not an organoid', cellCount=6, inMainOrganoid=True, metrics={}, representations={}))
        print(f'Generated {sample}: six watertight shapes', flush=True)
    data['membershipStackTemplates'] = copy.deepcopy(data['organoids'])
    from organoid_membership import rebuild
    rebuild(data)
    # Automatic biological membership classification does not apply to a gallery.
    data['provenance'] = data['provenance'].split(' Main-organoid membership:')[0]
    data['membershipAnalysis']['rule'] = 'Synthetic gallery: all six known objects are intentionally included in every set.'
    save(folder / 'dataset.json', data)
    save(folder / 'demo-generation.json', dict(seed=SEED, families=FAMILIES, variants=variants, shape=SHAPE,
         spacingZYXum=[SPACING] * 3, version=1, biologicalData=False, generatedLabels=True,
         scriptSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return data


def package(folder, archive):
    from atlas_interactive import analyze
    from atlas_missing_methods import inspect
    folder, archive = Path(folder).resolve(), Path(archive)
    if inspect(folder)['missing']:
        raise ValueError('Calculate all transferable methods before packaging the demo')
    data = json.loads((folder / 'dataset.json').read_text())
    # Publish self-contained PointNet reconstructions outside transient caches.
    import shutil
    for cell in data['cells']:
        for key, relative in cell.get('pointReconstructions', {}).copy().items():
            destination = Path('reconstructions') / key / cell['sample'] / f"{cell['label']}.points"
            (folder / destination).parent.mkdir(parents=True, exist_ok=True)
            if (folder / relative).resolve() != (folder / destination).resolve():
                shutil.copyfile(folder / relative, folder / destination)
            cell['pointReconstructions'][key] = str(destination)
    # Keep concise provenance without machine-specific cache paths/source lists.
    sources = data.pop('methodCompletion', [])
    data['demoInference'] = dict(methods=len(data['methods']), retrained=False, generatedLabels=True,
                                groups=[s['group'] for s in sources], biologicalValidation=False)
    save(folder / 'dataset.json', data)
    for level in ('Cells', 'Organoids'):
        for method in data['methods']:
            if level != 'Cells' and (method.get('scope') == 'cells' or method['metric'] in ('shape_hybrid95', 'precomputed')):
                continue
            for action in ('projection', 'groups'):
                request = dict(method=method['id'], level=level, action=action, minimum=10, samples=5, selection='eom', allowSingle=False)
                result = analyze(data, request)
                save(folder / 'demo-analysis' / level / f"{method['id']}.{action}.json", result)
    (folder / 'DEMO.md').write_text('''# Synthetic 3D Shape Gallery

72 generated objects: 12 variations each of Star, Sphere, Ellipsoid, Cactus,
Sausage and Donut. Twelve mixed-shape stacks support cross-stack neighbours.
The synthetic membrane images and exact generated labels are included. All
objects are deliberately included; this is not biological organoid membership.

All 36 transferable Atlas representations were calculated with the existing
frozen models and descriptors. Known family labels only control optional display
colours; exploratory groups come from each method's descriptors. These shapes
are outside the biological training distribution. This is a UI demonstration,
not a model-quality or biological benchmark.

The bundle includes meshes, nominal physical measurements, source stacks,
label previews, PointNet reconstructions, three-axis plots and default discovered
groups. Basic exploration needs no downloaded model or Python environment.
Custom group settings and new analyses need the normal analysis environment.
Demo copies are writable and changes do not alter the bundled original.
''')
    assets = [folder / 'dataset.json', folder / 'demo-generation.json', folder / 'DEMO.md']
    for directory in ['demo-analysis', 'reconstructions', 'sources'] + [s['id'] for s in data['samples']]:
        assets += sorted(p for p in (folder / directory).rglob('*') if p.is_file())
    inventory = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}
    payload = dict(files=inventory, cells=len(data['cells']), methods=len(data['methods']), version=1)
    demo_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    save(folder / 'demo-manifest.json', dict(payload, id=demo_id))
    assets.append(folder / 'demo-manifest.json')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in assets:
            entry = zipfile.ZipInfo(str(path.relative_to(folder)), date_time=(2026, 10, 23, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            bundle.writestr(entry, path.read_bytes())
    print(f'Packaged {len(assets)} assets, {len(data["cells"])} objects, {len(data["methods"])} methods, {archive.stat().st_size / 2**20:.1f} MiB', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['generate', 'package'])
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.operation == 'generate':
        generate(args.folder)
    else:
        package(args.folder, args.archive)
