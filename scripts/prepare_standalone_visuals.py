"""Prepare original-mesh turntables and an organoid-to-UMAP story for a new stack."""
import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from render_morphology_umap import BG, PALETTE, font, read_mesh
from atlas_shape_space import distances
from spatial_benchmark_core import surface_frame


def rgb(color):
    return [int(color[k:k+2], 16)/255 for k in (1, 3, 5)]


def positions(xy, rect):
    rect = np.asarray(rect)
    scale = min(rect[2:]/np.maximum(np.ptp(xy, axis=0), 1e-9))
    p = (xy-(xy.min(0)+xy.max(0))/2)*scale
    p[:, 1] *= -1
    return p+rect[:2]+rect[2:]/2


def save_mesh(path, v, f):
    normals = np.zeros_like(v)
    triangles = v[f]
    face = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    for k in range(3):
        np.add.at(normals, f[:, k], face)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    path.write_bytes(struct.pack('<II', len(v), len(f))+v.astype('<f4').tobytes()+normals.astype('<f4').tobytes()+f.astype('<u4').tobytes())


def main(dataset, output, story_method='spatial_robust'):
    from umap import UMAP
    path = dataset/'dataset.json'
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    data = json.loads(payload)
    keys = ['morph_all', 'spatial_robust']
    story_index = keys.index(story_method)
    story_name = 'Traditional nine-feature morphometry' if story_method == 'morph_all' else 'Frozen spatial shape descriptors'
    story_distance = 'full standardized nine-feature Euclidean distances' if story_method == 'morph_all' else 'full hybrid distances'
    cells = sorted([c for c in data['cells'] if all(k in c['representations'] for k in keys)], key=lambda c: c['label'])
    flagged = sum(not c['qc'] for c in cells)
    output.mkdir(parents=True, exist_ok=True)
    meshes = output/'meshes'
    meshes.mkdir(exist_ok=True)
    points, matrices = [], []
    for key in keys:
        metric = next(m['metric'] for m in data['methods'] if m['id'] == key)
        d = distances([c['representations'][key]['vector'] for c in cells], metric)
        fingerprint = hashlib.sha256(d.tobytes()).hexdigest()
        cache = output/f'{key}_embedding.npz'
        if cache.exists() and str(np.load(cache)['fingerprint']) == fingerprint:
            xy = np.load(cache)['xy']
        else:
            print(f'Fitting {key} UMAP for {len(cells)} objects', flush=True)
            xy = UMAP(n_components=2, n_neighbors=min(30, len(cells)-1), min_dist=.3,
                      metric='precomputed', random_state=42, n_jobs=1).fit_transform(d)
            np.savez(cache, xy=xy, fingerprint=fingerprint)
        points.append(xy.astype(float))
        matrices.append(d)
    with (output/'coordinates.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['cell_id', 'qc_pass', 'traditional_umap1', 'traditional_umap2', 'spatial_umap1', 'spatial_umap2'])
        for i, c in enumerate(cells):
            writer.writerow([c['id'], c['qc'], *points[0][i], *points[1][i]])
    geometry, original, centers, radii, rotations, hashes = [], [], [], [], [], {}
    for i, c in enumerate(cells):
        v, f, mesh_hash = read_mesh(dataset/c['sample']/f"{c['label']}.mesh")
        hashes[c['id']] = mesh_hash
        center = (v.min(0)+v.max(0))/2
        radius = max(np.linalg.norm(v-center, axis=1).max(), 1e-12)
        _, axes, _, _ = surface_frame(v, f)
        for k in (0, 1):
            canonical = (v-center)@axes
            if canonical[np.argmax(np.abs(canonical[:, k])), k] < 0:
                axes[:, k] *= -1
                axes[:, 2] *= -1
        local = (v-center)/radius
        source_mesh = meshes/f"{c['sample']}_L{c['label']}_source.mesh"
        canonical_mesh = meshes/f"{c['sample']}_L{c['label']}_canonical.mesh"
        save_mesh(source_mesh, local, f)
        save_mesh(canonical_mesh, local@axes, f)
        geometry.append((source_mesh, canonical_mesh))
        original.append(v)
        centers.append(center)
        radii.append(radius)
        rotations.append(Rotation.from_matrix(Rotation.from_euler('xy', [.32, -.4]).as_matrix()@axes.T).as_quat().tolist())
    width, height = 1920, 1200
    turn = output/'turntable'
    turn.mkdir(exist_ok=True)
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, width, 200), fill=BG)
    draw.rectangle((0, 1080, width, height), fill=BG)
    draw.text((60, 26), 'GFP 24 h  /  MV #5', font=font(18, True), fill='#62dcd4')
    for x, title, subtitle in [(60, 'Traditional morphometry', 'All 9 size + shape features'),
                               (1020, 'Spatial shape descriptors', 'Frozen spatial PCA + ambiguity fallback')]:
        draw.text((x, 68), title, font=font(36, True), fill='#f4f6fa')
        draw.text((x, 121), subtitle, font=font(19), fill='#aab9c9')
    draw.text((60, 162), f'{len(cells)} segmented objects in each map · {flagged} flagged for review · same cell identities', font=font(17), fill='#aab9c9')
    draw.text((60, 1101), 'Original segmented surfaces · fixed UMAP positions · full 360° turn', font=font(19), fill='#dbe5ef')
    draw.text((60, 1137), 'Display size normalized · colors show sphericity · exploratory geometry, not cell types', font=font(17), fill='#aab9c9')
    draw.text((1460, 1102), 'SPHERICITY', font=font(15, True), fill='#aab9c9')
    for x in range(350):
        color = tuple(int(t*255) for t in PALETTE(x/349)[:3])
        draw.line((1460+x, 1134, 1460+x, 1147), fill=color)
    draw.text((1460, 1155), '0.55', font=font(13), fill='#aab9c9')
    draw.text((1774, 1155), '0.95', font=font(13), fill='#aab9c9')
    overlay.save(turn/'overlay.png')
    nodes = []
    for j, xy in enumerate(points):
        pos = positions(xy, [65+j*960, 225, 830, 810])
        near = cdist(pos, pos)
        np.fill_diagonal(near, np.inf)
        sizes = np.clip(near.min(1)*.42, 7, 22)
        for i, c in enumerate(cells):
            nodes.append(dict(id=c['id'], mesh=str(geometry[i][1]), x=float(pos[i, 0]-width/2),
                y=float(height/2-pos[i, 1]), radius=float(sizes[i]),
                color=list(PALETTE(float(np.clip((c['metrics']['sphericity']-.55)/.4, 0, 1)))[:3])))
    (turn/'scene.json').write_text(json.dumps(dict(width=width, height=height, fps=20, frames=120,
        nodes=nodes, output=str(turn/'frames'), datasetSHA256=digest), indent=2))
    story = output/'organoid_story'
    story.mkdir(exist_ok=True)
    (story/'overlays').mkdir(exist_ok=True)
    # Pick a narrative query before reading its neighbor scores, using only QC
    # and closeness to the QC cohort's median physical cell volume.
    eligible = [i for i, c in enumerate(cells) if c['qc']]
    median = np.median([cells[i]['metrics']['volume_um3'] for i in eligible])
    qi = min(eligible, key=lambda i: (abs(cells[i]['metrics']['volume_um3']-median), cells[i]['id']))
    nearest = sorted([i for i in eligible if i != qi], key=lambda i: (matrices[story_index][qi, i], cells[i]['id']))[:3]
    pos = positions(points[story_index], [120, 225, 1680, 640])
    cloud = np.concatenate(original)
    orgcenter = (cloud.min(0)+cloud.max(0))/2
    orgscale = 310/np.linalg.norm(cloud-orgcenter, axis=1).max()
    colors = ['#77c9cb', '#aaa0ee', '#e49bb7', '#85b5eb', '#b8d293', '#d0aaec', '#89cdb1', '#e2bc82']
    nodes = []
    for i, c in enumerate(cells):
        match = nearest.index(i)+1 if i in nearest else 0
        color = '#f6a26e' if i == qi else ['#74ded5', '#b3a0ef', '#8fc3f3'][match-1] if match else colors[i % len(colors)]
        nodes.append(dict(id=c['id'], mesh=str(geometry[i][0]), source=True, sourceOrder=i,
            query=i == qi, match=match, color=rgb(color), organoidPosition=((centers[i]-orgcenter)*orgscale).tolist(),
            organoidRadius=float(radii[i]*orgscale), target=[float(pos[i, 0]-960), float(540-pos[i, 1]), 0.],
            targetRadius=22. if i == qi or match else 14., quaternion=rotations[i]))
    spec = dict(width=1920, height=1080, fps=20, frames=320, nodes=nodes, output=str(story/'frames'),
                method=story_method,
                query=cells[qi]['id'], neighbors=[dict(id=cells[i]['id'], distance=float(matrices[story_index][qi, i])) for i in nearest],
                querySelection='QC-passing cell nearest the QC median volume, before viewing neighbor scores',
                matchSelection='Three nearest QC-passing objects within the same stack, ranked by '+story_distance)
    (story/'scene.json').write_text(json.dumps(spec, indent=2))
    for frame in range(320):
        t = frame/20
        if t < 2 or t > 15.2:
            stage, title, subtitle = 0, 'Reconstruct the organoid', f'{len(cells)} segmented surfaces · original physical positions · {flagged} flagged for review'
        elif t < 3.5:
            stage, title, subtitle = 1, 'Meet the individual cells', 'Separate the surfaces while keeping each cell’s identity'
        elif t < 8:
            stage, title, subtitle = 2, 'Find each cell’s place in shape space', f'{story_name} · UMAP fitted on all {len(cells)} objects'
        elif t < 13:
            stage, title, subtitle = 3, 'Find similar shapes within the organoid', f"Query L{cells[qi]['label']} · nearest QC-passing matches · {'all 9 traditional features' if story_method == 'morph_all' else 'full spatial descriptors'}"
        else:
            stage, title, subtitle = 0, 'Every shape keeps its context', 'Return to the original reconstruction · unchanged cell geometry'
        image = Image.new('RGBA', (1920, 1080), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1920, 177), fill=BG)
        draw.rectangle((0, 950, 1920, 1080), fill=BG)
        draw.text((65, 30), 'GFP 24 h  /  '+('TRADITIONAL MORPHOMETRY ATLAS' if story_method == 'morph_all' else 'FROM ORGANOID TO SHAPE ATLAS'), font=font(18, True), fill='#6bdad4')
        draw.text((60, 67), title, font=font(44, True), fill='#f4f6fa')
        draw.text((65, 133), subtitle, font=font(21), fill='#aab9c9')
        for j, name in enumerate(['RECONSTRUCT', 'SEPARATE', 'MAP SHAPES', 'FIND MATCHES']):
            x = 65+j*460
            color = '#6bdad4' if j == stage else '#3b4a5d'
            draw.rounded_rectangle((x, 984, x+410, 989), radius=2, fill=color)
            draw.text((x, 1005), f'0{j+1}  {name}', font=font(17, True), fill=color)
        draw.text((65, 1043), 'Actual segmented surfaces · normalized display sizes in the map · illustrative transitions, not cell motion', font=font(16), fill='#8194a9')
        if 10.9 <= t < 13:
            for j, i in enumerate([qi]+nearest):
                cx = 300+j*440
                draw.rounded_rectangle((cx-195, 298, cx+195, 878), radius=22, outline='#627891', width=2)
                draw.text((cx-165, 326), 'QUERY' if j == 0 else f'MATCH 0{j}', font=font(17, True), fill='#f6a26e' if j == 0 else '#73d9d2')
                draw.text((cx-165, 774), f"Cell L{cells[i]['label']}", font=font(29, True), fill='#eef3fa')
                draw.text((cx-165, 824), 'GFP 24 h · same organoid', font=font(18), fill='#a6b9cd')
        image.save(story/'overlays'/f'{frame:04d}.png')
    provenance = dict(datasetSHA256=digest, sourceDataset=str(path), meshes=hashes,
        objects=len(cells), flagged=flagged, methods=keys, n_neighbors=30, min_dist=.3, seed=42,
        geometry='Original surfaces, proper rigid rotations and uniform display scaling only; no smoothing or morphing',
        inclusion='All reconstructed objects, including flagged fragments; these are not 186 validated cells',
        measurements='Original-dataset frozen feature scaling; comparison-only median imputation for 21 missing solidity measurements',
        story={k: spec[k] for k in ['method', 'query', 'neighbors', 'querySelection', 'matchSelection']})
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    print(f'Prepared {len(cells)} objects, {flagged} flagged; query {cells[qi]["id"]}; output {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--story-method', choices=['morph_all', 'spatial_robust'], default='spatial_robust')
    args = parser.parse_args()
    main(args.dataset, args.output.resolve(), args.story_method)
