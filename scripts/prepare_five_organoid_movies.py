"""Five original organoids flowing into one shared nine-feature UMAP."""
import csv
import colorsys
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from render_morphology_umap import BG, PALETTE, font, read_mesh
from prepare_standalone_visuals import positions, save_mesh, rgb
from spatial_benchmark_core import surface_frame
from atlas_shape_space import distances

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/Volumes/CHUNKY/Scientific_data/GFP-24hrs-2ndy-cleared-MV_all5_analysis/atlas_dataset')
OUT = ROOT/'results/five_organoid_movies'
SPHERICITY_RANGE = (.30, 1.00)

def cell_color(cell):
    value = float(cell['metrics']['sphericity'])
    return list(PALETTE(float(np.clip((value-SPHERICITY_RANGE[0])/(SPHERICITY_RANGE[1]-SPHERICITY_RANGE[0]), 0, 1))))[:3]

def main():
    from umap import UMAP
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'meshes').mkdir(exist_ok=True)
    payload = (DATA/'dataset.json').read_bytes()
    data = json.loads(payload)
    cells = sorted(data['cells'], key=lambda c: (c['sample'], c['label']))
    samples = sorted({c['sample'] for c in cells})
    assert len(cells) == 961 and len(samples) == 5
    d = distances([c['representations']['morph_all']['vector'] for c in cells], 'euclidean')
    fingerprint = hashlib.sha256(d.tobytes()).hexdigest()
    cache = OUT/'global_umap.npz'
    if cache.exists() and str(np.load(cache)['fingerprint']) == fingerprint:
        xy = np.load(cache)['xy']
    else:
        print('Fitting shared nine-feature UMAP on all 961 objects', flush=True)
        xy = UMAP(n_components=2, n_neighbors=30, min_dist=.3, metric='precomputed', random_state=42, n_jobs=1).fit_transform(d)
        np.savez(cache, xy=xy, fingerprint=fingerprint, ids=[c['id'] for c in cells])
    width, height, fps, frames = 1920, 1080, 20, 240
    display_xy = np.column_stack((xy[:, 1], -xy[:, 0]))  # Rigid 90-degree layout rotation; distances unchanged.
    pos = positions(display_xy, [105, 200, 1710, 700])
    near = cdist(pos, pos)
    np.fill_diagonal(near, np.inf)
    sizes = np.clip(near.min(1)*.43, 5, 15)
    geometry, bounds, hashes = [], {}, {}
    for i, c in enumerate(cells):
        v, f, digest = read_mesh(DATA/c['sample']/f"{c['label']}.mesh")
        hashes[c['id']] = digest
        lo, hi = v.min(0), v.max(0)
        center = (lo+hi)/2
        radius = max(np.linalg.norm(v-center, axis=1).max(), 1e-12)
        if c['sample'] not in bounds:
            bounds[c['sample']] = [lo, hi]
        else:
            bounds[c['sample']][0] = np.minimum(bounds[c['sample']][0], lo)
            bounds[c['sample']][1] = np.maximum(bounds[c['sample']][1], hi)
        _, axes, _, _ = surface_frame(v, f)
        for k in (0, 1):
            canonical = (v-center)@axes
            if canonical[np.argmax(abs(canonical[:, k])), k] < 0:
                axes[:, k] *= -1
                axes[:, 2] *= -1
        path = OUT/'meshes'/f"{c['sample']}_L{c['label']}.mesh"
        if not path.exists():
            save_mesh(path, (v-center)/radius, f)
        q = Rotation.from_matrix(Rotation.from_euler('xy', [.32, -.4]).as_matrix()@axes.T).as_quat().tolist()
        geometry.append((path, center, radius, q))
        if i%100 == 0:
            print(f'Prepared original surface {i+1}/961', flush=True)
    with (OUT/'global_umap_coordinates.csv').open('w') as h:
        w = csv.writer(h)
        w.writerow(['cell_id', 'organoid', 'qc_pass', 'UMAP1', 'UMAP2', 'screen_x', 'screen_y', 'sphericity', 'color_r', 'color_g', 'color_b'])
        for i, c in enumerate(cells):
            w.writerow([c['id'], c['sample'], c['qc'], *xy[i], *pos[i], c['metrics']['sphericity'], *cell_color(c)])
    for si, sample in enumerate(samples):
        folder = OUT/f'MV{si+1}'
        (folder/'overlays').mkdir(parents=True, exist_ok=True)
        lo, hi = bounds[sample]
        orgcenter = (lo+hi)/2
        orgscale = 330/(np.linalg.norm(hi-lo)/2)
        nodes = []
        order = 0
        for i, c in enumerate(cells):
            path, center, radius, q = geometry[i]
            source = c['sample'] == sample
            nodes.append(dict(id=c['id'], mesh=str(path), source=source, sourceOrder=order,
                color=cell_color(c), organoidColor=list(colorsys.hsv_to_rgb((c['label'] % 11)/11, .55, .83)), organoidPosition=((center-orgcenter)*orgscale).tolist(),
                organoidRadius=float(radius*orgscale), target=[float(pos[i, 0]-width/2), float(height/2-pos[i, 1]), 10. if source else -20.],
                targetRadius=float(sizes[i]), quaternion=q))
            order += int(source)
        spec = dict(width=width, height=height, fps=fps, frames=frames, nodes=nodes, output=str(folder/'frames'))
        (folder/'scene.json').write_text(json.dumps(spec))
        for frame in range(frames):
            t = frame/fps
            image = Image.new('RGBA', (width, height))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0,0,width,175), fill=BG)
            draw.rectangle((0,950,width,height), fill=BG)
            draw.text((64,28), f'ORGANOID ATLAS  /  MV #{si+1}', font=font(20, True), fill='#77ded3')
            title = 'A reconstructed organoid' if t<2 else 'Individual cells, separated' if t<3.5 else 'Finding their place in shape space' if t<7 else 'One shared atlas of all five organoids'
            draw.text((64,66), title, font=font(38,True), fill='#f4f6fa')
            draw.text((64,122), f'{order} cells from MV #{si+1}  →  961 segmented objects · all 9 traditional size + shape features', font=font(21), fill='#adbdcd')
            if t < 3.5:
                draw.text((64,970), 'INDIVIDUAL CELL COLOURS', font=font(18, True), fill='#dbe5ef')
                draw.text((500,970), 'Categorical label colours, as in Organoid Atlas', font=font(18), fill='#aab9c9')
            else:
                draw.text((64,970), 'CELL SPHERICITY', font=font(18, True), fill='#dbe5ef')
                for x in range(380):
                    color = tuple(int(v*255) for v in PALETTE(x/379)[:3])
                    draw.line((300+x,972,300+x,990), fill=color)
                draw.text((294,995), '0.30', font=font(14), fill='#aab9c9')
                draw.text((645,995), '1.00', font=font(14), fill='#aab9c9')
                caption = 'Cell identity colours → sphericity colours' if t < 7 else 'Sphericity colours in the shared UMAP'
                draw.text((750,970), caption, font=font(18), fill='#aab9c9')
            draw.text((64,1020), 'Original cell surfaces · UMAP fitted once to all 961 objects · 214 flagged objects retained', font=font(17), fill='#aab9c9')
            draw.text((1200,1020), 'Display sizes normalized in UMAP · illustrative motion', font=font(16), fill='#aab9c9')
            image.save(folder/'overlays'/f'{frame:04d}.png')
    manifest=dict(dataset=str(DATA), datasetSHA256=hashlib.sha256(payload).hexdigest(), method='morph_all',
        features=['volume','surface area','sphericity','elongation','flatness','solidity','major axis','middle axis','minor axis'],
        objects=len(cells), flagged=sum(not c['qc'] for c in cells), samples=samples,
        umap=dict(n_neighbors=30,min_dist=.3,metric='precomputed Euclidean',random_state=42,distanceSHA256=fingerprint),
        geometry='Original measured meshes, rigid transformations and uniform display scaling only; no simplification or deformation',
        organoidColorMapping=dict(method='Atlas categorical label palette', hue='(label % 11)/11', saturation=.55, brightness=.83, transition='Per-cell interpolation to sphericity synchronized with movement into UMAP'),
        colorMapping=dict(metric='sphericity', range=list(SPHERICITY_RANGE), clipping=True, palette='orange-lavender-teal', aboveRange=sum(c['metrics']['sphericity']>1 for c in cells)),
        sourceMeshSHA256=hashes, width=width,height=height,fps=fps,frames=frames)
    (OUT/'provenance.json').write_text(json.dumps(manifest, indent=2))
    print(f'Prepared five movies in {OUT}', flush=True)

if __name__ == '__main__':
    main()
