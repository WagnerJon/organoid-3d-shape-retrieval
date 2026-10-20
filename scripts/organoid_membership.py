"""Non-destructive membership of the largest connected organoid in each label stack.

26-neighbour connectivity on the native label grid, without filling, dilation or
surface smoothing. A cell belongs when a strict majority of its labelled voxels
is in the largest tissue component. Ambiguous/disconnected cells remain reviewable.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import numpy as np
import tifffile
from scipy import ndimage as ndi


def classify(labels):
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('Expected a nonnegative integer ZYX label stack')
    components, _ = ndi.label(labels > 0, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    if not sizes.any():
        return {}
    main = int(sizes.argmax())
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    inside_ids, inside_counts = np.unique(labels[components == main], return_counts=True)
    inside = dict(zip(inside_ids, inside_counts))
    return {int(label): dict(inMainOrganoid=bool(inside.get(label, 0) > count / 2),
                            mainOrganoidFraction=float(inside.get(label, 0) / count))
            for label, count in zip(ids, counts)}


def included_labels(labels):
    membership = classify(labels)
    return np.where(np.isin(labels, [k for k, v in membership.items() if v['inMainOrganoid']]), labels, 0)


def project(records, key, metric=None):
    """Fit on included records only; never use excluded cells in descriptive fits."""
    if not records:
        return [0., 0.]
    if metric == "precomputed":
        from harmonics_cajal_core import project_pairwise
        return project_pairwise(records, key)
    if metric == "shape_hybrid95":
        from atlas_shape_space import project_hybrid
        return project_hybrid(records, key)
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    vectors = np.array([r['representations'][key]['vector'] for r in records], dtype=float)
    if not np.isfinite(vectors).all():
        raise ValueError('Non-finite representation')
    if len(records) == 1 or np.max(np.ptp(vectors, axis=0)) < 1e-12:
        points = np.zeros((len(records), 2)); variance = [0., 0.]
    else:
        pca = PCA(n_components=min(2, *vectors.shape), svd_solver='full').fit(vectors)
        points = np.pad(pca.transform(vectors), ((0, 0), (0, 2 - pca.n_components_)))
        variance = np.pad(pca.explained_variance_ratio_, (0, 2 - pca.n_components_)).tolist()
    groups = KMeans(n_clusters=min(6, len(np.unique(vectors, axis=0))), random_state=42, n_init=10).fit_predict(vectors)
    for record, point, group in zip(records, points, groups):
        record['representations'][key].update(point=point.tolist(), cluster=int(group))
    return variance


def rebuild(data):
    included = [c for c in data['cells'] if c.get('inMainOrganoid', True)]
    features = data['traditionalFeatures']
    if included:
        values = np.array([[c['metrics'][f] for f in features] for c in included])
        mean, std = values.mean(0), np.maximum(values.std(0), 1e-12)
        data['normalization'] = dict(mean=mean.tolist(), std=std.tolist())
        for c in data['cells']:
            c['representations']['traditional']['vector'] = ((np.array([c['metrics'][f] for f in features])-mean)/std).tolist()
    stacks = []
    for original in data['organoids']:
        cells = [c for c in included if c['sample'] == original['sample']]
        if not cells:
            continue
        qc = [c for c in cells if c['qc']]
        stack = dict(original, metrics={f: float(np.mean([c['metrics'][f] for c in (qc or cells)])) for f in features},
                     representations={}, cellCount=len(cells), qc=bool(qc), flags='Main organoid only; shape means use QC-passing cells')
        stack['metrics']['volume_um3'] = sum(c['metrics']['volume_um3'] for c in cells)
        stacks.append(stack)
    for method in data['methods']:
        key = method['id']
        method['variance'] = project([c for c in included if key in c['representations']], key, method['metric'])
        if method.get('scope') == 'cells' or method['metric'] in ['shape_hybrid95', 'precomputed']:
            method['stackVariance'] = [0., 0.]
            continue  # A mean of ambiguity flags / sign frames is not an organoid embedding.
        for stack in stacks:
            cells = [c for c in included if c['sample'] == stack['sample'] and c['qc'] and key in c['representations']]
            if cells:
                vector = np.mean([c['representations'][key]['vector'] for c in cells], axis=0)
                if method['metric'] == 'cosine':
                    vector /= max(np.linalg.norm(vector), 1e-12)
                stack['representations'][key] = dict(vector=vector.tolist(), point=[0., 0.], cluster=0)
        method['stackVariance'] = project([s for s in stacks if key in s['representations']], key)
    data['organoids'] = stacks
    data['membershipAnalysis'] = dict(version=1, rule='Largest 26-connected foreground component on native grid; strict majority of each cell; manual overrides take precedence.',
                                      included=len(included), excluded=len(data['cells'])-len(included))
    # Keep the complete stack inventory even when a reviewer excludes its last cell.
    data['provenance'] = data['provenance'].split(' Main-organoid membership:')[0].replace('descriptive full-cohort fits', 'descriptive included-cohort fits').replace('stack table volume sums all cells', 'stack table volume sums included cells') + ' Main-organoid membership: detached cells are retained for review but excluded from feature normalization, PCA, clustering, neighbors and stack aggregates. Membership uses native labels, never altered surfaces. Existing learned embeddings are retained; models are not retrained. Stack volume sums included cells only.'
    return data


def update(folder, overrides=None):
    path = folder / 'dataset.json'
    data = json.loads(path.read_text())
    # Preserve empty stacks as templates for future re-inclusion.
    templates = data.setdefault('membershipStackTemplates', data['organoids'])
    data['organoids'] = templates
    by_sample = {s['id']: s for s in data['samples']}
    missing = sorted({c['sample'] for c in data['cells'] if 'automaticMainOrganoid' not in c})
    for n, sample in enumerate(missing):
        labels = tifffile.imread(by_sample[sample]['sourceLabels'])
        membership = classify(labels)
        for cell in data['cells']:
            if cell['sample'] == sample:
                if cell['label'] not in membership:
                    raise ValueError(f"Missing native label for {cell['id']}")
                cell.update(membership[cell['label']])
                cell['automaticMainOrganoid'] = cell['inMainOrganoid']
        print(f'Main organoid: {n+1}/{len(missing)} stacks · {sample}', flush=True)
    if overrides:
        for cell in data['cells']:
            if cell['id'] in overrides:
                value = overrides[cell['id']]
                cell['membershipOverride'] = value
                cell['inMainOrganoid'] = cell['automaticMainOrganoid'] if value is None else bool(value)
    rebuild(data)
    temp = path.with_suffix('.membership.tmp')
    temp.write_text(json.dumps(data, allow_nan=False, separators=(',', ':')))
    os.replace(temp, path)
    print(json.dumps(data['membershipAnalysis']), flush=True)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--overrides', type=Path)
    args = parser.parse_args()
    update(args.dataset, json.loads(args.overrides.read_text()) if args.overrides else None)
