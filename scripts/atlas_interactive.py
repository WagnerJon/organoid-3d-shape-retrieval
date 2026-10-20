"""Read-only, cached 3-axis projections and full-distance shape-group discovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA


def comparison_distances(vectors, metric):
    x = np.asarray(vectors, dtype=float)
    if metric in ('shape_hybrid95', 'precomputed'):
        from atlas_shape_space import distances
        d = distances(x, metric)
    elif metric == 'cosine':
        norms = np.linalg.norm(x, axis=1)
        if np.any(norms[:, None] * norms[None, :] <= 1e-12):
            raise ValueError('Cosine grouping requires non-zero comparison vectors')
        z = x / np.maximum(norms[:, None], 1e-12)
        d = 1 - z @ z.T
    elif metric == 'euclidean':
        d = cdist(x, x)
    else:
        raise ValueError('Unsupported comparison distance: ' + metric)
    d = np.maximum((d + d.T) / 2, 0.)
    np.fill_diagonal(d, 0.)
    if not np.isfinite(d).all():
        raise ValueError('Non-finite shape distances')
    return d


def project_three(x, metric):
    if metric in ('shape_hybrid95', 'precomputed'):
        d = comparison_distances(x, metric)
        b = -.5 * d ** 2
        b -= b.mean(0)[None, :] + b.mean(1)[:, None] - b.mean()
        values, vectors = np.linalg.eigh(b)
        order = np.argsort(values)[::-1][:3]
        positive = np.maximum(values[order], 0.)
        points = vectors[:, order] * np.sqrt(positive)
        variance = positive / max(np.maximum(values, 0).sum(), 1e-12)
        kind = 'mds'
    else:
        # Same unmodified descriptors as the existing display PCA; no whitening.
        k = min(3, *x.shape)
        if len(x) < 2 or np.max(np.ptp(x, axis=0)) < 1e-12:
            points = np.zeros((len(x), k)); variance = np.zeros(k)
        else:
            fit = PCA(n_components=k, svd_solver='full').fit(x)
            points = fit.transform(x); variance = fit.explained_variance_ratio_
        kind = 'pca'
    return dict(points=np.pad(points, ((0, 0), (0, 3 - points.shape[1]))).tolist(),
                variance=np.pad(variance, (0, 3 - len(variance))).tolist(), projection=kind)


def discover(d, ids, minimum=10, samples=5, selection='eom', allow_single=False):
    if selection not in ('eom', 'leaf'):
        raise ValueError('Unknown group selection: ' + str(selection))
    if not isinstance(allow_single, bool):
        raise ValueError('allowSingle must be a boolean')
    n = len(ids)
    minimum, samples = max(2, int(minimum)), max(1, int(samples))
    # Identical representations have no shape separation; avoid arbitrary
    # branches caused by tied zero-distance edges in the spanning tree.
    if n < max(minimum, samples + 1) or not np.any(np.asarray(d) > 0):
        labels = np.full(n, -1, dtype=int); probabilities = np.zeros(n)
    else:
        fit = HDBSCAN(min_cluster_size=minimum, min_samples=samples,
                      metric='precomputed', cluster_selection_method=selection,
                      allow_single_cluster=allow_single, copy=True).fit(d)
        labels, probabilities = fit.labels_, fit.probabilities_
    groups = []
    for label in sorted(set(labels) - {-1}):
        members = np.flatnonzero(labels == label)
        medoid = int(members[np.argmin(d[np.ix_(members, members)].mean(1))])
        nearest = sorted((int(i) for i in members if i != medoid), key=lambda i: (d[medoid, i], ids[i]))
        groups.append(dict(label=int(label), members=[ids[i] for i in members], representative=ids[medoid],
                           nearest=[dict(id=ids[i], distance=float(d[medoid, i])) for i in nearest[:5]],
                           membershipStrength=float(probabilities[members].mean())))
    return dict(labels=labels.tolist(), strengths=probabilities.tolist(), groups=groups,
                unassigned=int((labels < 0).sum()), minimum=minimum, samples=samples,
                selection=selection, allowSingle=allow_single)


def analyze(data, request):
    method = next(m for m in data['methods'] if m['id'] == request['method'])
    level = request.get('level', 'Cells')
    if level != 'Cells' and (method.get('scope') == 'cells' or method['metric'] in ('shape_hybrid95', 'precomputed')):
        raise ValueError('This method has cell-level results only')
    wanted = None if request.get('ids') is None else set(request['ids'])
    records = [r for r in data['cells' if level == 'Cells' else 'organoids']
               if r.get('inMainOrganoid', True) and method['id'] in r['representations']
               and (wanted is None or r['id'] in wanted)]
    ids = [r['id'] for r in records]
    if not ids:
        raise ValueError('No included objects have this method’s results in this cohort')
    x = np.array([r['representations'][method['id']]['vector'] for r in records], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite representation')
    result = dict(ids=ids, method=method['id'], level=level, metric=method['metric'])
    if request['action'] == 'projection':
        result.update(project_three(x, method['metric']))
    elif request['action'] == 'groups':
        result.update(discover(comparison_distances(x, method['metric']), ids,
                               request.get('minimum', 10), request.get('samples', 5),
                               request.get('selection', 'eom'), request.get('allowSingle', False)))
    else:
        raise ValueError('Unknown analysis')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    dataset = Path(request.pop('dataset')).resolve() / 'dataset.json'
    stat = dataset.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    key = hashlib.sha256(json.dumps([str(dataset), stamp, request,
                                     hashlib.sha256(Path(__file__).read_bytes()).hexdigest()], sort_keys=True).encode()).hexdigest()
    cache = Path(__file__).resolve().parents[1] / '.cache/atlas-interactive'
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / (key + '.json')
    if cached.exists():
        args.output.write_bytes(cached.read_bytes())
        return
    result = analyze(json.loads(dataset.read_text()), request)
    current = dataset.stat()
    if stamp != (current.st_mtime_ns, current.st_size):
        raise ValueError('Dataset changed during analysis; please retry')
    payload = json.dumps(result, allow_nan=False, separators=(',', ':'))
    # Distinct temporary files also support concurrent windows on the same plot.
    temporary = cache / (key + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(payload); os.replace(temporary, cached)
    args.output.write_text(payload)


if __name__ == '__main__':
    main()
