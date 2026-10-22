"""Convex diagonal spectral metric; no encoder, decoder or learned preprocessing."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from spectral64_core import ROOT, sha, save_json, powers
from spatial_benchmark_core import morphology, CONTROLS

OUT = ROOT / 'results/spectral_metric_D'
SOURCE = ROOT / 'results/spectral_autoencoder64'
DIM = 416


def folds(frame):
    """Similarity groups are stricter blocks than organoids in this dataset."""
    result = []
    outer = GroupKFold(5)
    for fold, (development, test) in enumerate(outer.split(frame, groups=frame.similarity_group)):
        sub = frame.iloc[development]
        # Choose usable validation coverage using only groups and fixed morphometry.
        # A139-cell block otherwise leaves one partition almost without eligible pairs.
        choices = list(GroupKFold(4).split(sub, groups=sub.similarity_group))
        coverage = [len(pools(frame, development[v], development[t])) for t, v in choices]
        train, validation = choices[int(np.argmax(coverage))]
        result.append(dict(fold=fold, train=development[train].tolist(),
                           validation=development[validation].tolist(),
                           development=development.tolist(), test=test.tolist()))
    return result


def controls(frame, fit_ids):
    x = morphology(frame, CONTROLS)
    return (x - x[fit_ids].mean(0)) / x[fit_ids].std(0).clip(1e-6)


def pools(frame, ids, fit_ids, matched=True):
    ids = np.asarray(ids); x = controls(frame, fit_ids)
    distance = cdist(x[ids], x[ids]); groups = frame.similarity_group.to_numpy()
    result = {}
    for local, i in enumerate(ids):
        ok = groups[ids] != groups[i]
        if matched: ok &= distance[local] <= 1
        candidates = ids[ok]
        if matched:
            if len(candidates) < 5: continue
            candidates = candidates[np.argsort(distance[local, ok], kind='stable')[:20]]
        if len(candidates): result[int(i)] = candidates
    return result


def training_pairs(frame, ids, seed):
    """At most96 balanced anchors; eight nearby controls plus four random partners."""
    rng = np.random.default_rng(seed); ids = np.asarray(ids)
    groups = frame.similarity_group.to_numpy(); x = controls(frame, ids)
    queues = {int(g): list(rng.permutation(ids[groups[ids] == g])) for g in np.unique(groups[ids])}
    anchors = []
    while len(anchors) < min(96, len(ids)):
        for g in rng.permutation(list(queues)):
            if queues[g]: anchors.append(queues[g].pop())
            if len(anchors) == min(96, len(ids)): break
    counts = {g: sum(groups[i] == g for i in anchors) for g in np.unique(groups[ids])}
    pairs = {}
    for i in anchors:
        allowed = ids[groups[ids] != groups[i]]
        near = allowed[np.argsort(np.linalg.norm(x[allowed] - x[i], axis=1), kind='stable')[:8]]
        remaining = np.setdiff1d(allowed, near)
        random = rng.choice(remaining, min(4, len(remaining)), replace=False)
        partners = np.r_[near, random]
        weight = 1 / (len(counts) * counts[groups[i]] * len(partners))
        for j in partners:
            key = tuple(sorted((int(i), int(j))))
            pairs[key] = pairs.get(key, 0.) + weight
    return [dict(i=i, j=j, weight=w) for (i, j), w in sorted(pairs.items())]


def smooth_edges():
    index = np.arange(DIM).reshape(32, 13)
    a = np.r_[index[:-1].ravel(), index[:, :-1].ravel()]
    b = np.r_[index[1:].ravel(), index[:, 1:].ravel()]
    return a, b


def objective(theta, x, y, pair_weight, noise_energy, ridge, smooth, noise):
    w = theta[:-1]; intercept = theta[-1]
    residual = x @ w + intercept - y
    value = np.dot(pair_weight, residual ** 2) + ridge * np.mean((w - 1) ** 2)
    grad = 2 * (x.T @ (pair_weight * residual)) + 2 * ridge * (w - 1) / DIM
    a, b = smooth_edges(); delta = w[a] - w[b]
    value += smooth * np.mean(delta ** 2) + noise * np.dot(noise_energy, w)
    np.add.at(grad, a, 2 * smooth * delta / len(a))
    np.add.at(grad, b, -2 * smooth * delta / len(a))
    grad += noise * noise_energy
    return float(value), np.r_[grad, 2 * np.dot(pair_weight, residual)]


def fit_metric(features, noisy, frame, fit_ids, pairs, labels, params):
    ids = np.asarray(fit_ids); allowed = set(ids.tolist())
    assert all(p['i'] in allowed and p['j'] in allowed for p in pairs)
    i = np.array([p['i'] for p in pairs]); j = np.array([p['j'] for p in pairs])
    x = np.square(features[i] - features[j]); x_scale = np.median(x.sum(1)).clip(1e-12)
    y = np.array([labels[tuple(sorted((int(a), int(b))))] for a, b in zip(i, j)])
    y_scale = np.median(y).clip(1e-12); pw = np.array([p['weight'] for p in pairs]); pw /= pw.sum()
    # Average physical perturbation variance equally across training groups.
    g = frame.iloc[ids].similarity_group.to_numpy()
    e = np.square(noisy[ids] - features[ids])
    e = np.mean([e[g == k].mean(0) for k in np.unique(g)], axis=0) / x_scale
    # The convex objective is quadratic. Build it once and diagonally precondition
    # the solver: outer shells have very different numerical signal magnitudes.
    # This changes only optimizer coordinates, never descriptor/distance scaling.
    design = np.c_[x / x_scale, np.ones(len(x))]
    target = y / y_scale
    hessian_half = design.T @ (pw[:, None] * design)
    linear = design.T @ (pw * target)
    hessian_half[np.arange(DIM), np.arange(DIM)] += params['ridge'] / DIM
    linear[:DIM] += params['ridge'] / DIM - .5 * params['noise'] * e
    a, b = smooth_edges(); smooth = params['smooth'] / len(a)
    np.add.at(hessian_half, (a, a), smooth); np.add.at(hessian_half, (b, b), smooth)
    np.add.at(hessian_half, (a, b), -smooth); np.add.at(hessian_half, (b, a), -smooth)
    constant = float(np.dot(pw, target ** 2) + params['ridge'])
    precondition = np.sqrt(np.diag(hessian_half)).clip(1e-12)
    h = hessian_half / precondition[:, None] / precondition[None, :]
    linear_scaled = linear / precondition
    def quadratic(z):
        product = h @ z
        return float(z @ product - 2 * linear_scaled @ z + constant), 2 * (product - linear_scaled)
    result = minimize(quadratic, np.r_[np.ones(DIM), 0.] * precondition,
        jac=True, method='L-BFGS-B',
        bounds=[(0., float(20*s)) for s in precondition[:-1]] + [(0., None)],
        options=dict(maxiter=10000, ftol=1e-12, gtol=1e-8, maxls=40, maxcor=30))
    if not result.success: raise RuntimeError(f'Metric optimization failed: {result.message}')
    theta = result.x / precondition; w = theta[:-1]
    exact, _ = objective(theta, x / x_scale, target, pw, e, params['ridge'], params['smooth'], params['noise'])
    if abs(exact - result.fun) > 1e-8: raise RuntimeError('Quadratic objective mismatch')
    if w.mean() < 1e-8: raise RuntimeError('Collapsed metric')
    return dict(weights=w, intercept=float(theta[-1]), x_scale=float(x_scale),
                y_scale=float(y_scale), noise_energy=e, objective=float(result.fun),
                iterations=int(result.nit), parameters=params)


def distance(features, query, gallery, weights=None):
    weights = np.ones(DIM) if weights is None else np.asarray(weights)
    return cdist(features[query] * np.sqrt(weights), features[gallery] * np.sqrt(weights), 'sqeuclidean')


def adjusted_features(frame, features, fit_ids):
    x = controls(frame, fit_ids)
    poly = PolynomialFeatures(2, include_bias=False)
    design = poly.fit_transform(x)
    model = Ridge(alpha=1).fit(design[fit_ids], features[fit_ids])
    return features - model.predict(design)


def geometry_rows(features, weights, candidate_pools, labels, frame):
    rows = []
    for i, candidates in candidate_pools.items():
        d = distance(features, [i], candidates, weights)[0]
        j = int(candidates[np.argmin(d)])
        costs = np.array([labels[tuple(sorted((i, int(k))))] for k in candidates])
        cost = labels[tuple(sorted((i, j)))]
        rows.append(dict(query=i, cell_id=frame.iloc[i].cell_id, group=int(frame.iloc[i].similarity_group),
                         neighbor=j, neighbor_cell=frame.iloc[j].cell_id, geometry_cost=cost,
                         oracle_cost=float(costs.min()), regret=float(cost - costs.min()),
                         in_best_quartile=float(cost <= np.quantile(costs, .25)), candidates=len(candidates)))
    return rows


def noise_rows(features, noisy, ids, weights, frame, matched_pools=None):
    ids = np.asarray(ids); w = np.ones(features.shape[1]) if weights is None else weights
    d = cdist(noisy[ids] * np.sqrt(w), features[ids] * np.sqrt(w), 'sqeuclidean')
    clean = distance(features, ids, ids, w)
    rows = []
    for local, i in enumerate(ids):
        ranking = np.argsort(d[local], kind='stable')
        permitted = np.flatnonzero(frame.iloc[ids].similarity_group.to_numpy() != frame.iloc[i].similarity_group)
        k = min(5, len(permitted))
        before = set(permitted[np.argsort(clean[local, permitted], kind='stable')[:k]])
        after = set(permitted[np.argsort(d[local, permitted], kind='stable')[:k]])
        rows.append(dict(query=int(i), cell_id=frame.iloc[i].cell_id, group=int(frame.iloc[i].similarity_group),
                         noise_correct=float(ranking[0] == local), noise_top5=float(local in ranking[:5]),
                         neighbor_jaccard=float(len(before & after) / max(len(before | after), 1))))
    return rows


def group_mean(rows, key):
    f = pd.DataFrame(rows)
    return float(f.groupby('group')[key].mean().mean())


def cluster_ci(values, groups, weighted=False):
    f = pd.DataFrame(dict(value=values, group=groups)).dropna()
    a = f.groupby('group').value.agg(['mean', 'sum', 'count'])
    ix = np.random.default_rng(73).integers(0, len(a), (4000, len(a)))
    if weighted:
        boot = a['sum'].to_numpy()[ix].sum(1) / a['count'].to_numpy()[ix].sum(1)
        mean = float(f.value.mean())
    else:
        boot = a['mean'].to_numpy()[ix].mean(1); mean = float(a['mean'].mean())
    return dict(mean=mean, ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                cells=len(f), groups=len(a))
