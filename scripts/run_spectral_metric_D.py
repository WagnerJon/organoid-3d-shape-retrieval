"""Offline nested grouped Experiment D. Stages deliberately separate fitting from evaluation."""
from spectral_metric_core import *
import argparse, hashlib, sqlite3, time
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import cdist
from spectral64_core import canonical, field_coefficients, trimesh
from spatial_benchmark_core import mesh_for, geometry_cost
from pointnet_shape_core import sample_surface


def load():
    frame = pd.read_csv(OUT / 'manifest.csv')
    data = np.load(OUT / 'features.npz')
    return frame, data['clean'], data['noise'], json.loads((OUT / 'folds.json').read_text())


def config_grid():
    configs = []
    for ridge in [.001, .01, .1]:
        configs.append(dict(family='ridge', ridge=ridge, smooth=0., noise=0.))
        for smooth in [.01, .1]:
            configs.append(dict(family='smooth', ridge=ridge, smooth=smooth, noise=0.))
            for noise in [.01, .1]:
                configs.append(dict(family='robust', ridge=ridge, smooth=smooth, noise=noise))
    return configs


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'protocol.json').exists():
        print('Existing frozen protocol retained', flush=True); return
    frame = pd.read_csv(SOURCE / 'manifest.csv')
    assert frame.groupby('sample').similarity_group.nunique().max() == 1
    frame.to_csv(OUT / 'manifest.csv', index=False)
    plans = folds(frame); save_json(OUT / 'folds.json', plans)
    sets = []
    for f in plans:
        inner = training_pairs(frame, f['train'], 7200 + f['fold'])
        full = training_pairs(frame, f['development'], 8200 + f['fold'])
        validation = pools(frame, f['validation'], f['train'])
        sets.append(dict(fold=f['fold'], inner_pairs=inner, refit_pairs=full,
                         validation_pools={str(i): c.tolist() for i, c in validation.items()}))
    save_json(OUT / 'pair_plans.json', sets)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [SOURCE / 'manifest.csv', SOURCE / 'coefficients.npz',
              ROOT / 'results/explorer_dataset/dataset.json', SOURCE / 'protocol.json']}
    protocol = dict(date='2026-10-21', experiment='D: learn a robust spectral distance, no autoencoder',
        cohorts=dict(cells=len(frame), organoids=frame['sample'].nunique(), blocks=frame.similarity_group.nunique(),
                     note='Previously analyzed descriptive data, not a new untouched biological cohort. All learned weights are out-of-fold.'),
        representation='Existing32-shell ×13-degree normalized radial-weighted band norms;416 dimensions, no whitening or compression',
        evaluation='Five outer GroupKFold partitions by similarity group, which contains entire organoids. Of four inner grouped partitions, choose the validation partition with most eligible matched queries using group membership/morphometry only, before geometry labels or any performance outcomes. Refit chosen settings on all outer development cells. No current-fold test endpoint participates in any pair fit, nuisance fit, noise estimate or model selection.',
        labels='Same historical geometric proxy: proper-rotation24-start ICP,8 iterations,384 deterministic area-sampled points, original per-object radius normalization; bidirectional squared nearest sampled-surface distance, no scale or reflection fit. Geometric surrogate, not biological ground truth.',
        training_pairs='At most96 group-balanced anchors per fit, eight nearest morphometry-control partners plus four uniformly random other-group partners. All endpoints must be in the fit partition. Group-balanced pair loss; no synthetic training objects.',
        metric='d²=sum(w[s,l]*(P_A-P_B)²). Nonnegative weights bounded0..20 before mean-one normalization for deployment. One nonnegative calibration intercept is fitted for geometric-regression loss only and never added to retrieval distances.',
        objective='Convex weighted squared regression to geometry costs, plus ridge mean((w-1)²), smooth mean((w_adjacent-w)²) across shell and degree neighbors, and noise_coefficient * sum(w * training_physical_noise_variance / typical_training_pair_distance²). Noise variance is group-balanced and fit-only. Global X/Y scales use training pairs only.',
        configurations=config_grid(),
        selection='Each family plus combined D_selected: minimize group-mean validation matched geometry cost among configurations whose cell-weighted validation2%-noise identity retrieval is within2 percentage points of raw shells. Raw equal weights are a declared fallback, so no forced learned winner. Ties prefer raw, then lower total penalty complexity by fixed configuration order. Validation selection precedes outer test. Refit can still worsen noise or geometry on outer test.',
        perturbation='One new smooth normal surface perturbation per cell, RMS displacement2% of surface-RMS radius before recentering/rescaling. SeedSHA256(cell_id+experimentD). Real geometry is perturbed and all shell coefficients recomputed. Training endpoints supply noise penalty; validation probes select settings; test probes are used only after models freeze. One realization per cell is a limitation.',
        references=['raw_shell', 'nuisance_adjusted: existing degree2 Ridge(alpha=1) procedure refitted strictly within each outer development fold'],
        primary='Out-of-fold matched-neighbor geometry cost: other similarity group, within1 standardized distance in log volume/elongation/sphericity, ≥5 candidates, at most20. Same pools for every method. Primary D_selected chosen on inner validation only.',
        secondary=['oracle regret and best geometry quartile in matched pools', '2% physical-noise same-cell recall@1/@5 within each held-out gallery', 'cross-group top5 neighborhood Jaccard under noise', 'physical-rotation descriptor drift/retrieval using previously cached actual rotated geometry'],
        uncertainty='4000 similarity-group bootstrap draws over pooled outer predictions. Geometry equal group weight; noise cell weight. Paired intervals against raw and fold-refitted nuisance adjustment. Shared training folds mean bootstrap is descriptive, not an independent-cohort guarantee. Report each outer fold and all family variants, no favorable fold selection.',
        gates='Improvement requires lower primary cost and paired95% interval for baseline-minus-D above0; noise noninferiority margin minus2 percentage points, paired95% lower bound. No target is an observed result. Prior0.0225/95% thresholds are descriptive references, not model-selection criteria.',
        scope='Offline experiment only. No Atlas changes, another autoencoder, nonlinear projection, biological-state claims, or publishing.',
        source_hashes=hashes)
    save_json(OUT / 'protocol.json', protocol)
    print('Frozen protocol, grouped partitions and pair plans', flush=True)


def perturb_worker(arg):
    i, row = arg; path = OUT / 'probes' / f'{i:04d}.npz'
    if path.exists(): return i
    v, f = mesh_for(row); vc, _ = canonical(v, f)
    seed = int(hashlib.sha256((row['cell_id'] + '|ExperimentD|noise2pct').encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    phase = rng.normal(size=(3, 3)) * 5; angle = rng.uniform(0, 2 * np.pi, 3)
    displacement = np.sin(vc @ phase.T + angle).mean(1)
    displacement -= displacement.mean(); displacement /= max(displacement.std(), 1e-12)
    normals = trimesh.Trimesh(vc, f, process=False).vertex_normals
    noisy, _ = canonical(vc + .02 * displacement[:, None] * normals, f)
    p = powers(field_coefficients(noisy, f))[0]
    np.savez_compressed(path, powers=p, cell_id=row['cell_id'], seed=seed)
    return i


def prepare():
    frame = pd.read_csv(OUT / 'manifest.csv'); (OUT / 'probes').mkdir(exist_ok=True)
    if not (OUT / 'features.npz').exists():
        with ProcessPoolExecutor(max_workers=2) as pool:
            for k, _ in enumerate(pool.map(perturb_worker, enumerate(frame.to_dict('records')))):
                if k % 40 == 0: print(f'New physical-noise descriptors {k+1}/{len(frame)}', flush=True)
        coefficients = np.load(SOURCE / 'coefficients.npz')['coefficients'][:len(frame)]
        clean = powers(coefficients)
        noise = np.stack([np.load(OUT / 'probes' / f'{i:04d}.npz')['powers'] for i in range(len(frame))])
        rotated = powers(np.load(SOURCE / 'rotated_training_views.npz')['coefficients'][:len(frame), 0])
        np.savez_compressed(OUT / 'features.npz', clean=clean, noise=noise, rotated=rotated,
                            ids=frame.cell_id.to_numpy(dtype=str))
    if not (OUT / 'surface_clouds.npy').exists():
        clouds = []
        for i, row in frame.iterrows():
            clouds.append(sample_surface(*mesh_for(row), 384, np.random.default_rng(20261030 + int(row.original_index))))
        np.save(OUT / 'surface_clouds.npy', np.stack(clouds))
    save_json(OUT / 'preparation.complete.json', dict(cells=len(frame), noise_probes=len(frame)))
    print('Noise descriptors and deterministic geometry surfaces prepared', flush=True)


def cloud_init():
    global CLOUDS
    CLOUDS = np.load(OUT / 'surface_clouds.npy', mmap_mode='r')


def geometry_worker(pair):
    i, j = pair
    return i, j, geometry_cost(CLOUDS[i], CLOUDS[j])


def cache_labels(requested, stage):
    db = sqlite3.connect(OUT / 'geometry.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS costs (i INTEGER, j INTEGER, value REAL, PRIMARY KEY(i,j))')
    labels = {(int(i), int(j)): float(value) for i, j, value in db.execute('SELECT i,j,value FROM costs')}
    missing = sorted(set(tuple(sorted((int(i), int(j)))) for i, j in requested) - labels.keys())
    if missing:
        print(f'{stage}: computing {len(missing)} new geometry pair labels', flush=True)
        with ProcessPoolExecutor(max_workers=2, initializer=cloud_init) as pool:
            for n, (i, j, value) in enumerate(pool.map(geometry_worker, missing, chunksize=8)):
                labels[(i, j)] = float(value)
                db.execute('INSERT OR REPLACE INTO costs VALUES (?,?,?)', (i, j, value))
                if n % 200 == 0:
                    db.commit(); print(f'{stage}: geometry pairs {n+1}/{len(missing)}', flush=True)
        db.commit()
    db.close()
    return labels


def pair_requests(plans):
    requested = set()
    for plan in plans:
        for key in ['inner_pairs', 'refit_pairs']:
            requested.update((p['i'], p['j']) for p in plan[key])
        for i, candidates in plan['validation_pools'].items():
            requested.update(tuple(sorted((int(i), int(j)))) for j in candidates)
    return requested


def labels():
    plans = json.loads((OUT / 'pair_plans.json').read_text())
    cache_labels(pair_requests(plans), 'Training/validation')
    print('All training and validation pair labels cached', flush=True)


def train():
    frame, features, noisy, plans = load()
    pair_plans = json.loads((OUT / 'pair_plans.json').read_text())
    labels = cache_labels(pair_requests(pair_plans), 'Training/validation')
    all_validation = []
    for plan, pair_plan in zip(plans, pair_plans):
        folder = OUT / f'fold{plan["fold"]}'; folder.mkdir(exist_ok=True)
        if (folder / 'complete.json').exists():
            all_validation.extend(json.loads((folder / 'validation.json').read_text())); continue
        vp = {int(i): np.array(c) for i, c in pair_plan['validation_pools'].items()}
        if not vp: raise RuntimeError('No eligible validation queries')
        raw_geom = geometry_rows(features, None, vp, labels, frame)
        raw_noise = noise_rows(features, noisy, plan['validation'], None, frame)
        raw_score = group_mean(raw_geom, 'geometry_cost')
        raw_rate = float(pd.DataFrame(raw_noise).noise_correct.mean())
        scores = [dict(config='raw', family='raw', geometry=raw_score, noise=raw_rate,
                       eligible=True, parameters=None)]
        for index, config in enumerate(config_grid()):
            params = {k: v for k, v in config.items() if k != 'family'}
            fit = fit_metric(features, noisy, frame, plan['train'], pair_plan['inner_pairs'], labels, params)
            w = fit['weights']; g = geometry_rows(features, w, vp, labels, frame)
            n = noise_rows(features, noisy, plan['validation'], w, frame)
            rate = float(pd.DataFrame(n).noise_correct.mean())
            scores.append(dict(config=f'config{index:02d}', family=config['family'],
                               geometry=group_mean(g, 'geometry_cost'), noise=rate,
                               eligible=bool(rate >= raw_rate - .02), parameters=params))
        save_json(folder / 'validation.json', scores); all_validation.extend(scores)
        selection = {}
        for family in ['ridge', 'smooth', 'robust', 'D_selected']:
            eligible = [s for s in scores if s['eligible'] and
                        (s['family'] in ['raw', family] or family == 'D_selected')]
            chosen = min(eligible, key=lambda s: s['geometry'])
            if chosen['config'] == 'raw':
                fitted = dict(weights=np.ones(DIM), parameters=None, fallback=True)
            else:
                fitted = fit_metric(features, noisy, frame, plan['development'], pair_plan['refit_pairs'],
                                    labels, chosen['parameters'])
                fitted['fallback'] = False
            # Normalize only by a single positive scalar: no effect on nearest neighbors.
            weights = fitted.pop('weights'); weights = weights / weights.mean()
            if 'noise_energy' in fitted: fitted['noise_energy'] = fitted['noise_energy'].tolist()
            np.savez_compressed(folder / f'{family}.npz', weights=weights.reshape(32, 13))
            selection[family] = dict(validation_selection=chosen, fit=fitted)
        save_json(folder / 'selection.json', selection)
        audit = dict(fold=plan['fold'], train_cells=len(plan['train']), validation_cells=len(plan['validation']),
                     development_cells=len(plan['development']), test_cells=len(plan['test']),
                     inner_pair_count=len(pair_plan['inner_pairs']), refit_pair_count=len(pair_plan['refit_pairs']),
                     no_test_endpoints=all(p['i'] not in plan['test'] and p['j'] not in plan['test']
                                           for key in ['inner_pairs', 'refit_pairs'] for p in pair_plan[key]))
        assert audit['no_test_endpoints']
        save_json(folder / 'complete.json', audit)
        print(f'Fold{plan["fold"]}: settings selected and models frozen; D={selection["D_selected"]["validation_selection"]["config"]}', flush=True)
    frozen = {str(p.relative_to(OUT)): sha(p) for p in OUT.glob('fold*/*.npz')}
    frozen.update({str(p.relative_to(OUT)): sha(p) for p in OUT.glob('fold*/selection.json')})
    save_json(OUT / 'models_frozen.json', dict(protocol_sha256=sha(OUT / 'protocol.json'), models=frozen,
                                            note='All outer-fold weights/settings frozen before final evaluation'))
    print('All folds frozen; ready for outer evaluation', flush=True)


def diagnostics():
    """Added after validation fallback, before outer scoring; never changes primary D."""
    assert not (OUT / 'evaluation.complete.json').exists(), 'Freeze diagnostics before outer outcomes'
    frame, features, noisy, plans = load()
    pair_plans = json.loads((OUT / 'pair_plans.json').read_text())
    labels = cache_labels(pair_requests(pair_plans), 'Diagnostic refits')
    choices = []
    for plan, pair_plan in zip(plans, pair_plans):
        folder = OUT / f'fold{plan["fold"]}'
        scores = json.loads((folder / 'validation.json').read_text()); selected = {}
        for family in ['ridge', 'smooth', 'robust']:
            chosen = min([s for s in scores if s['family'] == family], key=lambda s: s['geometry'])
            fitted = fit_metric(features, noisy, frame, plan['development'], pair_plan['refit_pairs'], labels,
                                chosen['parameters'])
            w = fitted['weights']; w = w / w.mean()
            np.savez_compressed(folder / f'{family}_unguarded.npz', weights=w.reshape(32, 13))
            selected[family] = chosen
        choices.append(dict(fold=plan['fold'], choices=selected))
    save_json(OUT / 'diagnostic_frozen.json', dict(
        status='post-validation diagnostic, frozen before any outer evaluation',
        reason='Every primary family chose raw fallback because all21 learned settings failed the validation noise guard. These forced learned variants expose the geometry/noise tradeoff; they do not replace the original primary analysis or support its confirmatory gate.',
        original_primary_freeze_sha256=sha(OUT / 'models_frozen.json'),
        choices=choices, hashes={str(p.relative_to(OUT)): sha(p) for p in OUT.glob('fold*/*_unguarded.npz')}))
    print('Diagnostic learned weights frozen; primary weights/settings unchanged', flush=True)


def evaluate():
    frozen = json.loads((OUT / 'models_frozen.json').read_text())
    assert sha(OUT / 'protocol.json') == frozen['protocol_sha256']
    for path, digest in frozen['models'].items(): assert sha(OUT / path) == digest
    diagnostic = json.loads((OUT / 'diagnostic_frozen.json').read_text()) if (OUT / 'diagnostic_frozen.json').exists() else None
    if diagnostic:
        assert sha(OUT / 'models_frozen.json') == diagnostic['original_primary_freeze_sha256']
        for path, digest in diagnostic['hashes'].items(): assert sha(OUT / path) == digest
    frame, features, noisy, plans = load(); rotated = np.load(OUT / 'features.npz')['rotated']
    requests = set(); candidate_sets = {}
    for plan in plans:
        cp = pools(frame, plan['test'], plan['development'])
        candidate_sets[plan['fold']] = cp
        for i, c in cp.items(): requests.update(tuple(sorted((i, int(j)))) for j in c)
    labels = cache_labels(requests, 'Frozen outer evaluation')
    geometry, stability, selections = [], [], []
    for plan in plans:
        fold = plan['fold']; folder = OUT / f'fold{fold}'; cp = candidate_sets[fold]
        adjusted = adjusted_features(frame, features, plan['development'])
        # Original nuisance controls held fixed under perturbation, matching the previous study.
        correction = features - adjusted
        methods = {'raw_shell': (features, noisy, rotated, None),
                   'nuisance_adjusted': (adjusted, noisy - correction, rotated - correction, None)}
        for family in ['ridge', 'smooth', 'robust', 'D_selected']:
            w = np.load(folder / f'{family}.npz')['weights'].ravel()
            methods[family] = (features, noisy, rotated, w)
        if diagnostic:
            for family in ['ridge', 'smooth', 'robust']:
                w = np.load(folder / f'{family}_unguarded.npz')['weights'].ravel()
                methods[family + '_unguarded'] = (features, noisy, rotated, w)
        for method, (x, n, r, w) in methods.items():
            for row in geometry_rows(x, w, cp, labels, frame): geometry.append(dict(fold=fold, method=method, **row))
            rows = noise_rows(x, n, plan['test'], w, frame)
            ww = np.ones(DIM) if w is None else w
            ids = np.array(plan['test']); prediction = cdist(r[ids] * np.sqrt(ww), x[ids] * np.sqrt(ww)).argmin(1)
            drift = np.linalg.norm((r[ids] - x[ids]) * np.sqrt(ww), axis=1)
            scale = np.median(cdist(x[ids] * np.sqrt(ww), x[ids] * np.sqrt(ww)))
            for local, row in enumerate(rows):
                stability.append(dict(fold=fold, method=method, rotation_correct=float(prediction[local] == local),
                                      relative_rotation_drift=float(drift[local] / max(scale, 1e-12)), **row))
        selections.append(dict(fold=fold, **json.loads((folder / 'selection.json').read_text())))
    pd.DataFrame(geometry).to_csv(OUT / 'outer_geometry.csv', index=False)
    pd.DataFrame(stability).to_csv(OUT / 'outer_stability.csv', index=False)
    save_json(OUT / 'all_selections.json', selections)
    save_json(OUT / 'evaluation.complete.json', dict(cells=len(frame), folds=len(plans),
              eligible_geometry_queries=int(pd.DataFrame(geometry).loc[lambda f: f.method == 'raw_shell', 'query'].nunique()),
              organoids=frame['sample'].nunique(), groups=frame.similarity_group.nunique()))
    print('Outer-fold evaluation complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['freeze', 'prepare', 'labels', 'train', 'diagnostics', 'evaluate'])
    globals()[parser.parse_args().stage]()
