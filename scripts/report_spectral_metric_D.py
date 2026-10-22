"""Report all predeclared Experiment D variants and paired out-of-fold uncertainty."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/organoid_metric_matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/organoid_metric_cache')
from spectral_metric_core import *
import shutil, platform, sys, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spatial_benchmark_core import mesh_for
from spectral64_core import canonical
from report_spherical_shape_study import mesh_axis

NAMES = {'raw_shell': 'Raw shell distance', 'nuisance_adjusted': 'Nuisance-adjusted shells',
         'ridge': 'Ridge weighting (guarded)', 'smooth': 'Smooth weighting (guarded)', 'robust': 'Noise-aware weighting (guarded)',
         'D_selected': 'D: validation-selected weighting',
         'ridge_unguarded': 'Ridge weighting (unguarded diagnostic)',
         'smooth_unguarded': 'Smooth weighting (unguarded diagnostic)',
         'robust_unguarded': 'Noise-aware weighting (unguarded diagnostic)'}
METHODS = list(NAMES)
PLOT_METHODS = ['raw_shell', 'nuisance_adjusted', 'ridge_unguarded', 'smooth_unguarded', 'robust_unguarded', 'D_selected']
COLORS = ['#6aada7', '#38827e', '#a0a0c4', '#a993c3', '#ca99b8', '#d08f63']


def fmt(x, percent=False):
    scale = 100 if percent else 1; digits = 1 if percent else 5
    a, b = x['ci95']; suffix = '%' if percent else ''
    return f'{scale*x["mean"]:.{digits}f}{suffix} [{scale*a:.{digits}f}, {scale*b:.{digits}f}]'


def main():
    complete = json.loads((OUT / 'evaluation.complete.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    frozen = json.loads((OUT / 'models_frozen.json').read_text())
    frame = pd.read_csv(OUT / 'manifest.csv')
    g = pd.read_csv(OUT / 'outer_geometry.csv'); s = pd.read_csv(OUT / 'outer_stability.csv')
    selections = json.loads((OUT / 'all_selections.json').read_text())
    result = {}; paired = []
    for method in METHODS:
        a = g[g.method == method]; b = s[s.method == method]
        result[method] = {}
        for key in ['geometry_cost', 'regret', 'in_best_quartile']:
            result[method][key] = cluster_ci(a[key], a.group)
        for key in ['noise_correct', 'noise_top5', 'neighbor_jaccard', 'rotation_correct', 'relative_rotation_drift']:
            result[method][key] = cluster_ci(b[key], b.group, weighted=True)
    for baseline in ['raw_shell', 'nuisance_adjusted']:
        ag = g[g.method == baseline].set_index('query'); bs = s[s.method == baseline].set_index('query')
        for method in METHODS:
            if method == baseline: continue
            a = g[g.method == method].set_index('query'); b = s[s.method == method].set_index('query')
            paired.append(dict(baseline=baseline, method=method,
                               geometry_improvement=cluster_ci((ag.loc[a.index].geometry_cost-a.geometry_cost).to_numpy(), a.group.to_numpy()),
                               noise_difference=cluster_ci((b.noise_correct-bs.loc[b.index].noise_correct).to_numpy(), b.group.to_numpy(), True)))
    gates = {}
    for baseline in ['raw_shell', 'nuisance_adjusted']:
        p = next(p for p in paired if p['baseline'] == baseline and p['method'] == 'D_selected')
        gates[baseline] = dict(geometry_improves_point=p['geometry_improvement']['mean'] > 0,
            geometry_improves_paired95=p['geometry_improvement']['ci95'][0] > 0,
            noise_noninferior_point=p['noise_difference']['mean'] >= -.02,
            noise_noninferior_paired95=p['noise_difference']['ci95'][0] >= -.02)
    primary = gates['raw_shell']
    success = primary['geometry_improves_paired95'] and primary['noise_noninferior_paired95']
    summary = dict(cohort=complete, methods=result, paired_comparisons=paired, primary_gates=gates,
                   success_against_raw=success, caveat='Geometry surrogate only; existing observed data and shared cross-validation training cohorts. No independent biological validation.')
    save_json(OUT / 'summary.json', summary)
    fold_rows = []
    for fold in range(5):
        for method in METHODS:
            a = g[(g.fold == fold) & (g.method == method)]; b = s[(s.fold == fold) & (s.method == method)]
            fold_rows.append(dict(fold=fold, method=method, geometry_cost=group_mean(a.to_dict('records'), 'geometry_cost'),
                                  noise_recall1=float(b.noise_correct.mean()), queries=len(a), test_cells=len(b)))
    pd.DataFrame(fold_rows).to_csv(OUT / 'fold_summary.csv', index=False)
    weights = np.stack([np.load(OUT / f'fold{fold}/D_selected.npz')['weights'] for fold in range(5)])
    np.savez_compressed(OUT / 'selected_weights_overview.npz', mean=weights.mean(0), std=weights.std(0), by_fold=weights)
    diagnostic_weights = np.stack([np.load(OUT / f'fold{fold}/robust_unguarded.npz')['weights'] for fold in range(5)])
    np.savez_compressed(OUT / 'diagnostic_weights_overview.npz', mean=diagnostic_weights.mean(0), std=diagnostic_weights.std(0), by_fold=diagnostic_weights)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(15, 11), layout='constrained')
    values = [result[m]['geometry_cost']['mean'] for m in PLOT_METHODS]
    errors = np.array([[v-result[m]['geometry_cost']['ci95'][0], result[m]['geometry_cost']['ci95'][1]-v]
                       for m, v in zip(PLOT_METHODS, values)]).T
    labels = ['Raw shells', 'Nuisance-adjusted', 'Ridge · diagnostic', 'Smooth · diagnostic', 'Noise-aware · diagnostic', 'D selected · guarded']
    ax[0, 0].barh(range(6), values, xerr=errors, color=COLORS, capsize=3)
    ax[0, 0].set_yticks(range(6), labels); ax[0, 0].invert_yaxis()
    ax[0, 0].set_xlabel('Matched-neighbor geometry cost ↓')
    ax[0, 0].set_title('Whole groups/organoids held out · 95% group-bootstrap intervals')
    values = np.array([result[m]['noise_correct']['mean'] for m in PLOT_METHODS])
    errors = np.array([[v-result[m]['noise_correct']['ci95'][0], result[m]['noise_correct']['ci95'][1]-v]
                       for m, v in zip(PLOT_METHODS, values)]).T
    ax[0, 1].bar(range(6), values, yerr=errors, color=COLORS, capsize=3)
    ax[0, 1].set_xticks(range(6), ['Raw', 'Adjusted', 'Ridge', 'Smooth', 'Noise-aware', 'D selected'], rotation=20)
    ax[0, 1].set_ylim(0, 1.05); ax[0, 1].set_ylabel('Same-cell recall@1 under 2% surface noise ↑')
    ax[0, 1].set_title('New physical perturbation for each of744 cells')
    image = ax[1, 0].imshow(diagnostic_weights.mean(0), origin='lower', aspect='auto', cmap='magma')
    ax[1, 0].set_xlabel('Harmonic degree'); ax[1, 0].set_ylabel('Radial shell')
    ax[1, 0].set_title('Noise-aware unguarded diagnostic · mean weight across folds')
    fig.colorbar(image, ax=ax[1, 0], label='Weight (mean over416 weights =1)')
    pp = [p for p in paired if p['baseline'] == 'raw_shell' and p['method'] in PLOT_METHODS]
    vv = [p['geometry_improvement']['mean'] for p in pp]
    ee = np.array([[v-p['geometry_improvement']['ci95'][0], p['geometry_improvement']['ci95'][1]-v] for p, v in zip(pp, vv)]).T
    ax[1, 1].errorbar(vv, range(len(pp)), xerr=ee, fmt='o', color='#90617b', capsize=3)
    ax[1, 1].axvline(0, color='gray', ls=':'); ax[1, 1].invert_yaxis()
    ax[1, 1].set_yticks(range(len(pp)), [labels[PLOT_METHODS.index(p['method'])] for p in pp])
    ax[1, 1].set_xlabel('Raw cost − method cost · positive favors method')
    ax[1, 1].set_title('Paired geometry improvement · same queries/candidates')
    fig.suptitle('Experiment D · geometry-aware spectral distance, no autoencoder', fontsize=18)
    fig.savefig(OUT / 'overview.png', dpi=170); plt.close(fig)
    # One deterministic query per fold; examples are never chosen for a good result.
    examples = []
    for fold in range(5):
        subset = g[(g.fold == fold) & (g.method == 'raw_shell')]
        q = int(subset['query'].min()); examples.append((fold, q))
    fig = plt.figure(figsize=(15, 17))
    for row, (fold, q) in enumerate(examples):
        selected = g[(g.fold == fold) & (g['query'] == q)].set_index('method')
        for column, method in enumerate(['query', 'raw_shell', 'nuisance_adjusted', 'robust_unguarded']):
            idx = q if method == 'query' else int(selected.loc[method].neighbor)
            v, f = mesh_for(frame.iloc[idx]); v, _ = canonical(v, f)
            label = 'Noise-aware diagnostic' if method == 'robust_unguarded' else NAMES.get(method, method)
            title = ('Query ' + frame.iloc[idx].cell_id) if method == 'query' else (
                label + '\n' + frame.iloc[idx].cell_id + f' · cost {selected.loc[method].geometry_cost:.4f}')
            mesh_axis(fig.add_subplot(5, 4, row*4+column+1, projection='3d'), v, f,
                      ['#d5ac81', '#6aada7', '#38827e', '#d08f63'][column], title)
    fig.suptitle('Fixed held-out queries: original geometry and retrieved neighbors', fontsize=17)
    fig.text(.5, .015, 'First eligible query in each fold; no outcome-based selection. Canonical pose and individual framing for visibility; distances use the frozen geometric proxy.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.01, right=.99, top=.92, bottom=.04, wspace=.02, hspace=.05)
    fig.savefig(OUT / 'retrieval_examples.png', dpi=140); plt.close(fig)
    save_json(OUT / 'visual_examples.json', [dict(fold=f, query=q) for f, q in examples])
    # Data audit checks exact identifiers and pair endpoints, not merely split sizes.
    plans = json.loads((OUT / 'folds.json').read_text()); pair_plans = json.loads((OUT / 'pair_plans.json').read_text())
    for plan, pair_plan in zip(plans, pair_plans):
        held = set(plan['test'])
        for key, allowed in [('inner_pairs', set(plan['train'])), ('refit_pairs', set(plan['development']))]:
            assert all(p['i'] in allowed and p['j'] in allowed for p in pair_plan[key])
        for key in ['train', 'validation']:
            for col in ['sample', 'similarity_group']:
                assert not set(frame.iloc[plan[key]][col]) & set(frame.iloc[list(held)][col])
    assert set(s[s.method == 'D_selected']['query']) == set(range(len(frame)))
    assert s.groupby(['method', 'query']).size().max() == 1
    assert g.groupby(['method', 'query']).size().max() == 1
    for source, digest in protocol['source_hashes'].items(): assert sha(ROOT / source) == digest
    for row in frame.itertuples(): assert sha(ROOT / 'results/explorer_dataset' / row.sample / f'{row.label}.mesh') == row.source_mesh_sha256
    assert sha(OUT / 'protocol.json') == frozen['protocol_sha256']
    for path, digest in frozen['models'].items(): assert sha(OUT / path) == digest
    diagnostic = json.loads((OUT / 'diagnostic_frozen.json').read_text())
    assert diagnostic['original_primary_freeze_sha256'] == sha(OUT / 'models_frozen.json')
    for path, digest in diagnostic['hashes'].items(): assert sha(OUT / path) == digest
    audit = dict(status='passed', cells=len(frame), outer_predictions_once_per_cell=True,
                 no_outer_test_pair_endpoints_in_fit=True, all_preprocessing_fit_within_fold=True,
                 source_meshes_verified=len(frame), atlas_unchanged=True, frozen_model_hashes_verified=True,
                 tests=5, latent_encoder_trained=False)
    save_json(OUT / 'validation.json', audit)
    pair_primary = next(p for p in paired if p['method'] == 'D_selected' and p['baseline'] == 'raw_shell')
    improvement = 100 * pair_primary['geometry_improvement']['mean'] / result['raw_shell']['geometry_cost']['mean']
    smooth_pair = next(p for p in paired if p['baseline'] == 'raw_shell' and p['method'] == 'smooth_unguarded')
    smooth_improvement = 100 * smooth_pair['geometry_improvement']['mean'] / result['raw_shell']['geometry_cost']['mean']
    status = ('The predeclared primary improvement and noise-noninferiority gates both pass.' if success else
              'The predeclared joint gate does not pass: this experiment does not establish a robust retrieval improvement over raw shells.')
    lines = ['# Experiment D — geometry-aware spectral retrieval', '',
        '**Offline metric learning only. No new autoencoder was trained and Atlas was not modified.**', '',
        f'{complete["cells"]} eligible segmented cells from{complete["organoids"]} organoids /{complete["groups"]} similarity groups. Five outer folds keep entire organoids and similarity groups out of fitting and model selection. {complete["eligible_geometry_queries"]} cells have an eligible matched candidate pool; all744 have perturbation retrieval readouts.', '',
        '## Result and decision', '', status, '',
        f'D_selected has {improvement:.1f}% relative lower matched geometry cost than raw shells (negative means worse). Its paired absolute improvement is **{fmt(pair_primary["geometry_improvement"])}**; the noise-retrieval change is **{fmt(pair_primary["noise_difference"], True)} percentage points**. The intervals must be considered alongside the point estimates.', '',
        f'The diagnostic learned metrics do show a trade-off: unguarded smooth weighting lowers geometry cost by {smooth_improvement:.1f}% (paired absolute improvement {fmt(smooth_pair["geometry_improvement"])}), but noise recall drops from {100*result["raw_shell"]["noise_correct"]["mean"]:.1f}% to {100*result["smooth_unguarded"]["noise_correct"]["mean"]:.1f}%. The training-noise penalty in the tested grid does not remove this problem: its unguarded variant reaches only {100*result["robust_unguarded"]["noise_correct"]["mean"]:.1f}% noise recall. These are diagnostic, not confirmatory-primary findings.', '',
        '![Out-of-fold overview](overview.png)', '',
        '| Method | Matched geometry cost ↓,95% CI | Recall@1 at2% surface noise ↑,95% CI | Cross-group top5 Jaccard ↑ |',
        '|---|---|---|---|']
    for method in PLOT_METHODS:
        r = result[method]
        lines.append(f'| {NAMES[method]} | {fmt(r["geometry_cost"])} | {fmt(r["noise_correct"], True)} | {fmt(r["neighbor_jaccard"])} |')
    lines += ['', 'Geometry costs weight similarity groups equally. Noise recall weights cells equally. Confidence intervals use4,000 similarity-group bootstrap draws. Pairwise comparisons use identical queries and galleries. Cross-validation models have overlapping training sets, so these intervals are descriptive and do not capture every source of training uncertainty.', '',
        'All five inner folds selected the raw fallback: every learned configuration failed the validation noise guard, even though some improved validation geometry costs. Therefore guarded ridge/smooth/robust variants also equal raw shells; their individual rows remain in the machine-readable results.', '',
        '**Post-validation diagnostics:** after observing that fallback, but before opening any outer outcomes, the best learned configuration in each family was additionally refitted without enforcing the noise guard. These unguarded rows expose the trade-off rather than disguising failed learning behind identical raw rows. They use the same grid, pairs and validation-only selection, and have separately frozen hashes in `diagnostic_frozen.json`. They are diagnostic additions, do not replace D_selected, and do not support the original primary gate.', '',
        'The previously reported0.023836 and95.8% were measured on different cohorts/probes. This study recomputes every reference on exactly its own held-out pools; those old values are not interchangeable baselines.', '',
        '## What was fitted', '',
        '- The unchanged descriptor is a32×13 grid: radial-weighted, globally normalized shell band norms from degree0–12. There is no per-coordinate whitening, dimension reduction, decoder or new representation-learning model.',
        '- Distance is the sum of416 nonnegative weights times squared band-norm differences. Before deployment, weights are rescaled by one positive scalar to mean1; this cannot change neighbor rankings. The optimization bounds are0–20 before that normalization. A large weight on an almost constant band contributes little distance; weights alone are not biological feature importance.',
        '- Geometric labels use proper-rotation,24-start ICP on384 deterministic area-sampled points, with8 refinement iterations, no scale/reflection fitting and the original source radius normalization. Bidirectional squared nearest-point distance is a useful geometric proxy, not biological similarity ground truth or an exact surface integral.',
        '- The convex objective regresses geometric costs, with ridge shrinkage toward equal weights, optional adjacent-shell/degree smoothness and an optional penalty for training-only perturbation variance. One nonnegative intercept handles a geometric-label floor during regression; it is never part of retrieval distance.',
        '- At most96 group-balanced anchors per fit have eight nearby morphometry-control partners and four random other-group partners. Every endpoint belongs to that fit partition. Full training-pair plans are saved before labels. Partners can include additional training cells beyond the96 anchors.',
        '- Each family searches ridge strengths0.001/0.01/0.1, smoothness0.01/0.1 where applicable and noise strength0.01/0.1 where applicable. There are21 declared configurations overall. Each selects lowest validation geometry cost subject to recall@1 within2 percentage points of raw validation shells. Equal weights are a declared fallback. D_selected chooses across all families using the same validation rule.',
        '- Selected settings are refitted on all outer-development cells. Refit can change the weights and need not preserve the validation noise guard on unseen cells. Test metrics never select a setting, family, seed or fold.',
        '- The nuisance reference uses the existing degree-two polynomial Ridge(alpha=1) procedure controlling log volume, sphericity and elongation. Its standardization and regression are refitted strictly on outer-development cells; it does not reuse a historical fit that might contain these folds’ held-out cells.', '',
        '## Split integrity and limits', '',
        'The source cohort was already examined in previous experiments. These are new out-of-fold metric-learning evaluations, not a newly collected or never-inspected prospective test cohort. Models in different folds may train on one another’s evaluation cells, as standard cross-validation permits; no model trains on its own evaluation cells or geometric pairs involving them.', '',
        'One pre-fit amendment fixed unusable validation coverage: of the four inner group partitions, the partition with most eligible matched queries is selected using fixed morphometry and group membership only. The original fixed-index design gave one validation fold just nine eligible queries. This amendment preceded every geometry label, fit and performance readout; the original design and reason are archived. Outer test partitions were not changed.', '',
        'All frozen weights/settings were written before final outer evaluation. `folds.json`, `pair_plans.json`, per-fold validation scores/selections, model hashes and endpoint audits make the boundary reviewable. Source descriptor settings are inherited from earlier work; only this study’s metric fitting and nuisance adjustment are fold-isolated.', '',
        '## Per-fold results and selections', '',
        '| Fold | Matched queries | Raw cost | D_selected cost | Raw noise recall@1 | D noise recall@1 | D validation choice |',
        '|---|---:|---:|---:|---:|---:|---|']
    fs = pd.DataFrame(fold_rows)
    for fold in range(5):
        raw = fs[(fs.fold == fold) & (fs.method == 'raw_shell')].iloc[0]
        d = fs[(fs.fold == fold) & (fs.method == 'D_selected')].iloc[0]
        selected = selections[fold]['D_selected']['validation_selection']
        lines.append(f'| {fold+1} | {int(raw.queries)} | {raw.geometry_cost:.5f} | {d.geometry_cost:.5f} | {100*raw.noise_recall1:.1f}% | {100*d.noise_recall1:.1f}% | {selected["family"]}: {selected["parameters"]} |')
    lines += ['', 'Do not select an apparently favorable fold after inspecting this table. Large similarity groups make fold gallery composition uneven; primary paired comparisons keep that composition identical for each method.', '',
        '## Noise and rotation', '',
        'Each cell has one newly generated smooth normal-displacement probe, with2% RMS surface-radius amplitude before re-centering and scale normalization. Shell coefficients are recomputed from actual perturbed geometry. Training-only probes determine the noise penalty, validation-only probes affect setting selection, and outer test probes are used only for final readouts. There is one noise realization per cell, not a full segmentation-error distribution.', '',
        'Same-cell recall includes the clean source in each held-out gallery. It measures robustness, whereas matched-neighbor retrieval excludes the query’s entire similarity group. Cross-group top5 Jaccard measures whether a noisy query retains its clean query’s neighbors. The nuisance-adjusted probe keeps original nuisance measurements fixed to isolate descriptor drift, not a full morphometry-remeasurement workflow.', '',
        '| Method | Noise recall@5 | Rotation recall@1 | Relative rotation drift | Best geometric quartile in matched pool |',
        '|---|---|---|---|---|']
    for method in METHODS:
        r = result[method]
        lines.append(f'| {NAMES[method]} | {fmt(r["noise_top5"], True)} | {fmt(r["rotation_correct"], True)} | {fmt(r["relative_rotation_drift"])} | {fmt(r["in_best_quartile"], True)} |')
    lines += ['', 'Rotation uses previously cached actual mesh rotations without PCA orientation correction before harmonic extraction. Nonnegative diagonal weighting preserves the mathematical invariance of ideal shell powers; finite quadrature can still cause numerical drift. This is an empirical cached-view check, not a new proof or an exhaustive set of poses.', '',
        '## Paired comparisons and gates', '',
        '| Reference | Geometry improvement,95% CI | Noise difference,95% CI (percentage points) | Geometry paired CI above0? | Noise paired CI above−2pp? |',
        '|---|---|---|---|---|']
    for baseline in ['raw_shell', 'nuisance_adjusted']:
        p = next(p for p in paired if p['method'] == 'D_selected' and p['baseline'] == baseline)
        gate = gates[baseline]
        lines.append(f'| {NAMES[baseline]} | {fmt(p["geometry_improvement"])} | {fmt(p["noise_difference"], True)} | {gate["geometry_improves_paired95"]} | {gate["noise_noninferior_paired95"]} |')
    lines += ['', 'The primary joint gate is improvement over raw shells plus noise noninferiority, both using paired95% interval limits. Improvement over adjusted shells is reported separately. A point estimate alone does not establish the gate. Multiple family results are diagnostic comparisons, not separate corrected confirmatory discoveries.', '',
        '![Fixed query examples](retrieval_examples.png)', '',
        '## How to proceed', '',
        ('The primary joint gate passes on this cross-validation benchmark. Before Atlas adoption, confirm the frozen weighting policy on an independent acquisition/specimen cohort and verify improvement with denser/resampled or point-to-mesh geometry metrics.' if success else
         'Keep raw/adjusted shell references as the practical baseline. The learned weights and per-fold results are useful diagnostics, but this run does not justify presenting a more robust retrieval method. Inspect where the validation-to-test change occurs, then test any revised policy on new held-out data rather than tuning this observed benchmark.'), '',
        'A small next development experiment could use much stronger shrinkage toward equal weights or a conservative blend between equal and learned weights, with its blend strength chosen inside validation. This is a proposed follow-up, not a result from this run, and should not be tuned against these observed outer outcomes.', '',
        'No biological treatment/state information is demonstrated. That requires verified labels and specimen/replicate-held-out prediction with morphometry controls. Also, reweighting invariant powers cannot recover phase or chirality information already discarded by those powers; reconstruction is deliberately outside Experiment D.', '',
        'The optional contrastive projection was not trained. This keeps the first experiment focused on the requested interpretable convex metric. A nonlinear extension should have a concrete justification from these results rather than being added because its architecture is available.', '',
        '## Files and reproduction', '',
        '`protocol.json` and `design_amendment.json` record design; `folds.json`/`pair_plans.json` record endpoints. `geometry.sqlite` caches geometric labels by source-manifest index. `features.npz` stores clean/noisy/rotated descriptors. `foldN/selection.json` records validation choices; normalized32×13 weights are in each fold’s `.npz` files. `outer_geometry.csv` and `outer_stability.csv` contain individual out-of-fold readouts, `summary.json` contains paired intervals/gates, and `validation.json` records preservation and leakage checks.', '',
        'Run the staged script with `freeze`, `prepare`, `labels`, `train`, `diagnostics`, `evaluate`, then run the report script. The optimizer uses bounded L-BFGS-B with an analytic gradient: [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html). A diagonal preconditioner and precomputed quadratic Hessian improve numerical conditioning without changing the objective or deployment metric. The first direct-coordinate solver hit its iteration limit before any model or outer outcome was selected; the equivalent preconditioned objective is checked numerically against the original expression. This is a custom diagonal geometric-regression metric, not an implementation of a named contrastive/LMNN method.', '',
        'Five study-specific tests passed, including an explicit check that changing held-out descriptors, noise probes and unrelated labels cannot alter fitted weights. Original Atlas dataset/source meshes and frozen D model hashes passed preservation checks. No Atlas application, previous encodings, checkpoints or source meshes were changed.', '']
    prose = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', '\n'.join(lines))
    prose = re.sub(r'(?<!\d),(?=\d)', ', ', prose)
    prose = prose.replace(' /43 similarity', ' / 43 similarity')
    (OUT / 'report.md').write_text(prose)
    snapshot = OUT / 'source_snapshot'; snapshot.mkdir(exist_ok=True)
    for p in [ROOT / 'scripts/spectral_metric_core.py', ROOT / 'scripts/run_spectral_metric_D.py',
              ROOT / 'scripts/report_spectral_metric_D.py', ROOT / 'tests/test_spectral_metric_D.py']:
        shutil.copy2(p, snapshot / p.name)
    save_json(snapshot / 'SHA256.json', {p.name: sha(p) for p in snapshot.glob('*.py')})
    save_json(OUT / 'environment.json', dict(python=sys.version, platform=platform.platform(),
                                           device='CPU; two geometry/preprocessing workers',
                                           optimizer='scipy.optimize.minimize, L-BFGS-B'))
    save_json(OUT / 'completion.json', dict(status='complete', report=str(OUT / 'report.md'),
                                         primary_joint_gate_passed=success))
    print('Experiment D report and preservation audit complete', flush=True)


if __name__ == '__main__': main()
