"""Independent test report and visual QA for the controlled A/B/C experiment."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import re,shutil,platform,sys
import numpy as np,pandas as pd
from spectral64_core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from report_spherical_shape_study import mesh_axis

NAMES={'A64':'A: invariant-power VAE','B64':'B: full-coefficient spectral AE','C64':'C: aligned-input reference','C_unaligned64':'C: unaligned rotated inputs','spectral_PCA64':'Full-coefficient PCA64','shell_power':'Raw shell powers (416D)','shell_adjusted_frozen':'Existing adjusted shell powers','shell_adjusted_train469':'Shell powers adjusted on469 training cells','traditional_nine_frozen':'Nine traditional features','coefficient_ceiling':'Full-coefficient reconstruction ceiling'}

def name(method):
    m=re.match(r'^(C_unaligned|A|B|C)_seed\d+(_adjusted)?$',method)
    return m[1]+'64'+('_adjusted' if m[2] else '') if m else method

def collapse(frame,identifiers,metrics):
    f=frame.copy();f['method']=f.method.map(name)
    return f.groupby(['method']+identifiers,dropna=False,as_index=False)[metrics].mean()

def uncertainty(values,groups,weighted=False):
    d=pd.DataFrame(dict(value=values,group=groups)).dropna();parts=d.groupby('group').value.agg(['sum','count','mean']);rng=np.random.default_rng(42);ix=rng.integers(0,len(parts),(2000,len(parts)))
    if weighted:
        means=parts['sum'].to_numpy()[ix].sum(1)/parts['count'].to_numpy()[ix].sum(1);point=float(d.value.mean())
    else:means=parts['mean'].to_numpy()[ix].mean(1);point=float(parts['mean'].mean())
    return dict(mean=point,ci95=np.percentile(means,[2.5,97.5]).tolist(),objects=len(d),groups=len(parts))

def fmt(value,percentage=False):
    if value is None:return '—'
    mul=100 if percentage else 1;digits=1 if percentage else 5
    a,b=value['ci95'];return f"{value['mean']*mul:.{digits}f}"+('%' if percentage else '')+f" [{a*mul:.{digits}f}, {b*mul:.{digits}f}]"


def main():
    complete=json.loads((OUT/'evaluation.complete.json').read_text());gate=json.loads((OUT/'B_information_gate.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    neighbors=pd.read_csv(OUT/'test_neighbors.csv');stability=pd.read_csv(OUT/'test_stability.csv');recon=pd.read_csv(OUT/'test_reconstruction.csv');ch=pd.read_csv(OUT/'physical_challenge_retrieval.csv')
    geom=collapse(neighbors,['query','group'],['geometry_cost'])
    stable=collapse(stability,['cell_id','group'],['noise_correct','rotation_correct','unaligned_rotation_correct','unaligned_noise_correct'])
    rec=collapse(recon,['cell_id','group','volume_eligible'],['volumetric_IoU','chamfer_squared','p95','high_degree9_12_R2','boundary_detail9_12_R2','surface_valid'])
    challenge=collapse(ch,['scope','query','pair'],['identity_correct','within_pair_correct'])
    result={}
    for method in geom.method.unique():
        f=geom[geom.method==method];entry=dict(name=NAMES.get(method,method),geometry_cost=uncertainty(f.geometry_cost,f.group))
        f=stable[stable.method==method]
        if len(f):
            for key in ['noise_correct','rotation_correct','unaligned_rotation_correct','unaligned_noise_correct']:
                if f[key].notna().any():entry[key]=uncertainty(f[key],f.group,True)
        f=rec[rec.method==method]
        if len(f):
            for key in ['volumetric_IoU','chamfer_squared','p95','high_degree9_12_R2','boundary_detail9_12_R2']:
                if f[key].notna().any():entry[key]=uncertainty(f[key],f.group)
            entry['volume_eligible']=int(f.volume_eligible.sum());entry['surface_valid_fraction']=float(f.surface_valid.mean())
        result[method]=entry
    for method in ['coefficient_ceiling']:
        f=rec[rec.method==method];result[method]=dict(name=NAMES[method])
        for key in ['volumetric_IoU','chamfer_squared','p95','high_degree9_12_R2','boundary_detail9_12_R2']:result[method][key]=uncertainty(f[key],f.group)
    paired=[]
    for baseline in ['shell_adjusted_frozen','shell_adjusted_train469','traditional_nine_frozen']:
        b=geom[geom.method==baseline].set_index('query')
        for method in ['A64','B64','C64','C_unaligned64','spectral_PCA64']:
            f=geom[geom.method==method].set_index('query');delta=b.loc[f.index].geometry_cost-f.geometry_cost
            paired.append(dict(method=method,baseline=baseline,**uncertainty(delta.to_numpy(),f.group.to_numpy())))
    summary=dict(test=complete,methods=result,paired_improvement=paired,development_gate=gate,biological_claim='No experimentally verified states/treatments or replicate-level labels; geometry-only evidence')
    save_json(OUT/'summary.json',summary)
    # Primary fixed proposal gates. Do not select aligned C instead of its
    # predeclared unaligned deployment variant after seeing test results.
    gates={}
    for method in ['A64','B64','C_unaligned64']:
        r=result[method];gates[method]=dict(latent64=True,geometry_below_0225=r['geometry_cost']['mean']<.0225,noise_at_least_95=r['noise_correct']['mean']>=.95,geometry_upperCI_below_0225=r['geometry_cost']['ci95'][1]<.0225,noise_lowerCI_at_least95=r['noise_correct']['ci95'][0]>=.95)
    save_json(OUT/'success_gates.json',dict(targets_are_proposed=True,gates=gates))
    # Figures show all seeds aggregated, not the best-performing run.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    methods=['traditional_nine_frozen','shell_adjusted_frozen','shell_power','spectral_PCA64','A64','B64','C_unaligned64'];colors=['#b2b2b2','#318f87','#68bcb2','#b5a2dc','#dba274','#789bc2','#ce85a9']
    ax=axes[0,0];values=[result[m]['geometry_cost']['mean'] for m in methods];err=np.array([[v-result[m]['geometry_cost']['ci95'][0],result[m]['geometry_cost']['ci95'][1]-v] for m,v in zip(methods,values)]).T
    ax.barh(range(len(methods)),values,color=colors,xerr=err,capsize=3);ax.set_yticks(range(len(methods)),['9 traditional','Existing adjusted shells','Raw shell powers','Spectral PCA64','A64','B64','C64 · unaligned']);ax.invert_yaxis();ax.axvline(.0225,color='#a04444',ls='--',label='Proposed <0.0225');ax.set_xlabel('Matched-neighbor geometry cost ↓');ax.set_title('Independent test retrieval · group-bootstrap 95% intervals');ax.legend(fontsize=8)
    ax=axes[0,1];methods2=['shell_power','spectral_PCA64','A64','B64','C_unaligned64']
    v=np.array([result[m]['noise_correct']['mean'] for m in methods2]);err=np.array([[v[i]-result[m]['noise_correct']['ci95'][0],result[m]['noise_correct']['ci95'][1]-v[i]] for i,m in enumerate(methods2)]).T
    ax.bar(range(5),v,color=['#68bcb2','#b5a2dc','#dba274','#789bc2','#ce85a9'],yerr=err,capsize=3);ax.axhline(.95,color='#a04444',ls='--',label='Proposed ≥95%');ax.set_xticks(range(5),['Shells','PCA64','A64','B64','C64\nunaligned']);ax.set_ylim(0,1.1);ax.set_ylabel('Same-cell retrieval under 2% surface noise');ax.set_title('135 held-out cells · cluster-bootstrap intervals');ax.legend(fontsize=8)
    ax=axes[1,0];phase=challenge[challenge.scope=='mirror'];methods3=['shell_power','spectral_PCA64','A64','B64','C64']
    vals=[];errs=[]
    for m in methods3:
        f=phase[phase.method==m];u=uncertainty(f.within_pair_correct,f.pair,True);vals.append(u['mean']);errs.append([u['mean']-u['ci95'][0],u['ci95'][1]-u['mean']])
    ax.bar(range(5),vals,color=['#68bcb2','#b5a2dc','#dba274','#789bc2','#ce85a9'],yerr=np.array(errs).T,capsize=3);ax.axhline(.5,color='gray',ls=':');ax.set_xticks(range(5),['Shells','PCA64','A64','B64','C64']);ax.set_ylim(0,1.1);ax.set_ylabel('Within-pair retrieval at 0.5% physical noise');ax.set_title('32 held-out mirror pairs · same pose / different chirality')
    ax=axes[1,1];methods4=['coefficient_ceiling','spectral_PCA64','B64','C_unaligned64'];v=[result[m]['volumetric_IoU']['mean'] for m in methods4]
    ax.bar(range(4),v,color=['#929292','#b5a2dc','#789bc2','#ce85a9']);ax.set_xticks(range(4),['Full coefficients','PCA64','B64','C64\nunaligned']);ax.set_ylim(0,1.05);ax.set_ylabel('Mean volumetric IoU ↑');ax.set_title('Separate field-reconstruction evaluation')
    fig.suptitle('Controlled 64D spectral experiment · models frozen before testing',fontsize=18);fig.savefig(OUT/'overview.png',dpi=170);plt.close(fig)
    # Fixed examples and fixed representative seed42, never chosen by appearance.
    kinds=['coefficient_ceiling','spectral_PCA64','B_seed42','C_unaligned_seed42'];titles=['Full-coefficient ceiling','PCA64','B64 · seed42','C64 · unaligned · seed42'];fig=plt.figure(figsize=(17,10))
    for row,q in enumerate([0,5,10]):
        source=np.load(OUT/'reconstructions'/f'{q:03d}_coefficient_ceiling.npz');v,f=source['vertices'],source['faces'];mesh_axis(fig.add_subplot(3,5,row*5+1,projection='3d'),v,f,'#e6ac7e',f'Test object {q+1}\nOriginal surface')
        for col,(kind,title,color) in enumerate(zip(kinds,titles,['#76b9b3','#b5a2dc','#789bc2','#ce85a9']),1):
            a=np.load(OUT/'reconstructions'/f'{q:03d}_{kind}.npz');field=a['field'];rv,rf,_,_=marching_cubes(field,.5,spacing=(5.2/39,)*3);rv-=2.6;mesh_axis(fig.add_subplot(3,5,row*5+col+1,projection='3d'),rv,rf,color,title)
    fig.suptitle('Held-out reconstructions: original vs representation ceiling vs compact codes',fontsize=17);fig.text(.5,.015,'Fixed test indices 0, 5, 10; seed42 fixed in advance. Each panel framed for shape visibility; quantitative metrics use the same normalized physical coordinates.',ha='center',fontsize=9);fig.subplots_adjust(left=.01,right=.99,top=.9,bottom=.04,wspace=0,hspace=.03);fig.savefig(OUT/'reconstructions.png',dpi=170);plt.close(fig)
    # Report.
    models=['A64','B64','C_unaligned64'];bphase=np.mean([r['phase_difference_R2'] for r in gate['rows'] if r['model']=='B']);test=complete
    lines=['# Controlled 64D spectral autoencoding: A / B / C','', '**All three models were implemented, trained and evaluated offline. Atlas was not modified.**','',
        f'Model B passed the development information-retention gate: its decoder recovered {100*bphase:.1f}% of the coefficient differences between mirrored shapes, while A’s invariant powers could not distinguish them. C was then trained on actual independently rotated meshes. The final test cohort contains **{test["test_cells"]} segmented cells / {test["organoids"]} organoids / {test["groups"]} similarity groups**. All model choices/checkpoints were frozen before test metrics.', '',
        '[MorphCell v3 paper/code review](paper_review.md) · [Frozen protocol](protocol.json) · [Machine-readable results](summary.json) · [Gate outcomes](success_gates.json)', '',
        '![Overview](overview.png)', '', '## Primary results and proposed gates','',
        'Lower geometry cost is better. Noise retrieval is correct source-cell identification against all135 test cells, not classification accuracy. Intervals resample similarity groups (2,000 cluster-bootstrap draws); model rows average all three seeds before bootstrapping. The geometry metric weights groups equally; stability rates weight cells equally.', '',
        '| Method | Latent/descriptor dimensions | Matched geometry cost, 95% CI | Retrieval at 2% surface noise, 95% CI |','|---|---:|---|---|']
    dimensions={'traditional_nine_frozen':9,'shell_adjusted_frozen':416,'shell_power':416,'spectral_PCA64':64,'A64':64,'B64':64,'C_unaligned64':64}
    for m in methods:
        lines.append(f'| {NAMES[m]} | {dimensions[m]} | {fmt(result[m]["geometry_cost"])} | {fmt(result[m].get("noise_correct"),True)} |')
    lines+=['', '**The targets <0.0225 and ≥95% were proposed gates, not prior results from this run.** The old0.023836 and95.8% references came from a reused validation cohort and a24-cell noise probe. They must not be compared directly with this independently held-out cohort; reference methods were recomputed above on the same test cells and candidate pools.', '',
        '| Model | Geometry point estimate passes? | Noise point estimate passes? | Both pass using conservative CI limits? |','|---|---|---|---|']
    for m in models:
        g=gates[m];lines.append(f'| {m} | {g["geometry_below_0225"]} | {g["noise_at_least_95"]} | {g["geometry_upperCI_below_0225"] and g["noise_lowerCI_at_least95"]} |')
    lines+=['', '### Development decision', '',
        f'B retains information that powers discard, but its independent matched retrieval cost is {result["B64"]["geometry_cost"]["mean"]:.5f}, versus {result["shell_adjusted_frozen"]["geometry_cost"]["mean"]:.5f} for the existing adjusted-shell reference. Its2% noise retrieval is {100*result["B64"]["noise_correct"]["mean"]:.1f}%. This is an information-retention milestone, not evidence that its learned distance improves our organoid retrieval.', '',
        f'C’s declared unaligned-input deployment has {100*result["C_unaligned64"]["rotation_correct"]["mean"]:.1f}% same-cell rigid-rotation retrieval and {100*result["C_unaligned64"]["noise_correct"]["mean"]:.1f}% retrieval with independent rotation plus2% surface noise. The consistency loss has therefore not established practically useful rotation invariance in this pilot. Canonicalizing its input must not be presented as a success of the invariant encoder.', '',
        'Keep the current shell descriptors as the practical reference; do not promote B/C into Atlas as improved methods. First diagnose view consistency/generalization and reconstruction loss on training/development data, using fresh held-out objects for any subsequent performance claim. A useful next control is stronger online pose augmentation against a simple full-coefficient PCA/canonical-frame decoder, before scaling up to MorphCell’s much larger model. The present test cohort is now an observed benchmark and cannot serve as untouched selection data in another round.', '']
    lines+=['',f'The matched retrieval analysis has {test["matched_queries"]} eligible queries. Every method uses the same pools: another similarity group, within one standardized distance unit in log volume/elongation/sphericity, at least five candidates, maximum20. Geometry is proper-rotation multi-start surface ICP on384 area samples with original radius normalization, matching the previous metric definition. This is an approximate geometric proxy, not semantic biological ground truth.', '',
        '## Experimental controls','',
        '- Real data:469 training cells/74 organoids,140 validation cells/17 organoids,135 test cells/17 organoids. No organoid or similarity group crosses partitions. The test objects are held out from this model fitting and selection; some frozen descriptive encodings already existed from previous Atlas work. This is not a newly collected prospective biological cohort.',
        '- Shared synthetic data:384 training,64 development and128 held-out meshes. No category labels enter training. Additional16 mirrored development pairs gate B;32 independently generated mirror pairs are held out for the final challenge.',
        '- All models:64-dimensional variational means, the same256/128-layer encoder and128/256-layer decoder, three seeds42/43/44,120 epochs, Adam.001, two forward passes per batch, weak KL1e-5. Checkpoints use reconstruction loss on validation objects only, with all seeds retained.',
        '- A is a new matched64D **power-only** VAE baseline, not the earlier128D489-input prototype. It uses416 shell powers, padded to5408 for the same nominal backbone size/dense matrix budget. B uses all5408 real degrees of freedom of the complex field. Conjugate-symmetric packing is exact, not lossy feature selection. A necessarily has fewer information-bearing input/output dimensions; padding does not add capacity or information.',
        '- B and C share full-coefficient target, radial weighting, scalar normalization and frequency weights. Both use the same weak variational bottleneck, so this comparison does not confound phase preservation with removal of the variational objective.',
        '- Source fields:32 shells, degrees0–12,32×64 angular quadrature, soft signed-distance occupancy. Center and surface-RMS scale are removed. A proper PCA/skewness frame supplies a declared reference orientation; physical scale and pose are kept separately for decoding. PCA degeneracies and sign sensitivity remain practical limitations.',
        '- B is the canonical-frame spectral baseline. C receives two independent actual mesh rotations per object, with no pose metadata at the encoder. The decoder applies the known spherical-harmonic rotation action. Its added consistency loss is normalized by batch latent variance to avoid a trivial global scale/offset loophole. This is a finite two-view pilot, not an exact-invariance theorem or exhaustive online SO(3) augmentation.',
        '- Training architecture/updates are matched; geometry preprocessing for C costs more and is recorded separately. The actual geometry is resampled before its harmonic transform. Fixed rotation actions were independently checked against oversampled quadrature, not learned or supplied as encoder features.',
        '- Full-complex coefficient error is radially weighted to approximate volumetric L2 error, with shared frequency weight sqrt(1+.05l(l+1)). There is no dangerous per-coefficient variance whitening. Bandwidth and radial/grid resolution still limit recoverable detail.', '',
        '## Information retention: the important distinction','',
        'The previous equal-*radial*-power challenge is retained, but multi-shell powers could already distinguish those objects. Passing it alone would not show improvement over the current shell representation. Mirrored chiral surfaces provide a stricter control: every shell’s degree powers are equal, although the full fields differ under proper rotations. Reflection invariance is a genuine information loss here; chirality is a controlled mathematical distinction, not an asserted cell-state marker.', '',
        'Development gate B was set before model fitting: mean pair-difference reconstruction R²≥0.25 and at least90% of mirrors with latent separation above1% of median between-pair distance, across all three seeds. This avoids rewarding meaningless floating-point code differences. All three B seeds passed; C training used no final test outcome.', '',
        'Final independent query meshes have0.5% smooth surface displacement and independent re-integration. Exact within-pair ties count0.5, not an arbitrary result from gallery ordering. The following scores use a common canonical pose to isolate information retention from the separate rotation challenge.', '',
        '| Method | Legacy radial-equal-power: within-pair retrieval | True mirror/shell-equal-power: within-pair retrieval | Mirror identity retrieval among64 objects |','|---|---:|---:|---:|']
    for m in ['shell_power','spectral_PCA64','A64','B64','C64']:
        vals=[]
        for scope in ['legacy','mirror']:
            f=challenge[(challenge.method==m)&(challenge.scope==scope)];vals.append(uncertainty(f.within_pair_correct,f.pair,True))
        f=challenge[(challenge.method==m)&(challenge.scope=='mirror')];v=uncertainty(f.identity_correct,f.pair,True)
        lines.append(f'| {NAMES.get(m,m)} | {fmt(vals[0],True)} | {fmt(vals[1],True)} | {fmt(v,True)} |')
    lines+=['', 'A strong linear PCA64 reconstruction/control is included. If it outperforms the neural model, that points to the training/objective or inductive bias rather than an inherent impossibility of compact full-coefficient encoding. Near-perfect within-pair discrimination does not imply all fine surface details were reconstructed.', '',
        '## Rotation invariance, separate from canonical alignment','',
        'A and B’s declared pipeline uses canonical orientation preprocessing; C is also shown with independently rotated inputs and no orientation correction. Raw-rotation stress applied to B is an out-of-training-domain diagnostic, not its normal deployed preprocessing. For C_unaligned, gallery and rigid-rotation query are different physical rotation views; combined noise queries have an independent rotation too.', '',
        '| Method/input scope | Rigid-rotation retrieval | Unaligned rigid-rotation retrieval | Unaligned rotation +2% noise |','|---|---|---|---|']
    for m in ['shell_power','spectral_PCA64','A64','B64','C64','C_unaligned64']:
        r=result[m];lines.append(f'| {NAMES.get(m,m)} | {fmt(r.get("rotation_correct"),True)} | {fmt(r.get("unaligned_rotation_correct"),True)} | {fmt(r.get("unaligned_noise_correct"),True)} |')
    lines+=['', '## Reconstruction: volume, surface and details are separate','',
        'No pose or scale is optimized against the test reconstruction: the stored source/known view transforms define the reference frame. C’s decoder predicts the canonical field and the known rotation places it in the input frame; measurements are made back in the known canonical frame, not with a fitted alignment. A predicts powers only, so it has no unique full-field decoder and cannot receive a fabricated 3D IoU score.', '',
        'IoU thresholds the field at0.5 on a40³ fixed physical-normalized grid. It is reported only for closed consistently oriented source meshes. Surface Chamfer uses bidirectional point-to-triangle distances from2048 area samples per surface; distances are in surface-RMS-normalized units, squared for Chamfer and unsquared forP95. The full-coefficient ceiling includes harmonic truncation,32 radial shells and40³ meshing. It is not the original segmentation at unlimited resolution.', '',
        '| Reconstruction | Volumetric IoU ↑ | Surface squared Chamfer ↓ | Surface P95 ↓ | Degree9–12 coefficient R² ↑ | Boundary detail R² ↑ |','|---|---|---|---|---|---|']
    for m in ['coefficient_ceiling','spectral_PCA64','B64','C64','C_unaligned64']:
        r=result[m];lines.append(f'| {NAMES.get(m,m)} | '+ ' | '.join(fmt(r[k]) for k in ['volumetric_IoU','chamfer_squared','p95','high_degree9_12_R2','boundary_detail9_12_R2'])+' |')
    lines+=['', 'Boundary detail R² compares reconstructed degrees9–12 of the occupancy field near the actual surface (within0.15 RMS radius), including phase/location, rather than rewarding high-frequency energy alone. It is a declared band-limited geometric-detail readout, not a claim to recover frequencies aboveL12 or native voxel stair steps. Negative R² means worse than predicting no signal in that band. `test_reconstruction.csv` records each cell and eligibility/failure status.', '',
        '![Reconstructions](reconstructions.png)', '',
        '## Nuisance-adjusted secondary comparison','',
        'Degree-two Ridge removes associations with log volume, elongation and sphericity using original469 training biological cells only. The same fit policy applies to learned codes and the new shell baseline. These are secondary variants, not choices made to pass the primary gates. Historical frozen adjustment used96 original training cells and remains a distinct reference. Noise robustness keeps the original nuisance measurements fixed, isolating descriptor drift; it does not test a complete remeasurement pipeline.', '',
        '| Secondary method | Geometry cost,95% CI | Noise retrieval,95% CI |','|---|---|---|']
    for m in ['shell_adjusted_train469','A64_adjusted','B64_adjusted','C64_adjusted']:
        r=result[m];lines.append(f'| {m} | {fmt(r["geometry_cost"])} | {fmt(r.get("noise_correct"),True)} |')
    lines+=['', '## Interpretation and limits','',
        'Model B’s gate demonstrates a narrow but concrete milestone: a64D learned representation can retain some geometric distinctions eliminated by shell powers. It does not establish that the learned metric is better for biological similarity, that all morphology is preserved, or that C inherits invariance merely from a consistency loss.', '',
        'A next development decision should use the independent test gates and paired intervals in `summary.json`, not a visually convincing cell or UMAP. Do not select a favorable seed/input scope or merge validation/test scores. Finite fixed rotation views, intrinsic-frame ambiguity, small biological sample count and the64D bottleneck are limitations of this controlled pilot. Training budgets are much smaller than MorphCell’s external pretraining and its384-wide representation; no MorphCell numerical benchmark is claimed.', '',
        '**Biological interpretation remains untested.** The135 test cells are held out segmented objects, not verified treatment/state labels or independent experimental replicates of those labels. Geometry metrics cannot prove information beyond ordinary morphometry about cell biology. That requires annotated cells, replicate-held-out prediction and suitable classical-feature/nuisance controls. The linked MorphCell review discusses why its single-patient colon and single-root results do not supply that missing replicate-level evidence for our project.', '',
        '## Reproduction and preservation','',
        '`protocol.json`, `test_freeze.json`, checkpoint hashes and per-seed training histories record design and selection. `coefficients.npz` contains full fields, `test_neighbors.csv`, `test_stability.csv`, `physical_challenge_retrieval.csv` and `test_reconstruction.csv` hold individual readouts. Source code is snapshotted under `source_snapshot/`. Four study-specific checks cover exact packing/energy, lost mirror information, proper pose action and disjoint splits. Existing Atlas data, meshes and prior model files were not edited.', '',
        'The implementation is an experimental spectral self-supervised study, not an official ShapeEmbedLite or MorphCell reproduction. No biological claims or publication-ready performance are asserted.', '']
    (OUT/'report.md').write_text('\n'.join(lines))
    # Integrity: original files and all frozen weights must remain unchanged.
    frozen=json.loads((OUT/'test_freeze.json').read_text())['checkpoint_sha256']
    for label,digest in frozen.items():k,s=label.split('_seed');assert sha(OUT/k/f'seed{s}.pt')==digest
    assert sha(ROOT/'results/pointnet_shape/data/manifest.csv')==protocol['source_hashes']['manifest']
    assert sha(ROOT/'results/explorer_dataset/dataset.json')==protocol['source_hashes']['atlas']
    frame=pd.read_csv(OUT/'manifest.csv')
    for row in frame.itertuples():assert sha(ROOT/'results/explorer_dataset'/row.sample/f'{row.label}.mesh')==row.source_mesh_sha256
    assert (OUT/'protocol.json').stat().st_mtime <= (OUT/'test_freeze.json').stat().st_mtime
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for filename in ['spectral64_core.py','prepare_spectral64.py','train_spectral64.py','noise_spectral64.py','rotate_spectral64.py','challenge_spectral64.py','evaluate_spectral64.py','report_spectral64.py']:shutil.copy2(ROOT/'scripts'/filename,snapshot/filename)
    shutil.copy2(ROOT/'tests/test_spectral64.py',snapshot/'test_spectral64.py');save_json(snapshot/'SHA256.json',{p.name:sha(p) for p in snapshot.glob('*.py')})
    from importlib.metadata import version
    packages={p:version(p) for p in ['numpy','scipy','torch','trimesh','scikit-learn','scikit-image']}
    save_json(OUT/'environment.json',dict(python=sys.version,platform=platform.platform(),packages=packages,training_device='Apple MPS',evaluation_device='CPU'))
    save_json(OUT/'validation.json',dict(status='passed',atlas_modified=False,source_meshes_verified=len(frame),frozen_checkpoint_hashes_verified=True,protocol_precedes_test_freeze=True,study_specific_tests=4,all_models_latent64=True))
    save_json(OUT/'completion.json',dict(status='complete',models=['A','B','C'],seeds=3,latent=64,test_cells=135,report=str(OUT/'report.md'),paper_review=str(OUT/'paper_review.md')))
    print('Controlled A/B/C report and preservation audit complete',flush=True)
if __name__=='__main__':main()
