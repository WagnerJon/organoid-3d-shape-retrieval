"""Auditable tables, scientific figures, and report for all five experiments."""
import os
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).resolve().parents[1]/'.cache/matplotlib')
import shutil,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from spherical_shape_study_core import *

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
NAMES={'radial_energy':'Radial summed power (energy)','traditional_nine':'9 traditional features','radial_power':'Radial power','radial_power_standardized':'Radial power · standardized','radial_power_bispectrum':'Radial power + bispectrum · raw','radial_power_bispectrum_standardized':'Radial power + bispectrum · standardized','bispectrum_standardized':'Bispectrum · standardized','shell_power':'Shell power · raw','frozen_shell_adjusted':'Existing shell power · nuisance-adjusted','all_invariants_raw':'All invariants · raw','all_invariants_standardized':'All invariants · standardized','block_balanced_invariants':'All invariants · block-balanced','PCA32':'PCA 32D','PCA128':'PCA 128D','VAE128_mean':'VAE 128D · mean of 3 seeds','general_XYZ_power':'General XYZ power · original','general_XYZ_power_balanced':'General XYZ power · Möbius-centered'}

def ci(values):
    x=np.array(values,float);rng=np.random.default_rng(42)
    return np.percentile(rng.choice(x,(2000,len(x)),replace=True).mean(1),[2.5,97.5]).tolist()

def mesh_axis(ax,v,f,color,title):
    if not len(v):ax.text2D(.1,.5,'Not topology-eligible',transform=ax.transAxes);ax.axis('off');return
    # Display original triangles; only the plotting camera changes.
    p=v[f];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-15)
    light=np.array([.4,-.6,.7]);shade=.45+.5*np.maximum(normal@light,0)
    base=np.array(matplotlib.colors.to_rgb(color));rgba=np.c_[np.minimum(base[None]*shade[:,None]*1.3,1),np.ones(len(f))]
    collection=Poly3DCollection(p,facecolors=rgba,edgecolors='none',rasterized=True);ax.add_collection3d(collection)
    extent=max(np.ptp(v,axis=0).max()/2,.1);center=(v.max(0)+v.min(0))/2
    for a,x in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],center):a(x-extent,x+extent)
    ax.set_box_aspect((1,1,1));ax.view_init(20,-60);ax.axis('off');ax.set_title(title,fontsize=10)

def main():
    rotation=pd.read_csv(OUT/'rotation.csv');sensitivity=pd.read_csv(OUT/'sensitivity.csv');mapping=json.loads((OUT/'mapping.json').read_text())
    balanced=json.loads((OUT/'balanced_mapping.json').read_text());blookup={r['id']:r for r in balanced}
    shell={x['cell_id']:x for x in json.loads((OUT/'shell_reconstruction.json').read_text())}
    matched=pd.read_csv(OUT/'matched_retrieval.csv')
    seeds=matched[matched.method.str.startswith('VAE_seed')]
    combined=seeds.groupby(['scope','query','neighbor','group']).geometry_cost.mean() # Neighbor differs across seeds: aggregate by query instead below.
    averaged=seeds.groupby(['scope','query','group'],as_index=False).geometry_cost.mean();averaged['method']='VAE128_mean';averaged['neighbor']='seed average'
    matched=pd.concat([matched,averaged],ignore_index=True)
    summaries=[]
    for scope,part in matched.groupby('scope'):
        baseline=part[part.method=='traditional_nine'].set_index('query').geometry_cost
        for method,g in part.groupby('method'):
            pergroup=g.groupby('group').geometry_cost.mean();q=g.set_index('query')
            improvement=pd.DataFrame({'value':baseline.loc[q.index]-q.geometry_cost,'group':q['group']}).groupby('group').value.mean()
            summaries.append(dict(scope=scope,method=method,name=NAMES.get(method,method),queries=len(g),groups=len(pergroup),geometry_cost=float(pergroup.mean()),improvement_vs_nine=float(improvement.mean()),improvement_ci95=ci(improvement.to_numpy())))
    save_json(OUT/'retrieval_summary.json',summaries)
    pd.DataFrame(summaries).to_csv(OUT/'retrieval_summary.csv',index=False)
    top={x['method']:x for x in summaries if x['scope']=='full'}
    phase=pd.read_csv(OUT/'supplementary_phase_challenge.csv')
    valid=[r for r in mapping if r['mapping']['valid']]
    reconstruction=[]
    for r in mapping:
        for method in ['radial','general']:
            if method in r:reconstruction.append(dict(cell_id=r['cell_id'],method=method,common_genus0=r['mapping']['valid'],**r[method]))
        reconstruction.append(dict(cell_id=r['cell_id'],method='shell_full_coefficients',common_genus0=r['mapping']['valid'],**shell[r['cell_id']]['metrics']))
        b=blookup[r['cell_id']]
        if 'reconstruction' in b:reconstruction.append(dict(cell_id=r['cell_id'],method='general_balanced',common_genus0=r['mapping']['valid'],**b['reconstruction']))
    reconstruction=pd.DataFrame(reconstruction);reconstruction.to_csv(OUT/'reconstruction_metrics.csv',index=False)
    recon=reconstruction[reconstruction.common_genus0].groupby('method')[['chamfer_squared','p95']].mean()
    # One figure covers all five questions, with separate focused geometry figure.
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    ax=axes[0,0];grid=rotation[rotation.rotations==16].pivot(index='L',columns='N',values='within_between_ratio')
    im=ax.imshow(grid,cmap='viridis',aspect='auto',norm=matplotlib.colors.LogNorm(vmin=grid.min().min(),vmax=grid.max().max()))
    ax.set_xticks(range(3),grid.columns);ax.set_yticks(range(4),grid.index);ax.set_xlabel('Fibonacci samples N');ax.set_ylabel('Harmonic degree L');ax.set_title('1. Rotation drift / between-shape distance')
    for i in range(4):
        for j in range(3):ax.text(j,i,f'{100*grid.iloc[i,j]:.2f}%',ha='center',va='center',color='white' if grid.iloc[i,j]<.07 else 'black')
    fig.colorbar(im,ax=ax,shrink=.7)
    ax=axes[0,1]
    rough=sensitivity[(sensitivity.family=='roughness')&(sensitivity.amount==.7)].groupby('width')[['radial_deformation_rms','power_distance']].mean()
    for k,label in [('radial_deformation_rms','Actual radial RMS deformation'),('power_distance','L12 radial-power distance')]:ax.plot([int(round(2+4/w)) for w in rough.index],rough[k],'-o',label=label)
    ax.set_yscale('log');ax.set_xlabel('Roughness harmonic degree');ax.set_title('2. Fine roughness disappears below the cutoff');ax.legend(fontsize=8)
    ax=axes[0,2];keys=['radial','general','general_balanced','shell_full_coefficients'];ax.bar(range(4),[recon.loc[k,'chamfer_squared'] for k in keys],color=['#df9d70','#b6a1da','#7e9fc7','#61b8b1'])
    ax.set_xticks(range(4),['Radial','cMCF','Centered\ncMCF','Shell full\ncoefficients'],fontsize=8);ax.set_ylabel('Mean squared Chamfer ↓');ax.set_title(f'3. Reconstruction on {len(valid)} topology-eligible cells')
    ax=axes[1,0];part=phase[phase.training_domain=='synthetic'];keys=['radial_power','radial_bispectrum','shell_power','VAE_seed42','VAE_seed43','VAE_seed44'];ax.bar(range(6),[float(part[part.method==k].identity.iloc[0]) for k in keys],color=['#df9d70','#b6a1da','#61b8b1']+['#7e9fc7']*3)
    ax.set_xticks(range(6),['Power','Bispectrum','Shells','VAE 42','VAE 43','VAE 44'],rotation=25,fontsize=8);ax.set_ylim(0,1.08);ax.set_ylabel('Top-1 identity retrieval');ax.set_title('4. Supplement: 64 unseen equal-power shapes')
    ax=axes[1,1];keys=['traditional_nine','radial_power','shell_power','PCA128','VAE128_mean','frozen_shell_adjusted'];ax.barh(range(6),[top[k]['geometry_cost'] for k in keys],color=['#b9b9b9','#df9d70','#61b8b1','#b6a1da','#7e9fc7','#31958e']);ax.set_yticks(range(6),['9 traditional','Radial power','Shell power','PCA 128D','VAE 128D','Existing adjusted shells'],fontsize=8);ax.invert_yaxis();ax.set_xlabel('Group-weighted geometry cost ↓');ax.set_title('5. Real-cell matched retrieval · 46 queries / 11 groups')
    ax=axes[1,2];real=pd.read_csv(OUT/'learning/real/retrieval.csv')
    for method in ['radial_power','shell_power','all_invariants_standardized','VAE128']:
        values=real[real.method==method].groupby('noise_index').identity.mean();ax.plot([0,.5,2],values,'-o',label={'radial_power':'Radial power','shell_power':'Shell power','all_invariants_standardized':'Standardized invariants','VAE128':'VAE 128D (3 seeds)'}[method])
    ax.set_xlabel('Smooth surface noise (% RMS radius)');ax.set_ylabel('Same-cell retrieval');ax.set_ylim(0,1.05);ax.set_title('5. Noise robustness · 24 validation cells');ax.legend(fontsize=8)
    fig.suptitle('Spherical ShapeEmbed 3D: five experiments',fontsize=20);fig.savefig(OUT/'overview.png',dpi=180);plt.close(fig)
    # Show a real cell and two difficult controls; same source geometry in each row.
    fig=plt.figure(figsize=(15,11));sphere=trimesh.creation.icosphere(subdivisions=4)
    examples=[('Real cell',OUT/'mapping_cells/000.npz','real_000'),('Curved tube',OUT/'arbitrary_controls/curved_tube.npz','curved_tube'),('Bent branches',OUT/'arbitrary_controls/bent_branched_tubes.npz','bent_branched_tubes')]
    for row,(name,path,bkey) in enumerate(examples):
        a=np.load(path);b=np.load(OUT/'balanced_mapping'/f'{bkey}.npz')
        if row==0:z=np.load(OUT/'shell_reconstructions/000.npz');sv,sf=z['vertices'],z['faces']
        else:sv,sf=a['shell_vertices'],a['shell_faces']
        parts=[(a['vertices'],a['faces'],'#f0a873','Original surface'),(a['radial'],sphere.faces,'#b8a0e5','Radial SH, L12'),(b['reconstruction'],sphere.faces,'#8dafd4','Centered general SH, L12'),(sv,sf,'#69c9bd','Full shell SH, L12')]
        for col,(v,f,color,title) in enumerate(parts):mesh_axis(fig.add_subplot(3,4,row*4+col+1,projection='3d'),v,f,color,(name+'\n' if col==0 else '')+title)
    fig.suptitle('Surface preservation: radial envelope vs spherical mapping vs shells',fontsize=17);fig.text(.5,.02,'Original meshes unchanged. SH reconstructions use full coefficients; invariant power descriptors alone cannot decode a unique surface.',ha='center',fontsize=10);fig.subplots_adjust(wspace=0,hspace=.05,top=.91,bottom=.04);fig.savefig(OUT/'reconstructions.png',dpi=180);plt.close(fig)
    # Exact coefficient pair: equal power, unequal geometry.
    pairs=np.load(OUT/'equal_power_pairs.npz')['coefficients'];a,b=pairs[0];sphere=trimesh.creation.icosphere(subdivisions=4);Y=basis(sphere.vertices,8)
    fig=plt.figure(figsize=(13,4));mesh_axis(fig.add_subplot(1,3,1,projection='3d'),sphere.vertices*(Y@a).real[:,None],sphere.faces,'#9b9de4','Shape A');mesh_axis(fig.add_subplot(1,3,2,projection='3d'),sphere.vertices*(Y@b).real[:,None],sphere.faces,'#69bdb8','Shape B');ax=fig.add_subplot(1,3,3);ax.plot(power(a,8),'o-',label='A');ax.plot(power(b,8),'x--',label='B');ax.set_yscale('log');ax.set_xlabel('Harmonic degree');ax.set_ylabel('Normalized band norm');ax.set_title('Same power spectrum');ax.legend();fig.tight_layout();fig.savefig(OUT/'equal_power_shapes.png',dpi=180);plt.close(fig)
    # Summary and markdown report.
    equal=json.loads((OUT/'equal_power_pairs.json').read_text());controls=json.loads((OUT/'arbitrary_controls.json').read_text());resolution=json.loads((OUT/'segmentation_resolution.json').read_text());realdata=np.load(OUT/'learning_real_data.npz')
    real_noise=pd.read_csv(OUT/'learning/real/retrieval.csv');noise_summary=real_noise.groupby(['method','noise_index']).identity.mean()
    gain=100*(1-top['VAE128_mean']['geometry_cost']/top['traditional_nine']['geometry_cost'])
    lag=100*(top['VAE128_mean']['geometry_cost']/top['frozen_shell_adjusted']['geometry_cost']-1)
    paired=matched[(matched.scope=='full')&matched.method.isin(['VAE128_mean','frozen_shell_adjusted'])].pivot(index=['query','group'],columns='method',values='geometry_cost');delta=paired['frozen_shell_adjusted']-paired['VAE128_mean'];groups=delta.groupby('group').mean();vs_shell=dict(improvement=float(groups.mean()),ci95=ci(groups))
    report=['# Spherical ShapeEmbed 3D — five experiments','', '**Offline research benchmark. Atlas, source surfaces, measurements and existing models were not changed.**', '',
        f'The strongest current result remains the existing nuisance-adjusted multi-shell harmonic descriptor. The new 128D VAE has mean geometric neighbor cost **{top["VAE128_mean"]["geometry_cost"]:.6f}**, {gain:.1f}% lower than nine traditional features, but **{lag:.1f}% higher (worse)** than adjusted shell harmonics. The more complex spherical mapper and bispectrum are useful diagnostics, but do not yet justify replacing that reference.', '',
        '![Overview](overview.png)','', '## 1. Rotation invariance and numerical stability','',
        'Eight analytic asymmetric shapes; 16 proper random rotations each; Fibonacci N = 256, 1,024, 4,096; L = 4, 8, 12, 16. Prefixes of 4 and 16 rotations share the same draws and are not independent experiments. Readout is within-object descriptor drift divided by median between-object distance.', '',
        '| N | L | Median rotation drift / between-object distance | Identity retrieval |','|---:|---:|---:|---:|']
    for r in rotation[(rotation.rotations==16)&rotation.L.isin([8,12,16])].itertuples():report.append(f'| {r.N} | {r.L} | {100*r.within_between_ratio:.3f}% | {100*r.identity:.1f}% |')
    report+=['', 'The primary descriptor uses band L2 norms (square roots of summed squared coefficient magnitudes), normalized by degree zero, as in the shell reference. A literal summed-power (squared band norm) radial baseline is also reported. At fixed N, requesting higher degrees increases integration error. L16 with N256 is especially undersampled (289 coefficients for 256 samples). These are numerical invariance tests, not a proof that finite sampling is exactly invariant.', '',
        'Voxelization is a separate source of error. Four synthetic shapes, an unrotated reference plus four rotated segmentations, at 32³, 64³ and 96³: median within/between ratios were '+', '.join(f'{100*r["ratio"]:.2f}%' for r in resolution)+'. Identity retrieval was 100% in this small four-object test. The angular-only test uses analytic rotations; the voxel test actually resegments onto each voxel grid.', '',
        '## 2. Morphological sensitivity and information loss','',
        '270 controlled shapes: elongation, protrusion, indentation, three-lobed radial branching proxy and roughness; three widths, three orientations, six deformation magnitudes. The radial branching proxy is supplemented by genuinely bent, non-star-shaped tubes in experiment 3. Roughness amplitudes are normalized to equal spherical RMS, separately from spatial frequency.', '',
        '| Roughness degree | True radial RMS deformation | L12 power distance |','|---:|---:|---:|']
    for width,r in rough.iterrows():report.append(f'| {int(round(2+4/width))} | {r.radial_deformation_rms:.6f} | {r.power_distance:.6f} |')
    report+=['', 'The same ~0.105 radial deformation can almost disappear when it lies above the harmonic cutoff. Tiny residuals are quadrature leakage, not faithful recovery. Monotonic distance alone is therefore an inadequate sensitivity test: it can be monotonic and still discard nearly all of a feature.', '',
        '## 3. Radial parameterization versus general spherical mapping','',
        f'Of the 48 real reference cells, **{len(valid)} / 48** have closed, connected, consistently oriented genus-zero topology. All eligible meshes passed the cMCF-inspired map checks (no flipped/degenerate spherical faces; degree one). Others were left ineligible: no holes were filled, handles removed, or disconnected components silently discarded.', '',
        'This is an independent finite-element cMCF-inspired implementation, not the authors’ binary, SPHARM-PDM, or SPHARM-OT. The flow uses 45 steps at dt0.04; validity does not prove full convergence or a globally optimal mapping. A supplementary Möbius centering against original surface area was added after the distortion audit. It is **not an equal-area map**.', '',
        f'Mean reconstruction scores below use the same {len(valid)} real cells. Chamfer is squared distance in surface-RMS-normalized coordinates; P95 is unsquared normalized distance. These are geometric proxies. Sampling differs slightly for the 40³ shell reconstruction.', '',
        '| Reconstruction | Mean squared Chamfer ↓ | Mean P95 ↓ |','|---|---:|---:|']
    for k in ['radial','general','general_balanced','shell_full_coefficients']:report.append(f'| {k} | {recon.loc[k,"chamfer_squared"]:.6f} | {recon.loc[k,"p95"]:.6f} |')
    report+=['', 'The following deliberately difficult controls expose the single-radius assumption:', '', '| Shape | Rays with any hit | Radial Chamfer | General L12 Chamfer | Centered general L12 Chamfer | Shell-coefficient Chamfer |','|---|---:|---:|---:|---:|---:|']
    for r in controls:
        b=blookup[r['shape']];gen=f'{r["general"]["chamfer_squared"]:.5f}' if 'general' in r else 'ineligible';bal=f'{b["reconstruction"]["chamfer_squared"]:.5f}' if 'reconstruction' in b else 'ineligible'
        report.append(f'| {r["shape"]} | {100*r["radial_qc"]["coverage"]:.1f}% | {r["radial"]["chamfer_squared"]:.5f} | {gen} | {bal} | {r["shell_full_coefficients"]["chamfer_squared"]:.5f} |')
    report+=['', 'For missing radial directions, nearest-angle filling is used only to keep descriptor arrays available; these shapes are not treated as valid radial mappings. Outermost intersections discard inner boundaries. The shell representation directly samples 3D occupancy and handles these non-star-shaped cases without a spherical surface map.', '',
        '**Important:** radial/general reconstructions use full complex coefficients; shell reconstructions use the full radial-shell coefficient field before reducing to powers. A power spectrum itself does not determine a unique reconstructable surface. The existing shell descriptor is therefore a retrieval reference, not an invertible decoder.', '',
        '![Reconstructions](reconstructions.png)','', '## 4. Power spectrum versus bispectrum','',
        f'32 deliberately constructed pairs have equal energy at every degree but different phase/band relationships. Maximum power-descriptor difference is **{max(r["power_distance"] for r in equal):.2e}**; all pairs have nonzero bispectrum difference. Eight pairs were additionally checked with proper-rotation surface ICP to establish nonzero alignment-aware shape difference.', '',
        'The bispectrum is implemented through Clebsch–Gordan contractions. Even-parity real and odd-parity imaginary terms are retained, with sorted triples through L8: 60 nonredundant entries. The power descriptor has 13 entries at L12. Raw coefficient arrays are not rotation invariant; powers and bispectra are invariant in the continuous theory, with numerical quadrature errors measured here.', '',
        '![Equal-power shapes](equal_power_shapes.png)','',
        'A supplementary 64-object phase-only challenge extends the pairs through L12 to match the training generator’s bandwidth. At 0.5% radial noise, power alone retrieves **'+f'{100*phase[(phase.training_domain=="synthetic")&(phase.method=="radial_power")].identity.iloc[0]:.1f}%**'+', while bispectrum and multi-shell power each retrieve **100%**. The shell representation can therefore recover some phase-sensitive morphology through its nonlinear occupancy fields without explicitly appending a bispectrum. This challenge was added after primary readouts, with all trained models frozen.', '',
        'On real cells, bispectrum does not improve the matched-neighbor result under either raw concatenation or training-only standardization. This is a capacity demonstration, not evidence that bispectrum is currently the best biological descriptor.', '',
        '## 5. Can learning improve the descriptors?','',
        'A 489D input combines 13 radial powers, 60 bispectrum terms and 416 shell powers. A small denoising VAE compresses it to 128D, trained without shape-family/biological labels; three seeds, 120 epochs, reconstruction + KL1e-5 + consistency0.1. Standardization is fit on training data only (std floor1e-6).', '',
        f'Synthetic split: 384 training / 64 development / 128 new test shapes. Real split: **{len(realdata["train"])} training / {len(realdata["dev"])} group-separated development / 140 original validation cells**. The 48-cell matched morphology cohort is reused from prior research; it is not a new biological holdout. Real training augmentation is small feature-space noise; physical surface perturbations are used for evaluation, not training. This limits what the robustness result can establish.', '',
        'The 128D bottleneck and self-supervised reconstruction are ShapeEmbed-inspired. This is **not a faithful spherical reproduction of ShapeEmbedLite**: its EDM input, permutation-matched EDM objective and coordinate decoder are replaced with an invariant-feature denoising objective. It cannot reconstruct a unique 3D surface from this invariant code. No claim is made of a finished ShapeEmbed3D method.', '',
        '### Matched real-cell retrieval','',
        '46 queries / 11 similarity groups. Candidate cells come from other groups and are matched in log volume, elongation and sphericity, using the same pools as previous experiments. Cost is proper-rotation geometry of the selected neighbor; lower is better. Group means receive equal weight. Confidence intervals are descriptive paired group bootstraps (2,000 draws), not multiple-comparison-corrected.', '',
        '| Method | Geometry cost ↓ | Improvement vs nine features, 95% CI |','|---|---:|---|']
    for key in ['traditional_nine','radial_power','radial_energy','radial_power_bispectrum_standardized','bispectrum_standardized','shell_power','block_balanced_invariants','all_invariants_standardized','PCA128','VAE128_mean','frozen_shell_adjusted']:
        r=top[key];lo,hi=r['improvement_ci95'];report.append(f'| {r["name"]} | {r["geometry_cost"]:.6f} | {r["improvement_vs_nine"]:+.6f} [{lo:+.6f}, {hi:+.6f}] |')
    report+=['',f'The mean VAE improvement against existing adjusted shell harmonics is {vs_shell["improvement"]:+.6f}, paired group-bootstrap interval [{vs_shell["ci95"][0]:+.6f}, {vs_shell["ci95"][1]:+.6f}]. Negative favors the existing reference. These compared methods have different fitting budgets; it is a descriptive reference comparison, not a controlled architecture-only ablation.', '',
        'The valid-mapping-only comparison uses 21 queries / 10 groups for **every** method. See `retrieval_summary.csv`; full-cohort scores must not be compared directly with this smaller selected cohort.', '',
        '### Physical noise robustness','',
        '24 real validation cells were properly rotated and displaced along vertex normals using a smooth random field with standard deviation 0%, 0.5% or 2% of RMS radius. Same-cell retrieval is measured against all 140 validation cells. These are synthetic surface perturbations, not repeated biological acquisitions.', '',
        '| Descriptor | No noise | 0.5% surface noise | 2% surface noise |','|---|---:|---:|---:|']
    for key in ['radial_power','shell_power','all_invariants_standardized','PCA128','VAE128']:
        report.append('| '+key+' | '+' | '.join(f'{100*noise_summary.loc[(key,n)]:.1f}%' for n in [0,1,2])+' |')
    report+=['', 'Whitening very small harmonic variances amplifies perturbations; the learned model does not solve this. On the easier random synthetic test, unscaled invariants already achieve 100% identity retrieval, leaving no room for a learned improvement. The phase-only supplement is harder and exposes additional loss in the learned codes. The supplementary controlled-family precision@5 test excludes duplicate orientations of the same generated shape; its labels are known deformation families, not biological cell types (`controlled_family_retrieval.csv`).', '',
        '## Recommendation','',
        'Keep multi-shell harmonics as the principal shape baseline. Use a radial representation only after explicit star-convexity/coverage checks; never silently treat a concave surface as a radial graph. The current general-map prototype is restricted by topology and parameter distortion, so it should remain a diagnostic branch.', '',
        'For the next ShapeEmbed-style model, prioritize reconstructing the full multi-shell occupancy/coefficient field, with robust block weighting and physically generated training views, instead of reconstructing only a power spectrum. Retain an invariant descriptor for retrieval and an equivariant/pose-conditioned path for reconstruction; this avoids asking an invariant code to choose a unique orientation or recover phases it never received. Any model must beat raw and nuisance-adjusted shell baselines on both matched geometry and surface-noise stability before Atlas integration.', '',
        'A more serious spherical-surface branch should test an author implementation and/or area-preserving SPHARM-OT, quantify distortion/convergence, and separately handle topology-ineligible shapes. This study does not establish a new methodological contribution or biological signal beyond morphometry.', '',
        '## Files, reproducibility and sources','',
        '`protocol.json` records the run settings and input hashes; `source_snapshot/` stores the study code; per-cell caches retain coefficients and mapping metadata. CSV/NPZ files contain all readouts, models and scalers. `validation.json` records preservation checks. Main entry points: `run_spherical_shape_study.py`, `learn_spherical_shape_study.py`, `extra_spherical_shape_study.py`; supplementary scale, balancing, phase and controlled-family scripts are included.', '',
        '- [ShapeEmbedLite authors’ repository](https://github.com/uhlmanngroup/ShapeEmbedLite): conceptual reconstruction-learning reference; the present feature VAE is an explicit departure.',
        '- [Kazhdan et al., rotation-invariant spherical harmonics](https://www.cs.jhu.edu/~misha/MyPapers/SGP03.pdf): shell-power reference.',
        '- [Kazhdan, Solomon and Ben-Chen, cMCF](https://arxiv.org/abs/1203.6819) and [authors’ spherical mapping code](https://github.com/mkazhdan/MoebiusRegistration): mapping/centering reference; our implementation is independent.',
        '- [Oreiller et al., spherical bispectrum in biomedical imaging](https://arxiv.org/abs/2004.13371): phase-sensitive invariant motivation, not an implementation of their CNN.',
        '- [SPHARM-OT](https://pmc.ncbi.nlm.nih.gov/articles/PMC10676763/): area-preserving mapping direction; not executed here.', '']
    (OUT/'report.md').write_text('\n'.join(report))
    # Verify source preservation after all computations.
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['source_hashes'].items():assert sha(ROOT/path)==digest,path
    frame=pd.read_csv(OUT/'real_manifest.csv')
    for row in frame.itertuples():assert sha(ROOT/'results/explorer_dataset'/row.sample/f'{row.label}.mesh')==row.source_mesh_sha256,row.cell_id
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    names=['spherical_shape_study_core.py','run_spherical_shape_study.py','learn_spherical_shape_study.py','extra_spherical_shape_study.py','diagnose_spherical_shape_study.py','balance_spherical_shape_study.py','enrich_spherical_readouts.py','controlled_spherical_readouts.py','report_spherical_shape_study.py']
    for name in names:shutil.copy2(ROOT/'scripts'/name,snapshot/name)
    shutil.copy2(ROOT/'tests/test_spherical_shape_study.py',snapshot/'test_spherical_shape_study.py')
    save_json(snapshot/'SHA256.json',{p.name:sha(p) for p in snapshot.glob('*.py')})
    protocol['actual_real_split']=dict(train=len(realdata['train']),development=len(realdata['dev']),validation=len(realdata['validation']))
    protocol['supplementary']='Scale-controlled ablations, Möbius area-centering, L32 map reconstruction, bandwidth-matched equal-power phase challenge, and controlled-family retrieval were added after primary diagnostics. Models remained frozen; these are exploratory analyses.'
    save_json(OUT/'protocol.json',protocol)
    save_json(OUT/'validation.json',dict(status='passed',source_dataset_and_source_caches_unchanged=True,source_meshes_verified=609,study_unit_checks=4,real_objects=609,topology_eligible=len(valid),mapping_checks='No flips and degree one; not a global injectivity/convergence certificate',atlas_modified=False))
    save_json(OUT/'completion.json',dict(status='complete',experiments=5,real_cells=609,synthetic_controlled_shapes=270,synthetic_learning_shapes=576,report=str(OUT/'report.md')))
    print('Report, figures and source preservation validation complete:',OUT/'report.md',flush=True)
if __name__=='__main__':main()
