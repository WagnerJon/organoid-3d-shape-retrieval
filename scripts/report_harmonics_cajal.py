"""Report published-method validation results and linked-space figure."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/harmonics_cajal'


def main():
    result=json.loads((OUT/'evaluation.json').read_text());rotation=json.loads((OUT/'rotation.json').read_text());rows=result['methods'];lookup={r['id']:r for r in rows};base=lookup['morph_all']['geometry_cost']
    lines=['# Spherical harmonics and CAJAL comparison','',result['scope'],'',
        f"{result['queries']} matched validation queries, {result['groups']} similarity groups; {result['candidate_pairs']} unique candidate pairs. Lower cost means better agreement with the existing segmented-cell surface geometry.",'',
        '| Method | Geometry cost ↓ | Change vs all 9 | Paired improvement, group bootstrap 95% CI |','|---|---:|---:|---|']
    for r in rows:
        ci=r['vs_morph_all'];lo,hi=ci['ci95'];lines.append(f"| {r['name']} | {r['geometry_cost']:.6f} | {100*(1-r['geometry_cost']/base):+.1f}% | {ci['mean']:+.6f} [{lo:+.6f}, {hi:+.6f}] |")
    approx=result['approximation']
    lines+=['','Positive percentage = lower geometric cost than all nine traditional features. Comparisons are exploratory and not corrected for selecting among many methods.','',
        f"Spherical harmonics preserved cell identity in {rotation['identity']:.1%} of 384 fresh-mesh rotations; mean five-neighbor Jaccard was {rotation['neighbor_jaccard']:.3f}. Mean code change was {rotation['mean_code_change']:.4f}; finite numerical quadrature is not exact rotation invariance.",'',
        f"Full-view quantized CAJAL and the unquantized validation solver selected the same nearest cell for {approx['top1_agreement']:.1%} of queries. Candidate-pair distance Spearman correlation was {approx['candidate_distance_spearman']:.3f}. The unquantized reference is also a local numerical solver, not a globally certified optimum.",'',
        'Atlas includes the quantized CAJAL method for all 1,088 cells, plus the raw and adjusted spherical harmonic spaces and an adjusted CAJAL landmark profile. The unquantized solver is separately available in Atlas under Validation benchmarks for all 48 validation cells. It is not silently substituted into the full-view matrix; all panels and neighbor marking use shared cells in that mode. Direct CAJAL neighbors use cached pairwise qGW values; the map is MDS, not PCA. The adjusted CAJAL profile uses distances to 96 training cells and a Euclidean residual-profile distance, which is a distinct approximation.','',
        'Both new shape methods discard orientation and scale; they also cannot distinguish mirror images. The harmonic power spectrum loses phase information. CAJAL depends on the finite surface sample, quantization and local solver optima. These are shape-comparison baselines, not pretrained biological foundation models.','',
        'Every adjusted comparison uses a degree-2 Ridge fit on the original 96 training cells to account for log volume, elongation and sphericity. The matched pools use the same three controls. Residualization is not proof of biological independence. Display projections and groups are descriptive included-cohort fits; ordinary filters do not refit them. Organoid plots summarize included QC cell vectors; direct CAJAL distances are cell-only.','',
        'The prior frozen spatial experiment is unchanged. No new test set was opened or used for model selection. Geometry agreement with a predicted source segmentation does not establish biological meaning; external conditions/phenotypes and independent specimens are still needed.','',
        'Implementation details and source references: [HARMONICS_CAJAL_COMPARISON.md](../../docs/HARMONICS_CAJAL_COMPARISON.md).','']
    (OUT/'report.md').write_text('\n'.join(lines))
    # Paired uncertainty remains on the absolute difference scale.
    names=[r['name'].replace(' · ','\n') for r in rows];means=np.array([r['vs_morph_all']['mean'] for r in rows]);ci=np.array([r['vs_morph_all']['ci95'] for r in rows])
    fig,ax=plt.subplots(figsize=(12,8));y=np.arange(len(rows));colors=['#009f97' if r['id'].startswith(('cajal','spherical')) else '#91a2b5' for r in rows]
    ax.barh(y,means,color=colors);ax.errorbar(means,y,xerr=np.maximum(np.stack([means-ci[:,0],ci[:,1]-means]),0),fmt='none',ecolor='#24374c',capsize=3)
    ax.set_yticks(y,names);ax.invert_yaxis();ax.axvline(0,color='#444',lw=.7);ax.set_xlabel('Reduction in normalized geometric neighbor cost vs all nine traditional features →')
    ax.set_title('Published-method comparison · validation only · 46 queries / 11 groups',loc='left',weight='bold');ax.spines[['right','top']].set_visible(False)
    fig.text(.02,.01,'Group bootstrap 95% intervals. Identical candidate pools. Geometry proxy uses predicted source surfaces, not biological labels.',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    data=json.loads((ROOT/'results/explorer_dataset/dataset.json').read_text());cells=[c for c in data['cells'] if c['qc'] and c['inMainOrganoid']]
    keys=['morph_all','spatial_robust','spherical_harmonics','cajal_qgw','morph_adjusted','spatial_adjusted','spherical_harmonics_adjusted','cajal_profile_adjusted']
    methods={m['id']:m for m in data['methods']};fig,axes=plt.subplots(2,4,figsize=(16,8));cmap=plt.get_cmap('tab10');colors=[cmap(c['representations']['morph_controls']['cluster']) for c in cells]
    for ax,key in zip(axes.ravel(),keys):
        m=methods[key];p=np.array([c['representations'][key]['point'] for c in cells]);ax.scatter(p[:,0],p[:,1],c=colors,s=6,alpha=.7,linewidths=0)
        ax.set_title(m['name'],fontsize=10);label='MDS' if m['projection']=='mds' else 'PC';ax.set_xlabel(label+' 1');ax.set_ylabel(label+' 2');ax.spines[['right','top']].set_visible(False)
    fig.suptitle('Same included QC cells · reference colors follow volume / elongation / sphericity groups',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'spaces.png',dpi=160);plt.close(fig)


    validation=set(data['harmonicsCajal']['validation_ids']);cells=[c for c in cells if c['id'] in validation]
    keys[3]='cajal_gw_validation';fig,axes=plt.subplots(2,4,figsize=(16,8));colors=[cmap(c['representations']['morph_controls']['cluster']) for c in cells]
    for ax,key in zip(axes.ravel(),keys):
        m=methods[key];p=np.array([c['representations'][key]['point'] for c in cells]);ax.scatter(p[:,0],p[:,1],c=colors,s=18,alpha=.8,linewidths=0)
        ax.set_title(m['name'],fontsize=9);label='MDS' if m['projection']=='mds' else 'PC';ax.set_xlabel(label+' 1');ax.set_ylabel(label+' 2');ax.spines[['right','top']].set_visible(False)
    fig.suptitle('Same 48 validation cells · unquantized CAJAL reference · descriptive projection fits',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'spaces_validation.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
