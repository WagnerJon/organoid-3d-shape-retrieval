"""Report paired, cross-organoid fractional Zernike experiments."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fractional_zernike_mpl')
from run_fractional_zernike import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def name(key):
    if key not in KEYS:return prior.NAMES.get(key,key)
    field,alpha=key.removesuffix('_minmax').split('_a')
    return f'{field.title()} Zernike α={alpha}'+(' · fold MinMax' if key.endswith('_minmax') else ' · L2')


def paired(a,b):
    """Paired percent cost reduction, bootstrap whole organoids only."""
    df=a[['query','organoid','geometry_cost']].merge(b[['query','geometry_cost']],on='query',validate='one_to_one',suffixes=('_new','_ref'))
    means=df.groupby('organoid')[['geometry_cost_new','geometry_cost_ref']].mean().to_numpy()
    draws=np.random.default_rng(42).integers(0,len(means),(4000,len(means)))
    boot=means[draws].mean(1);values=100*(1-boot[:,0]/boot[:,1])
    return dict(percent_reduction=float(100*(1-means[:,0].mean()/means[:,1].mean())),ci95=np.percentile(values,[2.5,97.5]).tolist())


def report():
    g=pd.read_csv(OUT/'retrieval.csv');s=pd.read_csv(OUT/'separation.csv');r=pd.read_csv(OUT/'robustness.csv')
    oldg=pd.read_csv(OLD/'retrieval_comparison.csv');olds=pd.read_csv(OLD/'separation_comparison.csv');oldr=pd.read_csv(OLD/'robustness.csv')
    g=pd.concat([g,oldg]);s=pd.concat([s,olds]);r=pd.concat([r,oldr]);result={}
    for key,a in g.groupby('method',sort=False):
        b=s[s.method==key]
        entry=dict(name=name(key),geometry=prior.bootstrap(a.geometry_cost,a.organoid),
                   ordering=prior.bootstrap(b.near_far_ordering,b.organoid),spearman=prior.bootstrap(b.spearman,b.organoid))
        for scope in ('clean','rotation','noise'):
            b=r[(r.method==key)&(r.scope==scope)]
            if len(b):entry[scope]=prior.bootstrap(b.correct,b.organoid,True)
        result[key]=entry
    comparisons={}
    for field in ('volume','surface'):
        for suffix in ('','_minmax'):
            key=f'{field}_a0.38{suffix}';baseline=f'{field}_a1{suffix}'
            comparisons[key+'_vs_'+baseline]=paired(g[g.method==key],g[g.method==baseline])
    for key in ('volume_a0.38','surface_a0.38'):
        comparisons[key+'_vs_shells']=paired(g[g.method==key],g[g.method=='spherical_harmonics'])
    summary=dict(methods=result,paired_comparisons=comparisons,represented=961,included=747,queries=100,probes=25,organoids=5,
        interpretation='Descriptive geometric retrieval on existing target benchmark. No alpha tuned on these queries. Multiple exploratory variants and only five organoids; not independent confirmation, human similarity annotation, or biological validation.')
    save_json(OUT/'summary.json',summary)
    rows=[]
    for key,v in result.items():rows.append(dict(method=key,name=v['name'],geometry=v['geometry']['mean'],geometry_ci_low=v['geometry']['ci95'][0],
        geometry_ci_high=v['geometry']['ci95'][1],ordering=v['ordering']['mean'],spearman=v['spearman']['mean'],
        noise=v.get('noise',{}).get('mean'),rotation=v.get('rotation',{}).get('mean')))
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    selected=['morph_all','spherical_harmonics','zernike20','volume_a1','volume_a0.38','surface_a1','surface_a0.38','volume_a0.38_minmax','surface_a0.38_minmax']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    means=[result[k]['geometry']['mean'] for k in selected]
    err=np.array([[v-result[k]['geometry']['ci95'][0],result[k]['geometry']['ci95'][1]-v] for k,v in zip(selected,means)]).T
    colors=['#999999','#3fa79d','#8ba1b4','#b4a4c4','#985bb5','#b4a4c4','#985bb5','#688cb5','#688cb5']
    axes[0].barh(range(len(selected)),means,xerr=err,capsize=2,color=colors)
    axes[0].set_yticks(range(len(selected)),[name(k) for k in selected]);axes[0].invert_yaxis()
    axes[0].set_xlabel('Cross-organoid neighbor geometry cost ↓')
    axes[0].set_title('100 fixed queries · five organoids · descriptive 95% intervals')
    robust=[k for k in selected if 'noise' in result[k]]
    axes[1].barh(np.arange(len(robust))-.18,[100*result[k]['noise']['mean'] for k in robust],height=.35,label='2% smooth surface noise',color='#3fa79d')
    axes[1].barh(np.arange(len(robust))+.18,[100*result[k]['rotation']['mean'] for k in robust],height=.35,label='Random mesh rotation',color='#985bb5')
    axes[1].set_yticks(range(len(robust)),[name(k) for k in robust]);axes[1].invert_yaxis();axes[1].set_xlim(0,105)
    axes[1].set_xlabel('Same-cell retrieval (%) ↑');axes[1].set_title('25 fixed probes · 747-cell gallery')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.10),fontsize=8,ncol=2)
    fig.suptitle('3D Zernike: does a fractional radial basis help cell-shape retrieval?',fontsize=16)
    fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    convergence=pd.read_csv(OUT/'convergence.csv');cross=json.loads((OUT/'classical_crosscheck.json').read_text());demo=pd.read_csv(OUT/'demo.csv')
    encoding=json.loads((OUT/'encoding.json').read_text())
    lines=['# Classical and fractional 3D Zernike descriptors','',
        'Computed eight 121D descriptor variants for all **961 objects** in the five-organoid dataset. Retrieval retains the existing **747 QC-passing/main-organoid cells**. Experimental files are separate from Atlas and its datasets.','',
        'The primary paired comparison fixes the input, sampling, normalization and distance, and changes only the radial exponent from classical α=1 to the paper-recommended α=0.38. α=0.14 and0.20 are prespecified secondary candidates. There is no target-dataset encoder training or geometry-label fitting.','',
        '![Comparison](comparison.png)','',
        '## Main comparison','',
        '| Descriptor | Geometric cost ↓ (95% descriptive CI) | Near/far ordering ↑ | Noise identity ↑ | Rotation identity ↑ |',
        '|---|---:|---:|---:|---:|']
    for key in selected:
        v=result[key];m=v['geometry'];noise=f"{v['noise']['mean']:.0%}" if 'noise' in v else 'Not retested';rotation=f"{v['rotation']['mean']:.0%}" if 'rotation' in v else 'Not retested'
        lines.append(f"| {name(key)} | {m['mean']:.5f} [{m['ci95'][0]:.5f}, {m['ci95'][1]:.5f}] | {v['ordering']['mean']:.1%} | {noise} | {rotation} |")
    lines+=['','Positive paired reductions favor fractional Zernike; intervals resample five whole organoids, preserving paired queries:','']
    for key,value in comparisons.items():lines.append(f"- {key}: {value['percent_reduction']:.1f}% reduction,95% interval [{value['ci95'][0]:.1f}%, {value['ci95'][1]:.1f}%].")
    lines+=['','Only five organoids and25 perturbation probes are available. Each query searches the other four organoids. Geometric cost uses the unchanged24-start proper-rotation ICP proxy on768 deterministic area samples, without reflection or scale fitting. Near/far ordering compares the four closest versus four most distant among16 fixed references per query. These are geometric proxies, not biological labels. Multiple secondary variants are exploratory; selecting the best row is not independent validation.','',
        '## Numerical checks','',
        'Unit checks establish radial orthogonality, the classical limit, agreement with the paper’s Section4 recurrence, constant-ball behavior, rigid/scale/translation stability, shape discrimination, and held-out-organoid scaling.','',
        f"Independent classical crosscheck against the existing ZMPY3D64³ implementation: median descriptor L2 difference {cross['median_l2']:.5f},95th percentile {cross['p95_l2']:.5f}. Changes between numerical integration schemes must not be attributed to the fractional exponent.",'',
        'A finer160×48×96 quadrature is checked on one prespecified probe per organoid, against96×32×64 production quadrature. This is a convergence diagnostic, not an accuracy label:','',
        '| Field / exponent | Median coarse–fine L2 drift | Median finer-grid rotation drift | Same-cell recall from finer encoding |','|---|---:|---:|---:|']
    for key,a in convergence.groupby('method',sort=False):lines.append(f"| {key} | {a.coarse_fine_l2.median():.5f} | {a.fine_rotation_l2.median():.5f} | {a.fine_recall.sum()}/{len(a)} |")
    lines+=['','## Synthetic diagnostic','',
        'All72 bundled demo objects were also encoded. Nearest-neighbor family recognition searches the other11 galleries, avoiding the query’s own gallery. Six known synthetic families are evaluation labels only. This easy synthetic task does not establish biological usefulness.','',
        '| Descriptor | Correct synthetic family |','|---|---:|']
    for key,a in demo.groupby('method',sort=False):lines.append(f"| {key} | {a.correct.sum()}/{len(a)} ({a.correct.mean():.1%}) |")
    lines+=['','## Implementation and departures from the paper','',
        'See [implementation note](../../docs/FRACTIONAL_ZERNIKE.md). We use a stable shifted-Jacobi representation of the radial basis satisfying the paper’s orthogonality condition and recurrence. The printed unit-step Pochhammer in Eq.9 is inconsistent with them, including at α=1. Our implementation follows the orthogonal basis rather than copying that apparent formula error.','',
        'The paper uses binary surface voxels and full-dataset MinMax scaling. We compare a paper-like surface band and a filled-volume control, both from unchanged meshes. Both use the exact volume centroid (area fallback for open meshes) and a shared0.9-unit-ball support. Spherical quadrature replaces Cartesian sums. Primary distances use L2-normalized invariant vectors. Secondary MinMax uses mass-normalized moments and fits only the other four organoids; constant coordinates are suppressed and query values are not clipped. These controlled adaptations are not a reproduction of Princeton benchmark numbers.','',
        'Invariant block norms discard phase and chirality. They cannot uniquely reconstruct geometry and cannot guarantee separation of every visibly different pair.','',
        f"CPU encoding: median {encoding['median_seconds']:.3f}s/object; peak recorded worker RSS {encoding['max_worker_rss_bytes']/1024**3:.2f}GiB; two workers. Runtime is measured on this machine; no image stack is loaded.",'',
        '## Outputs and reproduction','',
        '`codes.npz`: IDs and eight mass-normalized121D vectors for all961 objects. Normalize each row to unit L2 for the primary metric. `neighbors.csv`: ten cross-organoid neighbors for four primary variants, included objects only. `comparison.csv`, `retrieval.csv`, `separation.csv`, `robustness.csv`, `convergence.csv`, `demo.csv`: numerical results. `protocol.json`: frozen configuration, source hashes and scope.','',
        'Run `.venv/bin/python scripts/run_fractional_zernike.py encode`, then the `evaluate`, `probes`, `audit`, and `demo` stages, followed by `.venv/bin/python scripts/report_fractional_zernike.py`. Existing five-organoid inputs and benchmark caches are required; each original mesh hash is checked when encoding.','',
        '## Sources','',
        '- Ashton & Sit-Seth (2026), [New 3D Fractional-Order Zernike Moments for Shape Analysis](https://doi.org/10.3390/appliedmath6100174).','- [ZMPY3D author implementation](https://github.com/tawssie/ZMPY3D), used for the independently cached classical baseline.','']
    # Keep scientific prose readable when numeric values follow words.
    import re
    prose='\n'.join(lines)
    prose=re.sub(r'\b(All|all|only|Only|and|with|on|the|a|finer|against|unchanged|using|order|Section|Eq\.|width|radius|five|per|for)(?=\d)',r'\1 ',prose)
    (OUT/'report.md').write_text(prose)
    print(json.dumps(dict(paired_comparisons=comparisons,primary={k:result[k] for k in ('volume_a1','volume_a0.38','surface_a1','surface_a0.38')}),indent=2))


if __name__=='__main__':report()
