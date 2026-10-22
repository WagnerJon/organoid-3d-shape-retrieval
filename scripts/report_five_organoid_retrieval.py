"""Select, visualize, stage and publish the five-organoid retrieval result."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/organoid_retrieval_mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/organoid_retrieval_cache')
from five_organoid_retrieval_core import *
import argparse, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from report_spherical_shape_study import mesh_axis
from spectral64_core import canonical


def fmt(x,percent=False):
    mul=100 if percent else 1;digits=1 if percent else 5
    return f'{x["mean"]*mul:.{digits}f}'+('%' if percent else '')+f' [{x["ci95"][0]*mul:.{digits}f}, {x["ci95"][1]*mul:.{digits}f}]'


def selected_summary():
    g=pd.read_csv(OUT/'retrieval_comparison.csv');r=pd.read_csv(OUT/'robustness.csv');s=pd.read_csv(OUT/'separation_comparison.csv')
    result={}
    for key in METHODS:
        a=g[g.method==key];b=s[s.method==key]
        result[key]=dict(name=NAMES[key],geometry=bootstrap(a.geometry_cost,a.organoid),
                         near_far_ordering=bootstrap(b.near_far_ordering,b.organoid),
                         spearman=bootstrap(b.spearman,b.organoid))
        if key in PRIMARY:
            for scope in ['clean','noise','rotation']:
                f=r[(r.method==key)&(r.scope==scope)]
                result[key][scope]=bootstrap(f.correct,f.organoid,True)
                if scope=='noise':result[key]['noise_jaccard']=bootstrap(f.neighbor_jaccard,f.organoid,True)
    baseline=result['spherical_harmonics']['noise']['mean']
    qualified=[key for key in PRIMARY if result[key]['clean']['mean']==1. and
               result[key]['rotation']['mean']>=.96 and result[key]['noise']['mean']>=baseline-.04-1e-12]
    if 'spherical_harmonics' not in qualified:
        raise RuntimeError('Raw shell reference fails its own reproduction/invariance checks; inspect before selecting')
    selected=min(qualified,key=lambda key:result[key]['geometry']['mean'])
    save_json(OUT/'selection.json',dict(method=selected,name=NAMES[selected],qualified=qualified,
        rule='Predeclared practical shortlist, fresh clean100%, rotation>=96%, noise within one of25 probes of raw shells; lowest cross-organoid geometry cost',
        metrics=result[selected],benchmark_is_descriptive=True,noise_probes=25,queries=100))
    save_json(OUT/'summary.json',dict(selected=selected,methods=result,organoids=5,
        warning='Target benchmark used for method selection, only five organoids and25 perturbation probes; not independent confirmation or a universal-best claim'))
    return selected,result


def project_active(vectors,distances,included,hybrid=False):
    if hybrid:
        ids=np.flatnonzero(included);sq=distances[np.ix_(ids,ids)]**2
        means=sq.mean(0);grand=sq.mean();b=-.5*(sq-means[None]-means[:,None]+grand)
        eigen,vec=np.linalg.eigh(b);order=np.argsort(eigen)[::-1][:2];positive=eigen[order].clip(1e-12)
        rectangular=distances[:,ids]**2
        centered=-.5*(rectangular-rectangular.mean(1)[:,None]-means[None]+grand)
        xy=centered@vec[:,order]/np.sqrt(positive)
        variance=(positive/eigen.clip(0).sum()).tolist()
        # Display groups only; retrieval uses the original full metric.
        from sklearn.cluster import AgglomerativeClustering
        cluster=AgglomerativeClustering(n_clusters=6,metric='precomputed',linkage='average').fit_predict(distances[np.ix_(ids,ids)])
        groups=cluster[np.argmin(distances[:,ids],axis=1)]
    else:
        pca=PCA(n_components=2,svd_solver='randomized',random_state=42).fit(vectors[included])
        xy=pca.transform(vectors);variance=pca.explained_variance_ratio_.tolist()
        group=KMeans(n_clusters=6,n_init=10,random_state=42).fit(vectors[included]);groups=group.predict(vectors)
    return xy,groups,variance


def stage():
    selected,result=selected_summary();data=source();protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(ATLAS/'dataset.json')==protocol['source_dataset_sha256']
    frame=pd.read_csv(OUT/'manifest.csv');included=frame.included.to_numpy(bool);ids=np.flatnonzero(included)
    codes=np.load(OUT/'zernike20.npz')['vectors'];cached=np.load(OUT/'all_distances.npz');d=cached[selected]
    # Default search index excludes all QC-failed objects, including distant examples.
    nearest=[];farthest=[]
    for i in ids:
        allowed=ids[ids!=i];order=allowed[np.argsort(d[i,allowed],kind='stable')]
        for scope,pool in [('all_included',order),('other_organoids',order[frame.iloc[order]['sample'].to_numpy()!=frame.iloc[i]['sample']])]:
            for rank,j in enumerate(pool[:10],1):
                nearest.append(dict(query_cell=frame.iloc[i].cell_id,query_organoid=frame.iloc[i]['sample'],
                     scope=scope,rank=rank,neighbor_cell=frame.iloc[j].cell_id,neighbor_organoid=frame.iloc[j]['sample'],distance=float(d[i,j])))
        for rank,j in enumerate(order[::-1][:10],1):
            farthest.append(dict(query_cell=frame.iloc[i].cell_id,query_organoid=frame.iloc[i]['sample'],
                     rank=rank,dissimilar_cell=frame.iloc[j].cell_id,dissimilar_organoid=frame.iloc[j]['sample'],distance=float(d[i,j])))
    pd.DataFrame(nearest).to_csv(OUT/'nearest_neighbors.csv',index=False)
    pd.DataFrame(farthest).to_csv(OUT/'dissimilar_examples.csv',index=False)
    x=(codes if selected=='zernike20' else np.array([c['representations'][selected]['vector'] for c in data['cells']]))
    np.savez_compressed(OUT/'recommended_representations.npz',ids=frame.cell_id.to_numpy(dtype=str),
                        included=included,vectors=x,distances=d.astype(np.float32),method=selected)
    pd.DataFrame(dict(cell_id=frame.cell_id,qc_included=included,**{f'ZD{i:03d}':codes[:,i] for i in range(121)})).to_csv(OUT/'zernike20.csv',index=False)
    # Add Zernike and an explicit recommended entry; existing method vectors remain exact.
    specs={m['id']:m for m in data['methods']}
    additions=[('zernike20','3D Zernike ·121D',codes,'euclidean',cached['zernike20'],False)]
    if selected!='zernike20':
        additions.append(('shape_retrieval_recommended','Recommended similarity · '+NAMES[selected],x,
                          specs[selected]['metric'],d,selected=='spatial_robust'))
    else:additions[0]=('zernike20','Recommended similarity ·3D Zernike',codes,'euclidean',d,False)
    for key,name,vectors,metric,matrix,hybrid in additions:
        xy,groups,variance=project_active(vectors,matrix,included,hybrid)
        description=('Published ZMPY3D order20 volumetric Zernike invariant norms,121D,64³ soft occupancy, translation/scale/rotation-invariant construction with finite-grid error. CPU inference; no shape training, whitening or morphometry residualization.' if key=='zernike20' else
                     'Alias of existing '+selected+' using the same full vectors and distance. Recommended by the documented five-organoid cross-organoid comparison and25 physical perturbation probes; not a newly trained model.')
        description+=' All961 objects encoded;747 QC-passing/main-organoid objects determine display PCA/MDS and six exploratory display groups. Nearest-neighbor exports exclude QC-failed objects. Search uses full descriptor/metric, not plot coordinates. Scale is ignored for geometric shape methods.'
        for i,c in enumerate(data['cells']):c['representations'][key]=dict(vector=vectors[i].tolist(),point=xy[i].tolist(),cluster=int(groups[i]))
        data['methods'].append(dict(id=key,name=name,dimensions=vectors.shape[1],metric=metric,
            projection='mds' if hybrid else 'pca',variance=variance,stackVariance=[0.,0.],family='shape',
            scope='cells',coverage='all961 objects;747 included by existing QC',description=description))
    preferred='zernike20' if selected=='zernike20' else 'shape_retrieval_recommended'
    data['shapeRetrieval']=dict(recommendedMethod=preferred,sourceMethod=selected,
         includedCells=747,representedCells=961,benchmark='shape_retrieval/report.md',
         rule='Full descriptor distance; QC/main-organoid inclusion; source organoid exclusion only for cross-organoid benchmark and optional index.')
    data['provenance']+=' Added published ZMPY3D descriptors and a documented practical shape-retrieval recommendation. Original measurements, meshes, QC, inclusion and existing method values preserved.'
    (OUT/'dataset.staged.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False))
    save_json(OUT/'staged.json',dict(before_sha256=protocol['source_dataset_sha256'],after_sha256=sha(OUT/'dataset.staged.json'),
                                   preferred=preferred,selected=selected,added=[a[0] for a in additions]))
    print('Staged practical recommendation:',NAMES[selected],flush=True)


def report():
    selected,result=selected_summary();g=pd.read_csv(OUT/'retrieval_comparison.csv');plan=json.loads((OUT/'plan.json').read_text())
    frame=pd.read_csv(OUT/'manifest.csv');protocol=json.loads((OUT/'protocol.json').read_text());r=pd.read_csv(OUT/'robustness.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    ordered=sorted(METHODS,key=lambda k:result[k]['geometry']['mean']);fig,ax=plt.subplots(1,2,figsize=(16,8),layout='constrained')
    values=[result[k]['geometry']['mean'] for k in ordered]
    errors=np.array([[v-result[k]['geometry']['ci95'][0],result[k]['geometry']['ci95'][1]-v] for k,v in zip(ordered,values)]).T
    ax[0].barh(range(len(ordered)),values,xerr=errors,color=['#dd9c65' if k==selected else '#5ba9a6' for k in ordered],capsize=2)
    ax[0].set_yticks(range(len(ordered)),[NAMES[k] for k in ordered]);ax[0].invert_yaxis();ax[0].set_xlabel('Cross-organoid geometric neighbor cost ↓')
    ax[0].set_title('100 fixed queries · frozen encoders · same eligible galleries')
    values=[result[k]['noise']['mean'] for k in PRIMARY]
    ax[1].bar(range(4),values,color=['#5ba9a6','#b89cce','#7f9ac4','#dd9c65'])
    ax[1].set_xticks(range(4),['Raw shells','Zernike20','Adapted\nPoint-MAE','Spatial\nPCA']);ax[1].set_ylim(0,1.05)
    ax[1].set_ylabel('Same-cell retrieval after2% surface perturbation ↑');ax[1].set_title('25 actual mesh perturbations ·747-cell gallery')
    fig.suptitle('Five organoids: choose by shape retrieval and robustness',fontsize=18);fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    # Fixed query1 per organoid; no selection of attractive or favorable examples.
    queries=plan['queries'][::20];distances=np.load(OUT/'all_distances.npz')[selected]
    fig=plt.figure(figsize=(15,17));example_rows=[]
    for row,i in enumerate(queries):
        gallery=frame.index[frame.included&(frame['sample']!=frame.iloc[i]['sample'])].to_numpy()
        ranking=gallery[np.argsort(distances[i,gallery],kind='stable')]
        for column,j in enumerate([i,int(ranking[0]),int(ranking[1]),int(ranking[-1])]):
            v,f=mesh(frame.iloc[j].to_dict());v,_=canonical(v,f)
            title=('Query' if column==0 else ['','Nearest','Second nearest','Dissimilar'][column])+'\n'+frame.iloc[j].cell_id
            mesh_axis(fig.add_subplot(5,4,row*4+column+1,projection='3d'),v,f,
                ['#cda277','#62aaa6','#7bbab0','#b48bb2'][column],title)
        example_rows.append(dict(query=frame.iloc[i].cell_id,nearest=frame.iloc[ranking[0]].cell_id,dissimilar=frame.iloc[ranking[-1]].cell_id))
    fig.suptitle('Full-space shape neighbors: '+NAMES[selected],fontsize=17)
    fig.text(.5,.015,'Fixed first query per organoid; original surfaces in a canonical display pose. Other-organoid QC-passing galleries. Individual framing; physical size is ignored.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.01,right=.99,top=.92,bottom=.04,wspace=.02,hspace=.05);fig.savefig(OUT/'neighbors.png',dpi=140);plt.close(fig)
    save_json(OUT/'fixed_visual_examples.json',example_rows)
    lines=['# Practical shape retrieval for the five-organoid dataset','',
        '**Recommendation: '+NAMES[selected]+'.** This is the most useful supported choice from our runnable shortlist for this dataset and the stated goal, not a universal best descriptor.','',
        'All961 objects have published3D Zernike descriptors. The default search uses the747 cells passing existing QC and main-organoid inclusion. The214 flagged objects keep their descriptors for inspection but are excluded from the neighbor/dissimilar index. Original measurements and meshes are preserved.','',
        '## What I compared','',
        'Twelve runnable descriptors were compared using100 fixed queries,20 per organoid, each searching only the other four organoids. Learned weights/scalers came from the previous dataset and were not retrained here. This prevents a nearest neighbor from being merely another cell in the same acquisition, but does not turn the selected target benchmark into independent validation.','',
        '![Method comparison](comparison.png)','',
        '| Method | Cross-organoid geometry cost ↓,95% descriptive CI | Near/far ordering ↑ |2% noise identity retrieval ↑ |',
        '|---|---|---|---|']
    for key in ordered:
        value=result[key];lines.append('| '+NAMES[key]+' | '+fmt(value['geometry'])+' | '+fmt(value['near_far_ordering'],True)+' | '+(fmt(value['noise'],True) if 'noise' in value else 'Not retested here')+' |')
    lines+=['','Geometry is a proper-rotation24-start surface-ICP proxy with768 independent area-sampled surface points and exact surface-RMS normalization. Every method uses the same eligible gallery. Costs have a different sampling/normalization from earlier studies and should not be directly compared with their0.02x scores.','',
        'Near/far ordering tests the requested separation: for each query,16 fixed random other-organoid cells are ranked by geometry. Descriptor distances should put the four geometrically closest ahead of the four farthest;50% is chance ordering. It is a geometric diagnostic, not human annotation of what looks similar.','',
        'Intervals resample only five organoids; they are descriptive and uncertainty is substantial. The method recommendation reuses this comparison. It is not an independent performance guarantee, and no finite descriptor ensures that every visibly different object has a distinct vector.','',
        '## Practical shortlist and robustness','',
        'The practical shortlist was fixed before target retrieval scores: raw shells, new Zernike, adapted pose-averaged Point-MAE and spatial PCA with ambiguity fallback. These were supported by previous evidence or by a suitable published CPU implementation. Other tested methods supply context but were not all re-run under perturbations; a low geometric cost alone does not establish their robustness.','',
        'Among that shortlist, settings must reproduce fresh clean queries perfectly, recover≥96% of rigidly rotated queries, and lose at most one of25 noisy-query identifications relative to raw shells. The lowest geometric-cost qualifying descriptor is recommended. No neural model or learned spectral weighting is trained on these five organoids.','',
        '| Shortlisted method | Fresh clean recall | Rotation recall |2% noise recall | Noisy top5 neighbor Jaccard |',
        '|---|---|---|---|---|']
    for key in PRIMARY:
        a=result[key];lines.append('| '+NAMES[key]+' | '+' | '.join(fmt(a[k],True) for k in ['clean','rotation','noise'])+' | '+fmt(a['noise_jaccard'])+' |')
    lines+=['','There are25 preselected probes, five per organoid, against all747 included cells. Noise is one smooth normal-displacement field with2% surface-RMS amplitude. Rigid rotation is applied to actual geometry and frames/voxels are recomputed. One noise realization is not a test of segmentation splitting, merging or missing cells. Fresh clean re-encoding checks that a preprocessing mismatch is not being mislabeled as noise sensitivity.','',
        '## Why these methods','',
        '- **3D Zernike descriptors:** a published whole-object retrieval representation that combines radial polynomials and spherical harmonics in the volume. The author-maintained [ZMPY3D NumPy implementation](https://github.com/tawssie/ZMPY3D) runs on this Mac CPU without CUDA or training. We use order20,121 invariant norms,64³ soft mesh occupancy, consistent centering/unit-ball scale, and full-vector L2 distance. This adapts its volume-input routines to cell meshes; it is not a protein-trained model or an official organoid benchmark. [Original retrieval method](https://doi.org/10.1016/j.cad.2004.01.005).','',
        '- **Shell harmonics:** the previously implemented multi-shell power construction, which had the strongest robust-distance evidence in the recent experiments. It runs on CPU and does not depend on the training distribution of a neural encoder. [Published construction/source](https://www.cs.jhu.edu/~misha/Code/ShapeSPH/ShapeDescriptor/).','',
        '- **Point-MAE:** an available pretrained self-supervised point-cloud model, already ported and strictly loaded in this project. We test the frozen organoid-adapted model with four proper PCA-sign poses averaged, rather than its rotation-sensitive native coordinates. It runs on the Mac GPU with our existing portable geometric operations; that is a local port, not the authors’ unchanged CUDA installation. [Official source](https://github.com/Pang-Yatian/Point-MAE).','',
        '- **MorphCell:** relevant recent cell-specific learning, but the checked release requires CUDA/pointnet2-ops and provides no weight-download link in its README. It is not a currently reproduced Mac benchmark, so no random or substituted weights are reported as its performance. [Official repository](https://github.com/Visytudz/MorphCell).','',
        'For ordinary visual shape similarity, I do not subtract elongation, sphericity or other morphology from the default representation. Those differences are often precisely what the user wants to retrieve. Shape methods here remove pose and physical scale; physical volume is still available in Atlas for a separate size constraint.','',
        'Zernike and harmonic norm descriptors lose phase and chirality, so they are not complete shape invariants. Zernike finite voxelization and order limit fine detail; shell quadrature can miss extremely narrow features. Neural embeddings also have no guarantee of separating every new morphology. The fixed near/far examples and full-space neighbor lists let you inspect practical errors.','',
        'Some currently QC-passing segmentations still contain detached pieces, visible among the fixed farthest examples. Existing QC is preserved rather than silently redefining cells or cutting meshes. These outputs rank the provided segmented objects; the near/far score is not a separately verified benchmark restricted to intact biological cells.','',
        '## Using the result','',
        'After publication, reopen the existing five-organoid Atlas dataset and select its last **Recommended similarity** method. All original comparison methods remain available. Search/ranking uses the full method-specific distance; PCA/MDS coordinates are only a display. The display fit and six exploratory display groups use only747 included cells.','',
        '`nearest_neighbors.csv` has the ten closest cells for every included cell, both across all included objects and with an other-organoid constraint. `dissimilar_examples.csv` has ten distant objects. `recommended_representations.npz` contains IDs, inclusion flags, full selected vectors and distances; `zernike20.npz`/`.csv` contain all961 new published-descriptor vectors. Scores are descriptor distances, not probabilities of biological similarity.','',
        '![Fixed query examples](neighbors.png)','',
        '## Reproduction and audit','',
        'The frozen protocol, query/probe IDs, source hashes, author-code commit, per-cell outputs and per-query geometric/robustness readouts are saved beside this report. Four study-specific tests check Zernike dimensions/discrimination, transformed geometry, hybrid-metric parity and QC/cross-organoid boundaries. Original Atlas fields, existing representation values and source meshes are checked before additive publication. The native app and trained models are unchanged.','']
    import re
    prose=re.sub(r'(?<=[A-Za-z])(?=\d)',' ','\n'.join(lines));prose=re.sub(r'(?<!\d),(?=\d)',', ',prose)
    # Preserve URLs and literal filenames that contain digits.
    prose=prose.replace('pointnet 2-ops','pointnet2-ops').replace('zernike 20','zernike20').replace('ZMPY 3D','ZMPY3D')
    (OUT/'report.md').write_text(prose)
    for row in frame.to_dict('records'):mesh(row)
    assert sha(ATLAS/'dataset.json')==protocol['source_dataset_sha256']
    save_json(OUT/'validation.json',dict(status='passed',source_meshes_verified=961,represented=961,included=747,
              qc_failed_excluded_from_search=214,target_encoder_training=False,source_atlas_unchanged_before_publication=True,tests=4))
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in [ROOT/'scripts/five_organoid_retrieval_core.py',ROOT/'scripts/run_five_organoid_retrieval.py',
              ROOT/'scripts/report_five_organoid_retrieval.py',ROOT/'tests/test_five_organoid_retrieval.py']:
        shutil.copy2(p,snapshot/p.name)
    save_json(snapshot/'SHA256.json',{p.name:sha(p) for p in snapshot.glob('*.py')})
    print('Comparison report and original-data audit complete',flush=True)


def publish():
    staged=json.loads((OUT/'staged.json').read_text());assert sha(OUT/'dataset.staged.json')==staged['after_sha256']
    assert json.loads((OUT/'validation.json').read_text())['status']=='passed'
    if sha(ATLAS/'dataset.json')!=staged['before_sha256']:raise RuntimeError('Atlas changed since staging')
    before=source();after=json.loads((OUT/'dataset.staged.json').read_text())
    assert len(before['cells'])==len(after['cells'])==961
    for a,b in zip(before['cells'],after['cells']):
        for key in a:
            if key!='representations':assert a[key]==b[key],key
        for key,value in a['representations'].items():assert value==b['representations'][key]
    for a,b in zip(before['organoids'],after['organoids']):assert a==b
    for m in before['methods']:assert m==next(v for v in after['methods'] if v['id']==m['id'])
    target=JOINT/'shape_retrieval';target.mkdir(exist_ok=True)
    backup=target/f'dataset.before-{staged["before_sha256"][:12]}.json'
    if not backup.exists():shutil.copy2(ATLAS/'dataset.json',backup)
    for name in ['report.md','comparison.png','neighbors.png','nearest_neighbors.csv','dissimilar_examples.csv',
                 'recommended_representations.npz','zernike20.npz','zernike20.csv','protocol.json','plan.json',
                 'selection.json','summary.json','retrieval_comparison.csv','separation_comparison.csv',
                 'fixed_pair_distances.csv','robustness.csv','validation.json','unit_checks.log']:
        shutil.copy2(OUT/name,target/name)
    shutil.copytree(OUT/'source_snapshot',target/'source_snapshot',dirs_exist_ok=True)
    temp=ATLAS/'dataset.shape-retrieval.tmp';shutil.copy2(OUT/'dataset.staged.json',temp);os.replace(temp,ATLAS/'dataset.json')
    assert sha(ATLAS/'dataset.json')==staged['after_sha256']
    completion=dict(status='complete',selected=staged['selected'],preferredAtlasMethod=staged['preferred'],
                    cells=961,included=747,source_backup=str(backup),before=staged['before_sha256'],
                    after=staged['after_sha256'],output=str(target),existing_values_preserved=True)
    save_json(target/'complete.json',completion);save_json(OUT/'complete.json',completion)
    print('Published recommendation and complete retrieval index:',target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['stage','report','publish'])
    globals()[parser.parse_args().stage]()
