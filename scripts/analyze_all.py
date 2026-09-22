"""Full-cohort descriptive morphometry, quality screening and cell comparisons."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

OUT=ROOT/'results/cpsam_v2_all_analysis'
SOURCE=ROOT/'results/cpsam_v2_remaining'
FEATURES=['volume_um3','equivalent_diameter_um','surface_area_um2','sphericity','elongation','flatness','aspect_ratio','mesh_solidity']


def summary(frame):
    return {key:{'n':int(frame[key].count()),'mean':float(frame[key].mean()),'sd':float(frame[key].std()),
                  'median':float(frame[key].median()),'q25':float(frame[key].quantile(.25)),
                  'q75':float(frame[key].quantile(.75)),'min':float(frame[key].min()),'max':float(frame[key].max())} for key in FEATURES}


def main():
    OUT.mkdir(exist_ok=True)
    df=pd.read_csv(SOURCE/'all_cells_3d.csv').drop(columns=['spatial_group','distance_to_X001_um'],errors='ignore')
    df=df.sort_values(['sample','label']).reset_index(drop=True)
    metrics=pd.read_csv(SOURCE/'reconstruction_metrics.csv').sort_values('sample')
    assert len(df)==df.cell_id.nunique()
    assert metrics['sample'].nunique()==len(metrics)==108
    assert set(df['sample'])==set(metrics['sample'])
    assert int(df.reference_matched.sum())==int(metrics.true_positives.sum())
    assert np.isfinite(df[FEATURES]).all().all()
    df['substantial_fragmentation']=df.largest_component_fraction<.99
    df['invalid_sphericity']=(df.sphericity<=0)|(df.sphericity>1)
    df['invalid_solidity']=(df.mesh_solidity<=0)|(df.mesh_solidity>1+1e-6)
    df['shape_qc_pass']=~(df.touches_border|~df.mesh_watertight|df.substantial_fragmentation|df.invalid_sphericity|df.invalid_solidity)
    def flags(r):
        return ';'.join(name for name,bad in [('border',r.touches_border),('nonwatertight',not r.mesh_watertight),
            ('fragmented',r.substantial_fragmentation),('invalid_sphericity',r.invalid_sphericity),('invalid_solidity',r.invalid_solidity)] if bad) or 'pass'
    df['qc_flags']=df.apply(flags,axis=1)
    df['present_on_native_grid']=True
    df.loc[df.cell_id=='X100:L3','present_on_native_grid']=False
    # Verified in both saved label arrays: X100:L3 is ten working voxels, lost on nearest-neighbor resampling.
    df.to_csv(OUT/'all_cells.csv',index=False)
    df[~df.shape_qc_pass].to_csv(OUT/'flagged_objects.csv',index=False)
    matched=df[df.reference_matched]
    shape=matched[matched.shape_qc_pass]
    groups={'all_predictions':df,'matched_predictions':matched,'unmatched_predictions':df[~df.reference_matched],
            'all_shape_qc':df[df.shape_qc_pass],'matched_shape_qc':shape,'unmatched_shape_qc':df[~df.reference_matched & df.shape_qc_pass]}
    cohortrows=[];samplerows=[]
    for name,data in groups.items():
        for feature,values in summary(data).items():cohortrows.append(dict(cohort=name,feature=feature,**values))
        for sample,g in data.groupby('sample'):
            row=dict(cohort=name,sample=sample,n=len(g))
            for key in FEATURES:
                row.update({key+'_mean':g[key].mean(),key+'_median':g[key].median(),key+'_sd':g[key].std(),key+'_cv_pct':100*g[key].std()/g[key].mean()})
            samplerows.append(row)
    pd.DataFrame(cohortrows).to_csv(OUT/'cohort_summary.csv',index=False)
    samples=pd.DataFrame(samplerows).merge(metrics[['sample','reference_cells','foreground_dice','f1']],on='sample',how='left')
    samples.to_csv(OUT/'sample_summary.csv',index=False)
    metrics.to_csv(OUT/'reconstruction_metrics.csv',index=False)
    # Already measured reference comparisons; references do not alter predictions.
    refpaths=[ROOT/'results/cpsam_v2_first5/morphology/reference_comparison.csv',*SOURCE.glob('run_*/*/morphology/reference_comparison.csv')]
    ref=pd.concat([pd.read_csv(p) for p in refpaths],ignore_index=True)
    assert len(ref)==len(matched) and ref.cell_id.nunique()==len(matched)
    ref=ref.merge(df[['cell_id','shape_qc_pass']],on='cell_id',validate='one_to_one')
    ref.to_csv(OUT/'reference_comparison.csv',index=False)
    # Standardized shape space only: log ratios, no absolute volume or coordinates.
    valid=df[df.shape_qc_pass].copy()
    rawshape=pd.DataFrame({'sphericity':valid.sphericity,'log_elongation':np.log(valid.elongation),
                           'log_flatness':np.log(valid.flatness),'mesh_solidity':valid.mesh_solidity},index=valid.index)
    z=(rawshape-rawshape.mean())/rawshape.std()
    pca=PCA(n_components=2)
    scores=pca.fit_transform(z)
    valid['shape_pc1']=scores[:,0];valid['shape_pc2']=scores[:,1]
    valid.to_csv(OUT/'shape_coordinates.csv',index=False)
    pd.DataFrame(pca.components_,columns=z.columns,index=['PC1','PC2']).to_csv(OUT/'shape_pca_loadings.csv')
    distance=cdist(z,z)
    crosssample=valid['sample'].to_numpy()[:,None]!=valid['sample'].to_numpy()[None,:]
    distance[~crosssample]=np.inf
    neighborrows=[]
    for i,(_,cell) in enumerate(valid.iterrows()):
        for rank,j in enumerate(np.argsort(distance[i])[:3],1):
            neighbor=valid.iloc[j]
            neighborrows.append(dict(cell_id=cell.cell_id,reference_matched=cell.reference_matched,
               rank=rank,similar_cell=neighbor.cell_id,similar_cell_reference_matched=neighbor.reference_matched,
               shape_distance=distance[i,j],volume_ratio_neighbor_to_cell=neighbor.volume_um3/cell.volume_um3))
    pd.DataFrame(neighborrows).to_csv(OUT/'similar_cells_across_samples.csv',index=False)
    # Stack-level features give each stack equal weight; cells are not treated as independent replicates.
    sample_shape=shape.groupby('sample')[['sphericity','elongation','flatness','mesh_solidity']].median()
    ss=(sample_shape-sample_shape.mean())/sample_shape.std()
    sample_dist=cdist(ss,ss)
    pd.DataFrame(sample_dist,index=ss.index,columns=ss.index).to_csv(OUT/'sample_shape_distances.csv')
    sm=samples[samples.cohort=='matched_predictions'].set_index('sample').sort_index()
    extremes=[]
    for feature,data in [('volume_um3',matched),('sphericity',shape),('aspect_ratio',shape)]:
        for side,subset in [('lowest',data.nsmallest(5,feature)),('highest',data.nlargest(5,feature))]:
            extremes.extend(dict(feature=feature,extreme=side,cell_id=r.cell_id,value=r[feature]) for _,r in subset.iterrows())
    pd.DataFrame(extremes).to_csv(OUT/'extreme_cells.csv',index=False)
    stratified=[]
    for count,g in sm.groupby('reference_cells'):
        stratified.append(dict(reference_cell_count=int(count),stacks=len(g),mean_of_stack_mean_volumes=g.volume_um3_mean.mean(),
                              median_within_stack_volume_cv_pct=g.volume_um3_cv_pct.median(),
                              between_stack_mean_volume_cv_pct=100*g.volume_um3_mean.std()/g.volume_um3_mean.mean()))
    pd.DataFrame(stratified).to_csv(OUT/'cell_count_strata.csv',index=False)
    st={
        'stacks':len(metrics),'working_objects':len(df),'native_objects':int(metrics.predicted_cells.sum()),
        'reference_cells':int(metrics.reference_cells.sum()),'matched':len(matched),'unmatched_working':int((~df.reference_matched).sum()),
        'missed_reference_cells':int(metrics.false_negatives.sum()),
        'micro_precision':float(metrics.true_positives.sum()/metrics.predicted_cells.sum()),
        'micro_recall':float(metrics.true_positives.sum()/metrics.reference_cells.sum()),
        'micro_f1':float(2*metrics.true_positives.sum()/(metrics.predicted_cells.sum()+metrics.reference_cells.sum())),
        'mean_dice':float(metrics.foreground_dice.mean()),'median_dice':float(metrics.foreground_dice.median()),
        'qc_counts':{key:int(df[key].sum()) for key in ['touches_border','substantial_fragmentation','invalid_sphericity','invalid_solidity','shape_qc_pass']},
        'nonwatertight':int((~df.mesh_watertight).sum()),'matched_shape_qc':len(shape),
        'matched_summary':summary(matched),'matched_shape_summary':summary(shape),
        'volume_within_stack_median_cv_pct':float(sm.volume_um3_cv_pct.median()),
        'volume_between_stack_mean_cv_pct':float(100*sm.volume_um3_mean.std()/sm.volume_um3_mean.mean()),
        'sample_mean_volume_min':float(sm.volume_um3_mean.min()),'sample_mean_volume_max':float(sm.volume_um3_mean.max()),
        'smallest_mean_stack':sm.volume_um3_mean.idxmin(),'largest_mean_stack':sm.volume_um3_mean.idxmax(),
        'reference_volume_error_median_pct':float(ref.volume_um3_error_pct.median()),
        'reference_volume_error_mean_pct':float(ref.volume_um3_error_pct.mean()),
        'reference_volume_error_iqr_pct':[float(ref.volume_um3_error_pct.quantile(.25)),float(ref.volume_um3_error_pct.quantile(.75))],
        'shape_pca_explained_variance':pca.explained_variance_ratio_.tolist(),
        'shape_standardization_mean':rawshape.mean().to_dict(),'shape_standardization_sd':rawshape.std().to_dict()}
    (OUT/'statistics.json').write_text(json.dumps(st,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for ax,key,title in zip(axs.flat,['volume_um3','equivalent_diameter_um','sphericity','aspect_ratio','elongation','mesh_solidity'],
                           ['Volume (µm³)','Equivalent diameter (µm)','Sphericity','Major/minor axis ratio','Major/middle axis ratio','Mesh solidity']):
        surface=key in ['sphericity','mesh_solidity']
        for name,color,g in [('Matched','#2371b5',df[df.reference_matched]),('Unmatched','#e8913a',df[~df.reference_matched])]:
            if surface:g=g[g.shape_qc_pass]
            ax.hist(g[key],bins=30,density=True,alpha=.5,color=color,label=f'{name} (n={len(g)})')
        ax.set_title(title+(' — QC pass' if surface else ''));ax.set_ylabel('Density');ax.legend(fontsize=8)
    fig.suptitle('All 108 stacks: distributions of reconstructed cell size and shape',fontsize=16)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'distributions.png',dpi=180);fig.savefig(OUT/'distributions.pdf');plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(15,10),sharex=True)
    names=list(sm.index);xx=np.arange(len(names));median=matched.groupby('sample').volume_um3.median().reindex(names)
    q1=matched.groupby('sample').volume_um3.quantile(.25).reindex(names);q3=matched.groupby('sample').volume_um3.quantile(.75).reindex(names)
    axs[0].vlines(xx,q1,q3,color='#adcbe3',lw=3,label='Within-stack interquartile range')
    pts=axs[0].scatter(xx,median,c=sm.reference_cells,cmap='viridis',s=22,label='Matched-cell median')
    axs[0].set_ylabel('Cell volume (µm³)');axs[0].legend(loc='upper left',fontsize=8)
    fig.colorbar(pts,cax=axs[0].inset_axes([1.01,0,.012,1]),label='Annotated cells per stack')
    shp=shape.groupby('sample').sphericity.agg(['median','min','max']).reindex(names)
    axs[1].vlines(xx,shp['min'],shp['max'],color='#bed8c0',lw=2);axs[1].scatter(xx,shp['median'],s=15,color='#287d3c')
    axs[1].set_ylabel('Sphericity\nQC pass: median and range')
    axs[2].plot(xx,sm.volume_um3_cv_pct,color='#6d4491');axs[2].set_ylabel('Within-stack\nvolume CV (%)')
    axs[2].set_xticks(xx[::6],[names[i] for i in xx[::6]],rotation=60);axs[2].set_xlabel('Stack filename order — not a verified time axis')
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Variation across stacks — annotation-matched predictions',fontsize=16)
    fig.tight_layout(rect=[0,0,.95,.96]);fig.savefig(OUT/'across_samples.png',dpi=180);fig.savefig(OUT/'across_samples.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for name,color,g in [('Matched','#2371b5',valid[valid.reference_matched]),('Unmatched','#e8913a',valid[~valid.reference_matched])]:
        axs[0].scatter(g.shape_pc1,g.shape_pc2,s=12,alpha=.5,label=f'{name} (n={len(g)})',color=color)
    axs[0].legend();axs[0].set_xlabel(f'Shape PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)');axs[0].set_ylabel(f'Shape PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
    axs[0].set_title('Shape-only similarity (QC-passing objects)')
    axs[1].scatter(shape.volume_um3,shape.sphericity,c=shape.aspect_ratio,cmap='viridis',s=15,alpha=.6)
    axs[1].set_xlabel('Volume (µm³)');axs[1].set_ylabel('Sphericity');axs[1].set_title('Size versus shape (matched, QC pass)')
    fig.colorbar(axs[1].collections[0],ax=axs[1],label='Major/minor axis ratio')
    fig.tight_layout();fig.savefig(OUT/'individual_cells.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(sample_dist,cmap='viridis_r')
    ticks=np.arange(0,len(ss),8);ax.set_xticks(ticks,ss.index[ticks],rotation=90,fontsize=8);ax.set_yticks(ticks,ss.index[ticks],fontsize=8)
    fig.colorbar(im,ax=ax,label='Standardized median-shape distance (lower = more similar)')
    ax.set_title('108-stack shape comparison — matched, QC-passing cells')
    fig.tight_layout();fig.savefig(OUT/'sample_shape_similarity.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    axs[0].scatter(ref.volume_um3_reference,ref.volume_um3,s=12,alpha=.4)
    lim=max(ref.volume_um3_reference.max(),ref.volume_um3.max());axs[0].plot([0,lim],[0,lim],'k--')
    axs[0].set_xlabel('Reference volume (µm³)');axs[0].set_ylabel('Predicted volume (µm³)');axs[0].set_title('743 matched cells')
    axs[1].hist(ref.volume_um3_error_pct,bins=35,color='#2371b5');axs[1].axvline(0,color='black',ls='--')
    axs[1].set_xlabel('Volume difference from reference (%)');axs[1].set_ylabel('Cell count')
    fig.tight_layout();fig.savefig(OUT/'reference_bias.png',dpi=180);plt.close(fig)
    print(json.dumps({key:value for key,value in st.items() if not isinstance(value,dict)},indent=2))


if __name__=='__main__':main()
