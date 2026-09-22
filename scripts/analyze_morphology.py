"""Physical 3D morphometry and descriptive comparisons of reconstructed cells."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
import json
import numpy as np
import pandas as pd
import tifffile
import trimesh
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from skimage.measure import regionprops, marching_cubes
from skimage.transform import resize

BATCH = ROOT / 'results/cpsam_v2_first5'
OUT = BATCH / 'morphology'
FEATURES = ['volume_um3', 'equivalent_diameter_um', 'surface_area_um2',
            'sphericity', 'elongation', 'flatness', 'mesh_solidity']


def matches(ref, pred):
    rids, r = np.unique(ref, return_inverse=True)
    pids, p = np.unique(pred, return_inverse=True)
    counts = np.bincount(r.ravel() * len(pids) + p.ravel(), minlength=len(rids)*len(pids)).reshape(len(rids), len(pids))
    cross = counts[rids != 0][:, pids != 0]
    unions = counts.sum(1)[rids != 0, None] + counts.sum(0)[None, pids != 0] - cross
    iou = cross / np.maximum(unions, 1)
    rows, cols = linear_sum_assignment(-(iou >= .5).astype(float) - iou/(2*max(min(iou.shape), 1)))
    return {int(pids[pids != 0][c]): (int(rids[rids != 0][r]), float(iou[r,c])) for r,c in zip(rows,cols) if iou[r,c] >= .5}


def properties(labels, spacing):
    spacing = np.asarray(spacing)
    result = []
    for reg in regionprops(labels):
        xyz = reg.coords.astype(float) * spacing
        # Physical covariance of a uniform voxel solid, including within-voxel variance.
        cov = np.cov(xyz, rowvar=False, bias=True) + np.diag(spacing**2/12)
        eig = np.linalg.eigvalsh(cov)[::-1]
        axes = np.sqrt(20 * eig)  # Full diameters of equal-covariance solid ellipsoid.
        verts, faces, _, _ = marching_cubes(np.pad(reg.image, 1), .5, spacing=spacing)
        mesh = trimesh.Trimesh(verts, faces, process=False)
        mesh.fix_normals(multibody=True)
        volume = reg.area * np.prod(spacing)
        area = mesh.area
        components, n = ndimage.label(reg.image, structure=np.ones((3,3,3)))
        sizes = np.bincount(components.ravel())[1:]
        row = dict(label=int(reg.label), volume_um3=float(volume), surface_area_um2=float(area),
                   equivalent_diameter_um=float((6*volume/np.pi)**(1/3)),
                   surface_to_volume_per_um=float(area/volume),
                   sphericity=float(np.pi**(1/3)*(6*volume)**(2/3)/area),
                   major_axis_um=float(axes[0]), middle_axis_um=float(axes[1]), minor_axis_um=float(axes[2]),
                   elongation=float(axes[0]/axes[1]), flatness=float(axes[1]/axes[2]),
                   aspect_ratio=float(axes[0]/axes[2]),
                   mesh_solidity=float(abs(mesh.volume)/mesh.convex_hull.volume),
                   components_26=int(n), largest_component_fraction=float(sizes.max()/sizes.sum()),
                   touches_border=bool(any(a==0 for a in reg.bbox[:3]) or any(b==s for b,s in zip(reg.bbox[3:],labels.shape))),
                   mesh_watertight=bool(mesh.is_watertight))
        row.update(zip(['centroid_z_um','centroid_y_um','centroid_x_um'], xyz.mean(0)))
        result.append(row)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    rows, refrows, sensitivity = [], [], []
    for path in sorted(BATCH.glob('run_*/*/complete.json')):
        sample = path.parent.name
        info = json.loads(path.read_text())
        settings = json.loads((path.parent/'settings.json').read_text())['config']
        labels = tifffile.imread(path.parent/'labels_isotropic.tif')
        native = tifffile.imread(path.parent/'labels_native.tif')
        ref = tifffile.imread(info['mask'])
        matched = matches(ref, native)
        spacing = info['effective_spacing_zyx_um']
        measured = properties(labels, spacing)
        # Evaluation only: resample references to the prediction grid for comparable morphometry.
        ref_work = resize(ref, labels.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)
        references = properties(ref_work, spacing)
        for row in measured:
            rid, iou = matched.get(row['label'], (None, None))
            row.update(sample=sample, reference_label=rid, reference_iou=iou,
                       reference_matched=rid is not None, cell_id=f"{sample}:L{row['label']}")
            rows.append(row)
        refrows.extend(dict(sample=sample, **row) for row in references)
        # Same prediction measured on original grid: identifies grid-sensitive surface measures.
        for row in properties(native, settings['spacing_zyx_um']):
            orig = next(r for r in measured if r['label']==row['label'])
            sensitivity.append(dict(sample=sample,label=row['label'],**{f'{key}_native_change_pct':100*(row[key]/orig[key]-1) for key in ['volume_um3','surface_area_um2','sphericity']}))
        print('Measured',sample,flush=True)
    df = pd.DataFrame(rows).sort_values(['sample','label']).reset_index(drop=True)
    refdf = pd.DataFrame(refrows)
    # Spatial correspondence is descriptive, not proof of cell identity or tracking.
    df['spatial_group'] = ''
    df['distance_to_X001_um'] = np.nan
    coords = ['centroid_z_um','centroid_y_um','centroid_x_um']
    for matched in [True,False]:
        base = df[(df['sample']=='X001') & (df.reference_matched==matched)].sort_values('label')
        for sample in sorted(df['sample'].unique()):
            current = df[(df['sample']==sample) & (df.reference_matched==matched)]
            distances = cdist(base[coords],current[coords])
            ia, ib = linear_sum_assignment(distances)
            for a,b in zip(ia,ib):
                index = current.index[b]
                df.loc[index,'spatial_group'] = ('C' if matched else 'U') + str(a+1)
                df.loc[index,'distance_to_X001_um'] = distances[a,b]
    df.to_csv(OUT/'cells_3d.csv',index=False)
    refdf.to_csv(OUT/'reference_cells_3d.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'grid_sensitivity.csv',index=False)
    matched_df = df[df.reference_matched].copy()
    summaries=[]
    for cohort, data in [('all_predictions',df),('reference_matched',matched_df),('unmatched_predictions',df[~df.reference_matched])]:
        for sample, group in data.groupby('sample'):
            summary=dict(cohort=cohort,sample=sample,n=len(group))
            for key in FEATURES:
                summary.update({key+'_mean':group[key].mean(),key+'_sd':group[key].std(),key+'_cv_pct':100*group[key].std()/group[key].mean(),key+'_min':group[key].min(),key+'_max':group[key].max()})
            summaries.append(summary)
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'sample_summary.csv',index=False)
    spatial=matched_df.groupby('spatial_group')[FEATURES].agg(['mean','std','min','max'])
    spatial.to_csv(OUT/'spatial_group_summary.csv')
    joined=matched_df.merge(refdf,left_on=['sample','reference_label'],right_on=['sample','label'],suffixes=('','_reference'))
    for key in FEATURES:
        joined[key+'_error_pct']=100*(joined[key]/joined[key+'_reference']-1)
    joined.to_csv(OUT/'reference_comparison.csv',index=False)
    # Pairwise shape distances; equal weights, standardized over the 20 matched objects.
    shape_features=['sphericity','elongation','flatness','mesh_solidity']
    standardized=(matched_df[shape_features]-matched_df[shape_features].mean())/matched_df[shape_features].std()
    distances=squareform(pdist(standardized))
    pd.DataFrame(distances,index=matched_df.cell_id,columns=matched_df.cell_id).to_csv(OUT/'shape_distances.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    samples=sorted(df['sample'].unique())
    fig, axs=plt.subplots(2,3,figsize=(14,8))
    plotkeys=['volume_um3','equivalent_diameter_um','sphericity','elongation','flatness','mesh_solidity']
    names=['Volume (µm³)','Equivalent diameter (µm)','Sphericity (sphere = 1)','Elongation (major/middle)','Flatness (middle/minor)','Mesh solidity (convex = 1)']
    for ax,key,title in zip(axs.flat,plotkeys,names):
        for i,sample in enumerate(samples):
            group=df[df['sample']==sample]
            for j,(_,row) in enumerate(group.iterrows()):
                color=plt.get_cmap('tab10')(int(row.spatial_group[1:])-1) if row.reference_matched else '#a0a0a0'
                ax.scatter(i+(j-2.5)*.055,row[key],color=color,marker='o' if row.reference_matched else 'x',s=45)
            ax.plot([i-.18,i+.18],[group[group.reference_matched][key].mean()]*2,color='black',lw=2)
        ax.set_xticks(range(5),samples);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Individual reconstructed cells across five stacks',fontsize=17)
    fig.text(.5,.02,'C1 blue • C2 orange • C3 green • C4 red (provisional spatial groups) • Gray ×: unmatched • Black bar: matched mean',ha='center')
    fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(OUT/'cell_properties.png',dpi=180);fig.savefig(OUT/'cell_properties.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4.6))
    for ax,key,title in zip(axs,['volume_um3','sphericity','aspect_ratio'],['Volume (µm³)','Sphericity','Aspect ratio (major/minor)']):
        for name,group in matched_df.groupby('spatial_group'):
            group=group.sort_values('sample');ax.plot(group['sample'],group[key],'-o',label=name)
        ax.set_title(title);ax.grid(alpha=.2)
    axs[0].legend(title='Spatial group');fig.suptitle('Provisional spatial correspondences — not verified cell tracking')
    fig.tight_layout();fig.savefig(OUT/'spatial_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,7))
    order=matched_df.sort_values(['spatial_group','sample']).index
    loc=[list(matched_df.index).index(i) for i in order]
    image=ax.imshow(distances[np.ix_(loc,loc)],cmap='viridis_r',vmin=0)
    labels=matched_df.loc[order].apply(lambda r:f'{r.spatial_group} {r["sample"]}',axis=1)
    ax.set_xticks(range(len(loc)),labels,rotation=90,fontsize=7);ax.set_yticks(range(len(loc)),labels,fontsize=7)
    fig.colorbar(image,ax=ax,label='Standardized shape distance (lower = more similar)')
    ax.set_title('Pairwise cell-shape differences (size excluded)');fig.tight_layout();fig.savefig(OUT/'shape_distances.png',dpi=180);plt.close(fig)
    stats={
        'matched_ranges':{key:[float(matched_df[key].min()),float(matched_df[key].max())] for key in FEATURES},
        'matched_sample_means':matched_df.groupby('sample')[FEATURES].mean().to_dict(),
        'volume_within_sample_cv_pct':(100*matched_df.groupby('sample').volume_um3.std()/matched_df.groupby('sample').volume_um3.mean()).to_dict(),
        'volume_between_sample_mean_cv_pct':float(100*matched_df.groupby('sample').volume_um3.mean().std()/matched_df.groupby('sample').volume_um3.mean().mean()),
        'spatial_group_volume_cv_pct':(100*matched_df.groupby('spatial_group').volume_um3.std()/matched_df.groupby('spatial_group').volume_um3.mean()).to_dict(),
        'max_spatial_displacement_um':float(matched_df.distance_to_X001_um.max()),
        'mean_volume_error_pct':float(joined.volume_um3_error_pct.mean()),
        'median_volume_error_pct':float(joined.volume_um3_error_pct.median()),
        'grid_sensitivity_median_pct':pd.DataFrame(sensitivity).filter(like='_change_pct').median().to_dict(),
        'qc':{'border_touching':int(df.touches_border.sum()),'disconnected_labels':int((df.components_26>1).sum()),'nonwatertight':int((~df.mesh_watertight).sum())}}
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
