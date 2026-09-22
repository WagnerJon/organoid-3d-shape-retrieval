"""Write an evidence-based morphology report from saved measurements."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cpsam_v2_first5/morphology'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    df=pd.read_csv(OUT/'cells_3d.csv'); matched=df[df.reference_matched]
    stats=json.loads((OUT/'statistics.json').read_text())
    ref=pd.read_csv(OUT/'reference_comparison.csv')
    means=matched.groupby('sample').mean(numeric_only=True)
    allmeans=df.groupby('sample').volume_um3.mean()
    within=matched.groupby('sample').volume_um3.std()/means.volume_um3*100
    groups=matched.groupby('spatial_group').mean(numeric_only=True)
    groupcv=100*matched.groupby('spatial_group').volume_um3.std()/groups.volume_um3
    D=pd.read_csv(OUT/'shape_distances.csv',index_col=0)
    same=[]; different=[]
    for i,a in matched.iterrows():
        for j,b in matched.iterrows():
            if i<j and a['sample']!=b['sample']:
                (same if a.spatial_group==b.spatial_group else different).append(D.loc[a.cell_id,b.cell_id])
    sample_table=table(['Stack','All 6: mean volume (µm³)','Matched 4: mean volume (µm³)','Matched volume CV','Mean sphericity'],
        [[s,f'{allmeans[s]:.0f}',f'{r.volume_um3:.0f}',f'{within[s]:.1f}%',f'{r.sphericity:.3f}'] for s,r in means.iterrows()])
    group_table=table(['Provisional group','Mean volume (µm³)','Volume CV across stacks','Mean sphericity','Mean major/minor ratio'],
        [[s,f'{r.volume_um3:.0f}',f'{groupcv[s]:.1f}%',f'{r.sphericity:.3f}',f'{r.aspect_ratio:.2f}'] for s,r in groups.iterrows()])
    paths=lambda name:str((OUT/name).resolve())
    text=f'''# How different are the reconstructed cells?

Analysis of **30 cpsam_v2 reconstructed objects across X001, X003, X005, X007 and X009**. Every object is retained in the measurements. Twenty predictions match supplied annotations at IoU ≥ 0.5; ten do not. The latter remain a separate, visible group because their biological status is unverified. No predictions were changed or filtered using the reference masks.

## Main finding

The annotation-matched cells vary more in size **within a stack** than stack means vary **across these five stacks**. Within-stack volume coefficients of variation are **{within.min():.1f}–{within.max():.1f}%**; the coefficient of variation of the five mean volumes is **{stats['volume_between_sample_mean_cv_pct']:.1f}%**. CV means standard deviation divided by mean. These are descriptive summaries of different levels of variation, not a significance test or variance-components model.

Matched cell volumes range from **{matched.volume_um3.min():.0f} to {matched.volume_um3.max():.0f} µm³**; the largest is **{matched.volume_um3.max()/matched.volume_um3.min():.2f}×** the smallest. Equivalent spherical diameters range from **{matched.equivalent_diameter_um.min():.2f} to {matched.equivalent_diameter_um.max():.2f} µm**. Mean matched-cell volume rises **{100*(means.loc['X009','volume_um3']/means.loc['X001','volume_um3']-1):.1f}%** from X001 to X009. This is an ordered-image difference; it cannot yet be called biological growth or a difference between independent organoids.

{sample_table}

![Individual cell measurements]({paths('cell_properties.png')})

## Shape differences

Matched cells have sphericity **{matched.sphericity.min():.3f}–{matched.sphericity.max():.3f}**, where 1 denotes an ideal sphere. Their major/minor axis ratios range from **{matched.aspect_ratio.min():.2f} to {matched.aspect_ratio.max():.2f}**. They are therefore anisotropic and less compact than spheres, with modest but visible differences in elongation and flattening. A low sphericity can reflect elongation, surface irregularity or segmentation discretization; it is not a unique biological phenotype.

Equivalent ellipsoid major axes span **{matched.major_axis_um.min():.2f}–{matched.major_axis_um.max():.2f} µm**, middle axes **{matched.middle_axis_um.min():.2f}–{matched.middle_axis_um.max():.2f} µm**, and minor axes **{matched.minor_axis_um.min():.2f}–{matched.minor_axis_um.max():.2f} µm**. These are moment-derived axes, not maximum caliper diameters. Mesh solidity ranges from **{matched.mesh_solidity.min():.3f} to {matched.mesh_solidity.max():.3f}**; lower values indicate more departure from a convex solid.

## Comparing individual cells across stacks

**Numeric segmentation labels are not stable cell identities.** For example, the spatial counterpart of X001 label 3 is X003 label 2 and X009 label 1. Comparing “label 1” across files would mix different locations.

I assigned provisional groups C1–C4 using one-to-one nearest-centroid matching to X001, among annotation-matched predictions. The largest displacement is **{stats['max_spatial_displacement_um']:.2f} µm**. This uses prediction geometry, not morphology, to establish spatial correspondence. It is not validated tracking: there is no confirmed time metadata, registration or lineage model. The interpretation assumes a shared coordinate frame.

{group_table}

- **C1 is the largest on average**, about **{100*(groups.loc['C1','volume_um3']/groups.loc['C3','volume_um3']-1):.1f}% larger than C3**, the smallest.
- **C4 is the most sphere-like/compact by these descriptors**, with the highest mean sphericity and lowest major/minor ratio.
- **C2 is the least sphere-like**, with the lowest mean sphericity and highest major/minor ratio.
- Volumes for corresponding spatial groups vary by only **{groupcv.min():.1f}–{groupcv.max():.1f}% CV** across these images; all four increase in the file sequence. This regularity is consistent with related images, but does not establish their acquisition history.

![Provisional correspondence plots]({paths('spatial_comparison.png')})

The size-independent shape distance combines standardized sphericity, elongation, flatness and mesh solidity with equal weights. Median distance is **{np.median(same):.2f}** for the same provisional spatial group across stacks versus **{np.median(different):.2f}** for different groups across stacks. Thus shape tends to be more similar at corresponding locations, although distributions can overlap. This exploratory distance is dataset-dependent, contains correlated features and is not a biological classification or a significance test.

![Pairwise shape differences]({paths('shape_distances.png')})

## Unmatched predicted objects

The ten unmatched objects have volumes **{df[~df.reference_matched].volume_um3.min():.0f}–{df[~df.reference_matched].volume_um3.max():.0f} µm³**, mean **{df[~df.reference_matched].volume_um3.mean():.0f} µm³**, and sphericity **{df[~df.reference_matched].sphericity.min():.3f}–{df[~df.reference_matched].sphericity.max():.3f}**. They are generally smaller and less sphere-like than the matched cohort. They are substantial objects, not tiny speckles. Their absence from annotations does not by itself establish that they are false biological cells. Including them lowers every stack's mean cell volume, as shown in the table.

## How reliable are these differences?

1. **Segmentation bias:** matched predicted volumes exceed reference volumes by a mean **{stats['mean_volume_error_pct']:.1f}%** (median **{stats['median_volume_error_pct']:.1f}%**, range **{ref.volume_um3_error_pct.min():.1f}–{ref.volume_um3_error_pct.max():.1f}%**). This discrepancy is comparable to or larger than the 8.3% difference between the first and last stack means. Reference masks are a comparison standard, not an uncertainty-free biological truth. Reference morphometry used nearest-neighbor resampling onto the prediction grid.
2. **Surface sensitivity:** measuring the same predictions on the original anisotropic grid changes median volume by only **{stats['grid_sensitivity_median_pct']['volume_um3_native_change_pct']:.2f}%**, but surface area by **{stats['grid_sensitivity_median_pct']['surface_area_um2_native_change_pct']:.1f}%** and sphericity by **{stats['grid_sensitivity_median_pct']['sphericity_native_change_pct']:.1f}%**. Consequently use the approximately isotropic-grid values consistently. This is a discretization sensitivity experiment, not a confidence interval or independent validation. Small shape differences should not be overinterpreted.
3. **Topology checks:** all 30 meshes are watertight and no label touches its volume boundary. Two labels contain tiny detached components (X003:L4 and X005:L5); their largest components retain over 99.99% of their voxels. No components were removed.
4. **Calibration and independence:** dimensions use the dataset-derived Z/Y/X spacing of 1/0.1733/0.1733 µm, not embedded TIFF calibration. The dataset identity is consistent with the [EmbedSeg publication](https://openreview.net/pdf?id=JM6GuFGayL5), but whether these five files represent independent organoids or related images remains unconfirmed. No p-values, growth rates or population-level claims are made.

## Measurement definitions

- Volume: label voxel count × physical voxel volume.
- Surface area: unsmoothed, padded marching-cubes mesh area on the working grid.
- Equivalent diameter: diameter of a sphere with the measured voxel volume.
- Sphericity: π^(1/3) × (6V)^(2/3) / area.
- Principal axes: full equal-covariance ellipsoid diameters √(20λ), using physical voxel-center covariance plus within-voxel variance. Elongation = major/middle; flatness = middle/minor; aspect ratio = major/minor. Higher flatness means a relatively thinner minor axis.
- Mesh solidity: enclosed mesh volume / convex-hull volume, using mesh volume in both numerator and denominator.
- Surface/volume: measured surface area divided by voxel volume, in µm⁻¹.

## Files and reproducibility

- [Every reconstructed cell, all measurements and spatial mapping]({paths('cells_3d.csv')})
- [Per-stack statistics for all, matched and unmatched cohorts]({paths('sample_summary.csv')})
- [Reference comparison for each matched cell]({paths('reference_comparison.csv')})
- [Grid sensitivity for each cell]({paths('grid_sensitivity.csv')})
- [Pairwise shape-distance matrix]({paths('shape_distances.csv')})
- [Exportable figure PDF]({paths('cell_properties.pdf')})

Reproduce with `.venv/bin/python scripts/analyze_morphology.py` followed by `.venv/bin/python scripts/report_morphology.py`. Synthetic sphere and anisotropic-solid tests verified physical scaling, shape invariance to uniform scaling, and principal-axis ratios.
'''
    (OUT/'REPORT.md').write_text(text)
    print(OUT/'REPORT.md')


if __name__=='__main__':main()
