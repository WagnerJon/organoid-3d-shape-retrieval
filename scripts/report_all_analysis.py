"""Generate the full-cohort report from measured objects and analysis tables."""
from analyze_all import OUT,ROOT
import json
import pandas as pd


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    s=json.loads((OUT/'statistics.json').read_text())
    df=pd.read_csv(OUT/'all_cells.csv');samples=pd.read_csv(OUT/'sample_summary.csv')
    metrics=pd.read_csv(OUT/'reconstruction_metrics.csv')
    strata=pd.read_csv(OUT/'cell_count_strata.csv')
    cohorts=pd.read_csv(OUT/'cohort_summary.csv')
    g=pd.read_csv(OUT/'grid_sensitivity_12_stacks.csv');g=g[g.reference_matched & g.shape_qc_pass]
    gm=g.filter(like='_native_change_pct').median()
    sm=samples[samples.cohort=='matched_predictions'].set_index('sample')
    shape=samples[samples.cohort=='matched_shape_qc'].set_index('sample')
    features=[('volume_um3','Volume (µm³)'),('equivalent_diameter_um','Equivalent diameter (µm)'),
              ('surface_area_um2','Surface area (µm²)'),('sphericity','Sphericity'),
              ('elongation','Elongation: major/middle'),('flatness','Flatness: middle/minor'),
              ('aspect_ratio','Aspect ratio: major/minor'),('mesh_solidity','Mesh solidity')]
    rows=[]
    for key,label in features:
        surface=key in ['surface_area_um2','sphericity','mesh_solidity']
        v=s['matched_shape_summary' if surface else 'matched_summary'][key]
        rows.append([label,int(v['n']),f"{v['median']:.2f}",f"{v['q25']:.2f}–{v['q75']:.2f}",f"{v['min']:.2f}–{v['max']:.2f}"])
    feature_table=table(['Measurement','Objects','Median','Middle 50%','Full range'],rows)
    group_table=table(['Annotated cells per stack','Stacks','Median within-stack volume CV','CV of stack means'],
        [[int(r.reference_cell_count),int(r.stacks),f'{r.median_within_stack_volume_cv_pct:.1f}%',f'{r.between_stack_mean_volume_cv_pct:.1f}%' if pd.notna(r.between_stack_mean_volume_cv_pct) else 'Not estimable (one stack)'] for _,r in strata.iterrows()])
    cellids=['X001:L3','X206:L3','X212:L2','X216:L6']
    examples=df.set_index('cell_id').loc[cellids]
    example_table=table(['Cell','Volume (µm³)','Sphericity','Longest/shortest axis','Distinction'],
        [[ident,f'{r.volume_um3:.0f}',f'{r.sphericity:.3f}',f'{r.aspect_ratio:.2f}',desc] for (ident,r),desc in zip(examples.iterrows(),['Smallest matched object','Largest matched object','Most sphere-like matched QC-pass object','Most elongated matched QC-pass object'])])
    link=lambda f:str((OUT/f).resolve())
    ranks=pd.concat([sm.nsmallest(3,'volume_um3_mean'),sm.nlargest(3,'volume_um3_mean')])
    rank_table=table(['Stack','Matched objects','Mean volume (µm³)','Within-stack volume CV'],[[name,int(r.n),f'{r.volume_um3_mean:.0f}',f'{r.volume_um3_cv_pct:.1f}%'] for name,r in ranks.iterrows()])
    allmedian=cohorts[(cohorts.cohort=='all_predictions')&(cohorts.feature=='volume_um3')].iloc[0]['median']
    unmatchedmedian=cohorts[(cohorts.cohort=='unmatched_predictions')&(cohorts.feature=='volume_um3')].iloc[0]['median']
    text=f'''# Full-dataset 3D cell analysis

Completed analysis of **108 image stacks and {s['working_objects']:,} reconstructed objects**. All 103 remaining reconstructions and morphology jobs finished without recorded failures. All original predictions and annotations are preserved.

## Main conclusion: the first-five result does not describe the entire collection

The full collection contains substantially more size variation between stacks. For annotation-matched predictions, **the CV of stack mean cell volumes is {s['volume_between_stack_mean_cv_pct']:.1f}%**, compared with **3.1% in the first five stacks**. The median within-stack volume CV is **{s['volume_within_stack_median_cv_pct']:.1f}%**, although some stacks contain much broader mixtures (up to **{sm.volume_um3_cv_pct.max():.1f}%**).

Mean matched-cell volume ranges from **{s['sample_mean_volume_min']:.0f} µm³ in {s['smallest_mean_stack']}** to **{s['sample_mean_volume_max']:.0f} µm³ in {s['largest_mean_stack']}**, almost twofold. Individual matched volumes span **{s['matched_summary']['volume_um3']['min']:.0f}–{s['matched_summary']['volume_um3']['max']:.0f} µm³**, a **{s['matched_summary']['volume_um3']['max']/s['matched_summary']['volume_um3']['min']:.2f}×** range. These are observed differences in reconstructed morphology, not proof of growth, distinct cell types, or statistically independent organoid differences.

The 17.0% and 10.0% CV values summarize different levels of variation; they are not variance components or a significance test. The cell-count strata below show why a single overall comparison is incomplete.

![Variation across all stacks]({link('across_samples.png')})

## Which objects are included?

- **All predictions:** {s['working_objects']:,} objects, retained in `all_cells.csv`; pooled median volume **{allmedian:.0f} µm³**.
- **Annotation-matched:** {s['matched']} objects with one-to-one IoU ≥ 0.5. These form the main size comparison; no volume filtering was applied to them.
- **Unmatched predictions:** {s['unmatched_working']} objects, pooled median volume **{unmatchedmedian:.0f} µm³**. They include substantial unannotated objects, edge objects and tiny fragments. “Unmatched” does not establish biological falsehood.
- **Shape-quality screen:** {s['qc_counts']['shape_qc_pass']} total objects pass, of which **{s['matched_shape_qc']} are annotation-matched**. The conservative surface-shape comparison uses these 729 matched objects; all flagged values remain available.

The primary volume and moment-derived axis statistics retain all 743 matched objects. Surface area, sphericity and solidity summaries use the 729 matched objects that pass the shape screen. This distinction avoids throwing away valid voxel-volume measurements merely because a mesh has a topology issue. Conversely, passing the screen is not proof that segmentation is biologically correct.

## Size and shape of individual cells

{feature_table}

Equivalent diameter is the diameter of a sphere with the same volume, not the cell's maximum width. Sphericity and mesh solidity approach 1 for a sphere and a convex object, respectively. Larger major/minor ratios indicate stronger directional shape differences.

Most matched cells are moderately anisotropic: the middle half have longest/shortest axis ratios **{s['matched_summary']['aspect_ratio']['q25']:.2f}–{s['matched_summary']['aspect_ratio']['q75']:.2f}**. However, the full shape-quality cohort includes both nearly spherical cells and elongated cells with a ratio above 3. Stack median sphericity ranges from **{shape.sphericity_median.min():.3f} to {shape.sphericity_median.max():.3f}**. Shape variation is not explained simply by size: the most elongated example is much smaller than the largest example.

![Size and shape distributions]({link('distributions.png')})

### Concrete individual-cell differences

{example_table}

![Representative cells]({link('representative_cells.png')})

These examples are annotation-matched and pass the shape screen. The figure uses a common physical scale, centered objects and coarser surfaces for visualization only. All quantitative values come from full working-resolution measurements.

## Comparing stacks

The smallest and largest stack means are:

{rank_table}

Stacks differ in their number and composition of annotated cells. Stratifying by annotation count gives:

{group_table}

For 4- and 8-cell annotation groups, stack means vary substantially (about 18–19% CV) while typical within-stack variation is about 9–10%. For 6-, 10- and 12-cell groups, the opposite pattern appears: within-stack size mixtures are broad (about 31–33% median CV), but means across stacks in each group are relatively similar. These groups are descriptive annotation-count strata, not established biological stages or independent replicates. Groups with one or three stacks warrant especially limited interpretation.

The filename-ordered plots show smooth stretches and abrupt changes in size/composition. Acquisition/time/lineage metadata are still unconfirmed, so filename order is **not** treated as a time axis. The first-five spatial groups C1–C4 are deliberately **not extrapolated to all 108 stacks**: object numbers change and label IDs are not persistent identities.

## How to find cells with similar shapes in other samples

[`similar_cells_across_samples.csv`]({link('similar_cells_across_samples.csv')}) lists the three nearest shape matches in other stacks for every one of the **985 quality-passing objects**, including both matched and unmatched predictions. Each row gives both object IDs, their annotation-match status, shape distance and their volume ratio. Flagged objects remain in the measurement table but are excluded from these nearest-neighbor rankings.

Distance uses four equally standardized descriptors: sphericity, log(major/middle axis), log(middle/minor axis), and mesh solidity. Absolute size and centroid location are excluded, so similar shapes can have different sizes. Standardization parameters and PCA loadings are saved. These descriptors are partly correlated; this is an exploratory similarity measure, not cell identification, tracking, a learned biological classifier or a statistical test.

The first two shape PCA axes account for **{100*sum(s['shape_pca_explained_variance']):.1f}%** of standardized feature variation. Points near each other in this projection have broadly similar descriptors, but the remaining dimensions still matter.

![Individual-cell shape comparisons]({link('individual_cells.png')})

[`sample_shape_distances.csv`]({link('sample_shape_distances.csv')}) compares stack median shape profiles, giving every stack equal weight rather than allowing stacks with more cells to dominate. This sample-level metric standardizes raw stack medians; its numerical scale is different from the cell-level metric above. Aggregation also hides within-stack mixtures, so use the distributions alongside it.

![Stack shape similarity]({link('sample_shape_similarity.png')})

## Comparison with the supplied masks

Across the native-grid outputs, **743 of 750 reference cells match at IoU ≥ 0.5**; **7 are unmatched**. Pooled object precision is **{100*s['micro_precision']:.1f}%**, recall **{100*s['micro_recall']:.1f}%**, and F1 **{s['micro_f1']:.3f}**. Mean per-stack foreground Dice is **{s['mean_dice']:.3f}**. Precision is relative to the supplied annotations, which may not describe every biological object visible in the images.

Matched predicted volumes exceed reference volumes by a median **{s['reference_volume_error_median_pct']:.1f}%** (middle half **{s['reference_volume_error_iqr_pct'][0]:.1f}–{s['reference_volume_error_iqr_pct'][1]:.1f}%**; mean **{s['reference_volume_error_mean_pct']:.1f}%**). This systematic discrepancy matters when interpreting small volume differences. Reference morphometry was evaluated after nearest-neighbor resampling to the prediction grid, using identical physical spacing and measurement definitions. Annotations are a comparator, not an uncertainty-free truth.

![Prediction-reference volume comparison]({link('reference_bias.png')})

## Quality checks and limitations

- **68 objects touch an image boundary**, **30 meshes are not watertight**, **6 labels have more than 1% of their volume outside their largest connected component**, and **3 have sphericity >1**. Flags overlap: **103 unique objects** fail at least one screen. Fourteen of these are annotation-matched, all because of nonwatertight meshes; no matched object touches the image border. The three impossible sphericities arise in extremely small voxel fragments, where voxel-count volume and marching-cubes surface area are inconsistent. They must not be interpreted as real biological shape scores.
- **1,088 versus 1,087 objects:** ten-voxel object **X100:L3** exists on the working grid but disappears when labels are resampled onto the original anisotropic grid. Morphometry therefore includes 1,088 working-grid objects, while native-grid evaluation counts 1,087. This accounts exactly for the difference between 345 unmatched working-grid objects and 344 unmatched native-grid predictions.
- **Grid sensitivity:** a deterministic subset of 12 evenly filename-spaced stacks was remeasured on the native grid. Among {len(g)} matched, shape-quality-passing objects in that subset, median volume changes by **{gm['volume_um3_native_change_pct']:.2f}%**, surface area by **{gm['surface_area_um2_native_change_pct']:.1f}%**, and sphericity by **{gm['sphericity_native_change_pct']:.1f}%**. These are discretization sensitivity checks, not confidence intervals, and not measurements on all 108 stacks. Use working-grid surface metrics consistently; small surface-based differences are less secure than volume differences.
- **Physical calibration:** all dimensions use dataset-derived Z/Y/X spacing of 1/0.1733/0.1733 µm, with actual working-grid spacing corrected for rounded array dimensions. These values were not present in TIFF metadata. Absolute sizes depend on this assumption.
- **Biological independence is unknown.** Do not treat 743 matched cells as 743 independent biological replicates or 108 stacks as 108 proven independent organoids. No p-values, growth rates or cell-type classifications are asserted. The shape screen and annotation matching also create selection effects; all-object and unmatched summaries are supplied so those effects can be inspected.

## Definitions and reproducibility

Volume is voxel count × physical voxel volume. Surface area is the unsmoothed padded marching-cubes mesh area. Sphericity is π^(1/3)(6V)^(2/3)/A. Principal axes are full equal-covariance ellipsoid diameters √(20λ), including within-voxel covariance; they are not maximum caliper distances. Elongation = major/middle, flatness = middle/minor (higher means a relatively thinner minor axis), and aspect ratio = major/minor. Mesh solidity is enclosed mesh volume divided by convex-hull volume. Volume and axes are computed from labels, independently of mesh topology.

Quality thresholds: boundary-touching excluded for shape comparisons; watertight mesh required; largest 26-connected component fraction ≥0.99; 0<sphericity≤1; 0<solidity≤1 (numerical tolerance 10⁻⁶). There is no additional minimum-volume threshold. All exclusions are flags in the exported table, not modifications to segmentations.

The comparisons use predictions only for morphology. Supplied masks determine evaluation matches and reference measurements only; they were not used to alter any reconstruction.

Run `.venv/bin/python scripts/analyze_all.py`, `.venv/bin/python scripts/all_analysis_visual_checks.py`, then `.venv/bin/python scripts/report_all_analysis.py`. The analysis reuses the full-resolution per-object measurements computed during reconstruction. Counts, unique IDs, complete stack coverage and finite descriptors were verified; synthetic sphere and anisotropic-solid tests cover the measurement formulas.

## Data and figures

- [All 1,088 objects, measurements, match status and quality flags]({link('all_cells.csv')})
- [108-stack summaries, all/matched/unmatched and QC cohorts]({link('sample_summary.csv')})
- [Overall cohort summaries]({link('cohort_summary.csv')})
- [Three most similar shapes in other stacks]({link('similar_cells_across_samples.csv')})
- [Flagged objects]({link('flagged_objects.csv')})
- [Largest, smallest and most extreme shapes]({link('extreme_cells.csv')})
- [Reference comparisons]({link('reference_comparison.csv')})
- [Per-stack reconstruction accuracy]({link('reconstruction_metrics.csv')})
- [12-stack grid-sensitivity measurements]({link('grid_sensitivity_12_stacks.csv')})
- [Across-stack figure PDF]({link('across_samples.pdf')}) · [Distribution figure PDF]({link('distributions.pdf')}) · [Representative cells PDF]({link('representative_cells.pdf')})
'''
    (OUT/'REPORT.md').write_text(text)
    print(OUT/'REPORT.md')


if __name__=='__main__':main()
