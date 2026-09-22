"""Measure a completed sample and refresh cohort tables, without modifying masks."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import tifffile
from skimage.transform import resize
from analyze_morphology import properties,matches,FEATURES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cpsam_v2_remaining'


def main():
    sample=sys.argv[1]
    paths=list(OUT.glob(f'run_*/{sample}/complete.json'))
    if len(paths)!=1:raise ValueError(f'Expected one completed result for {sample}, got {len(paths)}')
    folder=paths[0].parent
    info=json.loads(paths[0].read_text())
    output=folder/'morphology';output.mkdir(exist_ok=True)
    labels=tifffile.imread(folder/'labels_isotropic.tif')
    native=tifffile.imread(folder/'labels_native.tif')
    ref=tifffile.imread(info['mask'])
    match=matches(ref,native)
    rows=properties(labels,info['effective_spacing_zyx_um'])
    for row in rows:
        rid,iou=match.get(row['label'],(None,None))
        row.update(sample=sample,cell_id=f"{sample}:L{row['label']}",reference_label=rid,reference_iou=iou,reference_matched=rid is not None)
    df=pd.DataFrame(rows)
    if len(df):df.to_csv(output/'cells_3d.csv',index=False)
    else:pd.DataFrame(columns=['sample','cell_id','reference_matched',*FEATURES]).to_csv(output/'cells_3d.csv',index=False)
    ref_work=resize(ref,labels.shape,order=0,preserve_range=True,anti_aliasing=False).astype(np.uint32)
    refdf=pd.DataFrame(properties(ref_work,info['effective_spacing_zyx_um']));refdf['sample']=sample
    refdf.to_csv(output/'reference_cells_3d.csv',index=False)
    if len(df) and df.reference_matched.any():
        joined=df[df.reference_matched].merge(refdf,left_on=['sample','reference_label'],right_on=['sample','label'],suffixes=('','_reference'))
        for key in FEATURES:joined[key+'_error_pct']=100*(joined[key]/joined[key+'_reference']-1)
        joined.to_csv(output/'reference_comparison.csv',index=False)
    # Original first five plus every completed remaining sample; no identity assumed across images.
    files=[ROOT/'results/cpsam_v2_first5/morphology/cells_3d.csv',*OUT.glob('run_*/*/morphology/cells_3d.csv')]
    allcells=pd.concat([pd.read_csv(p) for p in files if p.exists()],ignore_index=True)
    allcells.to_csv(OUT/'all_cells_3d.csv',index=False)
    summaries=[]
    for (name,matched),group in allcells.groupby(['sample','reference_matched']):
        row=dict(sample=name,reference_matched=bool(matched),n=len(group))
        for key in FEATURES:row.update({key+'_mean':group[key].mean(),key+'_sd':group[key].std()})
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(OUT/'sample_morphology_summary.csv',index=False)
    metrics=[]
    for batch in [ROOT/'results/cpsam_v2_first5',OUT]:
        for p in batch.glob('run_*/*/complete.json'):
            data=json.loads(p.read_text());metrics.append(dict(sample=p.parent.name,foreground_dice=data['foreground_dice'],**data['instance_metrics']))
    pd.DataFrame(metrics).sort_values('sample').to_csv(OUT/'reconstruction_metrics.csv',index=False)
    for filename,volume,spacing in [('labels_isotropic.tif',labels,info['effective_spacing_zyx_um']),
                                    ('labels_native.tif',native,json.loads((folder/'settings.json').read_text())['config']['spacing_zyx_um'])]:
        temporary=folder/('compressed_'+filename)
        tifffile.imwrite(temporary,volume,compression='zlib',metadata={'axes':'ZYX','spacing_zyx_um':spacing})
        if not np.array_equal(tifffile.imread(temporary),volume):raise RuntimeError('Lossless TIFF verification failed')
        temporary.replace(folder/filename)
    (output/'complete.json').write_text(json.dumps({'sample':sample,'cells':len(df),'reference_use':'comparison only'},indent=2))
    print(f'Morphometry saved: {sample}, {len(df)} cells; cumulative tables refreshed',flush=True)


if __name__=='__main__':main()
