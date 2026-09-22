"""Descriptive redundancy audit, not a biological performance benchmark."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/simclr3d'
m=pd.read_csv(OUT/'manifest.csv');a=pd.read_csv(ROOT/'results/cpsam_v2_all_analysis/all_cells.csv').set_index('cell_id').loc[m.cell_id]
h=np.load(OUT/'embeddings.npy');train=m.split.eq('train').to_numpy();val=m.split.eq('validation').to_numpy()
# Aspect ratio is omitted because it is redundant with elongation and flatness.
sets={'sphericity_only':['sphericity'],'traditional_shape':['sphericity','elongation','flatness','mesh_solidity'],'shape_plus_volume':['sphericity','elongation','flatness','mesh_solidity','volume_um3']}
results=[]
for name,features in sets.items():
    x=a[features].to_numpy();assert np.isfinite(x).all()
    scaler=StandardScaler().fit(x[train]);x=scaler.transform(x)
    corr=spearmanr(pdist(x[val]),pdist(h[val],metric='cosine')).statistic
    for modelname,model in [('ridge',Ridge(alpha=1)),('random_forest',RandomForestRegressor(n_estimators=200,min_samples_leaf=5,max_features=1.0,random_state=20260921,n_jobs=2))]:
        model.fit(x[train],h[train]);pred=model.predict(x[val])
        # Total squared-error score avoids giving tiny-variance coordinates equal weight.
        score=r2_score(h[val],pred,multioutput='variance_weighted')
        results.append(dict(features=name,model=modelname,validation_embedding_variance_explained=float(score),validation_pair_distance_spearman=float(corr)))
summary=dict(train_cells=int(train.sum()),validation_cells=int(val.sum()),validation_stacks=int(m.loc[val,'sample'].nunique()),feature_sets=sets,results=results,caveat='Diagnostic validation set was used for CNN checkpoint selection. Biological independence unknown. No biological target labels; this quantifies redundancy, not superiority. Forest and ridge settings fixed without validation tuning. Conventional measurements come from original masks, matching CNN input provenance; not strong-smoothed meshes.')
(OUT/'traditional_feature_comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
lines=['# CNN versus traditional features: redundancy audit','',f'Train: {train.sum()} cells. Diagnostic validation: {val.sum()} cells from {m.loc[val,"sample"].nunique()} whole stacks.','', '| Traditional inputs | Predictor | Validation embedding variance explained | Pair-distance Spearman |','|---|---|---:|---:|']
for r in results:lines.append(f"| {r['features']} | {r['model']} | {r['validation_embedding_variance_explained']:.1%} | {r['validation_pair_distance_spearman']:.3f} |")
lines+=['', 'Each predictor was fitted only on training cells to reconstruct the normalized 128-dimensional encoder embedding. Scores are variance-weighted R² on validation cells; they are not classification accuracy. Traditional features were standardized using training statistics. Shape features: sphericity, elongation, flatness and mesh solidity. Aspect ratio was omitted as redundant with the axis ratios. Pair-distance correlations compare standardized Euclidean traditional-feature distances with cosine embedding distances over all validation-cell pairs. These dependent pairs do not support an independent-observation significance test.','',summary['caveat'],'','A high prediction score means much of the CNN variation can be reproduced from conventional measurements. Remaining variation may contain additional shape information, discretization effects or noise; this analysis cannot distinguish those possibilities.','', 'A superiority benchmark needs an external target: blinded expert shape-similarity judgements or independently defined biological classes/conditions, with organoid/animal-level held-out groups. Compare traditional shape features, shape plus size, CNN alone, and their combination using the same split and downstream predictor. Rotation-repeat retrieval can additionally measure invariance, but does not establish biological usefulness.']
(OUT/'TRADITIONAL_COMPARISON.md').write_text('\n'.join(lines)+'\n')
