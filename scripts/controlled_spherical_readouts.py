"""Controlled-family retrieval and deformation preservation, frozen encoders."""
from learn_spherical_shape_study import *
from scipy.stats import spearmanr

def main():
    frame=pd.read_csv(OUT/'sensitivity.csv');cache=OUT/'controlled_learning_features.npz'
    if cache.exists():features=np.load(cache)['features']
    else:
        d,Y=fib_basis(2048,12);features=[]
        for row in frame.itertuples():features.append(radial_input(synthetic_radius(d,row.family,row.amount,row.width,row.seed),Y))
        features=np.array(features);np.savez_compressed(cache,features=features)
    rows=[];sensitivity=[]
    for domain in ['synthetic','real']:
        folder=OUT/'learning'/domain;z=np.load(folder/'scaler.npz');mean,sd=z['mean'],z['sd']
        codes={'radial_power':features[:,:13],'radial_power_bispectrum_standardized':((features-mean)/sd)[:,:73],'shell_power':features[:,73:]}
        for seed in [42,43,44]:
            cp=torch.load(folder/f'seed{seed}.pt',weights_only=True);model=Encoder(features.shape[1]);model.load_state_dict(cp['state']);model.eval()
            with torch.no_grad():codes[f'VAE_seed{seed}']=model.encode(torch.tensor((features-mean)/sd,dtype=torch.float32)).numpy()
        for method,x in codes.items():
            distance=cdist(x,x);precision=[]
            for i,r in frame.iterrows():
                if r.amount==0:continue
                identical=(frame.family==r.family)&(frame.amount==r.amount)
                if r.family!='elongation':identical&=frame.width==r.width
                allowed=np.flatnonzero((frame.amount>0)&~identical)
                top=allowed[np.argsort(distance[i,allowed])[:5]]
                precision.append(float(np.mean(frame.iloc[top].family==r.family)))
            rows.append(dict(training_domain=domain,method=method,queries=len(precision),family_precision5=float(np.mean(precision))))
            for (family,width,seed),part in frame.groupby(['family','width','seed']):
                ids=part.index.to_numpy();values=np.linalg.norm(x[ids]-x[ids[0]],axis=1)
                sensitivity.append(dict(training_domain=domain,method=method,family=family,width=width,seed=int(seed),magnitude_spearman=float(spearmanr(part.amount,values).statistic)))
    pd.DataFrame(rows).to_csv(OUT/'controlled_family_retrieval.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'learned_sensitivity.csv',index=False)
    print('Controlled-family and morphology-preservation readouts complete',flush=True)
if __name__=='__main__':main()
