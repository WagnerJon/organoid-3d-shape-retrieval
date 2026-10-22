"""Scale-controlled nonlearned baselines; models/checkpoints remain frozen."""
from learn_spherical_shape_study import *

def main():
    for name in ['synthetic','real']:
        folder=OUT/'learning'/name
        if name=='synthetic':
            a=np.load(OUT/'learning_synthetic.npz');clean=a['clean'];train=np.arange(384);gallery=clean[448:];queries=a['views'][448:];ids=np.arange(128)
        else:
            a=np.load(OUT/'learning_real_data.npz');clean=a['clean'];train=a['train'];validation=a['validation'];gallery=clean[validation];queries=np.load(OUT/'real_noise_queries.npz')['queries'];ids=np.searchsorted(validation,a['query_indices'])
        scale=np.load(folder/'scaler.npz');mean,sd=scale['mean'],scale['sd'];gx=(gallery-mean)/sd;qx=(queries-mean)/sd
        methods={
            'radial_energy':(gallery[:,:13]**2,queries[...,:13]**2),
            'radial_power_standardized':(gx[:,:13],qx[...,:13]),
            'bispectrum_standardized':(gx[:,13:73],qx[...,13:73]),
            'radial_power_bispectrum_standardized':(gx[:,:73],qx[...,:73])}
        block_scale=np.ones(clean.shape[1])
        for s in [slice(0,13),slice(13,73),slice(73,None)]:
            block_scale[s]=max(np.sqrt(np.mean(np.square(clean[train,s]-mean[s]).sum(1))),1e-12)
        methods['block_balanced_invariants']=((gallery-mean)/block_scale,(queries-mean)/block_scale)
        original=np.load(folder/'gallery_codes.npz');codes={k:original[k] for k in original.files}
        rows=pd.read_csv(folder/'retrieval.csv');rows=rows[~rows.method.isin(methods)].to_dict('records')
        for method,(g,q) in methods.items():
            codes[method]=g
            for noise in range(q.shape[1]):
                pred=np.argsort(cdist(q[:,noise],g),axis=1)
                rows.append(dict(domain=name,method=method,seed=-1,noise_index=noise,identity=float(np.mean(pred[:,0]==ids)),recall5=float(np.mean([i in x[:5] for i,x in zip(ids,pred)]))))
        np.savez_compressed(folder/'gallery_codes.npz',**codes)
        pd.DataFrame(rows).to_csv(folder/'retrieval.csv',index=False)
    matched_evaluation(pd.read_csv(OUT/'real_manifest.csv'))
    print('Scaled invariant ablations complete',flush=True)
if __name__=='__main__':main()
