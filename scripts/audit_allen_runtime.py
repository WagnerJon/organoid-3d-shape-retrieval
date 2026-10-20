"""Validate cached-head algebra and CPU/Metal agreement without touching Atlas."""
import json
import numpy as np
import pandas as pd
import torch
from allen_reference_core import AllenModel,OUT,chunked_encode


def main():
    torch.set_num_threads(2);frame=pd.read_csv(OUT/'manifest.csv');i=0;row=frame.iloc[i]
    with np.load(OUT/'data'/f'{i:04d}.npz') as a:
        rng=np.random.default_rng(20261112+i*17);p=a['points'][rng.integers(len(a['points']),size=8192)].copy()
    if row['split']=='train':p+=rng.normal(0,.005,p.shape).astype(np.float32)
    model=AllenModel().eval()
    with torch.no_grad():features=chunked_encode(model.encoder,torch.tensor(p[None]),chunk=512,return_features=True).numpy()[0]
    with np.load(OUT/'features'/f'{i:04d}_0.npz') as a:cached=a['features'];backend=str(a['device']);latent=a['latent']
    relative=float(np.linalg.norm(features-cached)/np.linalg.norm(features))
    with torch.no_grad():pred=model.encoder.final_conv[-1](torch.tensor(cached[None])).numpy()[0]
    np.testing.assert_allclose(pred,latent,rtol=1e-4,atol=1e-5)
    if relative>.001:raise ValueError(f'CPU/Metal relative difference {relative} exceeds prespecified tolerance')
    result=dict(cache_backend=backend,cpu_feature_relative_error=relative,threshold=.001,
        cached_head_maximum_error=float(np.max(np.abs(pred-latent))),scope='One full-size training cloud; additional CPU float64 forward, gradient and rotation tests check exact scheduling. Not a guarantee of bitwise backend equality.')
    (OUT/'runtime_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
