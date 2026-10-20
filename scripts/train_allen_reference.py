"""Separate partial fine-tuning contrasts of the official pretrained framework.

The 8192-point backbone stays fixed. Cached pre-head vectors are rotated jointly
with SDF queries; only the official final VNLinear and/or DecoderInner are trained.
This is not full end-to-end training and is explicitly labelled in the report.
"""
import argparse,copy,json,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from allen_reference_core import AllenModel,OUT,random_rotation
from pointnet_shape_core import sha256


class CachedAllen(nn.Module):
    def __init__(self):
        super().__init__();original=AllenModel()
        self.head=copy.deepcopy(original.encoder.final_conv[-1]);self.decoder=copy.deepcopy(original.decoder)
    def encode(self,features):return self.head(features)
    def decode(self,z,q):return self.decoder(q,z)
    def embedding(self,z):return z.norm(dim=-1)


def load_data():
    frame=pd.read_csv(OUT/'manifest.csv');features=[];query=[];target=[]
    for i in range(len(frame)):
        with np.load(OUT/'features'/f'{i:04d}_0.npz') as a:features.append(a['features'])
        with np.load(OUT/'data'/f'{i:04d}.npz') as a:query.append(a['query']);target.append(a['target'])
    return frame,np.asarray(features),np.asarray(query),np.asarray(target)


def validation(model,features,query,target,ids,device):
    model.eval();values=[]
    with torch.no_grad():
        for first in range(0,len(ids),8):
            index=ids[first:first+8];z=model.encode(torch.tensor(features[index],device=device));error=0.
            for start in range(0,20000,2048):
                q=torch.tensor(query[index,start:min(start+2048,20000)],device=device);t=torch.tensor(target[index,start:min(start+2048,20000)],device=device)
                error+=float((model.decode(z,q)-t).abs().sum().cpu())
            values.append((error,len(index)*20000))
    return sum(v for v,n in values)/sum(n for v,n in values)


def train_arm(name,frame,features,query,target,device):
    path=OUT/name;path.mkdir(exist_ok=True)
    if (path/'completed.json').exists():return
    torch.manual_seed(42);rng=np.random.default_rng(42);model=CachedAllen().to(device)
    for p in model.head.parameters():p.requires_grad_(name=='head_decoder')
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.0002)
    train=frame.index[frame.split=='train'].to_numpy();valid=frame.index[frame.split=='validation'].to_numpy()
    best=float('inf');history=[];started=time.monotonic();offset=0.;start_epoch=0
    if (path/'last.pt').exists():
        saved=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        if saved.get('manifest_sha256')!=sha256(OUT/'manifest.csv'):raise ValueError('Resume checkpoint has a different source cohort')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['numpy_rng']
        best=saved['best'];offset=saved['seconds'];start_epoch=int(saved['epoch'])+1
        history=pd.read_csv(path/'history.csv').to_dict('records')
        print(name,'resuming epoch',start_epoch,flush=True)
    for epoch in range(start_epoch,31):
        if epoch>0:
            model.train();losses=[];order=rng.permutation(train)
            for first in range(0,len(order),8):
                ids=order[first:first+8];indices=rng.integers(query.shape[1],size=(len(ids),2048))
                q=torch.tensor(query[ids[:,None],indices],device=device);t=torch.tensor(target[ids[:,None],indices],device=device)
                r=torch.tensor(np.stack([random_rotation(rng) for _ in ids]),device=device)
                f=torch.tensor(features[ids],device=device)@r;q=q@r;z=model.encode(f);prediction=model.decode(z,q)
                # Authors' sum over queried points, mean over cells. Mean absolute
                # error is logged separately so different query counts are comparable.
                loss=(prediction-t).abs().sum(-1).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach().cpu())/2048)
            training=float(np.mean(losses))
        else:training=None
        if epoch%5==0:
            value=validation(model,features,query,target,valid,device)
            record=dict(epoch=epoch,training_mean_l1=training,validation_mean_l1=value,seconds=offset+time.monotonic()-started)
            history.append(record)
            if value<best:
                best=value;torch.save(dict(model=model.state_dict(),epoch=epoch,validation_mean_l1=value,
                    manifest_sha256=sha256(OUT/'manifest.csv'),official_checkpoint_sha256=sha256(OUT/'checkpoints/Rotation_invariant_pointcloud_SDF.ckpt')),path/'best.pt')
            pd.DataFrame(history).to_csv(path/'history.csv',index=False);print(name,json.dumps(record),flush=True)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,numpy_rng=rng.bit_generator.state,best=best,seconds=offset+time.monotonic()-started,manifest_sha256=sha256(OUT/'manifest.csv')),path/'last.pt')
    torch.mps.synchronize()
    (path/'completed.json').write_text(json.dumps(dict(status='complete',epochs=30,selected_validation_mean_l1=best,seconds=offset+time.monotonic()-started,
        training_cells=len(train),validation_cells=len(valid),scope='Partial adaptation with frozen official 8192-point backbone; no end-to-end or full-paper training claim.'),indent=2))


def main():
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.55)
    frame,features,query,target=load_data()
    for name in ['decoder_only','head_decoder']:train_arm(name,frame,features,query,target,'mps')
    print('Both partial-adaptation arms completed',flush=True)


if __name__=='__main__':main()
