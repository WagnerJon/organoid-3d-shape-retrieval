"""Prepare and train the predeclared offline Point-MAE experiment."""
import json,time,math,subprocess,sys,os
import numpy as np
import pandas as pd
import torch
from point_mae_reference import OUT,ROOT,model,normalized
from pointnet_shape_core import sha256
from spatial_benchmark_core import mesh_for,surface_frame
from allen_reference_core import random_rotation


def prepare():
    frame=pd.read_csv(ROOT/'results/allen_reference/manifest.csv')
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha256(ROOT/'results/allen_reference/manifest.csv')==protocol['source_manifest_sha256']
    frame.to_csv(OUT/'manifest.csv',index=False)
    if (OUT/'prepared.json').exists():return frame
    pools=np.lib.format.open_memmap(OUT/'pools.npy',mode='w+',dtype=np.float32,shape=(len(frame),32768,3));axes=[];gaps=[];covariances=[]
    for i,row in frame.iterrows():
        with np.load(ROOT/'results/allen_reference/data'/f'{i:04d}.npz') as a:
            if str(a['mesh_sha256'])!=row.source_mesh_sha256:raise ValueError('Surface source mismatch')
            pools[i]=a['points']
        v,f=mesh_for(row);_,basis,eigen,gap=surface_frame(v,f)
        axes.append(basis);covariances.append(basis@np.diag(eigen)@basis.T);gaps.append(gap)
    pools.flush();np.savez(OUT/'frames.npz',axes=axes,covariance=covariances,gap=gaps)
    (OUT/'prepared.json').write_text(json.dumps(dict(cells=len(frame),train=int((frame.split=='train').sum()),validation=int((frame.split=='validation').sum()),test_used=False,source_sha256=sha256(OUT/'manifest.csv')),indent=2))
    print('Prepared609 eligible training/validation cells',flush=True)
    return frame


def draw(pool,index,view=0):
    rng=np.random.default_rng(20261019+index*3+view)
    return normalized(pool[rng.choice(len(pool),1024,replace=False)])


def validation(m,pools,ids,device):
    state=np.random.get_state();np.random.seed(1219);m.eval();losses=[]
    with torch.no_grad():
        for first in range(0,len(ids),8):
            ii=ids[first:first+8];x=np.stack([draw(pools[i],i) for i in ii])
            loss=m(torch.tensor(x,device=device));losses.append((float(loss.cpu()),len(ii)))
    np.random.set_state(state)
    return sum(x*n for x,n in losses)/sum(n for x,n in losses)


def train(frame):
    if (OUT/'adaptation.completed.json').exists():return
    torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.55);device='mps'
    torch.manual_seed(42);np.random.seed(42);rng=np.random.default_rng(42)
    m=model().to(device);opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.05)
    pools=np.load(OUT/'pools.npy',mmap_mode='r');train=frame.index[frame.split=='train'].to_numpy();val=frame.index[frame.split=='validation'].to_numpy()
    start=0;best=float('inf');history=[];offset=0.;began=time.monotonic()
    if (OUT/'last.pt').exists():
        saved=torch.load(OUT/'last.pt',map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==sha256(OUT/'protocol.json')
        m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['rng'];np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.mps.set_rng_state(saved['mps_rng'])
        start=saved['epoch']+1;best=saved['best'];offset=saved['seconds'];history=pd.read_csv(OUT/'history.csv').to_dict('records')
    for epoch in range(start,31):
        training=None
        if epoch:
            m.train();losses=[];order=rng.permutation(train)
            rate=.000001+(.0001-.000001)*.5*(1+math.cos(math.pi*(epoch-1)/30))
            for g in opt.param_groups:g['lr']=rate
            for first in range(0,len(order),8):
                ids=order[first:first+8];x=np.stack([normalized(pools[i][rng.choice(32768,1024,replace=False)])@random_rotation(rng) for i in ids])
                loss=m(torch.tensor(x,device=device));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                value=float(loss.detach().cpu())
                if not np.isfinite(value):raise ValueError('Nonfinite training loss')
                losses.append(value)
            training=float(np.mean(losses))
        if epoch%5==0:
            value=validation(m,pools,val,device)
            if not np.isfinite(value):raise ValueError('Nonfinite validation loss')
            record=dict(epoch=epoch,training_patch_cd=training,validation_patch_cd=value,seconds=offset+time.monotonic()-began)
            history.append(record)
            if value<best:
                best=value;torch.save(dict(model=m.state_dict(),epoch=epoch,validation_patch_cd=value,protocol_sha256=sha256(OUT/'protocol.json')),OUT/'best.pt')
            pd.DataFrame(history).to_csv(OUT/'history.csv',index=False)
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,rng=rng.bit_generator.state,numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),seconds=offset+time.monotonic()-began,protocol_sha256=sha256(OUT/'protocol.json')),OUT/'last.pt')
            (OUT/'status.json').write_text(json.dumps(dict(status='adapting',**record)));print(json.dumps(record),flush=True)
    (OUT/'adaptation.completed.json').write_text(json.dumps(dict(status='complete',epochs=30,best_validation_patch_cd=best,seconds=offset+time.monotonic()-began,scope='All parameters adapted with maskedpatch reconstruction; no labels.'),indent=2))


def main():
    frame=prepare();train(frame)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
    for script in ['evaluate_point_mae_reference.py','report_point_mae_reference.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script)],check=True,env=env)


if __name__=='__main__':main()
