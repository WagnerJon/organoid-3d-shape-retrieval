import time,json
import torch,numpy as np
from allen_reference_core import AllenModel,OUT,chunked_encode

def main():
 torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.55);m=AllenModel().to('mps').eval()
 a=np.load(OUT/'data/0000.npz')['points'];results=[]
 with torch.no_grad():
  for batch in [1,4,4]:
   x=torch.tensor(np.stack([a[:8192]]*batch),device='mps');torch.mps.synchronize();t=time.monotonic()
   z=chunked_encode(m.encoder,x,chunk=256,checkpoint=False);torch.mps.synchronize()
   r=dict(batch=batch,seconds=time.monotonic()-t,finite=bool(torch.isfinite(z).all().cpu()),driver_bytes=torch.mps.driver_allocated_memory());results.append(r);print(json.dumps(r),flush=True)
 (OUT/'inference_timing.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
