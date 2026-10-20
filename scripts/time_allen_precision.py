import time,json
import numpy as np,torch
from allen_reference_core import AllenModel,OUT,chunked_encode

def main():
 torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.55);m=AllenModel().to('mps').eval();p=np.load(OUT/'data/0000.npz')['points'];x=torch.tensor(p[:8192][None],device='mps');result=[]
 with torch.no_grad():
  t=time.monotonic();base=chunked_encode(m.encoder,x,chunk=1024);torch.mps.synchronize();print('FP32 chunk1024',time.monotonic()-t,flush=True)
  for dtype in [torch.bfloat16,torch.float16]:
   t=time.monotonic()
   with torch.autocast('mps',dtype=dtype):z=chunked_encode(m.encoder,x,chunk=1024)
   torch.mps.synchronize();r=dict(dtype=str(dtype),seconds=time.monotonic()-t,finite=bool(torch.isfinite(z).all().cpu()),relative_error=float(((z.float()-base).norm()/base.norm()).cpu()));result.append(r);print(json.dumps(r),flush=True)
 (OUT/'precision_probe.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
