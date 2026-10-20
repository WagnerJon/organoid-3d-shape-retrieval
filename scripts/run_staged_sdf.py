"""Run follow-up stages after the shared decoder finishes. Never exports to Atlas."""
import argparse
import json
import time
from prepare_staged_sdf import OUT


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='auto');p.add_argument('--steps',type=int,default=500)
    p.add_argument('--epochs',type=int,default=80);args=p.parse_args()
    for wait in range(240):
        status=OUT/'decoder/status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='complete':break
        if wait%12==0:print('Waiting for shared decoder training to finish',flush=True)
        time.sleep(5)
    else:raise RuntimeError('Decoder did not finish within wait limit')
    # Import after readiness, so the finalized evaluator is used.
    from evaluate_staged_sdf import evaluate
    from train_staged_sdf import train_encoder
    evaluate(args.device,args.steps,decoder_only=True)
    train_encoder(args.device,args.epochs)
    evaluate(args.device,args.steps)
