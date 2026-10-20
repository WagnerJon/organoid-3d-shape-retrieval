"""Run isolated experiment stages. Deliberately has no Atlas publishing option."""
import argparse
import time
from prepare_sdf_shape import OUT
from train_sdf_shape import train
from evaluate_sdf_shape import evaluate


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['overfit','pilot'],required=True)
    parser.add_argument('--device',default='auto');args=parser.parse_args()
    for wait in range(120):
        if (OUT/'manifest.csv').exists():break
        if wait%6==0:print('Waiting for signed-distance preparation to finish',flush=True)
        time.sleep(5)
    else:raise RuntimeError('Complete preparation before running the experiment')
    for kind in ['pointnet','vn']:
        train(kind,args.stage,args.device,250 if args.stage=='overfit' else 120,
              initial_checkpoint=OUT/f'overfit_{kind}'/'best.pt' if args.stage=='pilot' else None)
    evaluate(args.stage,args.device)
