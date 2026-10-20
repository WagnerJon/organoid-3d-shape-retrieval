"""Prepare -> train -> validation gate -> optional Atlas publication. Test evaluation is separate."""
import argparse
from pointnet_shape_core import ROOT,read_config
from prepare_pointnet_shape import prepare
from train_pointnet_shape import train
from evaluate_pointnet_shape import evaluate
from export_pointnet_shape import export
from benchmark_pointnet_shape import benchmark
from report_pointnet_shape import report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config/pointnet_shape.json');parser.add_argument('--device');parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();config=read_config(ROOT/args.config)
    if not (ROOT/config['output_dir']/'data/dataset.json').exists():prepare(config)
    train(config,args.smoke,args.device,args.resume)
    evaluate(config,args.smoke,args.device)
    if not args.smoke:
        benchmark(config,args.device)
        report(config)
    if args.publish:
        if args.smoke:raise ValueError('Smoke runs cannot be published')
        export(config,args.device)
