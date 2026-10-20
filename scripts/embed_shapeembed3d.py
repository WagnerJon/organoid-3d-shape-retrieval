"""Embed a new native .mesh or Nx3 .npy/.npz point set with the trained Lite3D pilot."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from shapeembed3d_core import OUT,ShapeEmbed3D,edm,fps_indices
from pointnet_shape_core import sample_surface,sha256
from prepare_pointnet_shape import read_mesh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=OUT/'invariant/best.pt')
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto');parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.output.resolve() in [args.input.resolve(),args.checkpoint.resolve()]:raise ValueError('Output must differ from input and checkpoint')
    rng=np.random.default_rng(args.seed)
    if args.input.suffix=='.mesh':
        v,f=read_mesh(args.input);candidates=sample_surface(v,f,2048,rng).astype(np.float32)
    else:
        if args.input.suffix=='.npy':p=np.load(args.input)
        else:
            with np.load(args.input) as a:p=a['points']
        p=np.asarray(p,dtype=np.float32)
        if p.ndim!=2 or p.shape[1]!=3 or len(p)<128 or not np.isfinite(p).all():raise ValueError('Expected at least128 finite XYZ points')
        candidates=p[rng.choice(len(p),min(2048,len(p)),replace=False)]
    if not np.isfinite(candidates).all():raise ValueError('Input geometry contains nonfinite coordinates')
    ids=fps_indices(candidates[None],128)[0];points=candidates[ids]
    device=('mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(4);state=torch.load(args.checkpoint,map_location='cpu',weights_only=False);kind=state.get('kind','invariant');m=ShapeEmbed3D(kind).to(device).eval();m.load_state_dict(state['model'],strict=True)
    with torch.no_grad():
        d=edm(torch.tensor(points[None],device=device));mean,log_var,normalized,scale=m.encode(d);pred,decoded=m.decode(mean)
    reconstructed=pred.cpu().numpy()[0];reconstructed=(reconstructed-reconstructed.mean(0))*float(scale.cpu().reshape(-1)[0])+points.mean(0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(model='Experimental ShapeEmbedLite3D',kind=kind,checkpoint_sha256=sha256(args.checkpoint),source_sha256=sha256(args.input),seed=args.seed,descriptor='Raw128D posterior mean; no nuisance adjustment',reconstruction='128 global decoded points in input units, centered near source; orientation/reflection arbitrary, no fitted alignment; not a surface mesh')
    np.savez_compressed(args.output,embedding=mean.cpu().numpy()[0],logvar=log_var.cpu().numpy()[0],sampled_points=points,normalization_scale=scale.cpu().numpy().reshape(-1)[0],source_center=points.mean(0),reconstruction_unaligned=reconstructed,normalized_input_edm=normalized.cpu().numpy()[0],decoded_edm=decoded.cpu().numpy()[0],metadata=json.dumps(metadata))
    print(json.dumps(dict(output=str(args.output),dimensions=128,points=128,device=device)))


if __name__=='__main__':main()
