"""Run pinned official Allen encoder/decoder source with minimal import adapters.

Only optional dependency imports are bypassed; model class/function bodies are
executed from the released source, not reimplemented. Grid/transformer branches
are disabled and fail closed if accessed. This does not install the full legacy
Lightning/CUDA environment.
"""
import ast
import logging
import types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from pointnet_shape_core import ROOT,sha256

CYTO=ROOT/'.cache/allen-cyto-dl'
BENCH=ROOT/'.cache/allen-benchmarking'
OUT=ROOT/'results/allen_reference'
SOURCES={
    'vnn':CYTO/'cyto_dl/nn/point_cloud/vnn.py',
    'graph':CYTO/'cyto_dl/nn/point_cloud/graph_functions.py',
    'encoder':CYTO/'cyto_dl/nn/point_cloud/dgcnn.py',
    'decoder':BENCH/'src/pointcloudutils/networks/simple_inner.py',
}


def disabled(*args,**kwargs):raise RuntimeError('Disabled optional Allen module branch accessed')


def load_module(path,name,extra=None,skip=()):
    tree=ast.parse(path.read_text(),filename=str(path))
    tree.body=[node for node in tree.body if not isinstance(node,ast.ImportFrom) or node.module not in skip]
    namespace={'__name__':'_allen_reference_'+name,'__file__':str(path)};namespace.update(extra or {})
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace


def official_classes():
    vnn=load_module(SOURCES['vnn'],'vnn');graph=load_module(SOURCES['graph'],'graph')
    extra=dict(graph,**{k:v for k,v in vnn.items() if not k.startswith('__')})
    extra.update(scatter_max=disabled,scatter_mean=disabled,VNTransformer=disabled,utils=types.SimpleNamespace(get_pylogger=logging.getLogger))
    encoder=load_module(SOURCES['encoder'],'encoder',extra,skip=('torch_scatter','VN_transformer','cyto_dl','graph_functions','vnn'))
    decoder=load_module(SOURCES['decoder'],'decoder',dict(VNLinear=vnn['VNLinear']),skip=('cyto_dl.nn.point_cloud.vnn',))
    return encoder['DGCNN'],decoder['DecoderInner']


class AllenModel(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();Encoder,Decoder=official_classes()
        self.encoder=Encoder(num_features=512,hidden_dim=342,k=20,mode='vector',hidden_conv2d_channels=[342]*4,
            hidden_conv1d_channels=[2736,512],scalar_inds=None,include_cross=True,include_coords=True,
            symmetry_breaking_axis=None,generate_grid_feats=False,neural_implicit=True,add_vn_transformer=False)
        self.decoder=Decoder(dim=3,c_dim=512,hidden_size=342,leaky=False)
        if pretrained:
            ckpt=torch.load(OUT/'checkpoints/Rotation_invariant_pointcloud_SDF.ckpt',map_location='cpu',weights_only=False)
            weights=ckpt['state_dict']
            encoder={k.removeprefix('encoder.pcloud.'):v for k,v in weights.items() if k.startswith('encoder.pcloud.')}
            decoder={k.removeprefix('decoder.pcloud.'):v for k,v in weights.items() if k.startswith('decoder.pcloud.')}
            self.encoder.load_state_dict(encoder,strict=True);self.decoder.load_state_dict(decoder,strict=True)

    def encode(self,points):return self.encoder(points)['pcloud']
    def decode(self,z,queries):return self.decoder(queries,z)
    def embedding(self,z):return torch.linalg.vector_norm(z,dim=-1)


def source_hashes():return {k:dict(path=str(v),sha256=sha256(v)) for k,v in SOURCES.items()}


def random_rotation(rng):
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    return q.astype(np.float32)


def chunked_encode(encoder,points,chunk=256,checkpoint=False,return_features=False):
    """Exact pointwise scheduling of the released neural-implicit vector path.

    Global means and the initial full-cloud kNN are unchanged. No channel or point
    reduction. Checkpointing recomputes pointwise activations during backward.
    """
    from torch.utils.checkpoint import checkpoint as recompute
    assert encoder.mode=='vector' and encoder.neural_implicit and not encoder.generate_grid_feats and not encoder.add_vn_transformer
    assert encoder.symmetry_breaking_axis is None and encoder.collate_intermediates
    def apply(fn,x):
        return recompute(fn,x,use_reentrant=False) if checkpoint and torch.is_grad_enabled() else fn(x)
    n=points.shape[1]
    raw=points.transpose(2,1)
    idx=chunked_knn(raw,encoder.k,chunk)
    graph_fn=encoder.get_graph_features.__globals__['get_graph_features']
    x=graph_fn(raw,k=encoder.k,idx=idx,mode='vector',scalar_inds=None,include_cross=encoder.include_cross,include_input=True);stages=[]
    for index,conv in enumerate(encoder.convs):
        if index==0:
            # Here the point axis is penultimate; last axis is k neighbors.
            x=torch.cat([apply(lambda block,layer=conv:layer(block).mean(-1),x[...,first:first+chunk,:])
                         for first in range(0,n,chunk)],dim=-1)
        else:
            pre=torch.cat([apply(conv,x[...,first:first+chunk]) for first in range(0,n,chunk)],dim=-1)
            x=torch.cat([pre.mean(-1,keepdim=True).expand_as(pre),pre],dim=1) if index<len(encoder.convs)-1 else pre
        stages.append(x)
    values=[]
    for first in range(0,n,chunk):
        x=torch.cat([s[...,first:first+chunk] for s in stages],dim=1)
        def final(block):
            layers=encoder.final_conv[:-1] if return_features else encoder.final_conv
            for layer in layers:block=layer(block)
            return block.sum(-1)
        values.append(apply(final,x))
    return torch.stack(values).sum(0)/n


@torch.no_grad()
def chunked_knn(x,k,chunk=256):
    """Exact all-points neighbors with bounded query-axis distance allocation."""
    batch,_,n=x.shape;xx=(x*x).sum(1);indices=[]
    for first in range(0,n,chunk):
        inner=-2*torch.matmul(x[:,:,first:first+chunk].transpose(2,1),x)
        pair=-xx[:,first:first+chunk,None]-inner-xx[:,None,:]
        indices.append(pair.topk(k=k,dim=-1)[1])
    idx=torch.cat(indices,dim=1)
    base=torch.arange(batch,device=x.device).view(-1,1,1)*n
    return (idx+base).reshape(-1)


def latent_rotation_distances(codes):
    """Secondary diagnostic: full vector-code distance modulo proper SO(3).

    This retains inter-channel angular structure that the published scalar-norm
    readout discards. It is a separate readout, not the paper's reported512 norms.
    """
    z=np.asarray(codes,dtype=float)
    cross=np.einsum('aci,bcj->abij',z,z)
    u,s,vt=np.linalg.svd(cross)
    orientation=np.linalg.det(u@vt)
    optimal=s[...,:2].sum(-1)+orientation*s[...,2]
    norm=np.square(z).sum((1,2))
    d=np.sqrt(np.maximum(norm[:,None]+norm[None,:]-2*optimal,0))
    np.fill_diagonal(d,0)
    return (d+d.T)/2
