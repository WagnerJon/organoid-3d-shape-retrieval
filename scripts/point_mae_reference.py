"""Pinned Point-MAE classes with CPU/Metal FPS, kNN and squared-Chamfer adapters."""
import ast
import random
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pointnet_shape_core import ROOT,sha256

OUT=ROOT/'results/point_mae'
SOURCE=ROOT/'.cache/point-mae-official/models/Point_MAE.py'


def fps_indices(points,n):
    """Greedy Euclidean FPS starting at point zero, as in PointNet2's kernel."""
    p=points.detach().cpu().numpy();b,size,_=p.shape
    distance=np.full((b,size),np.inf);current=np.zeros(b,dtype=int);result=[]
    for _ in range(n):
        result.append(current.copy())
        distance=np.minimum(distance,((p-p[np.arange(b),current,None])**2).sum(-1))
        current=distance.argmax(1)
    return torch.tensor(np.stack(result,1),device=points.device)


def fps(points,n):
    ids=fps_indices(points,n)
    return points.gather(1,ids[...,None].expand(-1,-1,3))


class KNN:
    def __init__(self,k,transpose_mode=True):self.k=k;assert transpose_mode
    def __call__(self,points,centers):
        distance=torch.cdist(centers,points,compute_mode='donot_use_mm_for_euclid_dist')
        return distance.topk(self.k,dim=-1,largest=False,sorted=True)


class DropPath(nn.Module):
    def __init__(self,drop_prob=0.):super().__init__();self.drop_prob=drop_prob
    def forward(self,x):
        if not self.training or not self.drop_prob:return x
        keep=1-self.drop_prob
        return x*(keep+torch.rand((x.shape[0],)+(1,)*(x.ndim-1),dtype=x.dtype,device=x.device)).floor()/keep


class SquaredChamfer(nn.Module):
    def forward(self,a,b):
        d=((a[:,:,None]-b[:,None])**2).sum(-1)
        return d.min(-1).values.mean()+d.min(-2).values.mean()


class Registry:
    def register_module(self):return lambda cls:cls


def classes():
    tree=ast.parse(SOURCE.read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    env=dict(__name__='_point_mae_reference',torch=torch,nn=nn,F=F,np=np,random=random,
        DropPath=DropPath,trunc_normal_=nn.init.trunc_normal_,MODELS=Registry(),misc=types.SimpleNamespace(fps=fps),
        KNN=KNN,print_log=lambda *a,**k:None)
    exec(compile(tree,str(SOURCE),'exec'),env)
    return env


def config():
    return types.SimpleNamespace(group_size=32,num_group=64,loss='cdl2',transformer_config=types.SimpleNamespace(
        mask_ratio=.6,mask_type='rand',trans_dim=384,encoder_dims=384,depth=12,drop_path_rate=.1,
        num_heads=6,decoder_depth=4,decoder_num_heads=6))


def model(pretrained=True):
    source=classes()['Point_MAE']
    class PortablePointMAE(source):
        def build_loss_func(self,loss_type):
            if loss_type!='cdl2':raise ValueError('Only official squared-Chamfer configuration is supported')
            self.loss_func=SquaredChamfer()
    m=PortablePointMAE(config())
    if pretrained:
        saved=torch.load(OUT/'checkpoints/pretrain.pth',map_location='cpu',weights_only=False)
        state=saved['base_model'];state={k.removeprefix('module.'):v for k,v in state.items()}
        m.load_state_dict(state,strict=True)
    return m


def encode_grouped(m,groups,centers):
    e=m.MAE_encoder;x=e.encoder(groups);x=e.norm(e.blocks(x,e.pos_embed(centers)))
    # Explicit custom readout: concatenate mean and max of all unmasked tokens.
    # No random/untrained classification token is introduced.
    return torch.cat([x.mean(1),x.max(1).values],-1)


def encode(m,points):return encode_grouped(m,*m.group_divider(points))


def masked_prediction(m,points):
    neighborhood,center=m.group_divider(points);x,mask=m.MAE_encoder(neighborhood,center)
    b,_,c=x.shape;n=int(mask.sum(1)[0]);visible=m.decoder_pos_embed(center[~mask]).reshape(b,-1,c)
    hidden=m.decoder_pos_embed(center[mask]).reshape(b,-1,c)
    full=torch.cat([x,m.mask_token.expand(b,n,-1)],1)
    rec=m.MAE_decoder(full,torch.cat([visible,hidden],1),n)
    prediction=m.increase_dim(rec.transpose(1,2)).transpose(1,2).reshape(b*n,32,3)
    target=neighborhood[mask].reshape(b*n,32,3)
    return prediction,target,center,mask,neighborhood


def normalized(points):
    p=points-points.mean(-2,keepdims=True)
    return (p/np.linalg.norm(p,axis=-1).max(-1,keepdims=True)[...,None]).astype(np.float32)
