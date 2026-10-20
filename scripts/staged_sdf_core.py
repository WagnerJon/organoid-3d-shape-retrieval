"""Staged, vector-code SDF experiments, isolated from Atlas.

Independently implemented reference-style local VN graph plus global context
blocks. This is not a CytoDL reproduction (2048 points rather than 8192).
"""
import numpy as np
import torch
from torch import nn
from pointnet_shape_core import gather
from sdf_shape_core import VectorLayer


class ReferenceVectorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.local=VectorLayer(3,32)
        self.local_expand=VectorLayer(32,64)
        self.local_reduce=nn.Linear(64,32,bias=False)
        self.context=nn.ModuleList([VectorLayer(64,32),VectorLayer(64,64),VectorLayer(128,64)])
        self.fuse=VectorLayer(192,128)
        self.head=nn.Linear(128,128,bias=False)

    @staticmethod
    def normalize_features(x):
        # One scalar per cloud: preserves vector directions, relative local scales,
        # point permutation invariance and SO(3), without batch-dependent statistics.
        return .2*x/(x.square().mean((1,2,3),keepdim=True)+1e-8).sqrt()

    def forward(self,points,neighbors):
        # Center every sampled cloud independently; reduces sampling-centroid leakage.
        points=points-points.mean(1,keepdim=True)
        center=points[:,:,None,None,:].expand(-1,-1,neighbors.shape[-1],1,-1)
        adjacent=gather(points[:,:,None,:],neighbors)
        edge=torch.cat([adjacent-center,center,torch.cross(adjacent,center,dim=-1)],dim=-2)
        x=self.local(edge).mean(2)
        x=self.local_reduce(self.local_expand(x).transpose(-1,-2)).transpose(-1,-2)
        x=self.normalize_features(x)
        stages=[x]
        for block in self.context:
            global_context=x.mean(1,keepdim=True).expand_as(x)
            x=self.normalize_features(block(torch.cat([x,global_context],dim=-2)));stages.append(x)
        x=self.normalize_features(self.fuse(torch.cat(stages,dim=-2))).mean(1)
        return 20*self.head(x.transpose(-1,-2)).transpose(-1,-2)


class VectorAdam:
    """Adam with one adaptive denominator per XYZ vector, preserving SO(3).

    Coordinatewise Adam does not commute with rotation. This version averages
    squared gradients over the spatial axis; only scalar per-channel adaptation.
    """
    def __init__(self,parameter,lr=.01):
        self.parameter=parameter;self.lr=lr;self.step_count=0
        self.first=torch.zeros_like(parameter)
        self.second=torch.zeros_like(parameter[...,:1])

    def zero_grad(self):self.parameter.grad=None

    @torch.no_grad()
    def step(self):
        self.step_count+=1;g=self.parameter.grad
        self.first.mul_(.9).add_(g,alpha=.1)
        self.second.mul_(.999).add_(g.square().mean(-1,keepdim=True),alpha=.001)
        m=self.first/(1-.9**self.step_count);v=self.second/(1-.999**self.step_count)
        self.parameter.addcdiv_(m,v.sqrt().add_(1e-8),value=-self.lr)


def code_loss(prediction,target,scale):
    return ((prediction-target)/scale).square().mean()


def stability(vectors,resampled,rotated):
    def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    z=np.asarray(vectors);u=unit(z);s=np.linalg.svd(z-z.mean(0),compute_uv=False)**2
    p=s/max(float(s.sum()),1e-15)
    return dict(resampling_self_top1=float((np.argmax(unit(resampled)@u.T,axis=1)==np.arange(len(u))).mean()),
        rotation_self_top1=float((np.argmax(unit(rotated)@u.T,axis=1)==np.arange(len(u))).mean()),
        effective_rank=float(np.exp(-(p*np.log(p.clip(1e-15))).sum())))
