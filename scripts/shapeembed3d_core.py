"""Experimental 3D ShapeEmbedLite adaptation, preserving EDM-to-VAE-to-EDM learning.

Surface point sets require simultaneous row/column permutation handling rather
than the cyclic/reversal subgroup of an ordered 2D contour. The invariant encoder
and approximate distance-based matching are explicitly new 3D components.
"""
import importlib.util
import torch
from torch import nn
import numpy as np
from scipy.optimize import linear_sum_assignment
from pointnet_shape_core import ROOT

OUT=ROOT/'results/shapeembed3d'
SOURCE=ROOT/'.cache/shapeembedlite-official/utils/models.py'


def fps_indices(points,n=128):
    """Geometry-only NumPy farthest-point sampling, first point as seed."""
    p=np.asarray(points);batch,size,_=p.shape
    if n>size:raise ValueError('Not enough candidate points')
    distance=np.full((batch,size),np.inf);current=np.zeros(batch,dtype=int);result=[]
    for _ in range(n):
        result.append(current.copy());distance=np.minimum(distance,((p-p[np.arange(batch),current,None])**2).sum(-1));current=distance.argmax(1)
    return np.stack(result,1)


def official():
    spec=importlib.util.spec_from_file_location('_official_shapeembedlite_models',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def edm(points):
    return torch.cdist(points,points,compute_mode='donot_use_mm_for_euclid_dist')


def normalize_edm(d):
    # Identical to Lite Normalize('fro') followed by Scale(N).
    scale=torch.linalg.vector_norm(d,dim=(-2,-1),keepdim=True)/d.shape[-1]
    return d/scale.clamp_min(1e-12),scale


class EDMSetEncoder(nn.Module):
    """Scalar distance-message network, equivariant under D -> P D P^T.

    No coordinates, frame, orientation, point indices or morphology features are
    supplied. All index reductions are symmetric means/maxima. This does not
    claim a mathematically complete graph-isomorphism invariant.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('distance_centers',torch.linspace(0,4,32))
        self.edge=nn.Sequential(nn.Linear(32,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.node=nn.Sequential(nn.Linear(64,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.pair=nn.Sequential(nn.Linear(161,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.update=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.global_head=nn.Sequential(nn.Linear(256,512),nn.LayerNorm(512),nn.SiLU())
    def forward(self,d):
        radial=torch.exp(-.5*((d[...,None]-self.distance_centers)/.15)**2)
        edge=self.edge(radial);h=self.node(torch.cat([edge.mean(2),edge.max(2).values],-1));n=d.shape[-1]
        pair=self.pair(torch.cat([h[:,:,None].expand(-1,-1,n,-1),h[:,None].expand(-1,n,-1,-1),edge,d[...,None]],-1))
        h=self.update(torch.cat([h,pair.mean(2),pair.max(2).values],-1))
        return self.global_head(torch.cat([h.mean(1),h.max(1).values],-1))


class ShapeEmbed3D(nn.Module):
    def __init__(self,kind='invariant',n=128,latent=128):
        super().__init__();self.kind=kind;self.n=n;source=official()
        self.encoder=EDMSetEncoder() if kind=='invariant' else source.my_resnet18(padding=False,weights=None)
        self.z_mean=nn.Linear(512,latent);self.z_log_var=nn.Linear(512,latent)
        # Keep initial stochastic noise below cell-to-cell signal. This changes
        # initialization, not the variational objective or information inputs.
        nn.init.constant_(self.z_log_var.bias,-6.)
        # Reuse Lite's actual decoder class/default hidden-layer recipe, changing
        # only the number of coordinates to3 and choosing128 surface points.
        self.decoder=source.DecoderMLP(input_dim=latent,space_dim=3,num_points=n)
    def encode(self,d):
        normalized,scale=normalize_edm(d)
        feature=self.encoder(normalized if self.kind=='invariant' else normalized[:,None])
        return self.z_mean(feature),self.z_log_var(feature),normalized,scale
    def decode(self,z):
        points,_=self.decoder(z)
        return points,edm(points)
    def forward(self,d,deterministic=False):
        mean,log_var,target,scale=self.encode(d)
        z=mean if deterministic else mean+torch.randn_like(mean)*torch.exp(.5*log_var.clamp(-20,10))
        points,reconstructed=self.decode(z)
        return reconstructed,target,mean,log_var,points,scale


def sinkhorn(cost,epsilon,steps=40):
    """Uniform probability coupling; log-domain row/column balancing."""
    n,m=cost.shape[-2:];logk=-cost/epsilon;u=torch.zeros_like(cost[...,0]);v=torch.zeros_like(cost[...,0,:])
    for _ in range(steps):
        u=-np.log(n)-torch.logsumexp(logk+v[...,None,:],-1)
        v=-np.log(m)-torch.logsumexp(logk+u[...,:,None],-2)
    t=torch.exp(logk+u[...,:,None]+v[...,None,:])
    # Finite Sinkhorn iteration can leave substantial row errors at small
    # entropy. Round to the uniform transport polytope rather than allowing the
    # reconstruction loss to ignore underweighted surface points.
    t=t*torch.minimum((1/n)/t.sum(-1).clamp_min(1e-30),torch.ones_like(u))[...,None]
    t=t*torch.minimum((1/m)/t.sum(-2).clamp_min(1e-30),torch.ones_like(v))[...,None,:]
    row=(1/n-t.sum(-1)).clamp_min(0);col=(1/m-t.sum(-2)).clamp_min(0)
    t=t+row[...,None]*col[...,None,:]/row.sum(-1,keepdim=True)[...,None].clamp_min(1e-20)
    return t


@torch.no_grad()
def distance_coupling(a,b):
    """Soft GW correspondence, generalized from Lite's cyclic EDM matching.

    Distance-profile initialization and annealed entropic block updates use only
    EDMs. Simultaneous input permutations transport this entire computation.
    Approximate relaxation, not an exact QAP/permutation-optimum certificate.
    """
    sa=a.sort(-1).values;sb=b.sort(-1).values
    profiles=((sa[:,:,None]-sb[:,None])**2).mean(-1)
    t=sinkhorn(profiles,.01)
    const=a.square().mean(-1)[...,None]+b.square().mean(-1)[...,None,:]
    for epsilon in [.03,.01,.003]:
        for _ in range(3):
            cost=const-2*a@t@b.transpose(-2,-1)
            t=sinkhorn(cost,epsilon)
    return t/t.sum((-2,-1),keepdim=True)


@torch.no_grad()
def permutation_coupling(a,b):
    """Approximate full reindexing: hard one-to-one candidates, EDM-only choice.

    Project the GW transport plan with Hungarian matching; also propose a
    distance-profile match. Keep the candidate with lower actual reindexed EDM
    MSE. No coordinates or alignment are used to choose correspondence.
    """
    soft=distance_coupling(a,b).cpu().numpy();aa=a.cpu().numpy();bb=b.cpu().numpy();plans=[];n=a.shape[-1]
    for x,y,t in zip(aa,bb,soft):
        sx=np.sort(x,axis=-1);sy=np.sort(y,axis=-1)
        profile=((sx[:,None]-sy[None])**2).mean(-1)
        candidates=[linear_sum_assignment(-t)[1],linear_sum_assignment(profile)[1]]
        best=min(candidates,key=lambda p:np.square(x-y[p[:,None],p[None,:]]).mean())
        plan=np.zeros((n,n),dtype=x.dtype);plan[np.arange(n),best]=1/n;plans.append(plan)
    return torch.tensor(np.stack(plans),device=a.device)


def gw_edm_loss(a,b,coupling=None):
    """Actual EDM MSE after GW-initialized hard reindexing by default.

    Explicit coupling is retained for transport-formula/oracle gradient audits.
    """
    if coupling is None:
        t=permutation_coupling(a.detach(),b.detach());perm=t.argmax(-1);batch=torch.arange(len(a),device=a.device)[:,None,None]
        reordered=b[batch,perm[:,:,None],perm[:,None,:]]
        return (a-reordered).square().mean((-2,-1)),t
    t=coupling
    row=t.sum(-1);col=t.sum(-2)
    aa=(a.square()*row[...,None]*row[...,None,:]).sum((-2,-1))
    bb=(b.square()*col[...,None]*col[...,None,:]).sum((-2,-1))
    cross=((a@t@b.transpose(-2,-1))*t).sum((-2,-1))
    return (aa+bb-2*cross).clamp_min(0),t


def vae_loss(output,beta=1e-10):
    prediction,target,mean,log_var,_,_=output
    rec,_=gw_edm_loss(prediction,target)
    kl=-.5*(1+log_var-mean.square()-log_var.exp()).sum(-1)
    # One reconstruction term; the upstream script adds it twice. We preserve
    # the stated paper objective rather than duplicating that bookkeeping bug.
    return rec.mean()+beta*kl.mean(),rec.mean(),kl.mean()
