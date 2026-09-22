"""Small 3D residual encoder, rigid/scale augmentations and standard NT-Xent loss."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.body=nn.Sequential(nn.Conv3d(cin,cout,3,stride=stride,padding=1,bias=False),nn.GroupNorm(4,cout),nn.ReLU(),
                                nn.Conv3d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(4,cout))
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Sequential(nn.Conv3d(cin,cout,1,stride=stride,bias=False),nn.GroupNorm(4,cout))
    def forward(self,x):return F.relu(self.body(x)+self.skip(x))


class ResNet3DSimCLR(nn.Module):
    def __init__(self,base=8,embedding_dim=128,projection_dim=64):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv3d(1,base,3,stride=2,padding=1,bias=False),nn.GroupNorm(4,base),nn.ReLU(),
            ResidualBlock(base,base),ResidualBlock(base,base*2,2),ResidualBlock(base*2,base*4,2),ResidualBlock(base*4,base*8,2),
            nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Linear(base*8,embedding_dim))
        self.projector=nn.Sequential(nn.Linear(embedding_dim,embedding_dim),nn.ReLU(),nn.Linear(embedding_dim,projection_dim))
    def forward(self,x):
        h=self.encoder(x)
        return h,F.normalize(self.projector(h),dim=1)


def augment(x,generator,scale_range=(.95,1.05),translation=.04,binary=True):
    n=len(x)
    # Unit random quaternions generate uniformly distributed 3D rotations.
    q=F.normalize(torch.randn(n,4,generator=generator),dim=1)
    w,a,b,c=q.unbind(1)
    rotation=torch.stack([1-2*(b*b+c*c),2*(a*b-w*c),2*(a*c+w*b),
                          2*(a*b+w*c),1-2*(a*a+c*c),2*(b*c-w*a),
                          2*(a*c-w*b),2*(b*c+w*a),1-2*(a*a+b*b)],dim=1).reshape(n,3,3)
    scale=torch.rand(n,1,1,generator=generator)*(scale_range[1]-scale_range[0])+scale_range[0]
    shift=(torch.rand(n,3,1,generator=generator)*2-1)*translation
    theta=torch.cat([rotation/scale,shift],dim=2).to(x.device)
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    # Threshold returns a binary view; geometric transforms do not require gradients.
    warped=F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    return (warped>=.5).float() if binary else warped


def nt_xent(z1,z2,temperature=.2):
    if len(z1)<2:raise ValueError('SimCLR requires at least two distinct objects in a batch')
    z=F.normalize(torch.cat([z1,z2]),dim=1);n=len(z1)
    logits=z@z.T/temperature
    logits=logits.masked_fill(torch.eye(2*n,device=z.device,dtype=torch.bool),-1e9)
    target=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(logits,target)


def pair_metrics(z1,z2):
    z=F.normalize(torch.cat([z1,z2]),dim=1);n=len(z1)
    sim=z@z.T;target=(torch.arange(2*n,device=z.device)+n)%(2*n)
    positive=sim[torch.arange(2*n,device=z.device),target].mean()
    selfmask=torch.eye(2*n,device=z.device,dtype=torch.bool)
    posmask=torch.zeros_like(selfmask);posmask[torch.arange(2*n,device=z.device),target]=True
    negative=sim[~(selfmask|posmask)].mean()
    accuracy=(sim.masked_fill(selfmask,-1e9).argmax(1)==target).float().mean()
    return dict(positive_cosine=float(positive.detach().cpu()),negative_cosine=float(negative.detach().cpu()),
                pair_retrieval=float(accuracy.detach().cpu()))


def augment_raw(x,generator,cfg):
    x=augment(x,generator,cfg['augmentation_scale'],cfg['augmentation_translation'],binary=False)
    shape=(len(x),1,1,1,1)
    def uniform(bounds):
        return (torch.rand(shape,generator=generator)*(bounds[1]-bounds[0])+bounds[0]).to(x.device)
    x=x.clamp(0,1).pow(uniform(cfg['intensity_gamma']))*uniform(cfg['intensity_gain'])
    noise=torch.randn(x.shape,generator=generator).to(x.device)*cfg['intensity_noise_std']
    return (x+noise).clamp(0,1)
