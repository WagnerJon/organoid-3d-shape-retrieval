"""Isolated paper-inspired VN-DGCNN/SDF pilot, not a CytoDL reproduction.

References: Vasan et al., Nature Methods 2025, doi:10.1038/s41592-025-02729-9;
Deng et al., Vector Neurons, arXiv:2104.12229. Independently implemented VN
equations; no author code is imported. Smaller graph/point budgets for Mac pilot.
"""
import numpy as np
import torch
from torch import nn
from pointnet_shape_core import PointNetEncoder, gather


class VectorLayer(nn.Module):
    """Channel mixing without spatial bias; learned-direction VN nonlinearity."""
    def __init__(self, source, target):
        super().__init__()
        self.value = nn.Linear(source, target, bias=False)
        self.direction = nn.Linear(source, target, bias=False)

    def forward(self, x):
        value = self.value(x.transpose(-1, -2)).transpose(-1, -2)
        direction = self.direction(x.transpose(-1, -2)).transpose(-1, -2)
        dot = (value * direction).sum(-1, keepdim=True)
        projection = dot.clamp_max(0) * direction / direction.square().sum(-1, keepdim=True).clamp_min(1e-8)
        return value - .8 * projection


def knn_vectors(x, k):
    """Graph distances invariant under joint SO(3); no scalar XYZ nonlinearities."""
    with torch.no_grad():
        flat = x.flatten(2)
        distance = (flat.square().sum(-1)[:, :, None] + flat.square().sum(-1)[:, None, :]
                    - 2 * flat @ flat.transpose(1, 2)).clamp_min(0)
        # Include self consistently; finite precision ties can change neighborhoods.
        return distance.topk(min(k, x.shape[1]), largest=False).indices


class VectorEncoder(nn.Module):
    def __init__(self, dimensions=128, neighbors=12):
        super().__init__()
        self.neighbors = neighbors
        self.blocks = nn.ModuleList([VectorLayer(3, 16), VectorLayer(32, 32), VectorLayer(64, 64)])
        self.fuse = VectorLayer(112, dimensions)

    def forward(self, points, groups=None):
        x = points[:, :, None, :]
        stages = []
        for stage, block in enumerate(self.blocks):
            neighbor = gather(x, knn_vectors(x, self.neighbors))
            center = x[:, :, None].expand_as(neighbor)
            edge = [neighbor-center, center]
            if stage == 0:
                edge.append(torch.cross(neighbor, center, dim=-1))
            x = block(torch.cat(edge, dim=-2)).mean(2)
            stages.append(x)
        # Mean pooling preserves vector equivariance, unlike coordinatewise max.
        # Fixed numerical gain keeps normalized-cell vector features at a useful
        # scale for the decoder. A scalar gain preserves SO(3) equivariance.
        return 10 * self.fuse(torch.cat(stages, dim=-2)).mean(1)


class SDFModel(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.encoder = (VectorEncoder() if kind == 'vn' else PointNetEncoder(
            dict(centers=[128,32], neighbors=16, embedding_dim=128)))
        # For VN: query-vector dot products + vector norms + query radius.
        self.decoder = nn.Sequential(nn.Linear(257 if kind == 'vn' else 131, 256), nn.Softplus(beta=10),
            nn.Linear(256,256), nn.Softplus(beta=10), nn.Linear(256,128), nn.Softplus(beta=10), nn.Linear(128,1))

    def encode(self, points, groups=None):
        return self.encoder(points, groups)

    def embedding(self, latent):
        return latent.norm(dim=-1) if self.kind == 'vn' else latent

    def decode(self, latent, query):
        if self.kind == 'vn':
            norms = latent.norm(dim=-1)[:, None].expand(-1, query.shape[1], -1)
            dot = torch.einsum('bqc,bdc->bqd', query, latent)
            features = torch.cat([dot, norms, query.norm(dim=-1, keepdim=True)], dim=-1)
        else:
            features = torch.cat([latent[:, None].expand(-1, query.shape[1], -1), query], dim=-1)
        return self.decoder(features).squeeze(-1)

    def forward(self, points, query, groups=None):
        latent = self.encode(points, groups)
        return latent, self.decode(latent, query)


def sdf_loss(prediction, target):
    # Balanced pools are sampled before this function. Clipping emphasizes boundary.
    # Do not clamp predictions: that can leave a wrongly saturated network with zero gradient.
    return (prediction-target.clamp(-.15,.15)).abs().mean()


def extract_surface(model, latent, resolution=56, extent=1.15):
    from skimage.measure import marching_cubes
    axis=np.linspace(-extent,extent,resolution,dtype=np.float32)
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    with torch.no_grad():
        values=np.concatenate([model.decode(latent,torch.as_tensor(q,device=latent.device)[None]).cpu().numpy()[0]
                               for q in np.array_split(grid, max(1,int(np.ceil(len(grid)/4096))))])
    field=values.reshape((resolution,)*3)
    if field.min()>=0 or field.max()<=0:
        return None, field
    vertices,faces,_,_=marching_cubes(field,0,spacing=(2*extent/(resolution-1),)*3)
    return (vertices-extent,faces),field
