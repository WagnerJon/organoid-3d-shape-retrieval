"""Geometry-only PointNet++ autoencoder. Pure PyTorch; no custom CUDA extensions.

Single-scale kNN set abstraction with deterministic farthest-point sampling.
Hierarchies may be prepared on CPU once: rigid rotation preserves their distances.
This conventional encoder is NOT rotation invariant. No PCA pose alignment or
invariance loss is imposed on a decoder that must reconstruct the input pose.
"""
import hashlib
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config['embedding_dim'] != 128:
        raise ValueError('This baseline requires a 128D latent space')
    if len(config['centers']) != 2 or not 1 < config['centers'][1] < config['centers'][0] <= config['points']:
        raise ValueError('Expected two decreasing set-abstraction center counts')
    if config['neighbors'] < 1 or config['points'] < 4:
        raise ValueError('Invalid point-cloud configuration')
    return config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sample_surface(vertices, faces, count, rng):
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    if not np.isfinite(triangles).all() or not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError('Surface has no finite nonzero-area triangles')
    chosen = triangles[rng.choice(len(triangles), size=count, p=areas/areas.sum())]
    uv = rng.random((count, 2)); root = np.sqrt(uv[:, :1])
    weights = np.concatenate([1-root, root*(1-uv[:, 1:]), root*uv[:, 1:]], axis=1)
    return (chosen * weights[:, :, None]).sum(1).astype(np.float32)


def normalize_cloud(points, target):
    center = points.mean(0, dtype=np.float64)
    scale = float(np.linalg.norm(points-center, axis=1).max())
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError('Degenerate surface extent')
    return ((points-center)/scale).astype(np.float32), ((target-center)/scale).astype(np.float32), center, scale


def farthest_points(points, count):
    """Order-independent for non-tied distances; deterministic tie handling, no random start."""
    points = np.asarray(points, dtype=np.float64)
    if not 0 < count <= len(points):
        raise ValueError('Invalid FPS count')
    distance = np.full(len(points), np.inf)
    current = int(np.argmax(np.square(points-points.mean(0)).sum(1)))
    output = np.empty(count, dtype=np.int64)
    for i in range(count):
        output[i] = current
        distance = np.minimum(distance, np.square(points-points[current]).sum(1))
        distance[output[:i+1]] = -1
        current = int(distance.argmax())
    return output


def hierarchy(points, centers=(512, 128), neighbors=32):
    output = {}
    for stage, count in enumerate(centers, 1):
        indices = farthest_points(points, count)
        k = min(neighbors, len(points))
        nearby = cKDTree(points).query(points[indices], k=k)[1].reshape(count, k)
        output[f'centers{stage}'] = indices
        output[f'neighbors{stage}'] = nearby
        points = points[indices]
    return output


def batch_hierarchy(points, centers, neighbors, device):
    items = [hierarchy(p, centers, neighbors) for p in points.detach().cpu().numpy()]
    return {key: torch.from_numpy(np.stack([item[key] for item in items])).long().to(device) for key in items[0]}


def gather(values, indices):
    batch = torch.arange(len(values), device=values.device).reshape(-1, *([1]*(indices.ndim-1)))
    return values[batch, indices]


class SetAbstraction(nn.Module):
    def __init__(self, feature_channels, channels):
        super().__init__()
        layers = []; previous = 3+feature_channels
        for width in channels:
            layers += [nn.Linear(previous, width), nn.LayerNorm(width), nn.ReLU()]
            previous = width
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features, centers, neighbors):
        center_xyz = gather(xyz, centers)
        relative = gather(xyz, neighbors)-center_xyz[:, :, None]
        values = relative if features is None else torch.cat([relative, gather(features, neighbors)], dim=-1)
        return center_xyz, self.mlp(values).max(dim=2).values


class PointNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.centers = config['centers']; self.neighbors = config['neighbors']
        self.local1 = SetAbstraction(0, [32, 64, 128])
        self.local2 = SetAbstraction(128, [128, 128, 256])
        self.global_mlp = nn.Sequential(nn.Linear(259, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU())
        self.latent = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, config['embedding_dim']))

    def forward(self, points, groups=None):
        if groups is None:
            groups = batch_hierarchy(points, self.centers, self.neighbors, points.device)
        xyz, features = self.local1(points, None, groups['centers1'], groups['neighbors1'])
        xyz, features = self.local2(xyz, features, groups['centers2'], groups['neighbors2'])
        return self.latent(self.global_mlp(torch.cat([xyz, features], dim=-1)).max(dim=1).values)


class PointCloudAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = PointNetEncoder(config)
        self.decoder = nn.Sequential(nn.Linear(config['embedding_dim'], 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, config['points']*3))

    def forward(self, points, groups=None):
        latent = self.encoder(points, groups)
        return latent, self.decoder(latent).reshape(len(points), self.config['points'], 3)


def chamfer(a, b, chunk=256):
    """Per-object symmetric mean *squared* Euclidean surface distance; bounded blocks."""
    def directed(x, y):
        result = []
        for first in range(0, x.shape[1], chunk):
            block = x[:, first:first+chunk]
            distance = (block.square().sum(-1, keepdim=True) + y.square().sum(-1)[:, None]
                        - 2 * block @ y.transpose(1, 2)).clamp_min(0)
            result.append(distance.min(-1).values)
        return torch.cat(result, dim=1).mean(1)
    return directed(a, b) + directed(b, a)


def rotations(count, rng):
    q = rng.normal(size=(count, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=1).reshape(count, 3, 3).astype(np.float32)


def select_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process; use an unsandboxed run or --device cpu')
    return torch.device(name)


def assert_split(manifest):
    if not manifest.cell_id.is_unique:
        raise ValueError('Duplicate cell IDs')
    if not set(manifest.split) <= {'train', 'validation', 'test'}:
        raise ValueError('Unknown split')
    if (manifest.groupby('sample').split.nunique() > 1).any() or (manifest.groupby('similarity_group').split.nunique() > 1).any():
        raise ValueError('Stack or near-duplicate group crosses splits')
    for split in ['train', 'validation', 'test']:
        if not ((manifest.split == split) & manifest.training_eligible).any():
            raise ValueError('No eligible cells in ' + split)


def load_data(folder):
    import pandas as pd
    metadata=json.loads((folder/'dataset.json').read_text())
    if metadata['manifest_sha256']!=sha256(folder/'manifest.csv'):
        raise ValueError('Prepared manifest fingerprint changed')
    for name,digest in metadata.get('array_sha256',{}).items():
        if sha256(folder/(name+'.npy'))!=digest:raise ValueError('Prepared geometry fingerprint changed: '+name)
    manifest = pd.read_csv(folder/'manifest.csv')
    assert_split(manifest)
    arrays = {name: np.load(folder/(name+'.npy'), mmap_mode='r')
              for name in ['points', 'targets', 'centers1', 'neighbors1', 'centers2', 'neighbors2']}
    if any(len(array) != len(manifest) for array in arrays.values()):
        raise ValueError('Manifest / geometry count mismatch')
    return manifest, arrays


def tensors(arrays, ids, device):
    values = {k: torch.from_numpy(np.array(v[ids], copy=True)).to(device) for k, v in arrays.items()}
    return values['points'].float(), values['targets'].float(), {k: values[k].long() for k in values if k not in {'points', 'targets'}}


def save_json(path, value, compact=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=None if compact else 2, separators=(',', ':') if compact else None, allow_nan=False)); temp.replace(path)
