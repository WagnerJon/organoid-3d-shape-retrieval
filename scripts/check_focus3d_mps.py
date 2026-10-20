"""Check Apple GPU operations needed by FOCUS-3D, without loading weights."""
import json
import torch
from torch.nn import functional as F

torch.manual_seed(0)
if not torch.backends.mps.is_available():
    raise RuntimeError('Apple MPS device is unavailable in this execution environment.')
tests = []
x = torch.randn(1,4,8,12,12)
w = torch.randn(8,4,3,3,3)
grid = torch.rand(1,4,7,1,3)*2-1
for name, operation in [
    ('Conv3d',lambda device:F.conv3d(x.to(device),w.to(device),padding=1)),
    ('3D trilinear grid sampling',lambda device:F.grid_sample(x.to(device),grid.to(device),align_corners=False)),
    ('3D trilinear interpolation',lambda device:F.interpolate(x.to(device),size=(12,20,20),mode='trilinear',align_corners=False)),
    ('GroupNorm',lambda device:F.group_norm(x.to(device),2)),
]:
    expected = operation('cpu')
    actual = operation('mps').cpu()
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
    tests.append(name)
print(json.dumps(dict(torch=torch.__version__,device='mps',passed=tests),indent=2))
