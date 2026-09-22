"""IConE's three unweighted objectives for persistent instance anchors."""
import torch
from torch.nn import functional as F


def losses(z1, z2, instance_anchors, all_anchor_parameters):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    anchors = F.normalize(instance_anchors, dim=1)
    all_anchors = F.normalize(all_anchor_parameters, dim=1)
    view_view = (1 - (z1 * z2).sum(1)).mean()
    view_instance = .5 * ((1 - (z1 * anchors).sum(1)).mean() +
                          (1 - (z2 * anchors).sum(1)).mean())
    gram = all_anchors @ all_anchors.T
    off_diagonal = ~torch.eye(len(all_anchors), dtype=torch.bool,
                              device=all_anchors.device)
    diversity = F.relu(gram[off_diagonal]).square().mean()
    return view_instance + view_view + diversity, {
        "view_instance": view_instance,
        "view_view": view_view,
        "diversity": diversity,
    }
