"""Foreground/boundary coverage and tolerant contour supervision (C)."""
import torch
import torch.nn.functional as F
from ..segmentation_head import get_uncertain_point_coords_with_randomness


def mask_boundary(mask, radius=1):
    """A two-sided morphological band; input is [N,1,H,W]."""
    kernel = 2 * radius + 1
    return (F.max_pool2d(mask, kernel, 1, radius) +
            F.max_pool2d(-mask, kernel, 1, radius)).clamp_min(0)


@torch.no_grad()
def sample_wafer_points(logits, targets, num_points, foreground_fraction=0.25,
                        boundary_fraction=0.25, boundary_radius=2):
    """Keep the SAME point budget, replacing part with GT-covered training points.

    Pixel-center coordinates match nearest GT sampling with align_corners=False.
    Empty support falls back to uniform points. GT is never used at inference.
    """
    n_fg = int(num_points * foreground_fraction)
    n_boundary = int(num_points * boundary_fraction)
    n_uncertain = num_points - n_fg - n_boundary
    uncertain = get_uncertain_point_coords_with_randomness(
        logits, lambda values: -values.abs(), n_uncertain, 3, 0.75)
    height, width = targets.shape[-2:]
    rows = []
    for i in range(targets.shape[0]):
        mask = targets[i:i+1].float()
        band = mask_boundary(mask, boundary_radius)[0, 0] > 0
        parts = [uncertain[i]]
        for support, count in ((mask[0, 0] > 0.5, n_fg), (band, n_boundary)):
            if count == 0:
                continue
            indices = support.nonzero(as_tuple=False)
            if indices.shape[0] == 0:
                coords = torch.rand(count, 2, device=logits.device)
            else:
                chosen = indices[torch.randint(indices.shape[0], (count,), device=indices.device)]
                coords = torch.stack(((chosen[:, 1].float()+0.5)/width,
                                      (chosen[:, 0].float()+0.5)/height), dim=1)
            parts.append(coords)
        rows.append(torch.cat(parts, dim=0))
    return torch.stack(rows)


def tolerant_boundary_loss(logits, targets, num_boxes, tolerance=1):
    """Soft boundary F1 loss at mask resolution, tolerant to annotation offsets.

    Max pooling retains narrow foreground when GT is reduced to mask resolution.
    GT-empty masks contribute zero. BCE/Dice still supervise interior/background.
    """
    if logits.shape[0] == 0:
        return logits.sum() * 0
    target = F.adaptive_max_pool2d(targets.float(), logits.shape[-2:])
    truth = mask_boundary(target)
    prediction = mask_boundary(logits.float().sigmoid())
    kernel = 2 * tolerance + 1
    truth_neighborhood = F.max_pool2d(truth, kernel, 1, tolerance)
    pred_neighborhood = F.max_pool2d(prediction, kernel, 1, tolerance)
    dims = (1, 2, 3)
    precision = (prediction * truth_neighborhood).sum(dims) / prediction.sum(dims).clamp_min(1e-6)
    recall = (truth * pred_neighborhood).sum(dims) / truth.sum(dims).clamp_min(1e-6)
    score = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    valid = truth.sum(dims) > 0
    return ((1 - score) * valid).sum() / num_boxes
