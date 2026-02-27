import torch
import torch.nn.functional as F
import numpy as np

def heatmaps_to_points_batch(
    heatmaps,
    kernel_size,
    threshold,
    output_hw=(640, 640),
    merge_radius=1.5,
    refine=True,
):
    """
    Recover point centers from batched 2-channel heatmaps.

    Parameters
    ----------
    heatmaps : torch.Tensor
        Tensor with shape ``(B, 2, H, W)`` and values in ``[0, 1]``.
    kernel_size : int
        Odd kernel for local-maxima suppression.
    threshold : float
        Peak threshold in heatmap space.
    output_hw : tuple[int, int] | None, default=(640, 640)
        If not ``None``, rescales points from heatmap coordinates ``(W, H)``
        to output-image coordinates ``(output_w, output_h)``.
    merge_radius : float, default=1.5
        Merge peaks closer than this radius (in heatmap pixels).
    refine : bool, default=True
        Use weighted local centroid refinement around each detected peak.
    """
    if heatmaps.ndim != 4 or heatmaps.shape[1] != 2:
        raise ValueError(f"Expected heatmaps with shape (B,2,H,W), got {tuple(heatmaps.shape)}")
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be an odd positive integer.")

    out_pos = []
    out_neg = []

    hm_h, hm_w = int(heatmaps.shape[-2]), int(heatmaps.shape[-1])

    for b in range(heatmaps.shape[0]):
        pos_pts_hm, neg_pts_hm = heatmap_to_points(
            heatmaps[b], kernel_size=kernel_size, threshold=threshold,
            merge_radius=merge_radius, refine=refine
        )

        if output_hw is not None:
            pos_pts = _rescale_points(pos_pts_hm, src_hw=(hm_h, hm_w), dst_hw=output_hw)
            neg_pts = _rescale_points(neg_pts_hm, src_hw=(hm_h, hm_w), dst_hw=output_hw)
        else:
            pos_pts = pos_pts_hm
            neg_pts = neg_pts_hm

        out_pos.append(pos_pts)
        out_neg.append(neg_pts)

    return out_pos, out_neg


def heatmap_to_points(heatmap2d, kernel_size, threshold, merge_radius=1.5, refine=True):
    """
    Recover points from a single sample heatmap of shape ``(2, H, W)``.
    Returns ``(pos_pts, neg_pts)`` as tensors of shape ``(N,2)`` in ``(x,y)`` order.
    """
    if heatmap2d.ndim != 3 or heatmap2d.shape[0] != 2:
        raise ValueError(f"Expected heatmap2d with shape (2,H,W), got {tuple(heatmap2d.shape)}")

    pos_heatmap = heatmap2d[0]
    neg_heatmap = heatmap2d[1]

    pos_pts = _channel_to_points(
        pos_heatmap, kernel_size=kernel_size, threshold=threshold,
        merge_radius=merge_radius, refine=refine
    )
    neg_pts = _channel_to_points(
        neg_heatmap, kernel_size=kernel_size, threshold=threshold,
        merge_radius=merge_radius, refine=refine
    )

    return pos_pts, neg_pts


def _channel_to_points(hm, kernel_size, threshold, merge_radius, refine):
    mask = _local_maxima(hm, kernel_size) & (hm >= threshold)
    ys, xs = torch.where(mask)  # IMPORTANT: torch.where returns (y, x)

    if xs.numel() == 0:
        return torch.zeros((0, 2), device=hm.device, dtype=torch.float32)

    scores = hm[ys, xs]
    pts = torch.stack((xs, ys), dim=1).to(torch.float32)  # (N,2) in (x,y)

    if merge_radius is not None and merge_radius > 0:
        pts, scores = _merge_close_points(pts, scores, merge_radius=merge_radius)

    if refine and pts.numel() > 0:
        pts = _refine_points(hm, pts)

    return pts


def _local_maxima(hm, kernel_size):
    """
    hm: [H, W]
    """
    pad = kernel_size // 2
    h = hm.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    pooled = F.max_pool2d(h, kernel_size=kernel_size, stride=1, padding=pad)
    return pooled[0, 0] == hm


@torch.no_grad()
def _merge_close_points(pts, scores, merge_radius=1.5):
    if pts.numel() == 0:
        return pts, scores

    order = scores.argsort(descending=True)
    pts = pts[order]
    scores = scores[order]

    keep_pts = []
    keep_scores = []
    suppressed = torch.zeros(len(pts), dtype=torch.bool, device=pts.device)
    r2 = merge_radius * merge_radius

    for i in range(len(pts)):
        if suppressed[i]:
            continue
        keep_pts.append(pts[i])
        keep_scores.append(scores[i])
        dx = pts[:, 0] - pts[i, 0]
        dy = pts[:, 1] - pts[i, 1]
        suppressed |= (dx * dx + dy * dy) <= r2

    return torch.stack(keep_pts, dim=0), torch.stack(keep_scores, dim=0)


@torch.no_grad()
def _refine_points(hm, pts, beta=4.0, radius=3, eps=1e-8):
    refined = torch.empty_like(pts)
    H, W = hm.shape

    for i, p in enumerate(pts):
        x0 = int(round(float(p[0].item())))
        y0 = int(round(float(p[1].item())))

        x1 = max(0, x0 - radius)
        x2 = min(W, x0 + radius + 1)
        y1 = max(0, y0 - radius)
        y2 = min(H, y0 + radius + 1)

        patch = hm[y1:y2, x1:x2].clamp_min(0.0)
        w = patch.pow(beta)
        s = w.sum()
        if s.item() < eps:
            refined[i, 0] = float(x0)
            refined[i, 1] = float(y0)
            continue

        ys = torch.arange(y1, y2, device=hm.device, dtype=torch.float32).view(-1, 1)
        xs = torch.arange(x1, x2, device=hm.device, dtype=torch.float32).view(1, -1)
        refined[i, 0] = (w * xs).sum() / s
        refined[i, 1] = (w * ys).sum() / s

    return refined


def _rescale_points(pts, src_hw, dst_hw):
    if pts.numel() == 0:
        return pts

    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)

    out = pts.clone()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out




