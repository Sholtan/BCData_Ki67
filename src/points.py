import torch
import torch.nn.functional as F


@torch.no_grad()
def merge_close_points(
    pts: torch.Tensor,          # (N,2) long, (x,y)
    scores: torch.Tensor,       # (N,)
    merge_radius: float = 1.5,  # in heatmap pixels
):
    if pts.numel() == 0:
        return pts, scores

    # sort by score desc
    order = scores.argsort(descending=True)
    pts = pts[order]
    scores = scores[order]

    kept_pts = []
    kept_scores = []
    suppressed = torch.zeros(len(pts), dtype=torch.bool, device=pts.device)
    r2 = merge_radius * merge_radius

    for i in range(len(pts)):
        if suppressed[i]:
            continue
        kept_pts.append(pts[i])
        kept_scores.append(scores[i])

        dx = (pts[:, 0] - pts[i, 0]).float()
        dy = (pts[:, 1] - pts[i, 1]).float()
        close = (dx * dx + dy * dy) <= r2
        suppressed |= close

    return torch.stack(kept_pts, dim=0), torch.stack(kept_scores, dim=0)


@torch.no_grad()
def heatmap_to_points(
    heatmap: torch.Tensor,
    thr: float = 0.3,
    nms_kernel: int = 3,
    topk: int | None = None,
    return_scores: bool = False,
    merge_radius: float = 1.5,   # <-- add this
):
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be (H,W), got shape {tuple(heatmap.shape)}")
    if nms_kernel % 2 == 0 or nms_kernel < 1:
        raise ValueError("nms_kernel must be an odd positive integer (e.g., 3, 5, 7).")

    H, W = heatmap.shape
    k = nms_kernel
    pad = k // 2

    h = heatmap.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    pooled = F.max_pool2d(h, kernel_size=k, stride=1, padding=pad)
    is_peak = (h == pooled) & (h >= thr)

    ys, xs = torch.where(is_peak[0, 0])
    if xs.numel() == 0:
        pts = torch.zeros((0, 2), dtype=torch.long, device=heatmap.device)
        if return_scores:
            scores = torch.zeros((0,), dtype=heatmap.dtype, device=heatmap.device)
            return pts, scores
        return pts

    scores = heatmap[ys, xs]
    pts = torch.stack([xs, ys], dim=1).to(torch.long)  # (N,2) (x,y)

    # --- HERE: merge plateau/tied peaks ---
    if merge_radius is not None and merge_radius > 0:
        pts, scores = merge_close_points(pts, scores, merge_radius=merge_radius)

    # --- THEN: optional topk ---
    if topk is not None and scores.numel() > topk:
        idx = torch.topk(scores, k=topk, largest=True).indices
        pts, scores = pts[idx], scores[idx]

    return pts


@torch.no_grad()
def refine_points_gaussian_centers(
    heatmap: torch.Tensor,      # (H,W), after sigmoid, in [0,1]
    pts: torch.Tensor,          # (N,2) long, (x,y)
    sigma: float,               # sigma in heatmap pixels
    radius: int | None = None,  # if None, uses ceil(3*sigma)
    beta: float = 4.0,          # sharpening; larger -> closer to argmax
    eps: float = 1e-8,

):
    if pts.numel() == 0:
        return pts.to(torch.float32)

    #print(heatmap.shape)
    H, W = heatmap.shape
    r = int(torch.ceil(torch.tensor(3.0 * sigma)).item()) if radius is None else int(radius)
    r = max(r, 1)

    refined = torch.empty((pts.shape[0], 2), device=heatmap.device, dtype=torch.float32)

    for i, (x0, y0) in enumerate(pts.tolist()):
        x0 = int(x0); y0 = int(y0)

        x1 = max(0, x0 - r); x2 = min(W, x0 + r + 1)
        y1 = max(0, y0 - r); y2 = min(H, y0 + r + 1)

        patch = heatmap[y1:y2, x1:x2].clamp_min(0.0)

        # weights (sharpen to reduce influence of tails/background)
        w = patch.pow(beta)

        s = w.sum()
        if s.item() < eps:
            refined[i] = torch.tensor([x0, y0], device=heatmap.device, dtype=torch.float32)
            continue

        ys = torch.arange(y1, y2, device=heatmap.device, dtype=torch.float32).view(-1, 1)
        xs = torch.arange(x1, x2, device=heatmap.device, dtype=torch.float32).view(1, -1)

        x_hat = (w * xs).sum() / s
        y_hat = (w * ys).sum() / s

        refined[i, 0] = x_hat
        refined[i, 1] = y_hat

    return refined  # (N,2) float, (x,y)

@torch.no_grad()
def heatmaps_to_points_batch(
    heatmaps: torch.Tensor,   # (B,1,H,W) or (B,H,W)
    sigma: float,
    thr: float = 0.3,
    nms_kernel: int = 3,
    topk: int | None = None,
    merge_radius: float = 1.5,
    refine_beta: float = 4.0,
):
    if heatmaps.ndim == 4:
        # (B,1,H,W) -> (B,H,W)
        heatmaps2d = heatmaps[:, 0]
    elif heatmaps.ndim == 3:
        heatmaps2d = heatmaps
    else:
        raise ValueError(f"Expected (B,1,H,W) or (B,H,W), got {tuple(heatmaps.shape)}")

    out: list[torch.Tensor] = []
    for b in range(heatmaps2d.shape[0]):
        hm = heatmaps2d[b]  # (H,W)

        pts = heatmap_to_points(
            hm,
            thr=thr,
            nms_kernel=nms_kernel,
            topk=topk,
            return_scores=False,
            merge_radius=merge_radius,
        )  # (N,2) long (x,y)

        refined = refine_points_gaussian_centers(
            hm,
            pts,
            sigma=sigma,
            beta=refine_beta,
        )  # (N,2) float (x,y)

        out.append(refined)

    return out  # length B, each is (Ni,2)