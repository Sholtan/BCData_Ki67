from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np
from typing import Optional, Tuple, Union

@dataclass
class PointsToGaussianHeatmap:
    """
    Gaussian heatmap generator for cell centers.
    The object of this class can be called to generate a heatmap.
    """
    out_hw: tuple[int, int]   # (H, W) of heatmap, e.g. (160, 160)
    in_hw: tuple[int, int]
    sigma: float = 2.0
    clip: bool = False
    dtype: torch.dtype = torch.float32

    def __call__(self, points_xy: torch.Tensor) -> torch.Tensor:
        H, W = self.out_hw
        heatmap = torch.zeros((1, H, W), dtype=self.dtype)

        if points_xy.numel() == 0:
            return heatmap

        if self.in_hw is not None:
            inH, inW = self.in_hw
            sx = W / float(inW)
            sy = H / float(inH)
            pts = points_xy.clone()
            pts[:, 0] *= sx
            pts[:, 1] *= sy

        # optionally clamp
        if self.clip:
            pts[:, 0] = pts[:, 0].clamp(0, W - 1)
            pts[:, 1] = pts[:, 1].clamp(0, H - 1)

        yy = torch.arange(H).view(H,1)
        xx = torch.arange(W).view(1,W)
        xx = xx.to(torch.float32)
        yy = yy.to(torch.float32)

        for x, y in pts:
            #x = int(torch.round(x).item())     # shouldn't round x and y
            #y = int(torch.round(y).item())
            g = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma ** 2))
            heatmap[0] = torch.maximum(heatmap[0], g)

        return heatmap


@dataclass
class PointsToGaussianHeatmapFast:
    """
    Fast Gaussian heatmap generator using windowed/truncated Gaussians.

    - Draws each point only in a local (2r+1)x(2r+1) window (r ~ truncate_sigma * sigma)
    - Combines multiple points via max (standard for center heatmaps)
    - Optionally rescales points from input image coordinates to heatmap coordinates

    Output shape: (1, H, W)
    Points expected: (N, 2) in (x, y)
    """
    out_hw: Tuple[int, int]                   # (H_out, W_out) e.g. (160, 160)
    sigma: float = 2.0
    truncate_sigma: float = 3.0               # radius = truncate_sigma * sigma
    in_hw: Optional[Tuple[int, int]] = (640, 640)  # (H_in, W_in) for scaling; None => already in out scale
    clip: bool = True
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None

    def __call__(self, points_xy: torch.Tensor) -> torch.Tensor:
        H, W = self.out_hw
        heatmap = torch.zeros((1, H, W), dtype=self.dtype, device=self.device)

        # Convert points to torch float32
        if isinstance(points_xy, np.ndarray):
            pts = torch.from_numpy(points_xy)
        else:
            pts = points_xy

        if pts.numel() == 0:
            return heatmap

        pts = pts.to(dtype=torch.float32, device=self.device)

        # Scale points from input image coords to heatmap coords if requested
        if self.in_hw is not None:
            inH, inW = self.in_hw
            sx = W / float(inW)
            sy = H / float(inH)
            pts = pts.clone()
            pts[:, 0] *= sx
            pts[:, 1] *= sy

        if self.clip:
            pts = pts.clone()
            pts[:, 0] = pts[:, 0].clamp(0, W - 1)
            pts[:, 1] = pts[:, 1].clamp(0, H - 1)

        # Truncation radius in pixels
        r = int(round(self.truncate_sigma * self.sigma))
        if r <= 0:
            # Degenerate: just set nearest pixel to 1
            xi = pts[:, 0].round().long().clamp(0, W - 1)
            yi = pts[:, 1].round().long().clamp(0, H - 1)
            heatmap[0, yi, xi] = 1.0
            return heatmap

        # Precompute a (2r+1)x(2r+1) Gaussian patch centered at (0,0)
        # Patch coordinate system: dx,dy in [-r, r]
        dx = torch.arange(-r, r + 1, device=self.device, dtype=torch.float32)
        dy = torch.arange(-r, r + 1, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(dy, dx, indexing="ij")
        g_patch = torch.exp(-(xx * xx + yy * yy) / (2.0 * self.sigma * self.sigma)).to(self.dtype)

        for x, y in pts:
            cx = int(torch.round(x).item())
            cy = int(torch.round(y).item())


            # Compute patch bounds in heatmap coordinates
            x0 = max(cx - r, 0)
            y0 = max(cy - r, 0)
            x1 = min(cx + r + 1, W)
            y1 = min(cy + r + 1, H)

            # Compute corresponding bounds in patch coordinates
            px0 = x0 - (cx - r)
            py0 = y0 - (cy - r)
            px1 = px0 + (x1 - x0)
            py1 = py0 + (y1 - y0)

            # Max-composite
            heatmap[0, y0:y1, x0:x1] = torch.maximum(
                heatmap[0, y0:y1, x0:x1],
                g_patch[py0:py1, px0:px1],
            )

        return heatmap