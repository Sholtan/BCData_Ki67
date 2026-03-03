"""
Image‑to‑heatmap transformation utilities for BCData.

This module defines dataclasses for converting lists of point annotations into
Gaussian heatmaps used for localisation and counting heads.  The base code
mirrors the original implementation but exposes a configurable Gaussian
``sigma`` parameter (default increased to 3.0) to broaden peaks and help
models see more context around each cell.

Two dataclasses are provided:

* :class:`PointsToLocalizationHeatmap` – Generates a peak‑normalized
  heatmap by taking the maximum of Gaussians centered at each point.

* :class:`PointsToCountHeatmap` – Generates a sum‑normalized heatmap
  where each Gaussian is normalized to have integral 1, and the final
  heatmap is the sum of these unit‑mass kernels.

The API remains compatible with the original code – both classes can be
called with a tensor of shape ``(N, 2)`` containing XY coordinates in
pixel units of the source image, and they return a heatmap tensor of
shape ``(1, H, W)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class PointsToLocalizationHeatmap:
    """Generate a localisation heatmap by taking the maximum of Gaussians.

    Parameters
    ----------
    out_hw : tuple[int, int]
        Height and width of the output heatmap.
    in_hw : tuple[int, int]
        Height and width of the input image that the points refer to.
    sigma : float, optional
        Standard deviation of the Gaussian kernel (in output pixel units).
        A larger value broadens the peaks and encourages smoother
        supervision.  Default is 3.0.
    clip : bool, optional
        If ``True``, clamps point coordinates to lie within the output grid.
    dtype : torch.dtype, optional
        Data type for the output heatmap.
    """

    out_hw: Tuple[int, int]
    in_hw: Tuple[int, int]
    sigma: float = 3.0
    clip: bool = False
    dtype: torch.dtype = torch.float32

    def __call__(self, points_xy: torch.Tensor) -> torch.Tensor:
        H, W = self.out_hw
        heatmap = torch.zeros((1, H, W), dtype=self.dtype)

        # Return empty heatmap if there are no points
        if points_xy.numel() == 0:
            return heatmap

        pts = points_xy.clone().to(torch.float32)

        # Rescale from input image coordinates to heatmap coordinates
        if self.in_hw is not None:
            inH, inW = self.in_hw
            sx = W / float(inW)
            sy = H / float(inH)
            pts[:, 0] *= sx
            pts[:, 1] *= sy

        # Optionally clamp points to valid range
        if self.clip:
            pts[:, 0] = pts[:, 0].clamp(0, W - 1)
            pts[:, 1] = pts[:, 1].clamp(0, H - 1)

        # Create coordinate grids
        yy = torch.arange(H, dtype=torch.float32).view(H, 1)
        xx = torch.arange(W, dtype=torch.float32).view(1, W)

        # For each point, compute a Gaussian and take the maximum
        for x, y in pts:
            g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * self.sigma ** 2))
            heatmap[0] = torch.maximum(heatmap[0], g)

        return heatmap


@dataclass
class PointsToCountHeatmap:
    """Generate a density heatmap by summing unit‑mass Gaussians.

    Parameters
    ----------
    out_hw : tuple[int, int]
        Height and width of the output heatmap.
    in_hw : tuple[int, int]
        Height and width of the input image that the points refer to.
    sigma : float, optional
        Standard deviation of the Gaussian kernel (in output pixel units).
        Default is 3.0.
    dtype : torch.dtype, optional
        Data type for the output heatmap.
    """

    out_hw: Tuple[int, int]
    in_hw: Tuple[int, int]
    sigma: float = 3.0
    dtype: torch.dtype = torch.float32

    def __call__(self, points_xy: torch.Tensor) -> torch.Tensor:
        H, W = self.out_hw
        heatmap = torch.zeros((1, H, W), dtype=self.dtype)

        # Return empty heatmap if there are no points
        if points_xy.numel() == 0:
            return heatmap

        pts = points_xy.clone().to(torch.float32)
        inH, inW = self.in_hw

        # Rescale points from input image coordinates to heatmap coordinates
        sx = W / float(inW)
        sy = H / float(inH)
        pts[:, 0] *= sx
        pts[:, 1] *= sy

        yy = torch.arange(H, dtype=torch.float32).view(H, 1)
        xx = torch.arange(W, dtype=torch.float32).view(1, W)

        # For each point compute a unit‑mass Gaussian and add it to the map
        for x, y in pts:
            g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * self.sigma ** 2))
            g = g / (g.sum() + 1e-12)  # Normalise so that the integral equals 1
            heatmap[0] += g

        return heatmap