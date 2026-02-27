from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np
from typing import Optional, Tuple, Union

@dataclass
class PointsToLocalizationHeatmap:
    """
    Gaussian heatmap generator for cell centers.
    The object of this class can be called to generate a heatmap.

    Used for both positive and negative cells.
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
            g = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma ** 2))
            heatmap[0] = torch.maximum(heatmap[0], g)

        return heatmap


@dataclass
class PointsToCountHeatmap:
    '''
    Gaussian heatmap for cells counting task.
    The object of this class can be called to generate a heatmap.

    For each cell discrete heatmap is calculated on a pixel grid using unnormalized kernel.
    Then it's normalized so that sum over all pixels is normalized to be == 1.
    The resulting heatmap is the sum of individual normalized grids

    Used for both positive and negative cells.
    '''
    out_hw: tuple[int, int]
    in_hw: tuple[int, int]
    sigma: float = 2.0
    dtype: torch.dtype = torch.float32

    def  __call__(self, points_xy: torch.Tensor) -> torch.Tensor:
        H, W = self.out_hw

        heatmap = torch.zeros((1, H, W), dtype=self.dtype)

        if points_xy.numel() == 0:
            return heatmap

        inH, inW = self.in_hw

        sx = W / float(inW)   # horizontal scaling factor
        sy = H / float(inH)

        pts = points_xy.clone()
        pts[:, 0] *= sx
        pts[:, 1] *= sy

        yy = torch.arange(H).view(H,1).to(torch.float32)
        xx = torch.arange(W).view(1,W).to(torch.float32)

        for x, y in pts:
            # x, y are center coordinates
            # xx, yy are arrays of grid coordinates
            g = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))
            g /= torch.sum(g)   # normalizing the sum over all grid
            heatmap[0] += g


        return heatmap










