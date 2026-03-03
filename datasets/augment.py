"""
Augmentation utilities for BCData.

This module defines joint transformations that operate on both the image
and the point annotations.  The most important transformation provided
here is :class:`RandomRotationWithPoints`, which rotates the image by
a random angle in the given range and adjusts the annotation
coordinates accordingly.  The rotation is centred on the middle of
the image, and coordinates outside the image bounds are not clipped
(they may wrap around when the heatmaps are generated).

Example
-------

>>> from datasets.augment import RandomRotationWithPoints
>>> augment = RandomRotationWithPoints(degrees=(0, 360))
>>> img_rot, pos_rot, neg_rot = augment(img, pos_pts, neg_pts)

"""

from __future__ import annotations

import math
import random
from typing import Tuple

import torch
from torchvision.transforms import functional as F


class RandomRotationWithPoints:
    """Rotate the image and associated point annotations by a random angle.

    Parameters
    ----------
    degrees : tuple[float, float], optional
        The lower and upper bounds of the rotation angle in degrees.
        The angle is sampled uniformly from this range.  Default is
        ``(0.0, 360.0)``.
    resample : int, optional
        Resampling filter to use for the image rotation.  See
        ``torchvision.transforms.functional.rotate`` for details.  The
        default uses bilinear interpolation.
    expand : bool, optional
        Whether the image is expanded to fit the whole rotated image.
        Setting this to ``False`` (default) keeps the output size the
        same as the input, cropping or wrapping as necessary.  A
        constant fill value of 0 is used for areas outside the original
        image.
    """

    def __init__(
        self,
        degrees: Tuple[float, float] = (0.0, 360.0),
        resample: int = 0,
        expand: bool = False,
    ) -> None:
        self.degrees = degrees
        self.resample = resample
        self.expand = expand

    def __call__(
        self,
        img: torch.Tensor,
        pos_pts: torch.Tensor,
        neg_pts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the rotation to an image and two sets of points.

        Parameters
        ----------
        img : torch.Tensor
            Image tensor of shape ``(C, H, W)``.
        pos_pts : torch.Tensor
            Positive point coordinates of shape ``(N_pos, 2)``.
        neg_pts : torch.Tensor
            Negative point coordinates of shape ``(N_neg, 2)``.

        Returns
        -------
        tuple
            ``(img_rot, pos_pts_rot, neg_pts_rot)`` where the image is
            rotated and the point coordinates are updated accordingly.
        """
        # Sample a random angle in degrees
        angle = random.uniform(self.degrees[0], self.degrees[1])

        # Rotate the image.  ``F.rotate`` expects channel‑first tensors.
        # ``expand=False`` keeps the same output size; fill with 0.
        img_rot = F.rotate(img, angle, interpolation=self.resample, expand=self.expand)

        # Compute rotation matrix around the centre of the image
        _, H, W = img.shape
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        def rotate_points(pts: torch.Tensor) -> torch.Tensor:
            if pts.numel() == 0:
                return pts
            # Shift points to origin
            x = pts[:, 0] - cx
            y = pts[:, 1] - cy
            # Apply rotation (note sign of sin for image coordinate system)
            x_rot = cos_a * x + sin_a * y
            y_rot = -sin_a * x + cos_a * y
            # Shift back
            x_rot = x_rot + cx
            y_rot = y_rot + cy
            return torch.stack([x_rot, y_rot], dim=1)

        pos_pts_rot = rotate_points(pos_pts)
        neg_pts_rot = rotate_points(neg_pts)

        return img_rot, pos_pts_rot, neg_pts_rot