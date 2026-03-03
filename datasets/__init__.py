"""Datasets package for the modified BCData_Ki67 project.

This package exposes the dataset classes and collation utilities used
throughout the training pipeline.  It mirrors the structure of the
original repository but lives under the ``BCData_Ki67_mod`` tree so
that the modified code is self‑contained.

``BCDataDataset`` loads images and cell annotations from the on‑disk
format and produces both localization and density heatmaps.  See
``bcdata.py`` for details.
"""

from .bcdata import BCDataDataset, collate_heatmap_points  # noqa: F401