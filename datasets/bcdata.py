"""
Dataset implementation for BCData dataset.
"""
#from __future__ import annotations
from typing import Callable, List, Tuple, Any
from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import h5py


class BCDataDataset(Dataset):
    """
    Parses images and annotations
    """
    SUPPORTED_SPLITS = {"train", "validation", "test"}

    def __init__(self,
                 root: Path,
                 split: str,
                 target_transform: Callable,
                 transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Must be one of {self.SUPPORTED_SPLITS}")

        self.image_dir = self.root / "images" / str(split)
        pos_ann_dir = self.root / "annotations" / str(split) / "positive"
        neg_ann_dir = self.root / "annotations" / str(split) / "negative"

        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)
        if not pos_ann_dir.exists():
            raise FileNotFoundError(pos_ann_dir)
        if not neg_ann_dir.exists():
            raise FileNotFoundError(pos_ann_dir)

        self.samples: List[Tuple[Path, Path, Path]] = self._build_index()

    def _build_index(self):
        """
        """
        image_paths = sorted(self.image_dir.glob("*.png"))
        if not image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

        samples = []

        for img_path in image_paths:
            stem = img_path.stem

            pos_ann_path = self.root / "annotations" / self.split / "positive" / (stem + ".h5")
            neg_ann_path = self.root / "annotations" / self.split / "negative" / (stem + ".h5")

            if not pos_ann_path.exists() or not neg_ann_path.exists():
                raise FileNotFoundError(f'Missing annotation: "{pos_ann_path}", "{neg_ann_path}"')

            samples.append((img_path, pos_ann_path, neg_ann_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_points(self, ann_path: Path) -> torch.Tensor:
        with h5py.File(ann_path, "r") as f:
            dset = f["coordinates"]
            assert isinstance(dset, h5py.Dataset)
            coords = torch.from_numpy(dset[:]).to(torch.float32)  # shape (N, 2)
        return coords

    def __getitem__(self, idx) -> Any:
        img_path, pos_ann_path, neg_ann_path = self.samples[idx]

        img: np.ndarray | torch.Tensor | None = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        if self.transform:
            img = self.transform(img)

        pos_pts: torch.Tensor = self._load_points(pos_ann_path)
        neg_pts: torch.Tensor = self._load_points(neg_ann_path)


        pos_heatmap = self.target_transform(pos_pts)
        neg_heatmap = self.target_transform(neg_pts)

        heatmaps = torch.cat([pos_heatmap, neg_heatmap], dim = 0)

        return img, heatmaps, pos_pts, neg_pts


def collate_heatmap_points(batch):
    # batch: list of tuples (img, heatmap, pos_pts, neg_pts)
    imgs, heatmaps, pos_points, neg_points = zip(*batch)

    imgs = torch.stack(imgs, dim=0)         # (B,C,H,W)

    heatmaps = torch.stack(heatmaps, dim=0) # (B,2,h,w)

    # points stays ragged: list length B, each is (Ni,2)
    pos_points = list(pos_points)
    neg_points = list(neg_points)

    return imgs, heatmaps, pos_points, neg_points