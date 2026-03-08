

import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)







from datasets.bcdata import BCDataDataset, collate_heatmap_points
from datasets.augment import RandomRotationWithPoints
from datasets.transforms import PointsToLocalizationHeatmap, PointsToCountHeatmap
from models.models import HybridModel
from models.losses import (
    weighted_sigmoid_mse_from_logits,
    weighted_softplus_mse_density_from_logits,
    weighted_l1_count_from_density_logits,
)
from src.weights_init import init_count_head_bias
from src.utils import create_next_experiment_dir, print_info







# -----------------------------------------------------------------------------
# Configuration constants
# Adjust these values to tune the training behaviour.
# -----------------------------------------------------------------------------
# Standard deviation for Gaussian kernels used in target generation
SIGMA = 2.0

# Loss weighting factors for density and count losses.  Larger values
# emphasise the corresponding loss relative to the localisation loss.
# Base lambda values derived from the original experiment
M_MAX = 0.174835
M_DENS = 5.013973e-05
M_COUNT = 59.171053
EPS = 1e-8
LAMBDA_DENS_BASE = M_MAX / (M_DENS + EPS)
LAMBDA_COUNT_BASE = M_MAX / (M_COUNT + EPS)

# Additional weighting applied to the negative channel for density and count
# losses.  Values greater than 1 increase the emphasis on negatives.
NEG_DENS_WEIGHT = 2.0
NEG_COUNT_WEIGHT = 2.0

# Calibration factor applied to the predicted negative density before
# computing the loss.  Empirically compensates for systematic undercounting.
NEG_DENSITY_SCALE = 1.2

# Initial expected count per class used to initialise count head biases.  If
# your dataset has a different average, you should update these values.
EXPECTED_POS_COUNT = 50.0
EXPECTED_NEG_COUNT = 50.0


def main() -> None:
    # -------------------------------------------------------------------------
    # Load configuration and set up paths
    # -------------------------------------------------------------------------
    with open(Path(__file__).resolve().parent.parent / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Use the h200_paths entry by default; adjust as needed
    data_root = Path(cfg["h200_paths"]["data_root"])
    checkpoint_root = Path(cfg["h200_paths"]["checkpoint_dir"])

    # Create a new experiment directory under checkpoint_root and write everything there
    exp_dir = create_next_experiment_dir(checkpoint_root)

    print(f"Data root: {data_root}")
    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Experiment directory: {exp_dir}")

    # -------------------------------------------------------------------------
    # Data transformations and dataset
    # -------------------------------------------------------------------------
    transform = T.Compose([
        T.ToPILImage(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    loc_heatmap_generator = PointsToLocalizationHeatmap(out_hw=(160, 160), in_hw=(640, 640), sigma=SIGMA)
    count_heatmap_generator = PointsToCountHeatmap(out_hw=(160, 160), in_hw=(640, 640), sigma=SIGMA)

    joint_transform = RandomRotationWithPoints(degrees=(0.0, 360.0))
    dataset = BCDataDataset(
        root=data_root,
        split="train",
        target_loc_transform=loc_heatmap_generator,
        target_count_transform=count_heatmap_generator,
        transform=transform,
        joint_transform=joint_transform,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_heatmap_points,
    )

    # -------------------------------------------------------------------------
    # Model, optimiser and initialisation
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel().to(device)

    H = W = 160
    init_count_head_bias(model.count_pos_head, EXPECTED_POS_COUNT, H, W, zero_last_weights=True)
    init_count_head_bias(model.count_neg_head, EXPECTED_NEG_COUNT, H, W, zero_last_weights=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2,
    )

    # Lists for logging losses
    Lmax_list: list[float] = []
    Ldens_list: list[float] = []
    Lcount_list: list[float] = []

    model.train()
    n_epochs = 100
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        for img, loc_heatmap, count_heatmap, pos_pts, neg_pts in tqdm(train_loader):
            optimizer.zero_grad()

            img = img.to(device, non_blocking=True)
            loc_heatmap = loc_heatmap.to(device, non_blocking=True)
            count_heatmap = count_heatmap.to(device, non_blocking=True)

            # Forward pass
            pred_loc_logits, pred_den_logits, _ = model(img)

            # Compute localisation loss using weighted sigmoid MSE
            Lmax = weighted_sigmoid_mse_from_logits(pred_loc_logits, loc_heatmap)

            # Density loss
            Ldens = weighted_softplus_mse_density_from_logits(
                pred_den_logits,
                count_heatmap,
                pos_weight=1.0,
                neg_weight=NEG_DENS_WEIGHT,
                alpha_pos=100.0,
                alpha_neg=100.0,
                neg_scale=NEG_DENSITY_SCALE,
            )
            Ldens = LAMBDA_DENS_BASE * Ldens

            # Ground truth counts per channel (constructed on the fly)
            gt_counts = torch.tensor(
                [(len(pp), len(nn)) for pp, nn in zip(pos_pts, neg_pts)],
                dtype=torch.float32,
                device=device,
            )

            # Count loss
            Lcount = weighted_l1_count_from_density_logits(
                pred_den_logits,
                gt_counts,
                pos_weight=1.0,
                neg_weight=NEG_COUNT_WEIGHT,
                neg_scale=NEG_DENSITY_SCALE,
            )
            Lcount = LAMBDA_COUNT_BASE * Lcount

            # Total loss
            loss = Lmax + Ldens + Lcount

            loss.backward()
            optimizer.step()

            # Logging
            Lmax_list.append(Lmax.item())
            Ldens_list.append(Ldens.item())
            Lcount_list.append(Lcount.item())

        # Print intermediate statistics at the end of each epoch
        with torch.no_grad():
            print(
                f"Lmax mean: {sum(Lmax_list[-len(train_loader):]) / len(train_loader):.4f}, "
                f"Ldens mean: {sum(Ldens_list[-len(train_loader):]) / len(train_loader):.4f}, "
                f"Lcount mean: {sum(Lcount_list[-len(train_loader):]) / len(train_loader):.4f}"
            )

        # Save checkpoint after each epoch (into exp_dir)
        if epoch % 20 == 19:
            ckpt_path = exp_dir / f"hybrid_mod_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt_path)
    
    ckpt_path = exp_dir / f"hybrid_mod_epoch_latest.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Save final loss logs (into exp_dir/loss_logs)
    loss_dir = exp_dir / "loss_logs"
    loss_dir.mkdir(parents=True, exist_ok=True)
    with open(loss_dir / "Lmax_list.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{x}\n" for x in Lmax_list)
    with open(loss_dir / "Ldens_list.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{x}\n" for x in Ldens_list)
    with open(loss_dir / "Lcount_list.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{x}\n" for x in Lcount_list)


if __name__ == "__main__":
    main()