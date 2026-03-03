# import os

# # BLAS / OpenMP
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# # OpenCV
# os.environ["OPENCV_FOR_THREADS_NUM"] = "1"



import yaml
from pathlib import Path
import matplotlib.pyplot as plt


import torch
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T



import cv2
#cv2.setNumThreads(0)

from tqdm import tqdm
import numpy as np

from datasets.bcdata import BCDataDataset, collate_heatmap_points
from datasets.transforms import PointsToLocalizationHeatmap, PointsToCountHeatmap


from visualization import overlay_heatmap


from models.models import HybridModel
from models.losses import weighted_sigmoid_mse_from_logits, softplus_mse_from_logits, l1_count_from_density_logits


from src.weights_init import init_count_head_bias
from src.utils import create_next_experiment_dir, print_info

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)






# -----------------------------------------------------------------------------
# Configuration constants
# Adjust these values to tune the training behaviour.
# -----------------------------------------------------------------------------
# Standard deviation for Gaussian kernels used in target generation
SIGMA = 3.0

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
NEG_DENSITY_SCALE = 1.5

# Initial expected count per class used to initialise count head biases.  If
# your dataset has a different average, you should update these values.
EXPECTED_POS_COUNT = 50.0
EXPECTED_NEG_COUNT = 80.0


def main() -> None:
    # -------------------------------------------------------------------------
    # Load configuration and set up paths
    # -------------------------------------------------------------------------
    with open(Path(__file__).resolve().parent.parent / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["h200_paths"]["data_root"])
    checkpoint_dir = Path(cfg["h200_paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create a new experiment folder and save everything there
    run_dir = create_next_experiment_dir(checkpoint_dir)

    print(f"Data root: {data_root}")
    print(f"Checkpoint root directory: {checkpoint_dir}")
    print(f"Run directory: {run_dir}")

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

    dataset = BCDataDataset(
        root=data_root,
        split="train",
        target_loc_transform=loc_heatmap_generator,
        target_count_transform=count_heatmap_generator,
        transform=transform,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    lambda_dens_pos = LAMBDA_DENS_BASE
    lambda_dens_neg = LAMBDA_DENS_BASE * NEG_DENS_WEIGHT
    lambda_count_pos = LAMBDA_COUNT_BASE
    lambda_count_neg = LAMBDA_COUNT_BASE * NEG_COUNT_WEIGHT

    Lmax_list: list[float] = []
    Ldens_list: list[float] = []
    Lcount_list: list[float] = []

    model.train()
    n_epochs = 100
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        for img, loc_heatmap, count_heatmap, pos_pts, neg_pts in train_loader:
            optimizer.zero_grad()

            img = img.to(device, non_blocking=True)
            loc_heatmap = loc_heatmap.to(device, non_blocking=True)
            count_heatmap = count_heatmap.to(device, non_blocking=True)

            pred_loc_logits, pred_den_logits, _ = model(img)

            Lmax = weighted_sigmoid_mse_from_logits(pred_loc_logits, loc_heatmap)

            dens_pred = F.softplus(pred_den_logits)
            dens_pred[:, 1] = dens_pred[:, 1] * NEG_DENSITY_SCALE

            Ldens_pos = ((dens_pred[:, 0] - count_heatmap[:, 0]) ** 2).mean()
            Ldens_neg = ((dens_pred[:, 1] - count_heatmap[:, 1]) ** 2).mean()
            Ldens = (lambda_dens_pos * Ldens_pos + lambda_dens_neg * Ldens_neg) / (lambda_dens_pos + lambda_dens_neg)

            pred_counts = dens_pred.sum(dim=(2, 3))  # (B, 2)
            gt_counts = torch.tensor([(len(pp), len(nn)) for pp, nn in zip(pos_pts, neg_pts)],
                                     dtype=torch.float32, device=device)

            Lcount_pos = F.l1_loss(pred_counts[:, 0], gt_counts[:, 0])
            Lcount_neg = F.l1_loss(pred_counts[:, 1], gt_counts[:, 1])
            Lcount = (lambda_count_pos * Lcount_pos + lambda_count_neg * Lcount_neg) / (lambda_count_pos + lambda_count_neg)

            loss = Lmax + Ldens + Lcount

            loss.backward()
            optimizer.step()

            Lmax_list.append(Lmax.item())
            Ldens_list.append(Ldens.item())
            Lcount_list.append(Lcount.item())

        with torch.no_grad():
            print(
                f"Lmax mean: {sum(Lmax_list[-len(train_loader):]) / len(train_loader):.4f}, "
                f"Ldens mean: {sum(Ldens_list[-len(train_loader):]) / len(train_loader):.4f}, "
                f"Lcount mean: {sum(Lcount_list[-len(train_loader):]) / len(train_loader):.4f}"
            )

        # Save checkpoint after each epoch
        if epoch % 10 == 0 and epoch != 0:
            ckpt_path = run_dir / f"hybrid_mod_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # Save final model
    ckpt_path = run_dir / "hybrid_mod_latest.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Save final loss logs
    loss_dir = run_dir / "loss_logs"
    loss_dir.mkdir(parents=True, exist_ok=True)

    with open(loss_dir / "Lmax_list.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{x}\n" for x in Lmax_list)
    with open(loss_dir / "Ldens_list.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{x}\n" for x in Ldens_list)
    with open(loss_dir / "Lcount_list.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{x}\n" for x in Lcount_list)

if __name__ == "__main__":
    main()