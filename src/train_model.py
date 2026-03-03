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




import cv2
#cv2.setNumThreads(0)

from tqdm import tqdm
import numpy as np

from datasets.bcdata import BCDataDataset, collate_heatmap_points
from datasets.transforms import PointsToLocalizationHeatmap, PointsToCountHeatmap


from visualization import overlay_heatmap
from training import train


from models.models import HybridModel
from models.losses import weighted_sigmoid_mse_from_logits, softplus_mse_from_logits, l1_count_from_density_logits

from src.debug import print_info

from src.weights_init import init_count_head_bias


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


data_root = Path(cfg["h200_paths"]["data_root"])
checkpoint_dir = Path(cfg["h200_paths"]["checkpoint_dir"])

print(f"data_root: {data_root}")
print(f"checkpoint_dir: {checkpoint_dir}")

loc_heatmap_generator = PointsToLocalizationHeatmap(out_hw=(160,160), in_hw=(640,640), sigma=2.0)
count_heatmap_generator = PointsToCountHeatmap(out_hw=(160,160), in_hw=(640,640), sigma=2.0)

dataset = BCDataDataset(root = data_root,
                        split="train",
                        target_loc_transform = loc_heatmap_generator,
                        target_count_transform = count_heatmap_generator)



train_loader = DataLoader(
    dataset,
    batch_size=16,        # choose based on GPU memory (640×640 images are large)
    shuffle=True,
    num_workers=4,       # use 0 if debugging
    pin_memory=True,     # recommended when using GPU
    drop_last=True,       # optional, useful for BatchNorm
    collate_fn = collate_heatmap_points
)



model = HybridModel()
device = 'cuda'
model.to(device);
C_pos, C_neg = 50.0, 50.0
H = W = 160

init_count_head_bias(model.count_pos_head, C_pos, H, W, zero_last_weights=True)
init_count_head_bias(model.count_neg_head, C_neg, H, W, zero_last_weights=True)

optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2
        )

m_max = 0.174835
m_dens = 5.013973e-05
m_count = 59.171053

eps = 1e-8
lambda_dens = m_max / (m_dens + eps)
lambda_count = m_max / (m_count + eps)


Lmax_list = []
Ldens_list = []
Lcount_list = []


model.train()

n_epochs = 100
for epoch in range(n_epochs):
    print(f"\nepoch: {epoch}", end=', ')

    for img, loc_heatmap, count_heatmap, pos_pts, neg_pts in train_loader:
        optimizer.zero_grad()

        img = img.to(device)
        loc_heatmap = loc_heatmap.to(device)
        count_heatmap = count_heatmap.to(device)


        pred_loc_hm, pred_den_hm, pred_count = model(img)
        pred_den_hm = pred_den_hm.to(device)

        Lmax = weighted_sigmoid_mse_from_logits(pred_logits = pred_loc_hm, target = loc_heatmap)
        Ldens = softplus_mse_from_logits(pred_logits = pred_den_hm, target = count_heatmap)

        gtN = [(len(tmp1), len(tmp2)) for tmp1, tmp2 in zip(pos_pts, neg_pts)]  # (B, 2)

        gtN = torch.tensor(gtN)
        gtN = gtN.to(device)

        Lcount = l1_count_from_density_logits(pred_logits = pred_den_hm, gtN = gtN)

        loss = Lmax + lambda_dens * Ldens + lambda_count * Lcount


        Lmax_list.append(Lmax.item())
        Ldens_list.append(Ldens.item())
        Lcount_list.append(Lcount.item())

        loss.backward()
        optimizer.step()


    with torch.no_grad():
        print(f"pred_count: \n{pred_count}\n\n\n")

version = '_04'

torch.save(model.state_dict(), "./checkpoints/hybrid" + version + ".pt")

with open("loss_log/Lmax_list" + version + ".txt", "w", encoding="utf-8") as f:
    for x in Lmax_list:
        f.write(f"{x}\n")


with open("loss_log/Ldens_list" + version + ".txt", "w", encoding="utf-8") as f:
    for x in Ldens_list:
        f.write(f"{x}\n")


with open("loss_log/Lcount_list" + version + ".txt", "w", encoding="utf-8") as f:
    for x in Lcount_list:
        f.write(f"{x}\n")


