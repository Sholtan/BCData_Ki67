import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.bcdata import BCDataDataset, collate_heatmap_points
from datasets.transforms import PointsToLocalizationHeatmap
from visualization import overlay_heatmap
from training import train
from models import NucleusLocalizationModel, heatmap_weighted_mse_loss
from utils.debug import print_info
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


data_root = Path(cfg["h200_paths"]["data_root"])
checkpoint_dir = Path(cfg["h200_paths"]["checkpoint_dir"])

print(f"data_root: {data_root}")
print(f"checkpoint_dir: {checkpoint_dir}")

heatmap_generator = PointsToLocalizationHeatmap(out_hw=(160,160), in_hw=(640,640), sigma=2.0)



dataset = BCDataDataset(root = data_root,
                        split="train",
                        target_transform = heatmap_generator)



train_loader = DataLoader(
    dataset,
    batch_size=24,        # choose based on GPU memory (640×640 images are large)
    shuffle=True,
    num_workers=0,       # use 0 if debugging
    pin_memory=True,     # recommended when using GPU
    drop_last=True,       # optional, useful for BatchNorm
    collate_fn = collate_heatmap_points
)



model = NucleusLocalizationModel()
device = 'cuda'


def count_metrics(preds, heatmaps, pos_points, neg_points):
    pass
losses = train(model=model, num_epochs=3, train_loader=train_loader, val_loader=None, loss_function=heatmap_weighted_mse_loss, count_metrics=count_metrics, checkpoint_dir=checkpoint_dir, forplot_img=None)
