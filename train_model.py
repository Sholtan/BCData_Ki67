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


data_root = Path(cfg["paths"]["data_root"])
print(data_root)