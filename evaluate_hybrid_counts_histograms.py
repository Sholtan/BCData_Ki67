"""
Script to evaluate counting performance of a trained HybridModel on BCData.

This script loads a trained hybrid model and the BCData test split, then
computes per-class counts using two strategies:

  1. Density-based counts: apply a softplus to the density head logits
     and sum over all pixels.
  2. Localization-based counts: apply a sigmoid to the localization head
     logits, extract local maxima via NMS and thresholding, and count
     detected peaks.

It bins each sample by the ground‑truth number of positive/negative cells
(e.g. [0–10], [11–30], [31–100], [101+]) and computes mean absolute
error (MAE), root mean squared error (RMSE), and mean absolute percentage
error (MAPE) for each bin and class. The results for both counting
strategies are printed for comparison.

Adjust the paths in the configuration section at the top to point to
where your data resides and where your model checkpoint is stored.
"""
import math
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.bcdata import BCDataDataset, collate_heatmap_points
from datasets.transforms import PointsToLocalizationHeatmap, PointsToCountHeatmap
from models.models import HybridModel
import yaml

# -----------------------------------------------------------------------------
# Configuration: update these paths according to your environment
# -----------------------------------------------------------------------------
# Location of BCData root directory (should contain ``images`` and ``annotations``)

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

checkpoint_dir = Path(cfg["h200_paths"]["checkpoint_dir"])

DATA_ROOT = Path(cfg["h200_paths"]["data_root"])

# Location of the hybrid model checkpoint to evaluate
CHECKPOINT_PATH = checkpoint_dir / "hybrid_02.pt"


# Split to evaluate ('test' recommended)
SPLIT = 'test'

# Device to run on ('cuda' or 'cpu')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Bins for stratified metrics (inclusive ranges; last bin upper bound may be math.inf)
COUNT_BINS: List[Tuple[int, float]] = [(0, 10), (11, 30), (31, 100), (101, math.inf)]

# Threshold and NMS kernel for localization head peak extraction
LOC_THRESHOLD = 0.5
NMS_KERNEL = 3  # must be odd
MERGE_RADIUS = 1.5  # in heatmap pixels
REFINE = True

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def softplus_count(pred_den_logits: torch.Tensor) -> torch.Tensor:
    """Convert density head logits to nonnegative densities via softplus and
    sum over H×W to get per‑class counts.

    Args:
        pred_den_logits: Tensor of shape (B, 2, H, W) containing raw logits.
    Returns:
        Tensor of shape (B, 2) with predicted counts for each class.
    """
    dens = F.softplus(pred_den_logits)  # (B,2,H,W)
    return dens.sum(dim=(2, 3))  # (B,2)


def heatmaps_to_point_counts(
    loc_logits: torch.Tensor,
    kernel_size: int = NMS_KERNEL,
    threshold: float = LOC_THRESHOLD,
    merge_radius: float = MERGE_RADIUS,
    refine: bool = REFINE,
) -> torch.Tensor:
    """Convert localization logits to predicted point counts per class.

    Applies a sigmoid to the localization head, extracts peaks with NMS and a
    threshold, optionally refines detected peaks, and counts them.

    Args:
        loc_logits: Tensor of shape (B, 2, H, W) with raw localization logits.
        kernel_size: NMS kernel size (odd positive integer).
        threshold: Minimum heatmap value to consider a peak.
        merge_radius: Radius in heatmap pixels within which peaks are merged.
        refine: Whether to refine peak coordinates via weighted centroid.
    Returns:
        Tensor of shape (B, 2) with counts of detected peaks per class.
    """
    from src.points_v2 import heatmaps_to_points_batch  # lazy import to avoid circular deps

    # Apply sigmoid to map logits to [0,1]
    heatmaps = torch.sigmoid(loc_logits)
    # Extract (pos, neg) points for each batch element; output lists length B
    pos_pts_batch, neg_pts_batch = heatmaps_to_points_batch(
        heatmaps=heatmaps,
        kernel_size=kernel_size,
        threshold=threshold,
        output_hw=None,  # counts only; no rescale needed
        merge_radius=merge_radius,
        refine=refine,
    )
    # Convert lists of tensors to counts
    counts = torch.tensor([
        (len(pos_pts), len(neg_pts))
        for pos_pts, neg_pts in zip(pos_pts_batch, neg_pts_batch)
    ], dtype=torch.float32, device=heatmaps.device)
    return counts  # (B,2)


def bin_index(n: int, bins: List[Tuple[int, float]]) -> int:
    """Return the index of the bin to which ``n`` belongs."""
    for idx, (lo, hi) in enumerate(bins):
        if lo <= n <= hi:
            return idx
    return -1


def update_metrics(
    accum: Dict[str, List[float]],
    gt: float,
    pred: float,
) -> None:
    """Update MAE, RMSE, and MAPE accumulators with a single ground truth and prediction."""
    err = pred - gt
    accum['abs'] += [abs(err)]
    accum['sq'] += [err ** 2]
    if gt != 0:
        accum['ape'] += [abs(err / gt)]
        accum['ape_n'] += [1.0]


def finalize_metrics(accum: Dict[str, List[float]]) -> Dict[str, float]:
    """Finalize metrics from accumulators into MAE, RMSE, and MAPE."""
    mae = float(sum(accum['abs']) / max(len(accum['abs']), 1))
    rmse = math.sqrt(sum(accum['sq']) / max(len(accum['sq']), 1))
    if sum(accum['ape_n']) > 0:
        mape = 100.0 * float(sum(accum['ape']) / sum(accum['ape_n']))
    else:
        mape = float('nan')
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def evaluate_counts_by_bins(
    model: HybridModel,
    loader: DataLoader,
    bins: List[Tuple[int, float]] = COUNT_BINS,
    device: str = DEVICE,
    loc_threshold: float = LOC_THRESHOLD,
    kernel_size: int = NMS_KERNEL,
) -> Tuple[Dict[str, Dict[str, List[Dict[str, float]]]], Dict[str, Dict[str, List[List[float]]]]]:
    """
    Evaluate counting metrics per bin for both density‑ and localization‑based counts and
    collect per‑sample errors (predicted minus ground truth) for histogram plotting.

    Args:
        model: Trained HybridModel.
        loader: DataLoader over the test split (yields img, loc_heatmap, count_heatmap, pos_pts, neg_pts).
        bins: List of inclusive (lo, hi) count ranges for binning.
        device: Device to run inference on.
        loc_threshold: Threshold for localization peaks.
        kernel_size: Kernel size for NMS in localization.
    Returns:
        A tuple of two dictionaries:
          1) metrics: same structure as before — keyed by class ('pos', 'neg') and strategy
             ('density', 'localization'), containing per‑bin MAE, RMSE and MAPE plus sample count.
          2) errors: keyed by class and strategy, containing a list of error lists per bin. Each
             inner list holds (predicted_count - true_count) values for that bin. These can be
             used to draw histograms of the raw counting errors.
    """
    # Initialize accumulators: for each class ('pos', 'neg') and counting strategy
    # ('dens', 'loc'), we have per‑bin accumulators for absolute error, squared error, etc.
    # Metrics accumulators
    metrics = {cls: {strategy: [
        {'abs': [], 'sq': [], 'ape': [], 'ape_n': []} for _ in bins
    ] for strategy in ['dens', 'loc']} for cls in ['pos', 'neg']}
    # Error lists: for each class and strategy we maintain a list per bin to store
    # (prediction - ground_truth) differences. These are useful for bias diagnostics
    # and histogram plotting.
    errors = {cls: {strategy: [ [] for _ in bins ] for strategy in ['dens', 'loc']}
              for cls in ['pos', 'neg']}

    model.eval()
    with torch.no_grad():
        for img, _loc_hm, _count_hm, pos_pts, neg_pts in loader:
            img = img.to(device)
            # Forward pass: get raw logits for loc and dens heads
            pred_loc_logits, pred_den_logits, _ = model(img)
            # Move logits to CPU for counting (to avoid GPU memory growth)
            pred_loc_logits_cpu = pred_loc_logits.cpu()
            pred_den_logits_cpu = pred_den_logits.cpu()
            # Ground‑truth counts from pos_pts and neg_pts lists
            batch_gt_counts = [
                (len(p), len(n)) for p, n in zip(pos_pts, neg_pts)
            ]
            gt_tensor = torch.tensor(batch_gt_counts, dtype=torch.float32)
            # Predicted density counts
            dens_counts = softplus_count(pred_den_logits_cpu)
            # Predicted localization counts
            loc_counts = heatmaps_to_point_counts(
                pred_loc_logits_cpu,
                kernel_size=kernel_size,
                threshold=loc_threshold,
                merge_radius=MERGE_RADIUS,
                refine=REFINE,
            )

            # Update metrics per sample
            for i, (gt_pos, gt_neg) in enumerate(gt_tensor):
                # Determine bin indices for pos and neg
                bin_pos = bin_index(int(gt_pos.item()), bins)
                bin_neg = bin_index(int(gt_neg.item()), bins)
                # Density counts
                pred_dens_pos, pred_dens_neg = dens_counts[i]
                # Localization counts
                pred_loc_pos, pred_loc_neg = loc_counts[i]

                # Update positive class metrics
                if bin_pos >= 0:
                    update_metrics(metrics['pos']['dens'][bin_pos], gt_pos.item(), pred_dens_pos.item())
                    update_metrics(metrics['pos']['loc'][bin_pos], gt_pos.item(), pred_loc_pos.item())
                # Store raw errors (prediction - ground truth) for histogram
                errors['pos']['dens'][bin_pos].append(pred_dens_pos.item() - gt_pos.item())
                errors['pos']['loc'][bin_pos].append(pred_loc_pos.item() - gt_pos.item())
                # Update negative class metrics
                if bin_neg >= 0:
                    update_metrics(metrics['neg']['dens'][bin_neg], gt_neg.item(), pred_dens_neg.item())
                    update_metrics(metrics['neg']['loc'][bin_neg], gt_neg.item(), pred_loc_neg.item())
                errors['neg']['dens'][bin_neg].append(pred_dens_neg.item() - gt_neg.item())
                errors['neg']['loc'][bin_neg].append(pred_loc_neg.item() - gt_neg.item())

    # Finalize metrics for each bin
    results = {'pos': [], 'neg': []}
    for cls in ['pos', 'neg']:
        for bin_idx in range(len(bins)):
            dens_met = finalize_metrics(metrics[cls]['dens'][bin_idx])
            loc_met = finalize_metrics(metrics[cls]['loc'][bin_idx])
            results[cls].append({
                'bin': bins[bin_idx],
                'density': dens_met,
                'localization': loc_met,
                'n': len(metrics[cls]['dens'][bin_idx]['abs'])  # number of samples in bin
            })
    return results, errors

# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def main():
    # Instantiate heatmap generators used in dataset
    loc_heatmap_generator = PointsToLocalizationHeatmap(out_hw=(160, 160), in_hw=(640, 640), sigma=2.0)
    count_heatmap_generator = PointsToCountHeatmap(out_hw=(160, 160), in_hw=(640, 640), sigma=2.0)

    # Load test dataset
    dataset = BCDataDataset(
        root=DATA_ROOT,
        split=SPLIT,
        target_loc_transform=loc_heatmap_generator,
        target_count_transform=count_heatmap_generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=8,  # adjust based on memory
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_heatmap_points,
    )

    # Instantiate and load model
    model = HybridModel()
    state = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)

    # Evaluate counts by bins and collect error lists
    results, errors = evaluate_counts_by_bins(
        model=model,
        loader=loader,
        bins=COUNT_BINS,
        device=DEVICE,
        loc_threshold=LOC_THRESHOLD,
        kernel_size=NMS_KERNEL,
    )

    # Print summary
    for cls in ['pos', 'neg']:
        print(f"\nResults for {cls} cells:")
        for res in results[cls]:
            lo, hi = res['bin']
            n = res['n']
            dens = res['density']
            loc = res['localization']
            print(f"  Bin [{lo}, {hi}] (n={n}):")
            print(f"    Density  - MAE: {dens['MAE']:.4f}, RMSE: {dens['RMSE']:.4f}, MAPE: {dens['MAPE']:.2f}%")
            print(f"    Localiz. - MAE: {loc['MAE']:.4f}, RMSE: {loc['RMSE']:.4f}, MAPE: {loc['MAPE']:.2f}%")

    # Draw histograms of raw errors (predicted minus true counts) for each class, strategy and bin
    # Create an output directory for histograms
    hist_dir = Path('histograms')
    hist_dir.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt is not None:
        for cls in ['pos', 'neg']:
            for bin_idx, (lo, hi) in enumerate(COUNT_BINS):
                for strategy_key, strategy_name in [('dens', 'density'), ('loc', 'localization')]:
                    errs = errors[cls][strategy_key][bin_idx]
                    if not errs:
                        continue
                    # Plot histogram without specifying colors or using subplots
                    plt.figure()
                    plt.hist(errs, bins=30)
                    plt.title(f"{cls} {strategy_name} errors for bin [{lo}, {hi}]")
                    plt.xlabel("Predicted count minus true count")
                    plt.ylabel("Frequency")
                    # Construct filename, replace inf with 'inf'
                    hi_str = 'inf' if hi == math.inf else str(int(hi))
                    fname = hist_dir / f"hist_{cls}_{strategy_key}_bin{lo}_{hi_str}.png"
                    plt.savefig(fname)
                    plt.close()
        print(f"\nHistogram images saved under: {hist_dir.resolve()}")


if __name__ == '__main__':
    main()
