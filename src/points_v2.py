import torch
import torch.nn.functional as F
import numpy as np

from utils.debug import print_info


def heatmaps_to_points_batch(heatmaps, kernel_size, threshold):
    '''
    heatmaps: gaussian heatmap with values in [0, 1]. shape: [B, 2, 160, 160]

    kernel_size: kernel size for max_pool2d
    '''

    out_pos = []
    out_neg = []

    for b in range(heatmaps.shape[0]):  # loop over batch
        print("batch start")
        heatmap2d = heatmaps[b]

        pos_pts, neg_pts = heatmap_to_points(heatmap2d, kernel_size, threshold)

        out_pos.append(pos_pts)
        out_neg.append(neg_pts)

        print('*'*20, '\n\n\n')

    return out_pos, out_neg

def heatmap_to_points(heatmap2d, kernel_size, threshold):
    '''
    heatmap2d: [2, 160, 160]

    '''
    print_info(heatmap2d, "heatmap2d")

    pos_heatmap = heatmap2d[0]
    neg_heatmap = heatmap2d[1]

    # Local maximums + threshold
    pos_mask = _local_maxima(pos_heatmap, kernel_size) & (pos_heatmap > threshold)  # true false matrix
    neg_mask = _local_maxima(neg_heatmap, kernel_size) & (neg_heatmap > threshold)

    pos_x, pos_y = torch.where(pos_mask[0])
    neg_x, neg_y = torch.where(neg_mask[0])

    pos_pts = torch.stack((pos_x, pos_y)).T
    neg_pts = torch.stack((neg_x, neg_y)).T

    return pos_pts, neg_pts

def _local_maxima(hm, kernel_size):
        '''
        hm: [H, W]
        kernel_size:
        '''
        print_info(hm, "hm")

        pad = kernel_size // 2

        pooled = F.max_pool2d(
            hm[None], kernel_size=kernel_size, stride=1, padding=pad
        )
        print_info(pooled, "pooled")

        res = pooled == hm
        print(f"res.shape:\n {res.shape}")
        return res

def loc_metrics_batch(pred_pts, pts):
    '''
    pred_pts: list with len B
    pts: list with len B
    '''
    metrics = []
    for b in range(len(pred_pts)):
        res = loc_metrics(pred_pts[b], pts[b])
        metrics.append(res)

    return metrics


def loc_metrics(pred_pts, pts):
    '''
    compute metrics for a single sample

    '''

    #dists = np.linalg.norm(pred_pts - pts, axis=-1)

    return
