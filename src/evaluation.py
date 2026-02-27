import torch.nn.functional as F
from utils.debug import print_info




def evaluate_count(val_loader, model):
    '''
    
    '''
    # pseudo code
    # model.eval()

    # total_abs_err = 0.0
    # total_sq_err = 0.0
    # total_n = 0

    # total_ape = 0.0
    # total_mape_n = 0

    # with torch.no_grad():
    #     for x, y in loader:
    #         x = x.to(device)
    #         y = y.to(device)

    #         pred = model(x)

    #         pred = pred.view(-1)
    #         y = y.view(-1)

    #         err = pred - y
    #         abs_err = torch.abs(err)

    #         total_abs_err += abs_err.sum().item()
    #         total_sq_err += (err ** 2).sum().item()
    #         total_n += y.numel()

    #         # MAPE (skip zero targets)
    #         nonzero_mask = (y != 0)
    #         if nonzero_mask.any():
    #             ape = torch.abs((pred[nonzero_mask] - y[nonzero_mask]) / y[nonzero_mask])
    #             total_ape += ape.sum().item()
    #             total_mape_n += ape.numel()

    # mae = total_abs_err / total_n
    # rmse = math.sqrt(total_sq_err / total_n)
    # mape = 100.0 * (total_ape / total_mape_n) if total_mape_n > 0 else float("nan")

    # return {
    #     "mae": mae,
    #     "rmse": rmse,
    #     "mape": mape,
    #     "n": total_n,
    #     "mape_n": total_mape_n,  # how many targets were non-zero
    # }

def count_metrics_batch(estimated_counts, gt_counts):
    '''
    arguments

    estimated_counts: shape (B,)
    gt_counts: shape (B,)
    



    '''

    assert len(estimated_counts) == len(gt_counts)
    B = len(estimated_counts)


    MAE = F.l1_loss()
    #MSQ = 
    #MAPE = 


    return



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

    # MAE
    # RMSQ
    # MAPE

    return



def count_metrics(count, pos_points_batch, neg_points_batch):
    '''

    '''

    # count pos in pos_points_batch
    # count neg in neg_points_batch

    print_info(count, 'count')
    print_info(pos_points_batch, 'pos_points_batch')
    print_info(neg_points_batch, 'neg_points_batch')

