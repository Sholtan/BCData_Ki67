import torch
import torch.nn.functional as F
from src.debug import print_info
import numpy as np



def evaluate_count(loader, model):
    '''
    loader: the loader to evaluate the metrics on

    Return:
    Mean Absolute Error
    Root Mean Squared Error
    Mean Absolute Persentage Error
    '''
    
    
    device = next(model.parameters()).device
    print(f"device: {device}")
    

    total_abs_error_pos = 0.
    total_abs_error_neg = 0.

    total_abs_perc_error_pos = 0.
    total_abs_perc_error_neg = 0.

    total_squared_error_pos = 0.
    total_squared_error_neg = 0.


    total_sample_count = 0
    total_nonzero_pos = 0
    total_nonzero_neg = 0

    pos_sum = 0
    neg_sum = 0

    model.eval()
    with torch.no_grad():
        for img, loc_heatmap, count_heatmap, pos_pts, neg_pts in loader:
            img = img.to(device)
            loc_heatmap = loc_heatmap.to(device)
            count_heatmap = count_heatmap.to(device)


            pred_loc_hm, pred_den_hm, pred_count = model(img)


            # pos_pts, neg_pts -> take gt count from them
            gtN = torch.tensor([(len(tmp1), len(tmp2)) for tmp1, tmp2 in zip(pos_pts, neg_pts)])  # (B, 2)

            pos_sum += gtN.sum(dim=0)[0].item()
            neg_sum += gtN.sum(dim=0)[1].item()


            gtN = gtN.float()
            pred_count_cpu = pred_count.detach().cpu().float()

            # Separate columns
            gt_pos = gtN[:, 0]
            gt_neg = gtN[:, 1]
            pred_pos = pred_count_cpu[:, 0]
            pred_neg = pred_count_cpu[:, 1]

            # Masks for nonzero GT
            mask_pos = gt_pos != 0
            mask_neg = gt_neg != 0
            # MAE calculation
            #***************************************************************************************************************************
            batch_abs_errors_pos = torch.abs(gtN - pred_count_cpu).sum(dim=0)[0].item()
            batch_abs_errors_neg = torch.abs(gtN - pred_count_cpu).sum(dim=0)[1].item()
            total_abs_error_pos += batch_abs_errors_pos
            total_abs_error_neg += batch_abs_errors_neg
            #***************************************************************************************************************************

            # MAPE calculation
            #***************************************************************************************************************************
            # Accumulate positive-cell percentage errors
            if mask_pos.any():
                abs_perc_error_pos = torch.abs((gt_pos[mask_pos] - pred_pos[mask_pos]) / gt_pos[mask_pos])
                total_abs_perc_error_pos += abs_perc_error_pos.sum().item()
                total_nonzero_pos += mask_pos.sum().item()

            # Accumulate negative-cell percentage errors
            if mask_neg.any():
                abs_perc_error_neg = torch.abs((gt_neg[mask_neg] - pred_neg[mask_neg]) / gt_neg[mask_neg])
                total_abs_perc_error_neg += abs_perc_error_neg.sum().item()
                total_nonzero_neg += mask_neg.sum().item()

            
            #***************************************************************************************************************************

            # RMSE calculation
            #***************************************************************************************************************************
            batch_squared_error_pos = torch.pow(gtN - pred_count_cpu, 2).sum(dim=0)[0].item()
            batch_squared_error_neg = torch.pow(gtN - pred_count_cpu, 2).sum(dim=0)[1].item()
            total_squared_error_pos += batch_squared_error_pos
            total_squared_error_neg += batch_squared_error_neg
            #***************************************************************************************************************************

            total_sample_count += len(gtN) 
            

    print(f"total_sample_count: {total_sample_count}")
    print(f"mean n pos: {pos_sum/total_sample_count},  mean n neg: {neg_sum/total_sample_count}")

    mae_pos = total_abs_error_pos / total_sample_count
    mae_neg = total_abs_error_neg / total_sample_count

    mape_pos = 100.0 * total_abs_perc_error_pos / total_nonzero_pos if total_nonzero_pos > 0 else None
    mape_neg = 100.0 * total_abs_perc_error_neg / total_nonzero_neg if total_nonzero_neg > 0 else None

    rmse_pos = np.sqrt(total_squared_error_pos / total_sample_count)
    rmse_neg = np.sqrt(total_squared_error_neg / total_sample_count)


    return mae_pos, mae_neg, rmse_pos, rmse_neg, mape_pos, mape_neg
        



























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


