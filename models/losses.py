import torch
import torch.nn.functional as F

def weighted_sigmoid_mse_from_logits(pred_logits, target, alpha_pos=100.0, alpha_neg=100.0):
    '''
    pred_logits: (B,2,H,W) raw logits from the model (NO sigmoid applied yet)
    target:      (B,2,H,W) Gaussian heatmap in [0,1]
    alpha:       how much more to weight peaks vs background
    '''

    pred = torch.sigmoid(pred_logits)


    # Weight map: background ~1, peaks approach (1+alpha)
    w_pos = 1.0 + alpha_pos * target[:, 0]
    w_neg = 1.0 + alpha_neg * target[:, 1]


    loss_pos = w_pos * (pred[:, 0] - target[:, 0]).pow(2)
    loss_neg = w_neg * (pred[:, 1] - target[:, 1]).pow(2)

    # normalize per image
    loss_pos = loss_pos.sum(dim=(1, 2)) / (w_pos.sum(dim=(1, 2)) + 1e-6)
    loss_neg = loss_neg.sum(dim=(1, 2)) / (w_neg.sum(dim=(1, 2)) + 1e-6)

    return (loss_pos.mean() + loss_neg.mean()) / 2

def softplus_mse_from_logits(pred_logits, target):
    '''
    pred_logits: (B,2,H,W) raw logits (NO softplus applied yet)
    target:      (B,2,H,W) Gaussian heatmap nonnegative
    '''
    pred = F.softplus(pred_logits)

    #print(f"grid_size: {grid_size}")

    loss_pos = (pred[:, 0] - target[:, 0]).pow(2).mean()
    loss_neg = (pred[:, 1] - target[:, 1]).pow(2).mean()



    return (loss_pos + loss_neg) / 2


def l1_count_from_density_logits(pred_logits, gtN):
    '''
    pred_logits: (B,2,H,W) raw logits from the model (NO softplus applied yet)
    gtN: (B, 2)
    '''
    pred = F.softplus(pred_logits)
    pred = torch.sum(pred, dim = (2, 3))
    
    gtN = gtN.to(pred.device).float()
    
    #L_count_mean = torch.abs(pred - gtN).mean()


    #return L_count_mean

    return F.l1_loss(pred, gtN, reduction="mean")

