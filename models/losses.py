import torch
def heatmap_weighted_mse_loss(pred_logits, target, alpha=100.0):
    """
    pred_logits: (B,1,H,W) raw logits from the model (NO sigmoid applied yet)
    target:      (B,1,H,W) Gaussian heatmap in [0,1]
    alpha:       how much more to weight peaks vs background
    """
    pred = torch.sigmoid(pred_logits)

    # Weight map: background ~1, peaks approach (1+alpha)
    w = 1.0 + alpha * target

    loss = w * (pred - target).pow(2)
    return loss.mean()



