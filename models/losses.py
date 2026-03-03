import torch
import torch.nn.functional as F

"""
Loss functions for the BCData_Ki67 hybrid localisation/counting model.

This module provides weighted losses for both the localisation and density/count heads.
The functions defined here mirror the original implementation but can be easily
extended to support per–class weighting or alternative activation functions.

Functions
---------
weighted_sigmoid_mse_from_logits(pred_logits, target, alpha_pos=100.0, alpha_neg=100.0)
    Computes a weighted mean‑squared error loss between a localisation head's raw
    logits and its Gaussian target.  A large `alpha` emphasises the error at
    peaks relative to background.

softplus_mse_from_logits(pred_logits, target)
    Computes the mean squared error between a density head's raw logits and
    the corresponding target.  The raw logits are passed through a softplus so
    that the output is non‑negative.

l1_count_from_density_logits(pred_logits, gtN)
    Computes a simple L1 loss on the predicted integral of the density head
    against the ground‑truth count.  The predicted density is obtained via
    softplus to ensure non‑negativity.

These functions operate on tensors of shape `(B, 2, H, W)` for the heads,
where channel 0 corresponds to the positive class and channel 1 to the
negative class.
"""

def weighted_sigmoid_mse_from_logits(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha_pos: float = 100.0,
    alpha_neg: float = 100.0,
) -> torch.Tensor:
    """Weighted mean squared error for localisation logits.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Raw logits from the localisation head with shape `(B, 2, H, W)`.
    target : torch.Tensor
        Gaussian heatmap targets in `[0, 1]` with the same shape as ``pred_logits``.
    alpha_pos : float, optional
        Weighting factor for positive peaks; larger values emphasise the loss on
        peak pixels, by default 100.0.
    alpha_neg : float, optional
        Weighting factor for negative peaks, by default 100.0.

    Returns
    -------
    torch.Tensor
        The scalar loss averaged over the batch and both channels.
    """
    pred = torch.sigmoid(pred_logits)

    # Weight maps emphasise regions where the target is 1 (peaks).  A value of
    # 1 means no weighting; values approaching (1 + alpha) heavily weight the
    # peak region.  We apply separate weights per channel so that the positive
    # and negative localisation tasks can be tuned independently.
    w_pos = 1.0 + alpha_pos * target[:, 0]
    w_neg = 1.0 + alpha_neg * target[:, 1]

    # Compute per‑pixel squared error and multiply by weights
    loss_pos = w_pos * (pred[:, 0] - target[:, 0]).pow(2)
    loss_neg = w_neg * (pred[:, 1] - target[:, 1]).pow(2)

    # Normalize each image by the sum of weights to avoid scale dependence on
    # the number of peak pixels.  Then average over the batch.
    loss_pos = loss_pos.sum(dim=(1, 2)) / (w_pos.sum(dim=(1, 2)) + 1e-6)
    loss_neg = loss_neg.sum(dim=(1, 2)) / (w_neg.sum(dim=(1, 2)) + 1e-6)

    return (loss_pos.mean() + loss_neg.mean()) / 2.0


def softplus_mse_from_logits(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error between softplus‑activated logits and the target.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Raw logits from the density head with shape `(B, 2, H, W)`.
    target : torch.Tensor
        Unit‑mass Gaussian heatmap targets with the same shape as ``pred_logits``.

    Returns
    -------
    torch.Tensor
        The scalar loss averaged over the batch and both channels.
    """
    pred = F.softplus(pred_logits)
    loss_pos = (pred[:, 0] - target[:, 0]).pow(2).mean()
    loss_neg = (pred[:, 1] - target[:, 1]).pow(2).mean()
    return (loss_pos + loss_neg) / 2.0


def l1_count_from_density_logits(
    pred_logits: torch.Tensor,
    gtN: torch.Tensor,
) -> torch.Tensor:
    """L1 loss on the predicted counts derived from the density logits.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Raw logits from the density head with shape `(B, 2, H, W)`.
    gtN : torch.Tensor
        Ground truth counts with shape `(B, 2)`.  The first column is the
        positive count and the second is the negative count.

    Returns
    -------
    torch.Tensor
        The scalar L1 loss on the predicted counts, averaged over the batch.
    """
    # Convert logits to densities via softplus so that predicted densities are
    # non‑negative.  Summing across height and width yields the predicted count
    # per channel.
    pred = F.softplus(pred_logits)
    pred_counts = pred.sum(dim=(2, 3))

    # Ensure gtN is float and on the same device as pred_counts
    gt_counts = gtN.to(pred_counts.device).float()

    return F.l1_loss(pred_counts, gt_counts, reduction="mean")


# -----------------------------------------------------------------------------
# Extended loss functions for density and count using sigmoid activation.
#
# These functions implement weighted losses for the density head and the
# per‑channel counts when the density head uses a sigmoid activation
# instead of softplus.  The negative channel can be scaled to compensate
# for systematic undercounting.  Positional weighting of the error
# (alpha parameters) is also supported.
# -----------------------------------------------------------------------------

def weighted_sigmoid_mse_density_from_logits(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 2.0,
    alpha_pos: float = 100.0,
    alpha_neg: float = 100.0,
    neg_scale: float = 1.0,
) -> torch.Tensor:
    """Weighted MSE for density logits using a sigmoid activation.

    This loss is similar to :func:`weighted_sigmoid_mse_from_logits` but
    includes per‑channel weighting and an optional scaling factor applied to
    the negative channel.  It is intended for use with the density head
    when using a sigmoid activation (i.e. bounded outputs).

    Parameters
    ----------
    pred_logits : torch.Tensor
        Raw logits from the density head with shape ``(B, 2, H, W)``.
    target : torch.Tensor
        Unit‑mass Gaussian targets of the same shape.
    pos_weight : float, optional
        Weight applied to the positive channel loss.  Default is 1.0.
    neg_weight : float, optional
        Weight applied to the negative channel loss.  Default is 2.0.
    alpha_pos : float, optional
        Positional weighting factor for the positive channel (peak emphasis).
        Default is 100.0.
    alpha_neg : float, optional
        Positional weighting factor for the negative channel.  Default is
        100.0.
    neg_scale : float, optional
        Multiplicative scale applied to the negative channel predictions to
        mitigate undercounting.  Default is 1.0 (no scaling).

    Returns
    -------
    torch.Tensor
        The scalar weighted loss averaged over the batch.
    """
    # Sigmoid activation to constrain predictions into [0, 1]
    pred = torch.sigmoid(pred_logits)
    # Apply scaling to negative channel
    pred[:, 1] = pred[:, 1] * neg_scale
    # Weight maps emphasise peaks
    w_pos = 1.0 + alpha_pos * target[:, 0]
    w_neg = 1.0 + alpha_neg * target[:, 1]
    # Per‑pixel squared error
    loss_pos = w_pos * (pred[:, 0] - target[:, 0]).pow(2)
    loss_neg = w_neg * (pred[:, 1] - target[:, 1]).pow(2)
    # Normalise per image by sum of weights
    loss_pos = loss_pos.sum(dim=(1, 2)) / (w_pos.sum(dim=(1, 2)) + 1e-6)
    loss_neg = loss_neg.sum(dim=(1, 2)) / (w_neg.sum(dim=(1, 2)) + 1e-6)
    # Combine with channel weights and average over batch
    combined = (pos_weight * loss_pos + neg_weight * loss_neg) / (pos_weight + neg_weight)
    return combined.mean()


def weighted_l1_count_from_sigmoid_logits(
    pred_logits: torch.Tensor,
    gtN: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 2.0,
    neg_scale: float = 1.0,
) -> torch.Tensor:
    """Weighted L1 loss on counts predicted via sigmoid‑activated logits.

    This function converts raw logits into densities via a sigmoid, applies a
    scaling factor to the negative channel, sums the densities to obtain
    predicted counts, and computes a weighted L1 loss against the ground
    truth counts.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Raw logits from the density head with shape ``(B, 2, H, W)``.
    gtN : torch.Tensor
        Ground truth counts with shape ``(B, 2)``.
    pos_weight : float, optional
        Weight applied to the positive channel count loss.  Default is 1.0.
    neg_weight : float, optional
        Weight applied to the negative channel count loss.  Default is 2.0.
    neg_scale : float, optional
        Multiplicative scale applied to the negative channel predictions.
        Default is 1.0 (no scaling).

    Returns
    -------
    torch.Tensor
        The scalar weighted L1 loss averaged over the batch.
    """
    pred = torch.sigmoid(pred_logits)
    pred[:, 1] = pred[:, 1] * neg_scale
    pred_counts = pred.sum(dim=(2, 3))
    gt_counts = gtN.to(pred_counts.device).float()
    loss_pos = F.l1_loss(pred_counts[:, 0], gt_counts[:, 0])
    loss_neg = F.l1_loss(pred_counts[:, 1], gt_counts[:, 1])
    combined = (pos_weight * loss_pos + neg_weight * loss_neg) / (pos_weight + neg_weight)
    return combined


# -----------------------------------------------------------------------------
# Weighted softplus loss functions for the density head.
#
# These functions mirror the weighted sigmoid losses but use a softplus
# activation, allowing the density head to produce unbounded non‑negative
# outputs.  They also support per‑channel weighting and an optional
# multiplicative scaling on the negative channel to correct for
# undercounting.
# -----------------------------------------------------------------------------

def weighted_softplus_mse_density_from_logits(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 2.0,
    alpha_pos: float = 100.0,
    alpha_neg: float = 100.0,
    neg_scale: float = 1.0,
) -> torch.Tensor:
    """Weighted MSE for density logits using a softplus activation.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Raw logits from the density head with shape ``(B, 2, H, W)``.
    target : torch.Tensor
        Unit‑mass Gaussian targets of the same shape.
    pos_weight : float, optional
        Weight applied to the positive channel loss.  Default is 1.0.
    neg_weight : float, optional
        Weight applied to the negative channel loss.  Default is 2.0.
    alpha_pos : float, optional
        Positional weighting factor for the positive channel (peak emphasis).
        Default is 100.0.
    alpha_neg : float, optional
        Positional weighting factor for the negative channel.  Default is
        100.0.
    neg_scale : float, optional
        Multiplicative scale applied to the negative channel predictions to
        mitigate undercounting.  Default is 1.0 (no scaling).

    Returns
    -------
    torch.Tensor
        The scalar weighted loss averaged over the batch.
    """
    # Softplus activation ensures non‑negativity and allows sums > 1
    pred = F.softplus(pred_logits)
    # Scale negative channel
    pred[:, 1] = pred[:, 1] * neg_scale
    # Weight maps emphasise peaks
    w_pos = 1.0 + alpha_pos * target[:, 0]
    w_neg = 1.0 + alpha_neg * target[:, 1]
    # Per‑pixel squared error
    loss_pos = w_pos * (pred[:, 0] - target[:, 0]).pow(2)
    loss_neg = w_neg * (pred[:, 1] - target[:, 1]).pow(2)
    # Normalise per image by sum of weights
    loss_pos = loss_pos.sum(dim=(1, 2)) / (w_pos.sum(dim=(1, 2)) + 1e-6)
    loss_neg = loss_neg.sum(dim=(1, 2)) / (w_neg.sum(dim=(1, 2)) + 1e-6)
    # Combine with channel weights and average over batch
    combined = (pos_weight * loss_pos + neg_weight * loss_neg) / (pos_weight + neg_weight)
    return combined.mean()


def weighted_l1_count_from_density_logits(
    pred_logits: torch.Tensor,
    gtN: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 2.0,
    neg_scale: float = 1.0,
) -> torch.Tensor:
    """Weighted L1 loss on counts predicted via softplus‑activated logits.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Raw logits from the density head with shape ``(B, 2, H, W)``.
    gtN : torch.Tensor
        Ground truth counts with shape ``(B, 2)``.
    pos_weight : float, optional
        Weight applied to the positive channel count loss.  Default is 1.0.
    neg_weight : float, optional
        Weight applied to the negative channel count loss.  Default is 2.0.
    neg_scale : float, optional
        Multiplicative scale applied to the negative channel predictions.
        Default is 1.0 (no scaling).

    Returns
    -------
    torch.Tensor
        The scalar weighted L1 loss averaged over the batch.
    """
    pred = F.softplus(pred_logits)
    pred[:, 1] = pred[:, 1] * neg_scale
    pred_counts = pred.sum(dim=(2, 3))
    gt_counts = gtN.to(pred_counts.device).float()
    loss_pos = F.l1_loss(pred_counts[:, 0], gt_counts[:, 0])
    loss_neg = F.l1_loss(pred_counts[:, 1], gt_counts[:, 1])
    combined = (pos_weight * loss_pos + neg_weight * loss_neg) / (pos_weight + neg_weight)
    return combined