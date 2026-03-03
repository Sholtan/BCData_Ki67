"""
Weight initialisation utilities for BCData Ki67 models.

The ``init_count_head_bias`` function sets the bias of a count head
so that the initial predicted count approximates a desired expected
value.  This is done by inverting the softplus activation such that
``softplus(bias) * H * W ≈ expected_count``.  Optionally, the
convolutional weights of the head can be zeroed to produce a nearly
uniform prediction at initialisation.
"""

import torch
import torch.nn as nn

def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Return the inverse of the softplus function for positive ``y``.

    The softplus activation is defined as ``softplus(x) = log(1 + exp(x))``.
    For positive ``y``, this function returns an ``x`` such that
    ``softplus(x) = y``.  This is useful for initialising biases to
    specific values after the softplus.

    Parameters
    ----------
    y : torch.Tensor
        A positive tensor representing the desired softplus output.

    Returns
    -------
    torch.Tensor
        The corresponding inverse softplus input.
    """
    return torch.log(torch.expm1(y))


@torch.no_grad()
def init_count_head_bias(
    head: nn.Module,
    expected_count: float,
    H: int,
    W: int,
    zero_last_weights: bool = True,
) -> None:
    """Initialise the bias of a count head to match an expected count.

    Parameters
    ----------
    head : nn.Module
        A module containing an attribute ``out`` which is a ``Conv2d`` with
        output channels equal to 1 (per class).  The bias of this layer
        will be set.
    expected_count : float
        Approximate number of cells expected in an image.  The bias is set
        so that ``softplus(bias) * H * W ≈ expected_count``.
    H : int
        Height of the output heatmap.
    W : int
        Width of the output heatmap.
    zero_last_weights : bool, optional
        Whether to zero the weights of the last convolutional layer.
        Zeroing the weights helps ensure that the initial map is nearly
        constant.  Default is ``True``.
    """
    # Compute desired density per pixel
    y = torch.tensor(expected_count / (H * W), dtype=head.out.bias.dtype, device=head.out.bias.device)
    b0 = inv_softplus(y)
    if zero_last_weights:
        nn.init.zeros_(head.out.weight)
    head.out.bias.fill_(b0.item())