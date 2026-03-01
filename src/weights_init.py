import torch
import torch.nn as nn

def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    # y must be > 0
    return torch.log(torch.expm1(y))





@torch.no_grad()
def init_count_head_bias(head: nn.Module, expected_count: float, H: int, W: int, zero_last_weights: bool = True):
    """
    head is your HeatmapHead (with head.out = Conv2d(..., 1, 1))
    Sets head.out.bias so that softplus(bias) * H * W ~= expected_count
    """
    y = torch.tensor(expected_count / (H * W), dtype=head.out.bias.dtype, device=head.out.bias.device)
    b0 = inv_softplus(y)

    if zero_last_weights:
        nn.init.zeros_(head.out.weight)   # makes initial map nearly constant
    head.out.bias.fill_(b0.item())

