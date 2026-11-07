import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorDiff3D(nn.Module):
    """
    3D Discrete Tensor Differential Field Operator.
    Approximates directional spatial gradients (dH, dW) and inter-channel
    variations (dC) via discrete finite differences.
    Contains one learnable scalar per channel (near-zero overhead).
    """
    def __init__(self, channels):
        super(TensorDiff3D, self).__init__()
        # Learnable channel-wise scaling coefficient for differential field
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # 1. Height-axis finite difference
        dh = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 1, 0))

        # 2. Width-axis finite difference
        dw = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (1, 0, 0, 0))

        # 3. Channel-axis finite difference
        dc = F.pad(x[:, 1:, :, :] - x[:, :-1, :, :], (0, 0, 0, 0, 1, 0))

        # 4. Total gradient magnitude field with hyperbolic tangent gating
        grad_field = torch.abs(dh) + torch.abs(dw) + torch.abs(dc)
        return torch.tanh(grad_field) * self.scale

