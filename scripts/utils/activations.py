"""Activation for export-friendly.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import torch
from torch import nn

class SiLU(nn.Module):
    """Export-friendly version of nn.SiLU()

    SiLU https://arxiv.org/pdf/1606.08415.pdf
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """Export-friendly version of nn.Hardswish()"""

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x * F.hardtanh(x + 3, 0., 6.) / 6.


class Mish(nn.Module):
    """Mish https://github.com/digantamisra98/Mish"""

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x * F.softplus(x).tanh()
