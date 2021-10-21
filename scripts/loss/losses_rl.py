"""Loss modules for representation learning.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""

from typing import Tuple

import torch


class RLLoss:
    """Loss for representation learning."""

    def __init__(self, ltype: str = "L1Loss") -> None:
        """Initialize RLLoss class.

        Args:
            ltype: loss type for representation learning (e.g. L1Loss).
        """
        self.ltype = ltype

    def __call__(
        self, pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
        """Loss function for representation learning.

        Args:
            pred: prediction results

        Returns:
            loss: the sum of losses for prediction results
            loss_items: mean losses for batch
            pred_shape: the shape of prediction results
        """
        preds1 = torch.stack([pred[i] for i in range(0, len(pred)) if i % 2 == 0])
        preds2 = torch.stack([pred[i] for i in range(0, len(pred)) if i % 2 != 0])
        loss = (preds1 - preds2).abs().sum() / torch.numel(preds1)
        bs = preds1.shape[0]

        batch_loss = loss * bs
        loss_items = torch.tensor([loss])
        pred_shape = preds1.shape
        return batch_loss, loss_items, pred_shape
