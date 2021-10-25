"""Loss modules for representation learning.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


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
            batch_loss: the sum of losses for prediction results per batch
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


class InfoNCELoss:
    """InfoNCE Loss for SimCLR https://arxiv.org/pdf/1807.03748v2.pdf."""

    def __init__(
        self,
        device: torch.device,
        batch_size: int = 32,
        n_trans: int = 2,
        temperature: float = 0.07,
    ) -> None:
        """Initialize InfoNCELoss class.

        Args:
            batch_size: batch size to use in iterator.
            n_trans: the number of times to apply transformations for representation learning.
            temperature: the value to adjust similarity scores.
                         e.g. # if the temperature is smaller than 1,
                              # similarity scores are enlarged than before.
                              # e.g. [100, 1] -> [1000, 10]
                              # It has an effect to train hard negative cases.
                              similarity_scores = np.array([100, 1])
                              temperature = 0.1
                              similarity_scores = similarity_scores / temperature
        """
        self.batch_size = batch_size
        self.n_trans = n_trans
        self.device = device
        self.temperature = temperature

    def __call__(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
        """Loss function for SimCLR.

        Args:
            features: encoded features from AYolov2 backbone

        Returns:
            batch_loss: the sum of InfoNCE losses for prediction results per batch
            loss_items: mean losses for batch
            feature_shape: the shape of encoded features
        """
        batch_size = min(self.batch_size, int(features.shape[0] / self.n_trans))
        # labels = torch.cat([torch.arange() for i in range(self.n_trans)], dim=0)
        labels = torch.cat(
            [torch.tensor([i for _ in range(self.n_trans)]) for i in range(batch_size)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (
            self.n_trans * batch_size,
            self.n_trans * batch_size,
        )
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        loss = CrossEntropyLoss()(logits, labels).to(self.device)

        batch_loss = loss * batch_size
        loss_items = torch.tensor([loss])
        feature_shape = features.shape
        return batch_loss, loss_items, feature_shape
