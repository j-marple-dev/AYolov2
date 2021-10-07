"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from scripts.train.abstract_trainer import AbstractTrainer
from scripts.utils.general import get_logger

LOGGER = get_logger(__name__)


class YoloTrainer(AbstractTrainer):
    """YoloTrainer class."""

    def __init__(
        self, model: nn.Module, hyp: Dict[str, Any], epochs: int, device: torch.device
    ) -> None:
        """Initialize YoloTrainer class.

        Args:
            model: yolo model to train.
            hyp: hyper parameter config.
            epochs: number of epochs to train.
            device: torch device to train on.
        """
        super().__init__(model, hyp, epochs, device)
        self.loss = eval(self.hyp["loss"])
        self.optimizer, self.scheduler = self.init_optimizer()

    def init_optimizer(
        self,
    ) -> Tuple[List[optim.Optimizer], List[lr_scheduler.LambdaLR]]:
        """Initialize optimizer and scheduler."""
        pg0: List[torch.Tensor] = []
        pg1: List[torch.Tensor] = []
        pg2: List[torch.Tensor] = []
        for _, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, torch.Tensor):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, torch.Tensor):
                pg1.append(v.weight)
        for _, v in self.model.named_parameters():
            v.requires_grad = True
        # TODO(ulken94): find fancy way to create optimizer
        optimizer: torch.optim.Optimizer = eval(self.hyp["optimizer"])(
            params=pg0, **self.hyp["optimizer_params"]
        )
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": self.hyp["weight_decay"]}
        )
        optimizer.add_param_group({"params": pg2})
        LOGGER.info(
            "Optimizer groups: %g .bias, %g conv.weight, %g other"
            % (len(pg2), len(pg1), len(pg0))
        )

        lf = (
            lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2)
            * (1 - self.hyp["lrf"])
            + self.hyp["lrf"]
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        return [optimizer], [scheduler]

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        y_hat = self.model(x)
        return self.loss(y_hat, y)

    # def training_step(
    #     self, batch: Union[List[torch.Tensor], torch.Tensor, Tuple[torch.Tensor, ...]],
    #     batch_idx: int
    # ) -> torch.Tensor:
