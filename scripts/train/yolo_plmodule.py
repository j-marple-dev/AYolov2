"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from scripts.train.abstract_pl_module import AbstractPLModule
from scripts.utils.torch_utils import select_device

logger = logging.getLogger(__name__)


class YoloPLModule(AbstractPLModule):
    """Yolo model trainer."""

    def __init__(
        self,
        model: nn.Module,
        hyp: Dict[str, Any],
        epochs: int,
        device: Optional[Union[torch.device, str]],
    ) -> None:
        """Initialize YoloTrainer class.

        Args:
            model: yolo model to train.
            hyp: hyper parameter config.
            epochs: number of epochs to train.
            device: torch device to train on.
        """
        if isinstance(device, str):
            torch_device = select_device(device)
        elif device is None:
            torch_device = torch.device("cpu")
        else:
            torch_device = device
        super().__init__(model, hyp, epochs, torch_device)
        self.loss = eval(self.hyp["loss"])
        # self.optimizer: torch.optim.Optimizer
        # self.scheduler: lr_scheduler.LambdaLR
        # Important: This property activates manual optimization.
        self.optimizer, self.scheduler = self.init_optimizer()

    def init_optimizer(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[lr_scheduler.LambdaLR]]:
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
        logger.info(
            "Optimizer groups: %g .bias, %g conv.weight, %g other"
            % (len(pg2), len(pg1), len(pg0))
        )

        lf = (
            lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2)
            * (1 - self.hyp["lrf"])
            + self.hyp["lrf"]
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # self.optimizer = optimizer
        # self.scheduler = scheduler
        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward the model."""
        output = self.model(x)
        return output

    def training_step(
        self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Train a step (a batch).

        Args:
            train_batch: train batch in tuple (input_x, true_y).
            batch_idx: current batch index.

        Returns:
            Result of loss function.
        """
        x, y = train_batch
        self.model.train()

        """
        TODO: This part should be out of class.
        if opt.image_weights:
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2
                iw = labels_to_image_weights(
                    dataset.labels, nc=nc, class_weights=cw
                )
                dataset.indices = random.choies(
                    range(dataset.n), weights=iw, k=dataset.n
                )
            if rank != -1:
                indices = (
                    torch.tensor(dataset.indices)
                    if rank == 0 else torch.zeros(dataset.n)
                )

        """
        # mloss = torch.zeros(4, device=self.device)
        output = self.model(x)
        loss = self.loss(output, y)  # type: ignore

        return loss

    def validation_step(
        self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validate a step (a batch).

        Args:
            val_batch: validation data batch in tuple (input_x, true_y).
            batch_idx: current batch index.

        Returns:
            Result of loss function.
        """
        x, y = val_batch
        output = self.model(x)
        loss = self.loss(output, y)  # type: ignore

        return loss

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        y_hat = self.model(x)
        return self.loss(y_hat, y)
