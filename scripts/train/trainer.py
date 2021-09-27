"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from scripts.train.abstract_trainer import AbstractTrainer
from scripts.utils.torch_utils import select_device


class YoloTrainer(AbstractTrainer):
    """Yolo model trainer."""

    def __init__(
        self,
        model: nn.Module,
        hyp: Dict[str, Any],
        device: Optional[Union[torch.device, str]],
    ) -> None:
        """Initialize YoloTrainer class.

        Args:
            model: yolo model to train.
            hyp: hyper parameter config.
        """
        if isinstance(device, str):
            self.device = select_device(device)
        elif device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        super().__init__(model, hyp, device)
        self.loss: object

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward the model."""
        output = self.model(x)
        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers and return optimizer."""
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

        optimizer: torch.optim.Optimizer = eval(self.hyp["optimizer"])(
            **self.hyp["optimizer_params"]
        )
        self.loss = eval(self.hyp["loss"])
        return optimizer

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
                dataset.indices = random.choices(
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
