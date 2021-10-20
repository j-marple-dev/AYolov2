"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from scripts.loss.losses import ComputeLoss
from scripts.train.abstract_pl_module import AbstractPLModule
from scripts.utils.logger import get_logger

LOGGER = get_logger(__name__)


class YoloPLModule(AbstractPLModule):
    """Yolo model trainer."""

    def __init__(self, model: nn.Module, cfg: Dict[str, Any],) -> None:
        """Initialize YoloTrainer class.

        Args:
            model: yolo model to train.
            cfg: model trainining hyper parameter config.
        """
        self.cfg_hyp, self.cfg_train = cfg["hyper_params"], cfg["train"]
        super().__init__(model, cfg)

        self.model.hyp = self.cfg_hyp
        self.loss = ComputeLoss(self.model)

    def _lr_function(self, x: float) -> float:
        return ((1 + math.cos(x * math.pi / self.cfg_train["epochs"])) / 2) * (
            1 - self.cfg_hyp["lrf"]
        ) + self.cfg_hyp["lrf"]

    def _init_optimizer(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[lr_scheduler.LambdaLR]]:
        """Initialize optimizer and scheduler.

        Returns:
            List of optimizers
            List of learning rate schedulers
        """
        nbs = 64
        accumulate = max(round(nbs / self.cfg_train["batch_size"]), 1)
        self.cfg_hyp["weight_decay"] *= self.cfg_train["batch_size"] * accumulate / nbs
        LOGGER.info(f"Scaled weight_decay = {self.cfg_hyp['weight_decay']}")

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

        optimizer = getattr(
            __import__("torch.optim", fromlist=[""]), self.cfg_hyp["optimizer"]
        )(params=pg0, **self.cfg_hyp["optimizer_params"])
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": self.cfg_hyp["weight_decay"]}
        )
        optimizer.add_param_group({"params": pg2})
        LOGGER.info(
            "Optimizer groups: %g .bias, %g conv.weight, %g other"
            % (len(pg2), len(pg1), len(pg0))
        )

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_function)
        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward the model."""
        output = self.model(x)
        return output

    def training_step(
        self,
        train_batch: Tuple[
            torch.Tensor,
            torch.Tensor,
            Tuple[str, ...],
            Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        """Train a step (a batch).

        Args:
            train_batch: train batch in tuple (input_x, true_y).
            batch_idx: current batch index.

        Returns:
            Result of loss function.
        """
        img, label, path, shape = train_batch

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
        output = self.model(img)
        loss = self.loss(output, label)  # type: ignore

        self.log_dict(
            {
                "loss": loss[0],
                "loss_box": loss[1][0],
                "loss_obj": loss[1][1],
                "loss_cls": loss[1][2],
            },
            prog_bar=True,
        )

        return loss[0]

    def validation_step(
        self,
        val_batch: Tuple[
            torch.Tensor,
            torch.Tensor,
            Tuple[str, ...],
            Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...],
        ],
        batch_idx: int,
    ) -> None:
        """Validate a step (a batch).

        Args:
            val_batch: validation data batch in tuple (input_x, true_y).
            batch_idx: current batch index.
        """
        img, label, path, shape = val_batch

        output = self.model(img)
        bboxes, train_out = output

        loss = self.loss(train_out, label)  # type: ignore

        self.log_dict(
            {
                "val_loss": loss[0],
                "val_loss_box": loss[1][0],
                "val_loss_obj": loss[1][1],
                "val_loss_cls": loss[1][2],
            },
            prog_bar=True,
            on_epoch=True,
        )

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        y_hat = self.model(x)
        return self.loss(y_hat, y)
