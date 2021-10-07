"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import math
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader

from scripts.loss.losses import ComputeLoss
from scripts.train.abstract_trainer import AbstractTrainer
from scripts.utils.general import (get_logger, labels_to_class_weights,
                                   labels_to_image_weights)

LOGGER = get_logger(__name__)


class YoloTrainer(AbstractTrainer):
    """YoloTrainer class."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        rank: int = -1,
    ) -> None:
        """Initialize YoloTrainer class.

        Args:
            model: yolo model to train.
            cfg: config.
            train_dataloader: dataloader for training.
            val_dataloader: dataloader for validation.
            rank: CUDA rank for DDP.
        """
        super().__init__(model, cfg, train_dataloader, val_dataloader)
        self.loss = ComputeLoss(self.model)
        self.optimizer, self.scheduler = self._init_optimizer()
        self.rank = rank
        self.maps = np.zeros(self.model.nc)  # map per class
        self.results = (
            {
                "total": (0, 0, 0, 0),
                "small": (0, 0, 0, 0),
                "medium": (0, 0, 0, 0),
                "large": (0, 0, 0, 0),
            },
            0,
            0,
            0,
        )  # P, R, mAP@0.5, mAP@0.5-0.95, val_loss(box, obj, cls)
        self.scaler: amp.GradScaler

    def update_image_weights(self) -> None:
        """Update image weights."""
        if self.cfg_train["image_weights"]:
            # Generate indices
            if self.rank in [-1, 0]:
                # number of total images
                n_imgs = len(self.train_dataloader.dataset.img_files)

                # class weights
                class_weights = (
                    self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2
                )
                # images weights
                image_weights = labels_to_image_weights(
                    self.train_dataloader.dataset.labels,
                    nc=self.model.nc,
                    class_weights=class_weights,
                )
                self.train_dataloader.dataset.indices = random.choices(
                    range(n_imgs), weights=image_weights, k=n_imgs
                )

            # Broadcast if DDP
            if self.rank != -1:
                indices = (
                    torch.tensor(self.train_dataloader.dataset.indices)
                    if self.rank == 0
                    else torch.zeros(n_imgs)
                ).int()
                dist.broadcast(indices, 0)
                if self.rank != 0:
                    self.train_dataloader.dataset.indices = indices.cpu().numpy()

    def _lr_function(self, x: float) -> float:
        return ((1 + math.cos(x * math.pi / self.cfg_train["epochs"])) / 2) * (
            1 - self.cfg_hyp["lrf"]
        ) + self.cfg_hyp["lrf"]

    def _init_optimizer(
        self,
    ) -> Tuple[List[optim.Optimizer], List[lr_scheduler.LambdaLR]]:
        """Initialize optimizer and scheduler."""
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

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        y_hat = self.model(x)
        return self.loss(y_hat, y)

    def training_step(
        self,
        train_batch: Tuple[
            torch.Tensor,
            torch.Tensor,
            Tuple[str, ...],
            Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...],
        ],
        batch_idx: int,
    ) -> None:
        """Train a step.

        Args:
            batch: batch data.
            batch_idx: batch index.

        Returns:
            Result of loss function.
        """
        img, label, path, shape = train_batch

        output = self.model(img)

        loss = self.loss(output, label)  # type: ignore

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

        return loss

    def on_start_epoch(self) -> None:
        """Run on start epoch."""

    def set_grad_scaler(self) -> amp.GradScaler:
        """Set GradScaler."""
        cuda = self.device.type != "cpu"
        return amp.GradScaler(enabled=cuda)

    def train(self) -> None:
        """Train model."""
        start_epoch = 0

        for scheduler in self.scheduler:
            scheduler.last_epoch = start_epoch - 1

        self.scaler = self.set_grad_scaler()

        LOGGER.info(
            "Image sizes %g train, %gt test\n"
            "Using %g dataloader workers\nLogging results to %s\n"
            "Starting training for %g epochs..."
            % (
                self.cfg_train["image_size"][0],
                self.cfg_train["image_size"][0],
                self.train_dataloader.num_workers,
                self.cfg_train["log_dir"] if self.cfg_train["log_dir"] else "exp",
                self.epochs,
            )
        )

        num_warmups = max(
            round(self.cfg_hyp["warmup_epochs"] * len(self.train_dataloader)), 1e3
        )

        # train epochs
        for i in range(start_epoch, self.epochs):

            mloss = torch.zeros(4, device=self.device)
            self.on_start_epoch()
            self.model.train()

            # train batch
            for i, batch in enumerate(self.train_dataloader):
                loss = self.training_step(batch, i)
            self.model.eval()
            for batch in self.val_dataloader:
                self.validation_step(batch)
            self.on_end_epoch()
