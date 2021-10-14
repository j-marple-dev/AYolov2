"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import math
import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.loss.losses import ComputeLoss
from scripts.train.abstract_trainer import AbstractTrainer
from scripts.utils.general import (check_img_size, get_logger,
                                   labels_to_image_weights)
from scripts.utils.plot_utils import plot_images
from scripts.utils.train_utils import YoloValidator

if TYPE_CHECKING:
    from scripts.utils.torch_utils import ModelEMA

LOGGER = get_logger(__name__)


class YoloTrainer(AbstractTrainer):
    """YoloTrainer class."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        ema: Optional["ModelEMA"],
        device: torch.device,
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
        super().__init__(model, cfg, train_dataloader, val_dataloader, device=device)

        self.loss = ComputeLoss(self.model)
        self.nbs = 64
        self.accumulate = max(round(self.nbs / self.cfg_train["batch_size"]), 1)
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
        self.mloss: torch.Tensor
        self.num_warmups = max(
            round(self.cfg_hyp["warmup_epochs"] * len(self.train_dataloader)), 1e3
        )
        if isinstance(self.cfg_train["image_size"], int):
            self.cfg_train["image_size"] = [self.cfg_train["image_size"]] * 2
        self.img_size, self.val_img_size = [
            check_img_size(x, max(self.model.stride))
            for x in self.cfg_train["image_size"]
        ]
        self.cfg_train["world_size"] = (
            int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        )
        self.ema = ema
        self.validator = YoloValidator(
            self.model, self.val_dataloader, self.device, cfg
        )

    def _lr_function(self, x: float) -> float:
        return ((1 + math.cos(x * math.pi / self.cfg_train["epochs"])) / 2) * (
            1 - self.cfg_hyp["lrf"]
        ) + self.cfg_hyp["lrf"]

    def _init_optimizer(
        self,
    ) -> Tuple[List[optim.Optimizer], List[lr_scheduler.LambdaLR]]:
        """Initialize optimizer and scheduler."""
        self.nbs = 64
        self.cfg_hyp["weight_decay"] *= (
            self.cfg_train["batch_size"] * self.accumulate / self.nbs
        )
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

    def warmup(self, ni: int, epoch: int) -> None:
        """Warmup before training.

        Args:
            ni: number integrated batches.
        """
        x_intp = [0, self.num_warmups]
        self.accumulate = max(
            1,
            np.interp(ni, x_intp, [1, self.nbs / self.cfg_train["batch_size"]]).round(),
        )
        for optimizer in self.optimizer:
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x["lr"] = np.interp(
                    ni,
                    x_intp,
                    [
                        self.cfg_hyp["warmup_bias_lr"] if j == 2 else 0.0,
                        x["initial_lr"] * self._lr_function(epoch),
                    ],
                )
                if "momentum" in x:
                    x["momentum"] = np.interp(
                        ni,
                        x_intp,
                        [self.cfg_hyp["warmup_momentum"], self.cfg_hyp["momentum"]],
                    )

    def multi_scale(self, imgs: torch.Tensor) -> torch.Tensor:
        """Set for multi scale image training.

        Args:
            imgs: torch tensor images.

        Returns:
            Reshaped images with scale factor.
        """
        grid_size = max(self.model.stride)
        sz = (
            random.randrange(self.img_size * 0.5, self.img_size * 1.5 + grid_size)
            // grid_size
            * grid_size
        )

        scale_factor = sz / max(imgs.shape[2:])
        if scale_factor != 1:
            new_shape = [
                math.ceil(x * scale_factor / grid_size) * grid_size
                for x in imgs.shape[2:]
            ]
            imgs = F.interpolate(
                imgs, size=new_shape, mode="bilinear", align_corners=False
            )
        return imgs

    def print_intermediate_results(
        self,
        loss_items: torch.Tensor,
        t_shape: torch.Size,
        img_shape: torch.Size,
        epoch: int,
        batch_idx: int,
    ) -> str:
        """Print intermediate_results during training batches.

        Args:
            loss_items: loss items from model.
            t_shape: torch label shape.
            img_shape: torch image shape.
            epoch: current epoch.
            batch_idx: current batch index.

        Returns:
            string for print.
        """
        self.mloss = (self.mloss * batch_idx + loss_items) / (batch_idx + 1)
        mem = "%.3gG" % (
            torch.cuda.memory_reserved() / 1e9  # to GBs
            if torch.cuda.is_available()
            else 0
        )
        s = ("%10s" * 2 + "%10.4g" * 6) % (
            "%g/%g" % (epoch, self.epochs - 1),
            mem,
            *self.mloss,
            t_shape[0],
            img_shape[-1],
        )

        self.pbar.set_description(s)

        return s

    def training_step(
        self,
        train_batch: Tuple[
            torch.Tensor,
            torch.Tensor,
            Tuple[str, ...],
            Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...],
        ],
        batch_idx: int,
        epoch: int,
    ) -> torch.Tensor:
        """Train a step.

        Args:
            batch: batch data.
            batch_idx: batch index.
            epoch: current epoch.

        Returns:
            Result of loss function.
        """
        num_integrated_batches = batch_idx + len(self.train_dataloader) * epoch

        if num_integrated_batches <= self.num_warmups:
            self.warmup(num_integrated_batches, epoch)

        imgs, labels, paths, shapes = train_batch
        imgs = self.prepare_img(imgs)
        labels = labels.to(self.device)

        if self.cfg_train["multi_scale"]:
            imgs = self.multi_scale(imgs)

        with amp.autocast(enabled=self.cuda):
            pred = self.model(imgs)
            loss, loss_items = self.loss(pred, labels)
            if self.rank != -1:
                loss *= self.cfg_train["world_size"]

        # backward
        self.scaler.scale(loss).backward()

        # Optimize
        if num_integrated_batches % self.accumulate == 0:
            for optimizer in self.optimizer:
                self.scaler.step(optimizer)  # optimizer.step
                self.scaler.update()
                optimizer.zero_grad()
                if self.ema:
                    self.ema.update(self.model)

        if self.rank in [-1, 0]:
            # TODO(ulken94): Log intermediate results to wandb. And then, remove noqa.
            print_string = self.print_intermediate_results(  # noqa
                loss_items, labels.shape, imgs.shape, epoch, batch_idx
            )

            if num_integrated_batches < 3:
                # plot images.
                f_name = os.path.join(
                    self.cfg_train["log_dir"],
                    f"train_batch{num_integrated_batches}.jpg",
                )
                # TODO(ulken94): Log images to wandb. And then, remove noqa.
                result = plot_images(  # noqa
                    images=imgs, targets=labels, paths=paths, fname=f_name
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
        pass

    def validation(self) -> None:
        """Validate model."""
        self.validator.validation()

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

    def set_datasampler(self, epoch: int) -> None:
        """Set dataloader's sampler epoch."""
        if self.rank != -1:
            self.train_dataloader.sampler.set_epoch(epoch)

    def on_start_epoch(self, epoch: int) -> None:
        """Run on an epoch starts.

        Args:
            epoch: current epoch.
        """
        self.update_image_weights()
        self.set_datasampler(epoch)
        self.log_train_stats()
        self.set_trainloader_tqdm()
        self.mloss = torch.zeros(4, device=self.device)
        for optimizer in self.optimizer:
            optimizer.zero_grad()

    def on_end_epoch(self, epoch: int) -> None:
        """Run on an epoch ends.

        Args:
            epoch: current epoch.
        """
        for optimizer in self.optimizer:  # for tensorboard
            lr = [x["lr"] for x in optimizer.param_groups]  # noqa
        self.scheduler_step()
        if self.rank in [-1, 0] and self.ema is not None:
            self.update_ema_attr()

    def scheduler_step(self) -> None:
        """Update scheduler parameters."""
        for scheduler in self.scheduler:
            scheduler.step()

    def update_ema_attr(self, include: Optional[List[str]] = None) -> None:
        """Update ema attributes.

        Args:
            include: a list of string which contains attributes.
        """
        if not include:
            include = ["yaml", "nc", "hyp", "gr", "names", "stride"]
        if self.ema:
            self.ema.update_attr(self.model, include=include)

    def set_grad_scaler(self) -> amp.GradScaler:
        """Set GradScaler."""
        return amp.GradScaler(enabled=self.cuda)

    def set_trainloader_tqdm(self) -> None:
        """Set tqdm object of train dataloader."""
        self.pbar = tqdm(
            enumerate(self.train_dataloader), total=len(self.train_dataloader)
        )

    def log_train_stats(self) -> None:
        """Log train information table headers."""
        LOGGER.info(
            ("\n" + "%10s" * 8)
            % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "targets", "img_size")
        )

    def on_train_start(self) -> None:
        """Run on start training."""
        for scheduler in self.scheduler:
            scheduler.last_epoch = self.start_epoch - 1  # type: ignore

        self.scaler = self.set_grad_scaler()

    def log_train_cfg(self) -> None:
        """Log train configurations."""
        LOGGER.info(
            "Image sizes %g train, %g test\n"
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

    # def train(self, test_every_epoch: int = 1) -> None:
    #     """Train model.

    #     Args:
    #         test_every_epoch: validate model in every {test_every_epoch} epochs.
    #     """
    #     self.start_epoch = 0

    #     # train epochs
    #     for epoch in range(self.start_epoch, self.epochs):
    #         is_final_epoch = epoch + 1 == self.epochs
    #         self.on_start_epoch(epoch)
    #         self.model.train()

    #         # train batch
    #         for i, batch in self.pbar:
    #             _loss = self.training_step(batch, i, epoch)  # noqa
    #         self.on_end_epoch(epoch)
    #         if is_final_epoch or epoch % self.cfg_train["validate_period"] == 0:
    #             self.model.eval()
    #             self.validation()  # TODO(ulken94): Save model and save validation logs.
