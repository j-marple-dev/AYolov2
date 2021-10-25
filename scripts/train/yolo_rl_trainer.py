"""Module Description.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""
import math
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.loss.losses_rl import InfoNCELoss, RLLoss
from scripts.train.abstract_trainer import AbstractTrainer
from scripts.utils.general import get_logger, labels_to_image_weights
from scripts.utils.torch_utils import is_parallel

LOGGER = get_logger(__name__)


def de_parallel(model: nn.Module) -> nn.Module:
    """Decapsule parallelized model.

    Args:
        model: Single-GPU modle, DP model or DDP model
    Return:
        a decapsulized single-GPU model
    """
    return model.module if is_parallel(model) else model  # type: ignore


class YoloRepresentationLearningTrainer(AbstractTrainer):
    """YoloTrainer class."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        rank: int = -1,
        n_trans: int = 2,
        rl_type: str = "base",
        temperature: float = 0.07,
    ) -> None:
        """Initialize YoloTrainer class.

        Args:
            model: yolo model to train.
            cfg: config.
            train_dataloader: dataloader for training.
            val_dataloader: dataloader for validation.
            rank: CUDA rank for DDP.
            n_trans: the number of times to apply transformations for representation learning.
            rl_type: Representation Learning types (e.g. base, simclr)
            temperature: the value to adjust similarity scores.
                         e.g. # if the temperature is smaller than 1,
                              # similarity scores are enlarged than before.
                              # e.g. [100, 1] -> [1000, 10]
                              # It has an effect to train hard negative cases.
                              similarity_scores = np.array([100, 1])
                              temperature = 0.1
                              similarity_scores = similarity_scores / temperature
        """
        super().__init__(model, cfg, train_dataloader, val_dataloader, device=device)

        self.rl_type = rl_type
        if self.rl_type == "base":
            self.loss = RLLoss(ltype="L1Loss")
        elif self.rl_type == "simclr":
            self.loss = InfoNCELoss(
                batch_size=cfg["train"]["batch_size"],
                n_trans=cfg["train"]["n_trans"],
                device=device,
                temperature=temperature,
            )
        else:
            assert "Wrong Representation Learning type."
        self.nbs = 64
        self.accumulate = max(round(self.nbs / self.cfg_train["batch_size"]), 1)
        self.optimizer, self.scheduler = self._init_optimizer()
        self.rank = rank
        self.scaler: amp.GradScaler
        self.mloss: torch.Tensor
        self.num_warmups = max(
            round(self.cfg_hyp["warmup_epochs"] * len(self.train_dataloader)), 1e3
        )
        if isinstance(self.cfg_train["image_size"], int):
            self.cfg_train["image_size"] = [self.cfg_train["image_size"]] * 2
        self.cfg_train["world_size"] = (
            int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        )
        self.n_trans = n_trans
        self.save_dir = f"{self.cfg_train['log_dir']}/weights"
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def _lr_function(self, x: float) -> float:
        return ((1 + math.cos(x * math.pi / self.cfg_train["epochs"])) / 2) * (
            1 - self.cfg_hyp["lrf"]
        ) + self.cfg_hyp["lrf"]

    def _init_optimizer(
        self,
    ) -> Tuple[
        List[optim.Optimizer],
        Union[List[lr_scheduler.LambdaLR], List[lr_scheduler.CosineAnnealingLR]],
    ]:
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

        if self.rl_type == "base":
            lambda_scheduler = lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self._lr_function
            )
            return [optimizer], [lambda_scheduler]
        else:  # self.rl_type == "simclr"
            assert self.rl_type == "simclr", "Wrong Representation Learning type."
            cos_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=len(self.train_dataloader), eta_min=0, last_epoch=-1
            )
            return [optimizer], [cos_scheduler]

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
        s = ("%10s" * 2 + "%10.4g" * 3) % (
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

        imgs, _, _ = train_batch

        with amp.autocast(enabled=self.cuda):
            pred = self.model(imgs)
            loss, loss_items, pred_shape = self.loss(pred)
            loss_items = loss_items.to(self.device)
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

        if self.rank in [-1, 0]:
            # TODO(ulken94): Log intermediate results to wandb. And then, remove noqa.
            self.print_intermediate_results(  # noqa
                loss_items, pred_shape, imgs.shape, epoch, batch_idx
            )

        return loss.item()

    def validation_step(
        self,
        val_batch: Tuple[
            torch.Tensor,
            Tuple[str, ...],
            Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...],
        ],
        batch_idx: int,
    ) -> float:
        """Validate a step (a batch).

        Args:
            val_batch: validation data batch in tuple (input_x, true_y).
            batch_idx: current batch index.
        Returns:
            Result of loss function.
        """
        imgs, _, _ = val_batch
        pred = self.model(imgs)
        loss, loss_items, _ = self.loss(pred)
        return loss.item()

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
        self.mloss = torch.zeros(1, device=self.device)
        for optimizer in self.optimizer:
            optimizer.zero_grad()

    def on_end_epoch(self, epoch: int) -> None:
        """Run on an epoch ends.

        Args:
            epoch: current epoch.
        """
        self.scheduler_step()

    def scheduler_step(self) -> None:
        """Update scheduler parameters."""
        for scheduler in self.scheduler:
            scheduler.step()

    def set_grad_scaler(self) -> amp.GradScaler:
        """Set GradScaler."""
        return amp.GradScaler(enabled=self.cuda)

    def set_trainloader_tqdm(self) -> None:
        """Set tqdm object of train dataloader."""
        self.pbar = tqdm(
            enumerate(self.train_dataloader), total=len(self.train_dataloader)
        )

    def set_valloader_tqdm(self) -> None:
        """Set tqdm object of train dataloader."""
        self.pbar = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader))

    def log_train_stats(self) -> None:
        """Log train information table headers."""
        LOGGER.info(
            ("\n" + "%10s" * 5) % ("Epoch", "gpu_mem", "loss", "targets", "img_size")
        )

    def log_val_stats(self) -> None:
        """Log train information table headers."""
        LOGGER.info(("\n" + "%10s" * 3) % ("", "", "loss"))

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

    def train(self) -> None:
        """Train model."""
        self.on_train_start()

        self.model.to(self.device)
        min_loss = np.inf
        for epoch in range(self.start_epoch, self.epochs):
            is_final_epoch = epoch + 1 == self.epochs
            self.on_start_epoch(epoch)
            self.model.train()
            for i, batch in (
                self.pbar if self.pbar else enumerate(self.train_dataloader)
            ):
                self.training_step(batch, i, epoch)

            self.on_end_epoch(epoch)
            if is_final_epoch or epoch % self.cfg_train["validate_period"] == 0:
                self.model.eval()
                val_loss = self.validation()

                ckpt = {
                    "epoch": epoch,
                    "model": deepcopy(de_parallel(self.model)).half(),
                    "optimizer": self.optimizer[0].state_dict(),
                }
                torch.save(ckpt, f"{self.save_dir}/last.pt")
                torch.save(ckpt, f"{self.save_dir}/epoch_{str(epoch).zfill(3)}.pt")
                if val_loss < min_loss:
                    min_loss = val_loss
                    best_name = f"{self.save_dir}/best_e{str(epoch).zfill(3)}.pt"
                    torch.save(ckpt, best_name)

    @torch.no_grad()
    def validation(self) -> float:
        """Run validation.

        Returns:
            mloss: mean loss for validation data
        """
        self.set_valloader_tqdm()
        self.log_val_stats()
        total_loss = 0.0
        for i, batch in self.pbar if self.pbar else enumerate(self.val_dataloader):
            loss = self.validation_step(batch, i)
            total_loss += loss
            if i != len(self.val_dataloader) - 1:
                loss_str = ("%10s" * 2 + "%10.4g") % (
                    "",
                    "",
                    total_loss / (i + 1) * self.cfg_train["batch_size"],
                )
            else:
                loss_str = ("%10s" * 2 + "%10.4g") % (
                    "",
                    "",
                    total_loss / len(self.val_dataloader.dataset),
                )
            self.pbar.set_description(loss_str)

        mloss = total_loss / len(self.val_dataloader.dataset)
        return mloss
