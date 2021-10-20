"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import math
import os
import random
from copy import deepcopy
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
from scripts.utils.anchors import check_anchors
from scripts.utils.general import check_img_size, labels_to_image_weights
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.plot_utils import plot_images, plot_label_histogram
from scripts.utils.torch_utils import de_parallel
from scripts.utils.train_utils import YoloValidator

if TYPE_CHECKING:
    import wandb.sdk.wandb_run.Run
    from scripts.utils.torch_utils import ModelEMA

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

LOGGER = get_logger(__name__)


class YoloTrainer(AbstractTrainer):
    """YoloTrainer class."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        ema: Optional["ModelEMA"],
        device: torch.device,
        log_dir: str = "exp",
        incremental_log_dir: bool = False,
        wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    ) -> None:
        """Initialize YoloTrainer class.

        Args:
            model: yolo model to train.
            cfg: config.
            train_dataloader: dataloader for training.
            val_dataloader: dataloader for validation.
        """
        super().__init__(
            model,
            cfg,
            train_dataloader,
            val_dataloader,
            device=device,
            log_dir=log_dir,
            incremental_log_dir=incremental_log_dir,
            wandb_run=wandb_run,
        )

        self.ema = ema
        self.best_score = 0.0
        self.loss = ComputeLoss(self.model)
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.cfg_train["batch_size"]), 1)
        self.optimizer, self.scheduler = self._init_optimizer()
        self.val_maps = np.zeros(self.model.nc)  # map per class
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

        if self.val_dataloader is not None:
            self.validator = YoloValidator(
                self.model if self.ema is None else self.ema.ema,
                self.val_dataloader,
                self.device,
                cfg,
                log_dir=self.log_dir,
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
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
            f"{len(pg0)} weight, {len(pg1)} weight (no decay), {len(pg2)} bias"
        )

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_function)

        pretrained = self.cfg_train.get("weights", "").endswith(".pt")
        if pretrained:
            ckpt = torch.load(self.cfg_train["weights"])
            if ckpt["optimizer"] is not None:
                optimizer.load_state_dict(ckpt["optimizer"][0])
                self.best_score = ckpt["best_score"]

            if self.ema and ckpt.get("ema"):
                self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
                self.ema.updates = ckpt["updates"]

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

        if self.pbar:
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
            if RANK != -1:
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

        if RANK in [-1, 0]:
            # TODO(ulken94): Log intermediate results to wandb. And then, remove noqa.
            print_string = self.print_intermediate_results(  # noqa
                loss_items, labels.shape, imgs.shape, epoch, batch_idx
            )

            if num_integrated_batches < 3:
                # plot images.
                f_name = os.path.join(
                    self.log_dir, f"train_batch{num_integrated_batches}.jpg",
                )
                # TODO(ulken94): Log images to wandb. And then, remove noqa.
                result = plot_images(  # noqa
                    images=imgs, targets=labels, paths=paths, fname=f_name
                )

        self.log_dict({"step_loss": loss[0].item()})

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

    def _save_weights(self, epoch: int, w_name: str) -> None:
        if RANK in [-1, 0]:
            ckpt = {
                "epoch": epoch,
                "best_score": self.best_score,
                "model": deepcopy(de_parallel(self.model)).half(),
                "optimizer": [optimizer.state_dict() for optimizer in self.optimizer],
            }
            if self.ema is not None:
                ckpt.update(
                    {"ema": deepcopy(self.ema.ema).half(), "updates": self.ema.updates}
                )

            torch.save(ckpt, os.path.join(self.wdir, w_name))
            del ckpt

    def log_dict(self, data: Dict[str, Any]) -> None:
        """Log dictionary data."""
        super().log_dict(data)
        self.update_loss()

    def update_loss(self) -> None:
        """Update train loss by `step_loss`."""
        if not self.state["is_train"]:
            return
        train_log = self.state["train_log"]
        if "loss" not in train_log:
            train_log["loss"] = 0
        train_log["loss"] += train_log["step_loss"]

    def validation(self) -> None:
        """Validate model."""
        if RANK in [-1, 0]:
            val_result = self.validator.validation()
            self.log_dict(
                {
                    "mP": val_result[0][0],
                    "mR": val_result[0][1],
                    "mAP50": val_result[0][2],
                    "mAP50_95": val_result[0][3],
                    "loss_box": val_result[0][4],
                    "loss_obj": val_result[0][5],
                    "loss_cls": val_result[0][6],
                    "mAP50_by_cls": {
                        k: val_result[1][i]
                        for i, k in enumerate(self.val_dataloader.dataset.names)
                    },
                }
            )

            self.val_maps = val_result[1]

            if val_result[0][2] > self.best_score:
                self.best_score = val_result[0][2]

            self._save_weights(self.current_epoch, "last.pt")

            # TODO(jeikeilim): Better metric to measure the best score so far.
            if val_result[0][2] == self.best_score:
                if self.wandb_run and RANK in [-1, 0]:
                    self.wandb_run.save(
                        os.path.join(self.wdir, "best.pt"), base_path=self.wdir
                    )
                self.best_score = val_result[0][2]
                self._save_weights(self.current_epoch, "best.pt")

    def update_image_weights(self) -> None:
        """Update image weights."""
        if self.cfg_train["image_weights"]:
            # Generate indices
            if RANK in [-1, 0]:
                # number of total images
                n_imgs = len(self.train_dataloader.dataset.img_files)

                # class weights
                class_weights = (
                    self.model.class_weights.cpu().numpy() * (1 - self.val_maps) ** 2
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
            if RANK != -1:
                indices = (
                    torch.tensor(self.train_dataloader.dataset.indices)
                    if RANK == 0
                    else torch.zeros(n_imgs)
                ).int()
                dist.broadcast(indices, 0)
                if RANK != 0:
                    self.train_dataloader.dataset.indices = indices.cpu().numpy()

    def set_datasampler(self, epoch: int) -> None:
        """Set dataloader's sampler epoch."""
        # if RANK != -1:
        #     self.train_dataloader.sampler.set_epoch(epoch)
        pass

    def on_epoch_start(self, epoch: int) -> None:
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

    def on_epoch_end(self, epoch: int) -> None:
        """Run on an epoch ends.

        Args:
            epoch: current epoch.
        """
        for optimizer in self.optimizer:  # for tensorboard
            lr = [x["lr"] for x in optimizer.param_groups]  # noqa

        self.scheduler_step()
        if RANK in [-1, 0] and self.ema is not None:
            self.update_ema_attr()
        # average the cumulated loss
        self.state["train_log"]["loss"] /= len(self.train_dataloader.dataset)

    def on_validation_end(self) -> None:
        """Run on validation end."""
        if self.state["val_log"]:
            self.state["val_log"].pop("mAP50_by_cls", None)

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
            include = ["yaml", "nc", "hyp", "gr", "names", "stride", "cfg"]
        if self.ema:
            self.ema.update_attr(self.model, include=include)

    def set_grad_scaler(self) -> amp.GradScaler:
        """Set GradScaler."""
        return amp.GradScaler(enabled=self.cuda)

    def set_trainloader_tqdm(self) -> None:
        """Set tqdm object of train dataloader."""
        if RANK in [-1, 0]:
            self.pbar = tqdm(
                enumerate(self.train_dataloader), total=len(self.train_dataloader)
            )

    def log_train_stats(self) -> None:
        """Log train information table headers."""
        if RANK in [-1, 0]:
            LOGGER.info(
                ("\n" + "%10s" * 8)
                % (
                    "Epoch",
                    "gpu_mem",
                    "box",
                    "obj",
                    "cls",
                    "total",
                    "targets",
                    "img_size",
                )
            )

    def on_train_start(self) -> None:
        """Run on start training."""
        # TODO(jeikeilim): Make load weight from wandb.
        labels = np.concatenate(self.train_dataloader.dataset.labels, 0)
        mlc = labels[:, 0].max()  # type: ignore
        nc = len(self.train_dataloader.dataset.names)
        # nb = len(self.train_dataloader)
        assert mlc < nc, (
            "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g"
            % (mlc, nc, self.cfg["data"], nc - 1)
        )

        grid_size = int(max(self.model.stride))  # type: ignore
        imgsz, _ = [check_img_size(x, grid_size) for x in self.cfg_train["image_size"]]

        if RANK in [-1, 0] and not self.cfg_train["resume"]:
            # c = torch.tensor(labels[:, 0])  # noqa
            plot_label_histogram(labels, save_dir=self.log_dir)

            if self.cfg_train["auto_anchor"]:
                check_anchors(
                    self.train_dataloader.dataset,
                    model=self.model,
                    thr=self.cfg_hyp["anchor_t"],
                    imgsz=imgsz,
                )

        for scheduler in self.scheduler:
            scheduler.last_epoch = self.start_epoch - 1  # type: ignore

        self.scaler = self.set_grad_scaler()

    def on_train_end(self) -> None:
        """Run on the end of the training."""
        self._save_weights(-1, "last.pt")

    def log_train_cfg(self) -> None:
        """Log train configurations."""
        if RANK in [-1, 0]:
            LOGGER.info(
                "Image sizes %g train, %g test\n"
                "Using %g dataloader workers\nLogging results to %s\n"
                "Starting training for %g epochs..."
                % (
                    self.cfg_train["image_size"][0],
                    self.cfg_train["image_size"][0],
                    self.train_dataloader.num_workers,
                    self.log_dir,
                    self.epochs,
                )
            )

    def log_wandb(self) -> None:
        """Log metrics to WanDB."""
        if not self.wandb_run:
            return
        wlogs = {
            "epoch": self.state["epoch"],
            "train_loss": self.state["train_log"]["loss"],
        }
        valid_log = self.state["val_log"]
        if valid_log:
            valid_loss = 0
            for key in valid_log:
                if key in ["mAP50_by_cls"]:
                    continue
                if key in ["loss_box", "loss_obj", "loss_cls"]:
                    loss = valid_log[key]
                    wlogs.update({"valid_" + key: loss})
                    valid_loss += loss  # ignoring weight for `box`, `obj`, `cls` losses
                wlogs.update({key: valid_log[key]})
            wlogs.update({"valid_loss": valid_loss})

        self.wandb_run.log(wlogs)
