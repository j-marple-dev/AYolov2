"""Knowledge Distillation Trainer module.

- Author: Hyung-Seok Shin
- Contact: hsshin@jmarple.ai
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import wandb.sdk.wandb_run.Run

import math

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader

from scripts.loss.losses import ComputeLoss
from scripts.train.abstract_trainer import AbstractTrainer
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.metrics import non_max_suppression

LOGGER = get_logger(__name__)


class SoftTeacherTrainer(AbstractTrainer):
    """Soft Teacher trainer.

    Ref: End-to-end semi-supervised object detection with soft teacher
    """

    def __init__(
        self,
        model: nn.Module,
        teacher: nn.Module,
        cfg: Dict[str, Any],
        train_dataloader: DataLoader,
        unlabeled_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        device: torch.device,
        log_dir: str = "exp",
        incremental_log_dir: bool = False,
        wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    ) -> None:
        """Initialize Soft Teacher trainer class."""
        super().__init__(
            model,
            cfg,
            train_dataloader,
            val_dataloader,
            device,
            log_dir,
            incremental_log_dir,
            wandb_run,
        )
        self.debug = True

        self.teacher = teacher.to(self.device).eval()
        self.unlabeled_dataloader = unlabeled_dataloader
        self.unlabeled_iterator = iter(unlabeled_dataloader)

        self.num_warmups = max(
            round(self.cfg_hyp["warmup_epochs"] * len(self.train_dataloader)), 1e3
        )

        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.cfg_train["batch_size"]), 1)
        self.optimizer, self.scheduler = self._init_optimizer()

        self.optimizer, self.scheduler = self._init_optimizer()
        self.loss = ComputeLoss(self.model)

    ####################################################
    # Abstact methods
    ####################################################
    def training_step(
        self,
        batch: Union[List[torch.Tensor], torch.Tensor, Tuple[torch.Tensor, ...]],
        batch_idx: int,
        epoch: int,
    ) -> torch.Tensor:
        """Train a step.

        Args:
            batch: batch data.
            batch_idx: batch index.

        Returns:
            Result of loss function.
        """
        import pdb

        pdb.set_trace()

        num_integrated_batches = batch_idx + len(self.train_dataloader) * epoch

        if num_integrated_batches <= self.num_warmups:
            self.warmup(num_integrated_batches, epoch)

        # compute loss for pseudo-labeled data
        unlabeled_imgs, pseudo_labels = self.get_pseudo_labeled_batch()

        # compute loss for labeled data
        imgs, labels, _, _ = batch
        imgs = self.prepare_img(imgs)
        if unlabeled_imgs is not None:
            imgs = torch.stack([imgs, unlabeled_imgs])
            labels = torch.stack([labels, pseudo_labels])

        labels = labels.to(self.device)
        with amp.autocast(enabled=self.cuda):
            pred = self.model(imgs)
            loss, loss_items = self.loss(pred, labels)

    def _lr_function(self, x: float, schedule_type: str = "") -> float:
        """Learning rate scheduler function."""
        if schedule_type == "cosine":
            if self.debug:
                print("Use Cosine Annealing.")
            return ((1 + math.cos(x * math.pi / self.cfg_train["epochs"])) / 2) * (
                1 - self.cfg_hyp["lrf"]
            ) + self.cfg_hyp["lrf"]
        else:
            if self.debug:
                print("Use Identity scheduler.")
            return x

    def _init_optimizer(self) -> Any:
        """Initialize optimizer."""
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

        # TODO(ulken94): Need to check weights from wandb.
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

    ####################################################
    # End of Abstract methods
    ####################################################

    ####################################################
    # Overrided methods
    ####################################################

    ####################################################
    # End of Override
    ####################################################
    @torch.no_grad()
    def get_pseudo_labeled_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct pseudo-labels using prediction of teacher."""
        # TODO: implement `get_batch` method for unlabeled dataset
        try:
            weak_augmented_batch = next(self.unlabeled_iterator)
        except StopIteration:
            self.unlabeled_iterator = iter(self.unlabeled_dataloader)
            weak_augmented_batch = next(self.unlabeled_iterator)

        imgs, _, _, _ = weak_augmented_batch  # img, path, shapes
        imgs = self.prepare_img(imgs)
        with amp.autocast(enabled=self.cuda):
            teacher_predicts_aggregated, _ = self.teacher(imgs)
        preds_after_nms = non_max_suppression(
            teacher_predicts_aggregated, conf_thres=0.25, iou_thres=0.45
        )

        pseudo_boxes_cls = self.score_filter(preds_after_nms)
        pseudo_boxes_reg = self.box_regression_variance_filter(preds_after_nms)

        # Strong augmented imgs and labels

        return 0, 0

    def score_filter(self, preds: List[torch.Tensor]) -> torch.Tensor:
        """Filter predicted bounding boxes."""
        # NMS
        # TODO: modify hardcoded params
        # filtered_bboxes = filter_invalid()

        pseudo_boxes_cls = None
        return pseudo_boxes_cls

    @staticmethod
    def filter_invalid(
        bbox: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        thr: float = 0.0,
        min_size: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def box_regression_variance_filter(self, preds: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        pseudo_boxes_reg = None
        return pseudo_boxes_reg

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


if __name__ == "__main__":
    trainer = SoftTeacherTrainer(
        model,
        teacher,
        cfg,
        train_dataloader,
        unlabeled_dataloader,
        val_dataloader,
        device,
        log_dir="debug",
    )
