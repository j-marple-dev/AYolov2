"""Knowledge Distillation Trainer module.

- Author: Hyung-Seok Shin
- Contact: hsshin@jmarple.ai
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import wandb.sdk.wandb_run.Run

import math
import os

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.augmentation.augmentation import AugmentationPolicy
from scripts.loss.losses import ComputeLoss
from scripts.train.abstract_trainer import AbstractTrainer
from scripts.utils.general import check_img_size, xyxy2xywh
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.metrics import non_max_suppression
from scripts.utils.plot_utils import plot_images

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
        # TODO(hsshin): ad-hoc attribute for debugging
        self.debug = True

        self.teacher = teacher.to(self.device).eval()
        self.unlabeled_dataloader = unlabeled_dataloader
        self.unlabeled_iterator = iter(unlabeled_dataloader)

        self.mloss: torch.Tensor  # mean loss
        self.num_warmups = max(
            round(self.cfg_hyp["warmup_epochs"] * len(self.train_dataloader)), 1e3
        )
        if isinstance(self.cfg_train["image_size"], int):
            self.cfg_train["image_size"] = [self.cfg_train["image_size"]] * 2

        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.cfg_train["batch_size"]), 1)
        self.optimizer, self.scheduler = self._init_optimizer()

        self.loss = ComputeLoss(self.model)

        self.pseudo_loss_weight = 0.5
        try:
            policy = cfg["strong_augmentation"]
            self.augment = AugmentationPolicy(policy)
        except KeyError:
            self.augment = None
            LOGGER.warn(
                "No augmentation policy is specified for pseudo-labeled images."
            )

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
        num_integrated_batches = batch_idx + len(self.train_dataloader) * epoch

        if num_integrated_batches <= self.num_warmups:
            self.warmup(num_integrated_batches, epoch)

        # compute loss for labeled data
        imgs, labels, _, _ = batch
        imgs = self.prepare_img(imgs)

        labels = labels.to(self.device)
        pred = self.model(imgs)
        loss, loss_items = self.loss(pred, labels)

        # compute loss for pseudo-labeled data
        unlabeled_imgs, pseudo_labels = self.get_pseudo_labeled_batch()
        pseudo_pred = self.model(unlabeled_imgs.to(self.device))
        pseudo_loss, pseudo_loss_items = self.loss(
            pseudo_pred, pseudo_labels.to(self.device)
        )

        # total loss as a weighted sum
        total_loss = loss + self.pseudo_loss_weight * pseudo_loss

        # backward
        total_loss.backward()

        # Optimize
        if num_integrated_batches % self.accumulate == 0:
            for optimizer in self.optimizer:
                optimizer.step()
                optimizer.zero_grad()

        # TODO(ulken94): Log intermediate results to wandb. And then, remove noqa.
        self.print_intermediate_results(  # noqa
            loss_items, labels.shape, imgs.shape, epoch, batch_idx
        )

        self.log_dict({"step_loss": total_loss[0].item()})

        return total_loss[0]

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

        for scheduler in self.scheduler:
            scheduler.last_epoch = self.start_epoch - 1  # type: ignore

    def on_train_end(self) -> None:
        """Run on the end of the training."""
        self._save_weights(-1, "last.pt")

    ####################################################
    # End of Override
    ####################################################
    @torch.no_grad()
    def get_pseudo_labeled_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct pseudo-labels using the prediction of teacher."""
        # TODO: implement `get_batch` method for unlabeled dataset
        try:
            weak_augmented_batch = next(self.unlabeled_iterator)
        except StopIteration:
            self.unlabeled_iterator = iter(self.unlabeled_dataloader)
            weak_augmented_batch = next(self.unlabeled_iterator)

        imgs, _, paths, _ = weak_augmented_batch  # img, labels, paths, shapes
        imgs = self.prepare_img(imgs)
        teacher_predicts_aggregated, _ = self.teacher(imgs)

        # TODO(hsshin): modify hard coded NMS params
        preds_after_nms = non_max_suppression(
            teacher_predicts_aggregated, conf_thres=0.25, iou_thres=0.45,
        )

        labels_yolo = self.prepare_labels_for_augmention(preds_after_nms, thr=0.26)

        if not self.augment:
            # TODO(hsshin) implement this case
            return None

        # Strong augmented imgs and labels
        augmented_imgs = []
        augmented_cls_id_bboxes = []
        imgs_np = imgs.cpu().numpy()
        imgs_np = (255 * imgs_np).astype(np.uint8)
        imgs_np = imgs_np.transpose((0, 2, 3, 1))  # (batch, H, W, C)
        for idx, (img, cls_ids_bboxes) in enumerate(zip(imgs_np, labels_yolo)):
            augmented_img, cls_id_bboxes = self.augment(img, cls_ids_bboxes)
            augmented_imgs.append(augmented_img)
            # add `batch idx` column
            batch_ids = np.array([idx] * len(cls_id_bboxes))
            augmented_cls_id_bboxes.append(
                np.hstack([batch_ids[:, np.newaxis], cls_id_bboxes])
            )

        batch_imgs = (
            np.stack(augmented_imgs, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
        )
        batch_imgs /= 255.0
        batch_imgs = torch.Tensor(batch_imgs)
        batch_cls_id_bboxes = np.vstack(augmented_cls_id_bboxes)
        batch_cls_id_bboxes = torch.Tensor(batch_cls_id_bboxes)

        if self.debug:
            # plot images.
            f_name = os.path.join(self.log_dir, "strong_augmented_batch.jpg",)
            plot_images(  # noqa
                images=batch_imgs,
                targets=batch_cls_id_bboxes,
                paths=paths,
                fname=f_name,
            )
            self.debug = False

        return batch_imgs, batch_cls_id_bboxes

    def prepare_labels_for_augmention(
        self, preds: List[torch.Tensor], thr: float = 0.0, min_size: float = 0.0
    ) -> List[np.ndarray]:
        """Filter out bboxes with an optional criterion and convert to yolo format."""
        width, height = self.cfg_train["image_size"]
        whwh = np.array([width, height, width, height])
        # NMS
        cls_ids_bboxes = []
        for pred in preds:
            bbox, label = self.filter_invalid(
                bbox=pred[:, :4],
                label=pred[:, 5],
                score=pred[:, 4],
                thr=thr,
                min_size=min_size,
            )
            cls_ids = label.cpu().numpy()
            bboxes = bbox.cpu().numpy()
            # Transform to normal coord (for Albumentation yolo format)
            # NOTE(hsshin): work only for square image
            bboxes /= whwh
            bboxes.clip(min=0, max=1, out=bboxes)
            # Transform to YOLO format (xyxy > xywh)
            bboxes = xyxy2xywh(bboxes)

            cls_ids_bboxes.append(np.hstack([cls_ids[:, np.newaxis], bboxes]))

        return cls_ids_bboxes

    @staticmethod
    def filter_invalid(
        bbox: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        thr: float = 0.0,
        min_size: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter out invalid bboxes."""
        if score is not None:
            valid = score > thr
            bbox = bbox[valid]
            if label is not None:
                label = label[valid]

        if min_size is not None:
            bw = bbox[:, 2] - bbox[:, 0]
            bh = bbox[:, 3] - bbox[:, 1]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
            if label is not None:
                label = label[valid]

        return bbox, label

    # def box_regression_variance_filter(self, preds: torch.Tensor) -> torch.Tensor:
    #     # TODO: implement
    #     pseudo_boxes_reg = None
    #     return pseudo_boxes_reg

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

    def on_epoch_start(self, epoch: int) -> None:
        """Run on an epoch starts."""
        LOGGER.info(
            ("\n" + "%10s" * 8)
            % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "targets", "img_size",)
        )
        self.pbar = tqdm(
            enumerate(self.train_dataloader), total=len(self.train_dataloader)
        )
        self.mloss = torch.zeros(4, device=self.device)

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
