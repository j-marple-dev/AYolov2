"""Knowledge Distillation Trainer module.

- Author: Hyung-Seok Shin
- Contact: hsshin@jmarple.ai
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import wandb.sdk.wandb_run.Run

import math
import os
import threading
import time
from copy import deepcopy

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.augmentation.augmentation import MultiAugmentationPolicies
from scripts.loss.losses import ComputeLoss
from scripts.train.abstract_trainer import AbstractTrainer
from scripts.utils.general import check_img_size, xyxy2xywh
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.metrics import non_max_suppression
from scripts.utils.plot_utils import plot_images
from scripts.utils.train_utils import YoloValidator

LOGGER = get_logger(__name__)
DEVICE = torch.device("cpu")


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
        teacher_device: torch.device = DEVICE,
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

        # self.teacher = teacher.to(self.device).eval()
        self.teacher_device = teacher_device
        self.teacher = teacher.to(self.teacher_device).eval()
        self.unlabeled_dataloader = unlabeled_dataloader
        # TODO(hsshin): shuffling does not work. fix it.
        # self.unlabeled_dataloader.dataset.shuffle()
        self.unlabeled_iterator = iter(self.unlabeled_dataloader)

        self.best_score = 0.0
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
            self.augment = MultiAugmentationPolicies(policy)
        except KeyError:
            self.augment = None
            LOGGER.warn(
                "No augmentation policy is specified for pseudo-labeled images."
            )

        # TODO(hsshin): revisit here
        self.nms_conf_thr: float = 0.4
        self.nms_iou_thr: float = 0.7

        self.conf_thr: float = 0.9
        self.bbox_size_thr: float = 20

        self.pseudo_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []

        if self.val_dataloader is not None:
            self.validator = YoloValidator(
                self.model, self.val_dataloader, self.device, cfg, log_dir=self.log_dir,
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

        # Parallel batch preparation for pseudo label and image.
        thread1 = threading.Thread(target=self.get_pseudo_labeled_batch,)

        if len(self.pseudo_buffer) < 3:
            thread1.start()

        # unlabeled_imgs, pseudo_labels = self.get_pseudo_labeled_batch()
        if len(self.pseudo_buffer) == 0:
            threading.Thread(target=self.get_pseudo_labeled_batch,).start()
            thread1.join()  # Wait until the buffer has at least one data.

        # Take pseudo image and label in pseudo_buffer.
        unlabeled_imgs, pseudo_labels = self.pseudo_buffer.pop(0)

        # compute loss for labeled data
        imgs, labels, _, _ = batch
        batch_size = imgs.shape[0]
        imgs = self.prepare_img(imgs)

        imgs = torch.cat((imgs, unlabeled_imgs.to(self.device)))

        pred = self.model(imgs)

        pred_origin = [pred[i][:batch_size] for i in range(len(pred))]
        pred_pseudo = [pred[i][batch_size:] for i in range(len(pred))]

        loss, loss_items = self.loss(pred_origin, labels.to(self.device))
        pseudo_loss, pseudo_loss_items = self.loss(
            pred_pseudo, pseudo_labels.to(self.device)
        )

        # total loss as a weighted sum
        total_loss = loss + self.pseudo_loss_weight * pseudo_loss
        total_loss.backward()

        # Optimize
        if num_integrated_batches % self.accumulate == 0:
            for optimizer in self.optimizer:
                optimizer.step()
                optimizer.zero_grad()

        # # TODO(ulken94): Log intermediate results to wandb. And then, remove noqa.
        # self.print_intermediate_results(  # noqa
        #     loss_items, labels.shape, imgs.shape, epoch, batch_idx
        # )

        self.log_dict({"step_loss": total_loss[0].item()})

        return total_loss[0]

    def _lr_function(self, x: float, schedule_type: str = "cosine") -> float:
        """Learning rate scheduler function."""
        if schedule_type == "cosine":
            return ((1 + math.cos(x * math.pi / self.cfg_train["epochs"])) / 2) * (
                1 - self.cfg_hyp["lrf"]
            ) + self.cfg_hyp["lrf"]
        else:
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

    def validation(self) -> None:
        """Validate model."""
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
            if self.wandb_run:
                self.wandb_run.save(
                    os.path.join(self.wdir, "best.pt"), base_path=self.wdir
                )
            self.best_score = val_result[0][2]
            self._save_weights(self.current_epoch, "best.pt")

    def on_validation_end(self) -> None:
        """Run on validation end."""
        if self.state["val_log"]:
            self.state["val_log"].pop("mAP50_by_cls", None)

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

    ####################################################
    # End of Override
    ####################################################
    @torch.no_grad()
    def get_pseudo_labeled_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct pseudo-labels using the prediction of teacher."""
        # TODO: implement `get_batch` method for unlabeled dataset
        while True:
            try:
                weak_augmented_batch = next(self.unlabeled_iterator)
                break
            except StopIteration:
                # self.unlabeled_dataloader.dataset.shuffle()
                self.unlabeled_iterator = iter(self.unlabeled_dataloader)
                weak_augmented_batch = next(self.unlabeled_iterator)
                break
            except ValueError:
                time.sleep(0.1)  # Wait until another thread has finished next()
                continue

        imgs, _, paths, _ = weak_augmented_batch  # img, labels, paths, shapes
        # imgs = self.prepare_img(imgs)
        imgs = imgs.to(self.teacher_device) / 255.0
        teacher_predicts_aggregated, _ = self.teacher(imgs)

        preds_after_nms = non_max_suppression(
            teacher_predicts_aggregated,
            conf_thres=self.nms_conf_thr,
            iou_thres=self.nms_iou_thr,
        )

        labels_yolo = self.prepare_labels_for_augmention(
            preds_after_nms, thr=self.conf_thr, min_size=self.bbox_size_thr
        )

        if not self.augment:
            bat_ids_cls_ids_bboxes: List[np.ndarray] = []
            for idx, cls_ids_bboxes in enumerate(labels_yolo):
                batch_ids = np.array([idx] * len(cls_ids_bboxes))
                bat_ids_cls_ids_bboxes.append(
                    np.hstack([batch_ids[:, np.newaxis], cls_ids_bboxes])
                )
            total_bat_ids_cls_ids_bboxes: np.ndarray = np.vstack(bat_ids_cls_ids_bboxes)
        else:
            # Strong augmented imgs and labels
            augmented_cls_id_bboxes: List[np.ndarray] = []
            augmented_imgs: List[np.ndarray] = []
            imgs *= 255.0
            imgs_np = imgs.cpu().numpy().astype(np.uint8)
            imgs_np = imgs_np.transpose((0, 2, 3, 1))  # (batch, H, W, C)
            for idx, (img, cls_ids_bboxes) in enumerate(zip(imgs_np, labels_yolo)):
                augmented_img, cls_id_bboxes = self.augment(img, cls_ids_bboxes)
                augmented_imgs.append(augmented_img)
                # add `batch idx` column
                batch_ids = np.array([idx] * len(cls_id_bboxes))
                augmented_cls_id_bboxes.append(
                    np.hstack([batch_ids[:, np.newaxis], cls_id_bboxes])
                )
            imgs_np = (
                np.stack(augmented_imgs, axis=0)
                .transpose((0, 3, 1, 2))
                .astype(np.float32)
            )
            imgs = torch.Tensor(imgs_np) / 255.0
            total_bat_ids_cls_ids_bboxes = np.vstack(augmented_cls_id_bboxes)

        labels = torch.Tensor(total_bat_ids_cls_ids_bboxes)
        if self.debug:
            # plot images.
            f_name = os.path.join(self.log_dir, "strong_augmented_batch.jpg",)
            plot_images(  # noqa
                images=imgs.clone(),
                targets=total_bat_ids_cls_ids_bboxes,
                paths=paths,
                fname=f_name,
            )
            LOGGER.info(f"Strong augmented image is saved in {f_name}")
            self.debug = False

        self.pseudo_buffer.append((imgs, labels))
        return imgs, labels

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
            if len(bbox) == 0:
                cls_ids_bboxes.append(np.zeros((0, 5)))
                continue

            cls_ids = label.cpu().numpy()  # type: ignore
            bboxes = bbox.cpu().numpy()
            # Transform to normal coord (for Albumentation yolo format)
            # NOTE(hsshin): work only for square image
            bboxes /= whwh
            bboxes.clip(min=0, max=1, out=bboxes)
            # Transform to YOLO format (xyxy > xywh)
            bboxes = xyxy2xywh(bboxes)

            cls_ids_bboxes.append(np.hstack([cls_ids[:, np.newaxis], bboxes]))  # (n, 5)

        return cls_ids_bboxes

    @staticmethod
    def filter_invalid(
        bbox: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        thr: float = 0.0,
        min_size: float = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

    def _save_weights(self, epoch: int, w_name: str) -> None:
        ckpt = {
            "epoch": epoch,
            "best_score": self.best_score,
            "model": deepcopy(self.model).half(),
            "optimizer": [optimizer.state_dict() for optimizer in self.optimizer],
        }
        torch.save(ckpt, os.path.join(self.wdir, w_name))
        del ckpt
