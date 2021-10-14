"""Train utilities.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.loss.losses import ComputeLoss
from scripts.utils.general import scale_coords, xywh2xyxy
from scripts.utils.metrics import (ConfusionMatrix, ap_per_class, box_iou,
                                   non_max_suppression)


class AbstractValidator(ABC):
    """Model validator class."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        cfg: Dict[str, Any],
    ) -> None:
        """Initialize Validator class.

        Args:
            model: a torch model.
            dataloader: dataloader with validation dataset.
            device: torch device.
        """
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.cfg_train = cfg["train"]
        self.cfg_hyp = cfg["hyper_params"]

    @abstractmethod
    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        """Validate a batch."""
        pass

    @abstractmethod
    def validation(self, *args: Any, **kwargs: Any) -> Any:
        """Validate model."""
        pass


class YoloValidator(AbstractValidator):
    """YOLO model validator."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        cfg: Dict[str, Any],
        compute_loss: bool = True,
    ) -> None:
        """Initialize YoloValidator class.."""
        super().__init__(model, dataloader, device, cfg)
        self.class_map = list(range(self.model.nc))  # type: ignore
        self.names = {k: v for k, v in enumerate(self.dataloader.dataset.names)}  # type: ignore
        self.confusion_matrix: ConfusionMatrix
        self.statistics: Dict[str, Any]
        self.nc = 1 if self.cfg_train["single_cls"] else int(self.model.nc)  # type: ignore
        self.iouv = torch.linspace(0.5, 0.95, 10).to(
            self.device, non_blocking=True
        )  # IoU vecot
        self.niou = self.iouv.numel()
        if compute_loss:
            self.loss_fn = ComputeLoss(self.model)

        self.loss = torch.zeros(3, device=self.device)
        self.seen: int = 0

    def set_confusion_matrix(self) -> None:
        """Set confusion matrix."""
        self.confusion_matrix = ConfusionMatrix(nc=self.model.nc)

    def init_statistics(self) -> None:
        """Set statistics default."""
        self.statistics = {
            "dt": [0.0, 0.0, 0.0],
            "p": 0.0,
            "r": 0.0,
            "f1": 0.0,
            "mp": 0.0,
            "mr": 0.0,
            "map50": 0.0,
            "map": 0.0,
            "jdict": [],
            "stats": [],
            "ap": [],
            "ap_class": [],
        }

    def init_attrs(self, s: str) -> None:
        """Initialize attributes before validation."""
        self.set_confusion_matrix()
        self.init_statistics()
        self.seen = 0
        self.tqdm = tqdm(enumerate(self.dataloader), desc=s)

    def prepare_img(self, img: torch.Tensor) -> torch.Tensor:
        """Prepare img for model."""
        img = img.to(self.device, non_blocking=True)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        return img

    def run_nms(
        self,
        out: torch.Tensor,
        targets: torch.Tensor,
        width: int,
        height: int,
        nb: int,
        save_hybrid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Run non-max-suppression.

        Args:
            out: model output.
            targets: target label.
            width: image width.
            height: image height.
            nb: batch size.
            save_hybrid: save hybrid or not

        Returns:
            a tuple with [nms_out, modified_target, time]
        """
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(
            self.device, non_blocking=True
        )
        label = (
            [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        )
        t3 = time.time()
        out = non_max_suppression(
            out,
            self.cfg_hyp["conf_t"],
            self.cfg_hyp["iou_t"],
            labels=label,
            multi_label=True,
            agnostic=self.cfg_train["single_cls"],
        )
        self.statistics["dt"][2] += time.time() - t3

        return (out, targets, t3)

    def compute_loss(self, train_out: torch.Tensor, targets: torch.Tensor) -> None:
        """Compute loss.

        Args:
            train_out: output from model (detected)
            targets: target labels.
        """
        self.loss += self.loss_fn([x.float() for x in train_out], targets)[1][:3]

    def process_batch(
        self,
        detections: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """Return correct predictions matrix.

        Both sets of boxes are in (x1, y1, x2, y2) format.
        Args:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        correct = torch.zeros(
            detections.shape[0],
            self.iouv.shape[0],
            dtype=torch.bool,
            device=self.iouv.device,
        )
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where(
            (iou >= self.iouv[0]) & (labels[:, 0:1] == detections[:, 5])
        )  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(self.iouv.device, non_blocking=True)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= self.iouv
        return correct

    def statistics_per_image(
        self,
        img: torch.Tensor,
        out: list,
        targets: torch.Tensor,
        shapes: tuple,
        paths: tuple,
    ) -> None:
        """Calculate statistics per image.

        Args:
            img: input image.
            out: model output of input image (img).
            targets: target label.
            shapes: batch image shape.
            paths: image path.
        """
        for si, pred in enumerate(out):

            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]  # noqa
            self.seen += 1

            if len(pred) == 0:
                if nl:
                    self.statistics["stats"].append(
                        (
                            torch.zeros(0, self.niou, dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            tcls,
                        )
                    )
                continue

            # Predictions
            if self.cfg_train["single_cls"]:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape)  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape)  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = self.process_batch(predn, labelsn)
                if self.cfg_train["plot"]:
                    self.confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool)
            self.statistics["stats"].append(
                (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
            )  # (correct, conf, pcls, tcls)

            # Save/log
            # if save_txt:
            #     save_one_txt(predn, save_conf, shape, file=self.cfg_train["log_dir"] / 'labels' / (path.stem + '.txt'))
            # if save_json:
            #     save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            # callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

    def validation_step(  # type: ignore
        self,
        val_batch: Tuple[
            torch.Tensor,
            torch.Tensor,
            Tuple[str, ...],
            Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...],
        ],
        batch_idx: int,
    ) -> None:
        """Validate a step.

        Args:
            val_batch: a validation batch.
            batch_idx: batch index.
        """
        imgs, targets, paths, shapes = val_batch
        t1 = time.time()
        imgs = self.prepare_img(imgs)
        targets = targets.to(self.device, non_blocking=True)
        batch_size, _, height, width = imgs.shape
        t2 = time.time()
        self.statistics["dt"][0] += t2 - t1

        # Run model
        out, train_out = self.model(imgs)  # inference and training outputs
        self.statistics["dt"][1] += time.time() - t2

        # Compute loss
        if self.loss_fn:
            self.compute_loss(train_out, targets)

        out, targets, t3 = self.run_nms(out, targets, width, height, batch_size)
        self.statistics_per_image(imgs, out, targets, shapes, paths)

        # TODO(ulken94): Plot images.

    def compute_statistics(self) -> None:
        """Compute statistics for dataset."""
        self.statistics["stats"] = [
            np.concatenate(x, 0) for x in zip(*self.statistics["stats"])
        ]  # to numpy
        if len(self.statistics["stats"]) and self.statistics["stats"][0].any():
            (
                self.statistics["p"],
                self.statistics["r"],
                self.statistics["ap"],
                self.statistics["f1"],
                self.statistics["ap_class"],
            ) = ap_per_class(
                *self.statistics["stats"],
                plot=False,
                save_dir=self.cfg_train["log_dir"],
                names=self.names,
            )
            self.statistics["ap50"], self.statistics["ap"] = (
                self.statistics["ap"][:, 0],
                self.statistics["ap"].mean(1),
            )  # AP@0.5, AP@0.5:0.95
            (
                self.statistics["mp"],
                self.statistics["mr"],
                self.statistics["map50"],
                self.statistics["map"],
            ) = (
                self.statistics["p"].mean(),
                self.statistics["r"].mean(),
                self.statistics["ap50"].mean(),
                self.statistics["ap"].mean(),
            )
            self.statistics["nt"] = np.bincount(
                self.statistics["stats"][3].astype(np.int64), minlength=self.nc
            )  # number of targets per class
        else:
            self.statistics["nt"] = torch.zeros(1)

    def print_results(self, verbose: bool = True) -> tuple:
        """Print validation results.

        Args:
            verbose: print validation result per class or not.

        Returns:
            a tuple with dt for statistics.
        """
        # print result
        pf = "%20s" + "%11i" * 2 + "%11.3g" * 4  # print format
        print(
            pf
            % (
                "all",
                self.seen,
                self.statistics["nt"].sum(),
                self.statistics["mp"],
                self.statistics["mr"],
                self.statistics["map50"],
                self.statistics["map"],
            )
        )

        # print result per class
        if (
            (verbose or (self.nc < 50 and not self.model.training))
            and self.nc > 1
            and len(self.statistics["stats"])
        ):
            for i, c in enumerate(self.statistics["ap_class"]):
                print(
                    pf
                    % (
                        self.names[c],
                        self.seen,
                        self.statistics["nt"][c],
                        self.statistics["p"][i],
                        self.statistics["r"][i],
                        self.statistics["ap50"][i],
                        self.statistics["ap"][i],
                    )
                )
        # print speed
        t = tuple(
            x / self.seen * 1e3 for x in self.statistics["dt"]
        )  # speeds per image
        if not self.model.training:
            shape = (
                self.cfg_train["batch_size"],
                3,
                self.cfg_train["image_size"],
                self.cfg_train["image_size"],
            )
            print(
                f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}"
                % t
            )
        return t

    @torch.no_grad()
    def validation(self) -> Tuple[Tuple[list, ...], np.ndarray, tuple]:  # type: ignore
        """Validate model."""
        s = ("%20s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Labels",
            "P",
            "R",
            "mAP@.5",
            "mAP@.5:.95",
        )
        self.init_attrs(s)
        # dt, precision, recall, f1 score, mean-precision, mean-recall, mAP@.5, mAP@.5:.95

        for batch_i, batch in self.tqdm:
            self.validation_step(batch, batch_i)
        self.compute_statistics()
        t = self.print_results()
        maps = np.zeros(self.nc) + self.statistics["map"]
        for i, c in enumerate(self.statistics["ap_class"]):
            maps[c] = self.statistics["ap"][i]

        return (
            (
                self.statistics["mp"],
                self.statistics["mr"],
                self.statistics["map50"],
                self.statistics["map"],
                *(self.loss.cpu() / len(self.dataloader)).tolist(),
            ),
            maps,
            t,
        )
