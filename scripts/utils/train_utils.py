"""Train utilities.

- Author: Haneol Kim, Jongkuk Lim
- Contact: hekim@jmarple.ai, limjk@jmarple.ai
"""
import importlib
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.loss.losses import ComputeLoss
from scripts.utils.general import increment_path, scale_coords, xywh2xyxy
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.metrics import (ConfusionMatrix, ap_per_class, box_iou,
                                   non_max_suppression)

if importlib.util.find_spec("tensorrt") is not None:
    from scripts.utils.tensorrt_runner import TrtWrapper  # noqa: F401

LOGGER = get_logger(__name__)


class AbstractValidator(ABC):
    """Model validator class."""

    def __init__(
        self,
        model: Union[nn.Module, "TrTWrapper"],  # type: ignore  # noqa: F821
        dataloader: DataLoader,
        device: torch.device,
        cfg: Dict[str, Any],
        log_dir: str = "exp",
        incremental_log_dir: bool = False,
        half: bool = False,
        export: bool = False,
        nms_type: str = "nms",
    ) -> None:
        """Initialize Validator class.

        Args:
            model: a torch model or TensorRT Wrapper.
            dataloader: dataloader with validation dataset.
            device: torch device.
            cfg: validate config which includes
                {
                    "train": {
                        "single_cls": True or False,
                        "plot": True or False,
                        "batch_size": number of batch size,
                        "image_size": image size
                    },
                    "hyper_params": {
                        "conf_t": confidence threshold,
                        "iou_t": IoU threshold.
                    }
                }
            log_dir: log directory path.
            incremental_log_dir: use incremental directory.
                If set, log_dir will be
                    {log_dir}/val/{DATE}_runs,
                    {log_dir}/val/{DATE}_runs1,
                    {log_dir}/val/{DATE}_runs2,
                            ...
            half: use half precision input.
            export: export validation results to file.
            nms_type: NMS type (e.g. nms, batched_nms, fast_nms, matrix_nms)
        """
        super().__init__()
        self.n_class = len(dataloader.dataset.names)  # type: ignore
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.cfg_train = cfg["train"]
        self.cfg_hyp = cfg["hyper_params"]
        self.half = half
        self.export = export
        self.nms_type = nms_type

        if incremental_log_dir:
            self.log_dir = increment_path(
                os.path.join(log_dir, "val", datetime.now().strftime("%Y_%m%d_runs"))
            )
        else:
            self.log_dir = log_dir

        if self.export and not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
            LOGGER.info("Export directory: " + colorstr("bold", str(self.log_dir)))

    def convert_target(
        self, targets: torch.Tensor, width: int, height: int, n_batch: int
    ) -> torch.Tensor:
        """Convert targets from normalized coordinates 0.0 ~ 1.0 to pixel coordinates.

        Args:
            targets: (n, 6) tensor.
                targets[:, 0] represents index number of the batch.
                targets[:, 1] represents class index number.
                targets[:, 2:] represents normalized xyxy coordinates.
            width: image width size.
            height: image height size.
            n_batch: batch size
        """
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(
            self.device, non_blocking=True
        )
        return targets

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
        model: Union[nn.Module, "TrTWrapper"],  # type: ignore # noqa: F821
        dataloader: DataLoader,
        device: torch.device,
        cfg: Dict[str, Any],
        compute_loss: bool = True,
        log_dir: str = "exp",
        incremental_log_dir: bool = False,
        half: bool = False,
        export: bool = False,
        hybrid_label: bool = False,
        nms_type: str = "nms",
    ) -> None:
        """Initialize YoloValidator class.

        Args:
            model: a torch model or TensorRT Wrapper.
            dataloader: dataloader with validation dataset.
            device: torch device.
            cfg: validate config which includes
                {
                    "train": {
                        "single_cls": True or False,
                        "plot": True or False,
                        "batch_size": number of batch size,
                        "image_size": image size
                    },
                    "hyper_params": {
                        "conf_t": confidence threshold,
                        "iou_t": IoU threshold.
                    }
                }
            log_dir: log directory path.
            incremental_log_dir: use incremental directory.
                If set, log_dir will be
                    {log_dir}/val/{DATE}_runs,
                    {log_dir}/val/{DATE}_runs1,
                    {log_dir}/val/{DATE}_runs2,
                            ...
            half: use half precision input.
            export: export validation results to file.
            hybrid_label: Run NMS with hybrid information (ground truth label + predicted result.)
                    (PyTorch only) This is for auto-labeling purpose.
            nms_type: NMS type (e.g. nms, batched_nms, fast_nms, matrix_nms)
        """
        super().__init__(
            model,
            dataloader,
            device,
            cfg,
            log_dir=log_dir,
            incremental_log_dir=incremental_log_dir,
            half=half,
            export=export,
            nms_type=nms_type,
        )
        self.class_map = list(range(self.n_class))  # type: ignore
        self.names = {k: v for k, v in enumerate(self.dataloader.dataset.names)}  # type: ignore
        self.confusion_matrix: ConfusionMatrix
        self.statistics: Dict[str, Any]
        self.nc = 1 if self.cfg_train["single_cls"] else int(self.n_class)  # type: ignore
        self.iouv = torch.linspace(0.5, 0.95, 10).to(
            self.device, non_blocking=True
        )  # IoU vecot
        self.niou = self.iouv.numel()
        if compute_loss:
            self.loss_fn = ComputeLoss(self.model)
        else:
            self.loss_fn = None

        self.loss = torch.zeros(3, device=self.device)
        self.seen: int = 0
        self.hybrid_label = hybrid_label

    def set_confusion_matrix(self) -> None:
        """Set confusion matrix."""
        self.confusion_matrix = ConfusionMatrix(nc=self.n_class)

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

    def init_attrs(self) -> None:
        """Initialize attributes before validation."""
        self.set_confusion_matrix()
        self.init_statistics()
        self.seen = 0
        self.tqdm = tqdm(
            enumerate(self.dataloader),
            desc="Validating ...",
            total=len(self.dataloader),
        )

    def prepare_img(self, img: torch.Tensor) -> torch.Tensor:
        """Prepare img for model."""
        img = img.to(self.device, non_blocking=True)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        return img

    def convert_trt_out(
        self, out: torch.Tensor, n_objs: torch.Tensor
    ) -> List[torch.Tensor]:
        """Convert output from TensorRT model to validation ready format.

        Args:
            out: (batch_size, keep_top_k, 6) tensor.
            n_objs: (batch_size,) tensor which contains detected objects on each image.

        Return:
            List of detected results (n_obj, 6) (x1, y1, x2, y2, confidence, class_id)
        """
        result = [
            torch.zeros((n_obj, 6)).to(self.device, non_blocking=True)
            for n_obj in n_objs
        ]
        # result = torch.zeros((n_objs.sum(), 6)).to(self.device, non_blocking=True)

        for i, n_obj in enumerate(n_objs):
            result[i] = out[i][:n_obj]

        return result

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
            if si >= len(shapes):  # TensorRT works with fixed batch size only.
                break

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
            scale_coords(
                img[si].shape[1:], predn[:, :4], shape, shapes[si][1]
            )  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(
                    img[si].shape[1:], tbox, shape, shapes[si][1]
                )  # native-space labels
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
        outs = self.model(
            imgs.half() if self.half else imgs
        )  # inference and training outputs
        self.statistics["dt"][1] += time.time() - t2

        if len(outs) == 2:
            out, train_out = outs
        else:
            out, train_out = outs[0], None

        targets = self.convert_target(targets, width, height, batch_size)
        labels_for_hybrid = (
            [targets[targets[:, 0] == i, 1:] for i in range(batch_size)]
            if self.hybrid_label
            else []
        )

        # Compute loss
        if self.loss_fn:
            self.compute_loss(train_out, targets)

        t3 = time.time()
        if isinstance(train_out, torch.Tensor):  # TensorRT case.
            out = self.convert_trt_out(out.clone(), train_out.clone())
        else:
            out = non_max_suppression(
                out,
                self.cfg_hyp["conf_t"],
                self.cfg_hyp["iou_t"],
                multi_label=True,
                labels=labels_for_hybrid,
                agnostic=self.cfg_train["single_cls"],
                nms_type=self.nms_type,
            )
        self.statistics["dt"][2] += time.time() - t3

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
                plot=self.export,
                save_dir=self.log_dir,
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
        log_str = str(
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

        LOGGER.info(log_str)

        # print result per class
        if (verbose or self.nc < 50) and self.nc > 1 and len(self.statistics["stats"]):
            for i, c in enumerate(self.statistics["ap_class"]):
                LOGGER.info(
                    str(
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
                )
        # print speed
        t = tuple(
            x / self.seen * 1e3 for x in self.statistics["dt"]
        )  # speeds per image
        if verbose:
            shape = (
                self.cfg_train["batch_size"],
                3,
                self.cfg_train["image_size"],
                self.cfg_train["image_size"],
            )
            LOGGER.info(
                f"Speed: {t[0]:.1f}ms pre-process, {t[1]:.1f}ms inference, {t[2]:.1f}ms NMS per image at shape {shape}"
            )
        return t

    @torch.no_grad()
    def validation(self, verbose: bool = True) -> Tuple[Tuple[list, ...], np.ndarray, tuple]:  # type: ignore
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
        self.init_attrs()
        # dt, precision, recall, f1 score, mean-precision, mean-recall, mAP@.5, mAP@.5:.95

        for batch_i, batch in self.tqdm:
            self.validation_step(batch, batch_i)
        self.compute_statistics()

        LOGGER.info(s)
        t = self.print_results(verbose=verbose)

        maps = np.zeros(self.nc) + self.statistics["map"]
        for i, c in enumerate(self.statistics["ap_class"]):
            maps[c] = self.statistics["ap50"][i]

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
