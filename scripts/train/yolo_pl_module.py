"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import argparse
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import yaml

from scripts.train.abstract_pl_module import AbstractPLModule
from scripts.utils.general_utils import check_img_size, init_seeds
from scripts.utils.torch_utils import ModelEMA, select_device

logger = logging.getLogger(__name__)


class YoloPLModule(AbstractPLModule):
    """Yolo model trainer."""

    def __init__(
        self,
        model: nn.Module,
        hyp: Dict[str, Any],
        opt: argparse.Namespace,
        log_dir: str,
        device: Optional[Union[torch.device, str]],
        mlc: np.int64,
        nb: int,
        nc: int = 80,
        ckpt: Optional[Dict[str, Any]] = None,
        freeze: Optional[str] = None,
    ) -> None:
        """Initialize YoloTrainer class.

        Args:
            model: Yolo model to train. The pretrained weights should be loaded before if you attempt to use the weights.
            hyp: Hyper parameter config.
            opt: Train options from argparse.
            log_dir: Log directory.
            device: Torch device to train on.
            mlc: Max lable class (np.concatenate(dataset.labels, 0)[:, 0].max())
            nb: Number of batches.
            nc: Number of classes.
            ckpt: A checkpoint of pretrained model.
            freeze: Path to save frozen model.
        """
        if isinstance(device, str):
            torch_device = select_device(device)
        elif device is None:
            torch_device = torch.device("cpu")
        else:
            torch_device = device
        logger.info(f"Hyper-parameters {hyp}")
        self.ckpt = ckpt
        self.opt = vars(opt)
        self.wdir = os.path.join(log_dir, "weights")

        os.makedirs(self.wdir, exist_ok=True)

        self.last = os.path.join(self.wdir, "last.pt")
        self.best = os.path.join(self.wdir, "best.pt")
        self.results_file = os.path.join(log_dir, "results.txt")

        epochs, batch_size, self.total_batch_size, self.weights, self.rank = (
            opt.epochs,
            opt.batch_size,
            opt.total_batch_size if opt.total_batch_size else opt.batch_size,
            opt.weights,
            opt.global_rank,
        )
        super().__init__(model, hyp, epochs, torch_device, batch_size)
        self.loss = eval(self.hyp["loss"])
        # Important: This property activates manual optimization.
        self.optimizer, self.scheduler = self.init_optimizer()
        init_seeds(2 + self.rank)

        self.model.nc = nc
        # TODO(ulken94): Add exponential moving average.
        # ema = ModelEMA(model) if rank in [-1, 0] else None
        assert mlc < nc, (
            "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g"
            % (mlc, nc, opt.data, nc - 1)
        )
        self.mlc = mlc
        self.nb = nb
        self.ema: Optional[ModelEMA] = None
        self.accumulate: int

    def init_optimizer(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[lr_scheduler.LambdaLR]]:
        """Initialize optimizer and scheduler."""
        nbs = 64  # nominal batch size.
        self.accumulate = max(round(nbs / self.total_batch_size), 1)
        self.hyp["weights_decay"] *= self.total_batch_size * self.accumulate / nbs

        pg0: List[torch.Tensor] = []  # batch normalization
        pg1: List[torch.Tensor] = []  # weights
        pg2: List[torch.Tensor] = []  # biases
        for _, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, torch.Tensor):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, torch.Tensor):
                pg1.append(v.weight)
        for _, v in self.model.named_parameters():
            v.requires_grad = True

        # TODO(ulken94): find fancy way to create optimizer
        # Create optimizer from hyp config.
        optimizer: torch.optim.Optimizer = eval(self.hyp["optimizer"])(
            params=pg0, **self.hyp["optimizer_params"]
        )
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": self.hyp["weight_decay"]}
        )
        optimizer.add_param_group({"params": pg2})
        logger.info(
            "Optimizer groups: %g .bias, %g conv.weight, %g other"
            % (len(pg2), len(pg1), len(pg0))
        )

        lf = (
            lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2)
            * (1 - self.hyp["lrf"])
            + self.hyp["lrf"]
        )
        # Learning rate scheduler
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        if self.ckpt and self.ckpt["optimizer"] is not None:
            optimizer.load_state_dict(self.ckpt["optimizer"])

        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed forward the model.

        Args:
            x: Input tensor.

        Returns:
            Result of model of input x.
        """
        output = self.model(x)
        return output

    def training_step(
        self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Train a step (a batch).

        Args:
            train_batch: Train batch in tuple (input_x, true_y).
            batch_idx: Current batch index.

        Returns:
            Result of loss function.
        """
        x, y = train_batch
        self.model.train()

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
        # mloss = torch.zeros(4, device=self.device)
        output = self.model(x)
        loss = self.loss(output, y)  # type: ignore

        return loss

    def validation_step(
        self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validate a step (a batch).

        Args:
            val_batch: Validation data batch in tuple (input_x, true_y).
            batch_idx: Current batch index.

        Returns:
            Result of loss function.
        """
        x, y = val_batch
        output = self.model(x)
        loss = self.loss(output, y)  # type: ignore

        return loss

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        y_hat = self.model(x)
        return self.loss(y_hat, y)

    def save_run_settings(self) -> None:
        """Save run settings."""
        log_dir = Path(self.opt["logdir"]) / "evolve"
        with open(log_dir / "hyp.yaml", "w") as f:
            yaml.dump(self.hyp, f, sort_keys=False)
        with open(log_dir / "opt.yaml", "w") as f:
            yaml.dump(self.opt, f, sort_keys=False)

    def freeze_params(self, names: List[str]) -> None:
        """Freeze parameters.

        Args:
            names: Parameter names which is to be frozen.
        """
        if any(names):
            for k, v in self.model.named_parameters():
                if any(x in k for x in names):
                    print("freezing %s" % k)
                    v.requires_grad = False

    def prepare_training(self, freeze: Optional[List[str]]) -> None:
        """Prepare for training.

        Args:
            freeze: Parameter names which is to be frozen.
        """
        pretrained = self.ckpt is not None
        if freeze is None:
            freeze = [""]
        self.freeze_params(freeze)
        self.start_epoch = 0
        self.best_fitness: Union[float, np.ndarray] = 0.0
        if pretrained and self.ckpt is not None:
            if self.ckpt.get("training_results") is not None:
                with open(self.results_file, "w") as file:
                    file.write(self.ckpt["training_results"])  # write results.txt

            self.start_epoch = self.ckpt["epoch"] + 1
            if self.opt["resume"]:
                assert self.start_epoch > 0, (
                    "%s training to %g epochs is finished, noting to resume."
                    % (self.weights, self.epochs,)
                )

            if self.epochs < self.start_epoch:
                logger.info(
                    f"The model has been trained for {self.ckpt['epoch']} epochs. Fine-tuning for {self.epochs} additional epochs."
                )
                self.epochs += self.ckpt["epoch"]

            del self.ckpt

        gs = int(max(self.model.stride))
        imgsz, imgsz_test = [check_img_size(x, gs) for x in self.opt["img_size"]]

        if self.rank in [-1, 0] and self.ema is not None:
            self.ema.updates = self.start_epoch * self.nb // self.accumulate
            if not self.opt["resume"]:
                labels = np.concatenate()  # noqa

        # TODO(ulken94): How to get number of classes.
        self.hyp["cls"] *= self.model.nc / 80.0
        self.model.hyp = self.hyp
        self.model.gr = 1.0

    def prepare_ema(self) -> None:
        """Prepare ema."""
        self.ema = ModelEMA(self.model) if self.rank in [-1, 0] else None
