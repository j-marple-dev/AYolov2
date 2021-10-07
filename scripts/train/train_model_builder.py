"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.utils.anchors import check_anchors
from scripts.utils.general_utils import check_img_size, labels_to_class_weights
from scripts.utils.plot_utils import plot_labels
from scripts.utils.torch_utils import ModelEMA, intersect_dicts

logger = logging.Logger(__name__)


class TrainModelBuilder:
    """Train model builder class."""

    def __init__(
        self,
        model: nn.Module,
        # weight_path: str,
        device: torch.device,
        opt: Optional[Union[argparse.Namespace, Dict[str, Any]]],
        hyp: Dict[str, Any],
        log_dir: str,
    ) -> None:
        """Initialize TrainModelBuilder class.

        Args:
            model: a torch model to train.
            weight_path: weight path for load state dict.
            device: torch device.
            opt: an option for model.
        """
        self.model = model
        self.weight_path = opt.weights
        self.device = device
        if isinstance(opt, argparse.Namespace):
            self.opt = vars(self.opt)
        elif isinstance(opt, dict):
            self.opt = opt
        else:
            raise TypeError("opt should be a dictionary or Namespace.")
        self.hyp = hyp
        self.log_dir = log_dir
        self.wdir = Path(os.path.join(log_dir, "weights"))
        os.makedirs(self.wdir, exist_ok=True)

    def model_to_ddp(self) -> nn.Module:
        """Convert model to DDP model."""
        model = DDP(
            self.model,
            device_ids=[self.opt["local_rank"]],
            output_device=TrainModelBuilder.opt["local_rank"],
        )
        return model

    def model_to_data_parallel(self) -> nn.Module:
        """Convert model to data parallel model."""
        return torch.nn.DataParallel(self.model)

    def model_to_sync_bn(self) -> nn.Module:
        """Convert model batch norm to sync batch norm."""
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(
            self.device
        )
        return model

    def load_model_weights(
        self,
        weights: Union[Dict, str],
        model: Optional[nn.Module] = None,
        exclude: Optional[list] = None,
    ) -> nn.Module:
        """Load model's pretrained weights."""
        if model is None:
            model = self.model

        if isinstance(weights, str):
            ckpt = torch.load(weights)
        else:
            ckpt = weights

        state_dict = ckpt["model"].float().state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)  # load weights
        logger.info(
            "Transferred %g/%g items from %s"
            % (len(state_dict), len(model.state_dict()), weights)
        )
        return model

    def set_model_params(
        self, dataset: torch.utils.data.Dataset, nc: int = 80, names: str = ""
    ) -> None:
        """Set model parameters.

        Args:
            nc: Number of classes.
        """
        self.model.nc = nc
        self.model.hyp = self.hyp
        self.model.gr = 1.0
        self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(
            self.device
        )
        self.model.names = names

    def prepare_model_for_training(
        self,
        dataset: torch.utils.data.Dataset,
        dataloader: torch.utils.data.DataLoader,
        nc: int = 80,
    ) -> Tuple[nn.Module, Optional[ModelEMA]]:
        """Prepare model for training.

        Args:
            dataset: Dataset for training
            dataloader: Dataloader for training
            nc: Number of classes.

        Returns:
            a model which is prepared for training.
        """
        rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
        cuda = self.device.type != "cpu"

        pretrained = self.weights.endswith(".pt")
        ckpt = torch.load(self.weights, map_location=self.device)
        if pretrained:
            exclude = ["anchor"] if self.opt["cfg"] or self.hyp.get("anchors") else []
            self.model = self.load_model_weights(weights=ckpt, exclude=exclude)
            start_epoch = ckpt["epoch"] + 1
            if self.opt["resume"]:
                assert start_epoch > 0, (
                    "%s training to %g epochs is finished, nothing to resume."
                    % (self.opt["weights"], self.opt["epochs"],)
                )
                shutil.copytree(
                    self.wdir,
                    self.wdir.parent / f"weights_backup_epoch{start_epoch - 1}",
                )  # save previous weights
            if self.opt["epochs"] < start_epoch:
                logger.info(
                    "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                    % (self.opt["weights"], ckpt["epoch"], self.opt["epochs"])
                )
                # epochs += ckpt["epoch"]  # finetune additional epochs

        if cuda and rank == -1 and torch.cuda.device_count() > 1:
            self.model = self.model_to_data_parallel()

        if self.opt["sync_bn"] and self.cuda and rank != -1:
            self.model = self.model_to_sync_bn()

        ema = ModelEMA(self.model) if rank in [-1, 0] else None

        if cuda and rank != -1:
            self.model = self.model_to_ddp()

        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()
        nb = len(dataloader)
        assert mlc < nc, (
            "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g"
            % (mlc, nc, self.opt["data"], nc - 1)
        )
        nbs = 64  # nominal batch size
        gs = int(max(self.model.stride))
        imgsz, _ = [check_img_size(x, gs) for x in self.opt["img_size"]]
        if rank in [-1, 0] and ema is not None:
            accumulate = max(round(nbs / self.opt["total_batch_size"]), 1)
            ema.updates = start_epoch * nb // accumulate

            if not self.opt["resume"]:
                labels = np.concatenate(self.dataset.labels, 0)
                c = torch.tensor(labels[:, 0])  # noqa
                plot_labels(labels, save_dir=self.log_dir)

                if not self.opt["noautoanchor"]:
                    check_anchors(
                        dataset, model=self.model, thr=self.hyp["anchor_t"], imgsz=imgsz
                    )
