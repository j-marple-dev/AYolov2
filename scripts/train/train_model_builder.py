"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from scripts.utils.general import increment_path, labels_to_class_weights
from scripts.utils.logger import get_logger
from scripts.utils.torch_utils import (ModelEMA, init_seeds, is_parallel,
                                       load_model_weights, select_device)

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

LOGGER = get_logger(__name__)


class TrainModelBuilder:
    """Train model builder class."""

    def __init__(self, model: nn.Module, cfg: Dict[str, Any], log_dir: str) -> None:
        """Initialize TrainModelBuilder.

        Args:
            model: a torch model to train.
            opt: train config
            log_dir: logging root directory
        """
        self.model = model
        self.cfg = cfg
        if hasattr(self.model, "model_parser"):
            self.yaml = self.model.model_parser.cfg  # type: ignore
        else:
            self.yaml = None
        self.device = select_device(cfg["train"]["device"], cfg["train"]["batch_size"])
        self.cuda = self.device.type != "cpu"
        self.log_dir = increment_path(
            os.path.join(log_dir, "train", datetime.now().strftime("%Y_%m%d_runs"))
        )
        self.wdir = Path(os.path.join(self.log_dir, "weights"))
        if RANK in [-1, 0]:
            os.makedirs(self.wdir, exist_ok=True)

    def to_ddp(self) -> nn.Module:
        """Convert model to DDP model."""
        self.model = DDP(self.model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        return self.model

    def to_data_parallel(self) -> nn.Module:
        """Convert model to DataParallel model."""
        self.model = torch.nn.DataParallel(self.model)

        return self.model

    def to_sync_bn(self) -> nn.Module:
        """Convert model to SyncBatchNorm model."""
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(
            self.device
        )
        return self.model

    def set_model_params(
        self, dataset: torch.utils.data.Dataset, nc: int = 80, names: str = ""
    ) -> None:
        """Set model parameters.

        Args:
            nc: Number of classes.
        """
        head = (
            self.model.module.model[-1] if is_parallel(self.model) else self.model.model[-1]  # type: ignore
        )  # YOLOHead module

        # TODO(jeikeilim): Re-visit here
        # TODO(jeikeilim): Double check if the model contains nc
        self.model.nc = nc  # type: ignore
        self.model.hyp = self.cfg["hyper_params"]
        self.model.gr = 1.0  # type: ignore
        self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(  # type: ignore
            self.device
        )
        self.model.names = names  # type: ignore
        self.model.stride = head.stride  # type: ignore
        self.model.cfg = self.cfg  # type: ignore
        self.model.yaml = self.yaml  # type: ignore

    def ddp_init(self) -> None:
        """Initialize DDP device."""
        if not torch.cuda.is_available():
            return

        # DDP INIT
        if LOCAL_RANK != -1:
            assert (
                torch.cuda.device_count() > LOCAL_RANK
            ), "insufficient CUDA devices for DDP command"
            assert (
                self.cfg["train"]["batch_size"] % WORLD_SIZE == 0
            ), "--batch-size must be multiple of CUDA device count"
            assert not self.cfg["train"][
                "image_weights"
            ], "--image-weights argument is not compatible with DDP training"

            torch.cuda.set_device(LOCAL_RANK)
            self.device = torch.device("cuda", LOCAL_RANK)
            dist.init_process_group(
                backend="nccl" if dist.is_nccl_available() else "gloo"
            )

    def prepare(
        self,
        dataset: torch.utils.data.Dataset,
        dataloader: torch.utils.data.DataLoader,
        nc: int = 80,
    ) -> Tuple[nn.Module, Optional[ModelEMA], torch.device]:
        """Prepare model for training.

        Args:
            dataset: Dataset for training
            dataloader: Dataloader for training
            nc: Number of classes.

        Returns:
            a model which is prepared for training.
            EMA model if supports, otherwise None
        """
        self.set_model_params(dataset, nc=nc)
        init_seeds(1 + RANK)

        self.model.to(self.device)

        start_epoch = 0
        weight_fp = self.cfg["train"]["weights"]
        if weight_fp:
            if not weight_fp.endswith(".pt"):
                best_weight = wandb.restore("best.pt", run_path=weight_fp)
                weight_fp = best_weight.name
        pretrained = weight_fp.endswith(".pt")
        if pretrained:
            ckpt = torch.load(weight_fp, map_location=self.device)

            # TODO(jeikeilim): Re-visit here.
            exclude = []
            # (
            #     ["anchor"]
            #     if self.cfg["cfg"] or self.cfg["hyper_params"].get("anchors")
            #     else []
            # )
            self.model = load_model_weights(self.model, weights=ckpt, exclude=exclude)
            start_epoch = ckpt["epoch"] + 1
            if self.cfg["train"]["resume"]:
                assert start_epoch > 0, (
                    "%s training to %g epochs is finished, nothing to resume."
                    % (self.cfg["train"]["weights"], self.cfg["train"]["epochs"],)
                )
                shutil.copytree(
                    self.wdir,
                    self.wdir.parent / f"weights_backup_epoch{start_epoch - 1}",
                )  # save previous weights
            if self.cfg["train"]["epochs"] < start_epoch:
                LOGGER.info(
                    "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                    % (
                        self.cfg["train"]["weights"],
                        ckpt["epoch"],
                        self.cfg["train"]["epochs"],
                    )
                )

        if isinstance(self.cfg["train"]["image_size"], int):
            self.cfg["train"]["image_size"] = [self.cfg["train"]["image_size"]] * 2

        nbs = 64  # nominal batch size
        nb = len(dataloader)

        ema = ModelEMA(self.model) if RANK in [-1, 0] else None

        if self.cuda and RANK == -1 and torch.cuda.device_count() > 1:
            self.to_data_parallel()

        if self.cfg["train"]["sync_bn"] and self.cuda and RANK != -1:
            self.to_sync_bn()

        if RANK in [-1, 0] and ema is not None:
            # TODO(jeikeilim): Double check here.
            accumulate = max(round(nbs / self.cfg["train"]["batch_size"]), 1)
            ema.updates = start_epoch * nb // accumulate

        if self.cuda and RANK != -1:
            self.to_ddp()

        # TODO(jeikeilim): How to pass names here?
        self.set_model_params(dataset, nc=nc)

        return (self.model, ema, self.device)
