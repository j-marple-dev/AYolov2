"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.utils.anchors import check_anchors
from scripts.utils.general import (check_img_size, get_logger,
                                   labels_to_class_weights)
from scripts.utils.plot_utils import plot_label_histogram
from scripts.utils.torch_utils import (ModelEMA, init_seeds, intersect_dicts,
                                       is_parallel, select_device)

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
        self.log_dir = log_dir
        self.wdir = Path(os.path.join(log_dir, "weights"))
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

    def load_model_weights(
        self, weights: Union[Dict, str], exclude: Optional[list] = None,
    ) -> nn.Module:
        """Load model's pretrained weights.

        Args:
            weights: model weight path.
            exclude: exclude list of layer names.

        Return:
            self.model which the weights has been loaded.
        """
        # TODO(jeikeilim): Make a separate function to load_model_weight in utils
        # and call that function in below.

        if isinstance(weights, str):
            ckpt = torch.load(weights)
        else:
            ckpt = weights

        state_dict = ckpt["model"].float().state_dict()
        state_dict = intersect_dicts(
            state_dict, self.model.state_dict(), exclude=exclude
        )
        self.model.load_state_dict(state_dict, strict=False)  # load weights
        LOGGER.info(
            "Transferred %g/%g items from %s"
            % (len(state_dict), len(self.model.state_dict()), weights)
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

        # TODO(jeikeilim): Make load weight from wandb.
        start_epoch = 0
        pretrained = self.cfg["train"]["weights"].endswith(".pt")
        if pretrained:
            ckpt = torch.load(self.cfg["train"]["weights"], map_location=self.device)

            # TODO(jeikeilim): Re-visit here.
            exclude = (
                ["anchor"]
                if self.cfg["cfg"] or self.cfg["hyper_params"].get("anchors")
                else []
            )
            self.model = self.load_model_weights(weights=ckpt, exclude=exclude)
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
        gs = int(max(self.model.stride))  # type: ignore
        imgsz, _ = [check_img_size(x, gs) for x in self.cfg["train"]["image_size"]]

        ema = ModelEMA(self.model) if RANK in [-1, 0] else None

        if self.cuda and RANK == -1 and torch.cuda.device_count() > 1:
            self.to_data_parallel()

        if self.cfg["train"]["sync_bn"] and self.cuda and RANK != -1:
            self.to_sync_bn()

        # Max label class
        # TODO(jeikeilim): Re-visit here to check if
        # dataset, dataloader, nc argument is really necessary.
        # TODO(jeikeilim): mlc does not need to be called here. This can go outside of this method.
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # type: ignore
        nb = len(dataloader)
        assert mlc < nc, (
            "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g"
            % (mlc, nc, self.cfg["data"], nc - 1)
        )
        if RANK in [-1, 0] and ema is not None:
            # TODO(jeikeilim): Double check here.
            accumulate = max(round(nbs / self.cfg["train"]["batch_size"]), 1)
            ema.updates = start_epoch * nb // accumulate

            if not self.cfg["train"]["resume"]:
                labels = np.concatenate(dataset.labels, 0)  # type: ignore
                c = torch.tensor(labels[:, 0])  # noqa
                plot_label_histogram(labels, save_dir=self.log_dir)

                if self.cfg["train"]["auto_anchor"]:
                    check_anchors(
                        dataset,
                        model=self.model,
                        thr=self.cfg["hyper_params"]["anchor_t"],
                        imgsz=imgsz,
                    )

        if self.cuda and RANK != -1:
            self.to_ddp()

        # TODO(jeikeilim): How to pass names here?
        self.set_model_params(dataset, nc=nc)

        return (self.model, ema, self.device)
