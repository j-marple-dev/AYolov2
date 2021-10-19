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
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.utils.general import increment_path
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.torch_utils import ModelEMA, init_seeds, select_device

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

LOGGER = get_logger(__name__)


class TrainModelBuilder:
    """Train model builder class."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        log_dir: str,
        full_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize TrainModelBuilder.

        Args:
            model: a torch model to train.
            opt: train config
            log_dir: logging root directory
            full_cfg: full config contents.
        """
        self.model = model
        self.cfg = cfg
        self.device = select_device(cfg["train"]["device"], cfg["train"]["batch_size"])
        self.cuda = self.device.type != "cpu"
        self.log_dir = increment_path(
            os.path.join(log_dir, "train", datetime.now().strftime("%Y_%m%d_runs"))
        )
        self.wdir = Path(os.path.join(self.log_dir, "weights"))
        if RANK in [-1, 0]:
            os.makedirs(self.wdir, exist_ok=True)

        if full_cfg is not None:
            for k, v in full_cfg["args"].items():
                if isinstance(v, str) and Path(v).is_file():
                    src = Path(v)
                    dst = self.log_dir / f"{k}{src.suffix}"
                    shutil.copyfile(src, dst)
                    LOGGER.info(
                        "Copying "
                        + colorstr("bold", str(src))
                        + " to "
                        + colorstr("bold", str(dst))
                    )
            with open(self.log_dir / "full_cfg.yaml", "w") as f:
                yaml.dump(full_cfg, f)

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

    def prepare(self) -> Tuple[nn.Module, Optional[ModelEMA], torch.device]:
        """Prepare model for training.

        Args:
            total_n_batch: Total number of batch step in 1 epoch.

        Returns:
            a model which is prepared for training.
            EMA model if supports, otherwise None
        """
        init_seeds(1 + RANK)

        self.model.to(self.device)

        if isinstance(self.cfg["train"]["image_size"], int):
            self.cfg["train"]["image_size"] = [self.cfg["train"]["image_size"]] * 2

        ema = ModelEMA(self.model) if RANK in [-1, 0] else None

        if self.cuda and RANK == -1 and torch.cuda.device_count() > 1:
            self.to_data_parallel()

        if self.cfg["train"]["sync_bn"] and self.cuda and RANK != -1:
            self.to_sync_bn()

        if self.cuda and RANK != -1:
            self.to_ddp()

        return (self.model, ema, self.device)
