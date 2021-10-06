"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import os
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


class TrainModelBuilder:
    """Train model builder class."""

    def __init__(
        self, model: nn.Module, opt: Dict[str, Any], device: torch.device
    ) -> None:
        """Initialize TrainModelBuilder.

        Args:
            model: a torch model to train.
            opt: train config
            device: torch device.
        """
        self.model = model
        self.opt = opt
        self.device = device
        self.cuda = self.device.type != "cpu"

    def _to_ddp(self) -> nn.Module:
        """Convert model to DDP model."""
        self.model = DDP(self.model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        return self.model

    def _to_data_parallel(self) -> nn.Module:
        """Convert model to DataParallel model."""
        self.model = torch.nn.DataParallel(self.model)

        return self.model

    def _to_sync_bn(self) -> nn.Module:
        """Convert model to SyncBatchNorm model."""
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(
            self.device
        )
        return self.model

    def __call__(self) -> nn.Module:
        """Prepare model for training.

        Returns:
            a model which is prepared for training.
        """
        if self.cuda and RANK == -1 and torch.cuda.device_count() > 1:
            self._to_data_parallel()

        if self.opt["sync_bn"] and self.cuda and RANK != -1:
            self._to_sync_bn()

        if self.cuda and RANK != -1:
            self._to_ddp()

        if LOCAL_RANK != -1:
            assert (
                torch.cuda.device_count() > LOCAL_RANK
            ), "insufficient CUDA devices for DDP command"
            assert (
                self.opt["batch_size"] % WORLD_SIZE == 0
            ), "--batch-size must be multiple of CUDA device count"
            assert not self.opt[
                "image_weights"
            ], "--image-weights argument is not compatible with DDP training"

            torch.cuda.set_device(LOCAL_RANK)
            self.device = torch.device("cuda", LOCAL_RANK)
            dist.init_process_group(
                backend="nccl" if dist.is_nccl_available() else "gloo"
            )

        return self.model
