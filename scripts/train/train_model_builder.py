"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import argparse
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainModelBuilder:
    """Train model builder class."""

    opt: Dict[str, Any] = {}
    device: torch.device = torch.device("cpu")
    cuda: bool = False

    @staticmethod
    def _model_to_ddp(model: nn.Module) -> nn.Module:
        model = DDP(
            model, device_ids=[], output_device=TrainModelBuilder.opt["local_rank"]
        )
        return model

    @staticmethod
    def _model_to_data_parallel(model: nn.Module) -> nn.Module:
        return torch.nn.DataParallel(model)

    @staticmethod
    def _model_to_sync_bn(model: nn.Module) -> nn.Module:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(
            TrainModelBuilder.device
        )
        return model

    @staticmethod
    def prepare_model_for_training(
        model: nn.Module,
        weight_path: str,
        device: torch.device,
        opt: Optional[Union[argparse.Namespace, Dict[str, Any]]],
    ) -> nn.Module:
        """Prepare model for training.

        Args:
            model: a torch model to train.
            weight_path: weight path for load state dict.
            device: torch device.
            opt: an option for model.

        Returns:
            a model which is prepared for training.
        """
        if opt:
            if isinstance(opt, argparse.Namespace):
                TrainModelBuilder.opt.update(vars(opt))
            else:
                TrainModelBuilder.opt.update(opt)

        TrainModelBuilder.device = device
        TrainModelBuilder.cuda = TrainModelBuilder.device.type != "cpu"

        rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1

        if TrainModelBuilder.cuda and rank == -1 and torch.cuda.device_count() > 1:
            model = TrainModelBuilder._model_to_data_parallel(model)

        if TrainModelBuilder.opt["sync_bn"] and TrainModelBuilder.cuda and rank != -1:
            model = TrainModelBuilder._model_to_sync_bn(model)

        if TrainModelBuilder.cuda and rank != -1:
            model = TrainModelBuilder._model_to_ddp(model)

        return model

