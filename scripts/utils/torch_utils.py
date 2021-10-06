"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import os
from typing import Optional

import torch
from torch import nn

from scripts.utils.general import get_logger

LOGGER = get_logger(__name__)


def select_device(device: str = "", batch_size: Optional[int] = None) -> torch.device:
    """Select torch device.

    Args:
        device: 'cpu' or '0' or '0, 1, 2, 3' format string.
        batch_size: distribute batch to multiple gpus.

    Returns:
        A torch device.
    """
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert torch.cuda.is_available(), (
            "CUDA unavailable, invalid device %s requested" % device
        )

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:
            assert (
                batch_size % ng == 0
            ), "batch-size %g not multiple of GPU count %g" % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = "Using CUDA "
        for i in range(0, ng):
            if i == 1:
                s = " " * len(s)
                LOGGER.info(
                    "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                    % (s, i, x[i].name, x[i].total_memory / c)
                )

    else:
        LOGGER.info("Using CPU")

    LOGGER.info("")
    return torch.device("cuda:0" if cuda else "cpu")


def is_parallel(model: nn.Module) -> bool:
    """Check if the model is DP or DDP.

    Args:
        model: PyTorch nn.Module

    Return:
        True if the model is DP or DDP,
        False otherwise.
    """
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
