"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import math
import os
import random
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
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


def init_torch_seeds(seed: int = 0) -> None:
    """Set random seed for torch.

    If seed == 0, it can be slower but more reproducible.
    If not, it would be faster but less reproducible.
    Speed-reproducibility tradedoff https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(seed)

    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False

    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def init_seeds(seed: int = 0) -> None:
    """Initialize random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def intersect_dicts(
    da: dict, db: dict, exclude: Union[List[str], Tuple[str, ...]] = ()
) -> dict:
    """Check dictionary intersection of matching keys and shapes.

    Omitting 'exclude' keys, using da values.
    """
    return {
        k: v
        for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def copy_attr(
    a: object,
    b: object,
    include: Union[List[str], Tuple[str, ...]] = (),
    exclude: Union[List[str], Tuple[str, ...]] = (),
) -> None:
    """Copy attributes from b to a, options to only include and to exclude."""
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Model Exponential Moving Average.

    from https://github.com/rwightman/pytorch-image-
    models Keep a moving average of everything in the model state_dict (parameters and
    buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(
        self, model: nn.Module, decay: float = 0.9999, updates: int = 0
    ) -> None:
        """Initialize ModelEMA class."""
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / 2000)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(
        self,
        model: nn.Module,
        include: Union[List[str], Tuple[str, ...]] = (),
        exclude: tuple = ("process_group", "reducer"),
    ) -> None:
        """Update EMA attributes."""
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
