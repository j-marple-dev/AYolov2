"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import math
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from kindle import YOLOModel
from torch import nn

from scripts.utils.logger import colorstr, get_logger

LOGGER = get_logger(__name__)


def count_param(model: nn.Module) -> int:
    """Count number of all parameters.

    Args:
        model: PyTorch model.

    Return:
        Sum of # of parameters
    """
    return sum(list(x.numel() for x in model.parameters()))


@contextmanager
def torch_distributed_zero_first(local_rank: int) -> Generator:
    """Make sure torch distributed call is run on only local_rank -1 or 0.

    Decorator to make all processes in distributed training wait for each local_master
    to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])  # type: ignore
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])  # type: ignore


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


def de_parallel(model: nn.Module) -> nn.Module:
    """Decapsule parallelized model.

    Args:
        model: Single-GPU modle, DP model or DDP model

    Return:
        a decapsulized single-GPU model
    """
    return model.module if is_parallel(model) else model  # type: ignore


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
    """Copy attributes from b to a, options to only include and to exclude.

    Args:
        a: destination
        b: source
        include: key names to copy
        exclude: key names NOT to copy
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def load_model_weights(
    model: nn.Module, weights: Union[Dict, str], exclude: Optional[list] = None,
) -> nn.Module:
    """Load model's pretrained weights.

    Args:
        model: model instance to load weight.
        weights: model weight path.
        exclude: exclude list of layer names.

    Return:
        self.model which the weights has been loaded.
    """
    if isinstance(weights, str):
        ckpt = torch.load(weights)
    else:
        ckpt = weights

    exclude_list = [] if exclude is None else exclude

    state_dict = ckpt["model"].float().state_dict()
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude_list)
    model.load_state_dict(state_dict, strict=False)  # load weights
    LOGGER.info(
        "Transferred %g/%g items from %s"
        % (
            len(state_dict),
            len(model.state_dict()),
            weights if isinstance(weights, str) else weights.keys(),
        )
    )
    return model


def load_pytorch_model(
    weight_path: str, model_cfg_path: str = "", load_ema: bool = True
) -> Optional[nn.Module]:
    """Load PyTorch model.

    Args:
        weight_path: weight path which ends with .pt
        model_cfg_path: if provided, the model will first construct by the model_cfg,
                        and transfer weights to the constructed model.
                        In case of model_cfg_path was provided but not weight_path,
                        the model weights will be randomly initialized
                        (for experiment purpose).
        load_ema: load EMA weights if possible.

    Return:
        PyTorch model,
        None if loading PyTorch model has failed.
    """
    if weight_path == "":
        LOGGER.warning(
            "Providing "
            + colorstr("bold", "no weights path")
            + " will validate a randomly initialized model. Please use only for a experiment purpose."
        )
    else:
        ckpt = torch.load(weight_path)
        if isinstance(ckpt, dict):
            model_key = (
                "ema"
                if load_ema and "ema" in ckpt.keys() and ckpt["ema"] is not None
                else "model"
            )
            ckpt_model = ckpt[model_key]
        elif isinstance(ckpt, nn.Module):
            ckpt_model = ckpt

        ckpt_model = ckpt_model.cpu().float()

    if ckpt_model is None and model_cfg_path == "":
        LOGGER.warning("No weights and no model_cfg has been found.")
        return None

    if model_cfg_path != "":
        model = YOLOModel(model_cfg_path, verbose=True)
        model = load_model_weights(model, {"model": ckpt_model}, exclude=[])
    else:
        model = ckpt_model

    return model


def sparsity(model: nn.Module) -> float:
    """Compute global model sparsity.

    Args:
        model: PyTorch model.

    Return:
        sparsity ratio (sum of zeros / # of parameters)
    """
    n_param, zero_param = 0.0, 0.0
    for p in model.parameters():
        n_param += p.numel()
        zero_param += (p == 0).sum()  # type: ignore
    return zero_param / n_param


def prune(model: nn.Module, amount: float = 0.3) -> None:
    """Prune model to requested global sparsity.

    Note that this is in-place operation.

    Args:
        model: PyTorch model.
        amount: target sparsity ratio.
            i.e. 0.1 = 10% of sparsity. (Weak prunning)
                 0.9 = 90% of sparsity. (Strong prunning)
    """
    import torch.nn.utils.prune as prune

    LOGGER.info("Pruning model... ")

    for _, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)  # prune
            prune.remove(m, "weight")  # make permanent
    LOGGER.info(" |---  %.3g global sparsity" % sparsity(model))


def scale_img(
    img: torch.Tensor, ratio: float = 1.0, same_shape: bool = False, gs: int = 32
) -> torch.Tensor:
    """Scales img(bs,3,y,x) by ratio constrained to gs-multiple.

       Reference: https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py#L257-L267

    Args:
        img: image tensor
        ratio: scale ratio for image tensor
        same_shape: whether to make same shape or not
        gs: stride

    Returns:
        scaled image tensor
    """
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(
            img, [0, w - s[1], 0, h - s[0]], value=0.447
        )  # value = imagenet mean


class EarlyStopping:
    """YOLOv5 simple early stopper."""

    def __init__(self, patience: int = 30) -> None:
        """Initialize Early stopping class.

        Args:
            patience: early stopping patience.
        """
        self.best_score = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float(
            "inf"
        )  # epochs to wait after score stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch: int, score: float) -> bool:
        """Decide stop early or not.

        Args:
            epoch: training epoch.
            score: score of current epoch.

        Returns:
            finish training or not.
        """
        if (
            score >= self.best_score
        ):  # >= 0 to allow for early zero-score stage of training
            self.best_epoch = epoch
            self.best_score = score
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (
            self.patience - 1
        )  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(
                f"EarlyStopping patience {self.patience} exceeded, stopping training."
            )
        return stop


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
                    key = k if k in msd else f"module.{k}"
                    v *= d
                    v += (1.0 - d) * msd[key].detach()

    def update_attr(
        self,
        model: nn.Module,
        include: Union[List[str], Tuple[str, ...]] = (),
        exclude: tuple = ("process_group", "reducer"),
    ) -> None:
        """Update EMA attributes."""
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
