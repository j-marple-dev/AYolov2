"""Test utilities.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from scripts.utils.torch_utils import scale_img


def descale_pred(
    p: torch.Tensor, flips: Optional[int], scale: float, img_size: tuple
) -> torch.Tensor:
    """De-scale predictions following augmented inference (inverse operation).

       Reference: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py#L156-L171

    Args:
        p: augmented inferences
        flips: filp type (2: vertical flip, 3: horizontal flip)
        scale: scale ratio for input image tensor
        img_size: input image tensor size (height, width)

    Returns:
        p: de-scaled and de-flipped tensor
    """
    p[..., :4] /= scale  # de-scale
    if flips == 2:
        p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
    elif flips == 3:
        p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
    return p


def clip_augmented(model: nn.Module, y: List) -> List:
    """Clip YOLOv5 augmented inference tails.

       Reference: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py#L173-L182

    Args:
        model: YOLOModel or nn.Module which last layer is YOLOHead.
        y: augmented inferences

    Returns:
        y: clipped tensors for augmented inferences
    """
    # number of detection layers (P3-P5)
    nl = model.model[-1].nl  # type: ignore
    g = sum(4 ** x for x in range(nl))  # grid points
    e = 1  # exclude layer count
    i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
    y[0] = y[0][:, :-i]  # large
    i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
    y[-1] = y[-1][:, i:]  # small
    return y


def inference_with_tta(
    model: nn.Module, x: torch.Tensor, s: List, f: List
) -> Tuple[torch.Tensor, None]:
    """Inference with TTA.

       Reference: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py#L129-L141

    Args:
        model: YOLOModel or nn.Module which last layer is YOLOHead.
        x: input image tensors for model
        s: scale ratios of each augmentation for TTA
        f: flip types of each augmentation for TTA

    Returns:
        agumented inferences, train outputs
    """
    img_size = x.shape[-2:]  # height, width
    y = []  # outputs
    for si, fi in zip(s, f):
        xi = scale_img(x.flip(fi) if fi else x, si, gs=int(model.stride.max()))  # type: ignore
        yi = model(xi)[0]  # forward
        yi = descale_pred(yi, fi, si, img_size)
        y.append(yi)
    y = clip_augmented(model, y)  # clip augmented tails
    return torch.cat(y, 1), None  # augmented inference
