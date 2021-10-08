"""General utilities.

- Author: Jongkuk Lim, Haneol Kim
- Contact: limjk@jmarple.ai, hekim@jmarple.ai
"""

import logging
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch

from scripts.utils.constants import LOG_LEVEL


def make_divisible(x: int, divisor: int, minimum_check_number: int = 0) -> int:
    """Return 'x' evenly divisible by 'divisor'.

    Args:
        x: Input which want to make divisible with 'divisor'.
        divisor: Divisor.
        minimun_check_number: Minimum number to check.

    Returns:
        ceil(x / divisor) * divisor
    """
    if x <= minimum_check_number:
        return math.floor(x)
    else:
        return math.ceil(x / divisor) * divisor


def check_img_size(img_size: int, s: int = 32) -> int:
    """Verify image size is a multiple of stride s.

    Args:
        img_size: Current image size.
        s: Stride.

    Returns:
        New image size verified with stride s.
    """
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print(
            "WARNING --img-size %g must be multiple of max stride %g, updating to %g"
            % (img_size, s, new_size)
        )
    return new_size


def labels_to_class_weights(
    labels: Union[list, np.ndarray, torch.Tensor], nc: int = 80
) -> torch.Tensor:
    """Get class weights from training labels."""
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    c_labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = c_labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(
    labels: Union[list, np.ndarray],
    nc: int = 80,
    class_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Produce image weights based on class mAPs."""
    if class_weights is None:
        np_class_weights: np.ndarray = np.ones(80)
    else:
        np_class_weights = class_weights

    n = len(labels)
    class_counts = np.array(
        [np.bincount(labels[i][:, 0].astype(int), minlength=nc) for i in range(n)]
    )
    image_weights = (np_class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def get_logger(name: str, log_level: Optional[int] = None) -> logging.Logger:
    """Get logger with formatter.

    Args:
        name: logger name
        log_level: logging level if None is given, constants.LOG_LEVEL will be used.

    Return:
        logger with string formatter.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(LOG_LEVEL if log_level is None else log_level)

    logger.addHandler(ch)

    return logger


def clip_coords(
    boxes: Union[torch.Tensor, np.ndarray],
    wh: Tuple[float, float],
    inplace: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Clip bounding boxes with xyxy format to given wh (width, height).

    Args:
        boxes: bounding boxes (n, 4) (x1, y1, x2, y2)
        wh: image size (width, height)
        inplace: inplace modification.
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        if not inplace:
            boxes = boxes.clone()

        boxes[:, 0].clamp_(0, wh[0])  # x1
        boxes[:, 1].clamp_(0, wh[1])  # y1
        boxes[:, 2].clamp_(0, wh[0])  # x2
        boxes[:, 3].clamp_(0, wh[1])  # y2
    else:  # np.array (faster grouped)
        if not inplace:
            boxes = np.copy(boxes)

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, wh[0])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, wh[1])  # y1, y2

    return boxes


def xyxy2xywh(
    x: Union[torch.Tensor, np.ndarray],
    wh: Tuple[float, float] = (1.0, 1.0),
    clip_eps: Optional[float] = None,
    check_validity: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Convert (n, 4) bound boxes from xyxy to xywh format.

        [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right.

    Args:
        xy: (n, 4) xyxy coordinates
        wh: image size (width, height).
            Give image size only if you want to
            normalized pixel coordinates to normalized coordinates.
        clip_eps: clip coordinates by wh with epsilon margin. If clip_eps is not None.
            epsilon value is recommended to be 1E-3
        check_validity: bounding box width and height validity check.
            whichi make bounding boxes to the following conditions.
            1) (x1 - width / 2) >= 0
            2) (y1 - height / 2) >= 0
            3) (x2 + width / 2) <= 1
            4) (y2 + height / 2) <= 1
    Return:
        return coordinates will be (xywh / wh) with centered xy
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    if clip_eps is not None:
        y = clip_coords(y, (wh[0] - clip_eps, wh[1] - clip_eps))

    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / wh[0]  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / wh[1]  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / wh[0]  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / wh[1]  # height

    if check_validity:
        y[:, 2] = y[:, 2] + (np.minimum((y[:, 0] - (y[:, 2] / 2)), 0) * 2)
        y[:, 2] = y[:, 2] - ((np.maximum((y[:, 0] + (y[:, 2] / 2)), 1) - 1) * 2)
        y[:, 3] = y[:, 3] + (np.minimum((y[:, 1] - (y[:, 3] / 2)), 0) * 2)
        y[:, 3] = y[:, 3] - ((np.maximum((y[:, 1] + (y[:, 3] / 2)), 1) - 1) * 2)

        y = y.clip(1e-12, 1)

    return y


def xywh2xyxy(
    x: Union[torch.Tensor, np.ndarray],
    ratio: Tuple[float, float] = (1.0, 1.0),
    wh: Tuple[float, float] = (1.0, 1.0),
    pad: Tuple[float, float] = (0.0, 0.0),
) -> Union[torch.Tensor, np.ndarray]:
    """Convert (n, 4) bound boxes from xywh to xyxy format.

        [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.

    Args:
        x: (n, 4) xywh coordinates
        ratio: label ratio adjustment. Default value won't change anything other than xywh to xyxy.
        wh: Image size (width and height). If normalized xywh to pixel xyxy format, place image size here.
        pad: image padded size.

    Return:
        return coordinates will be (ratio * wh * xyxy + pad)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ratio[0] * wh[0] * (x[:, 0] - x[:, 2] / 2) + pad[0]  # top left x
    y[:, 1] = ratio[1] * wh[1] * (x[:, 1] - x[:, 3] / 2) + pad[1]  # top left y
    y[:, 2] = ratio[0] * wh[0] * (x[:, 0] + x[:, 2] / 2) + pad[0]  # bottom right x
    y[:, 3] = ratio[1] * wh[1] * (x[:, 1] + x[:, 3] / 2) + pad[1]  # bottom right y
    return y
