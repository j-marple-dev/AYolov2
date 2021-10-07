"""General utilities.

- Author: Jongkuk Lim, Haneol Kim
- Contact: limjk@jmarple.ai, hekim@jmarple.ai
"""

import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.utils.constants import LOG_LEVEL, PLOT_COLOR
from scripts.utils.plot_utils import hist2d


def resample_segments(segments: List[np.ndarray], n: int = 1000) -> List[np.ndarray]:
    """Interpolate segments by (n, 2) segment points.

    Args:
        segments: segmentation coordinates list [(m, 2), ...]
        n: number of interpolation.

    Return:
        Interpolated segmentation (n, 2)
    """
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)])
            .reshape(2, -1)
            .T
        )  # segment xy

    return segments


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


def segment2box(segment: np.ndarray, width: int = 640, height: int = 640) -> np.ndarray:
    """Convert 1 segment label to 1 box label.

    Applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)

    Args:
        segment: one segmentation coordinates. (n, 2)
        width: width constraint.
        height: height constraint

    Return:
        bounding box contrained by (width, height)
    """
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))
    )  # xyxy


def segments2boxes(segments: List[np.ndarray]) -> np.ndarray:
    """Convert segment labels to box labels, i.e. (xy1, xy2, ...) to (xywh).

    Args:
        segments: List of segments. [(n1, 2), (n2, 2), ...]

    Return:
        box labels (n, 4)
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes), clip_eps=None, check_validity=False)  # cls, xywh


def box_candidates(
    box1: np.ndarray,
    box2: np.ndarray,
    wh_thr: float = 2,
    ar_thr: float = 20,
    area_thr: float = 0.1,
    eps: float = 1e-16,
) -> np.ndarray:  # box1(4,n), box2(4,n)
    """Compute candidate boxes.

    Args:
        box1: before augment
        box2: after augment
        wh_thr: width and height threshold (pixels),
        ar_thr: aspect ratio threshold
        area_thr: area_ratio

    Return:
        Boolean mask index of the box candidates.
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + eps) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def plot_label_histogram(labels: np.ndarray, save_dir: Union[str, Path] = "") -> None:
    """Plot dataset labels."""
    c, b = labels[:, 0], labels[:, 1:].transpose()
    nc = int(c.max() + 1)  # number of classes

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_xlabel("classes")
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap="jet")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap="jet")
    ax[2].set_xlabel("width")
    ax[2].set_ylabel("height")
    plt.savefig(Path(save_dir) / "labels.png", dpi=200)
    plt.close()

    # seaborn correlogram
    try:
        import pandas as pd
        import seaborn as sns

        x = pd.DataFrame(b.transpose(), columns=["x", "y", "width", "height"])
        sns.pairplot(
            x,
            corner=True,
            diag_kind="hist",
            kind="scatter",
            markers="o",
            plot_kws=dict(s=3, edgecolor=None, linewidth=1, alpha=0.02),
            diag_kws=dict(bins=50),
        )
        plt.savefig(Path(save_dir) / "labels_correlogram.png", dpi=200)
        plt.close()
    except Exception:
        pass


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


def xyn2xy(
    x: Union[torch.Tensor, np.ndarray],
    wh: Tuple[float, float] = (640, 640),
    pad: Tuple[float, float] = (0.0, 0.0),
):
    """Convert normalized xy (n, 2) to pixel coordinates xy.

    wh: Image size (width and height). If normalized xywh to pixel xyxy format, place image size here.
    pad: image padded size (width and height).
    """
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = wh[0] * x[:, 0] + pad[0]  # top left x
    y[:, 1] = wh[1] * x[:, 1] + pad[1]  # top left y
    return y


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
        pad: image padded size (width and height).

    Return:
        return coordinates will be (ratio * wh * xyxy + pad)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ratio[0] * wh[0] * (x[:, 0] - x[:, 2] / 2) + pad[0]  # top left x
    y[:, 1] = ratio[1] * wh[1] * (x[:, 1] - x[:, 3] / 2) + pad[1]  # top left y
    y[:, 2] = ratio[0] * wh[0] * (x[:, 0] + x[:, 2] / 2) + pad[0]  # bottom right x
    y[:, 3] = ratio[1] * wh[1] * (x[:, 1] + x[:, 3] / 2) + pad[1]  # bottom right y
    return y


def draw_labels(
    img: np.ndarray, label_list: np.ndarray, label_info: Dict[int, str]
) -> np.ndarray:
    """Draw label informations on the image.

    Args:
        img: image to draw labels
        label_list: (n, 5) label informations of img with normalized xywh format.
                    (class_id, centered x, centered y, width, height)
        label_info: label names. Ex) {0: 'Person', 1:'Car', ...}

    Returns:
        label drawn image.
    """
    overlay_alpha = 0.3
    label_list = np.copy(label_list)
    label_list[:, 1:] = xywh2xyxy(
        label_list[:, 1:], wh=(float(img.shape[1]), float(img.shape[0]))
    )

    for label in label_list:
        class_id = int(label[0])
        class_str = label_info[class_id]

        xy1 = tuple(label[1:3].astype("int"))
        xy2 = tuple(label[3:5].astype("int"))
        plot_color = tuple(map(int, PLOT_COLOR[class_id]))

        overlay = img.copy()
        overlay = cv2.rectangle(overlay, xy1, xy2, plot_color, -1)
        img = cv2.addWeighted(overlay, overlay_alpha, img, 1 - overlay_alpha, 0)
        img = cv2.rectangle(img, xy1, xy2, plot_color, 1)

        (text_width, text_height), baseline = cv2.getTextSize(class_str, 3, 0.5, 1)
        overlay = img.copy()
        overlay = cv2.rectangle(
            overlay,
            (xy1[0], xy1[1] - text_height),
            (xy1[0] + text_width, xy1[1]),
            (plot_color[0] // 0.3, plot_color[1] // 0.3, plot_color[2] // 0.3),
            -1,
        )
        img = cv2.addWeighted(overlay, overlay_alpha + 0.2, img, 0.8 - overlay_alpha, 0)
        cv2.putText(
            img,
            class_str,
            xy1,
            3,
            0.5,
            (plot_color[0] // 3, plot_color[1] // 3, plot_color[2] // 3),
            1,
            cv2.LINE_AA,
        )
    return img
