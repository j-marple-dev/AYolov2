"""General utilities.

- Author: Jongkuk Lim, Haneol Kim
- Contact: limjk@jmarple.ai, hekim@jmarple.ai
"""

import glob
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from scripts.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Settings START
torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(
    linewidth=320, formatter={"float_kind": "{:11.5g}".format}
)  # format short g, %precision=5
# pd.options.display.max_columns = 10
cv2.setNumThreads(
    0
)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(min(os.cpu_count(), 8))  # NumExpr max threads
# Settings END


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
        LOGGER.warn(
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
    return xyxy2xywh(np.array(boxes), clip_eps=None, check_validity=False)  # type: ignore


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
) -> Union[torch.Tensor, np.ndarray]:
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


def scale_coords(
    img1_shape: Tuple[float, float],
    coords: Union[torch.Tensor, np.ndarray],
    img0_shape: Tuple[float, float],
    ratio_pad: Optional[Union[tuple, list, np.ndarray]] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """Rescale coords (xyxy) from img1_shape to img0_shape.

    Args:
        img1_shape: current image shape. (h, w)
        coords: (xyxy) coordinates.
        img0_shape: target image shape. (h, w)
        ratio_pad: padding ratio.  (w, h)

    Returns:
        scaled coordinates.
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape[::-1])  # clip_coord use wh image shape.
    return coords


def increment_path(
    path_: str, exist_ok: bool = False, sep: str = "", mkdir: bool = False
) -> Path:
    """Increment file or directory path.

    i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    Args:
        path: path to use increment path
        exist_ok: Check if the path already exists and uses the path if exists.
        sep: separator string
        mkdir: create directory if the path does not exist.

    Return:
        incremented path.
    """
    path = Path(path_)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path

    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory

    return path


class TimeChecker:
    """Time analyzer class."""

    def __init__(
        self,
        title: str = "",
        ignore_thr: float = 0.05,
        sort: bool = True,
        add_start: bool = True,
        cuda_sync: bool = False,
    ) -> None:
        """Initialize TimeChecker class.

        Args:
            title: name of the time analysis
            ignore_thr: time percentage that took below {ignore_thr}% will be ignored for the logging.
            sort: log sorted by time consumption ratios
            add_start: auto add start time
                       TimeChecker requires at least two time checks.
                       The first time will always be used as the start time.
            cuda_sync: Use cuda synchronized time.
        """
        self.times: Dict[str, List[float]] = dict()
        self.name_idx: Dict[str, int] = dict()
        self.idx_name: List[str] = []

        self.title = title
        self.ignore_thr = ignore_thr
        self.sort = sort
        self.cuda_sync = cuda_sync

        if add_start:
            self.add("start")

    def __getitem__(self, name: str) -> Tuple[float, int]:
        """Get time taken.

        Returns:
            time took(s)
            Number of times that {name} event occur.
        """
        idx = self.name_idx[name]
        name_p = self.idx_name[idx - 1]

        times_0 = self.times[name_p]
        times_1 = self.times[name]

        n_time = min(len(times_0), len(times_1))
        time_took = 0.0
        for i in range(n_time):
            time_took += times_1[i] - times_0[i]

        return time_took, n_time

    def add(self, name: str) -> None:
        """Add time point."""
        if self.cuda_sync:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        if name not in self.name_idx:
            self.name_idx[name] = len(self.times)
            self.idx_name.append(name)
            self.times[name] = [time.monotonic()]
        else:
            self.times[name].append(time.monotonic())

    def clear(self) -> None:
        """Clear time records."""
        self.times.clear()
        self.name_idx.clear()
        self.idx_name.clear()

    def _convert_unit_str(self, value: float) -> Tuple[float, str]:
        """Convert second unit to s, ms, ns metric.

        Args:
            value: time(s)
        Returns:
            Converted time value.
            Unit of the time value(s, ms, ns).
        """
        if value < 0.001:
            value *= 1000 * 1000
            unit = "ns"
        elif value < 1:
            value *= 1000
            unit = "ms"
        else:
            unit = "s"

        return value, unit

    @property
    def total_time(self) -> float:
        """Get total time."""
        time_tooks = [self[self.idx_name[i]][0] for i in range(1, len(self.times))]

        return sum(time_tooks)

    def __str__(self) -> str:
        """Convert time checks to the log string."""
        msg = f"[{self.title[-15:]:>15}] "
        time_total = self.total_time
        time_tooks = [self[self.idx_name[i]] for i in range(1, len(self.times))]

        if self.sort:
            idx = np.argsort(np.array(time_tooks)[:, 0])[::-1]
        else:
            idx = np.arange(0, len(self.times) - 1)

        for i in idx:
            time_took = time_tooks[i][0]
            time_ratio = time_took / time_total

            time_took, unit = self._convert_unit_str(time_took)

            if time_ratio > self.ignore_thr:
                msg += f"{self.idx_name[i+1][:10]:>11}: {time_took:4.1f}{unit}({time_ratio*100:4.1f}%), "

        time_total, unit = self._convert_unit_str(time_total)
        msg += f"{'Total':>11}: {time_total:4.1f}{unit}"
        return msg
