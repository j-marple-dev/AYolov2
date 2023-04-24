"""General utilities.

- Author: Jongkuk Lim, Haneol Kim
- Contact: limjk@jmarple.ai, hekim@jmarple.ai
"""

import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


def count_param(model: nn.Module) -> int:
    """Count number of all parameters.

    Args:
        model: PyTorch model.

    Return:
        Sum of # of parameters
    """
    return sum(list(x.numel() for x in model.parameters()))


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


class TimeChecker:
    """Time analyzer class."""

    def __init__(
        self,
        title: str = "",
        ignore_thr: float = 0.05,
        sort: bool = True,
        add_start: bool = True,
        cuda_sync: bool = False,
        enable: bool = True,
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
            enable: whether to use time checker
        """
        self.times: Dict[str, List[float]] = dict()
        self.name_idx: Dict[str, int] = dict()
        self.idx_name: List[str] = []

        self.title = title
        self.ignore_thr = ignore_thr
        self.sort = sort
        self.cuda_sync = cuda_sync
        self.enable = enable

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
        if not self.enable:
            return

        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        if name not in self.name_idx:
            self.name_idx[name] = len(self.times)
            self.idx_name.append(name)
            self.times[name] = [time.time()]
        else:
            self.times[name].append(time.time())

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

        if not self.enable:
            return f"{msg} disabled."

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
