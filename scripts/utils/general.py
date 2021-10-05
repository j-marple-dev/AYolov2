"""General utilities.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from scripts.utils.constants import PLOT_COLOR


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


def plot_labels(
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
