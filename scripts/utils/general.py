"""General utilities.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

from typing import Union, Tuple
import numpy as np
import torch


def xyxy2xywh(x: Union[torch.Tensor, np.ndarray],
              wh: Tuple[float, float] = (1.0, 1.0),
              ) -> Union[torch.Tensor, np.ndarray]:
    """Convert (n, 4) bound boxes from xyxy to xywh format

	[x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right.

    Args:
        xy: (n, 4) xyxy coordinates
        wh: image size (width, height).
            Give image size only if you want to
            normalized pixel coordinates to normalized coordinates.

    Return:
        return coordinates will be (xywh / wh)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / wh[0]  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / wh[1]  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / wh[0]  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / wh[1]  # height

    return y


def xywh2xyxy(x: Union[torch.Tensor, np.ndarray],
              ratio: Tuple[float, float] = (1.0, 1.0),
              wh: Tuple[float, float] = (1.0, 1.0),
              pad: Tuple[float, float] = (0.0, 0.0),
              ) -> Union[torch.Tensor, np.ndarray]:
    """Convert (n, 4) bound boxes from xywh to xyxy format

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


