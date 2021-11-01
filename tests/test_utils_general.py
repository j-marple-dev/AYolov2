"""Unit test for general utils.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""
import random

import numpy as np
import torch

from scripts.utils.general import xywh2xyxy, xyxy2xywh


def test_xyxy2xywh() -> None:
    if random.random() > 0.5:
        return
    xy = np.random.uniform(0, 1, (100, 2))
    wh = (1 - xy) * np.random.uniform(0.2, 0.8, (100, 1))

    xywh = np.hstack([xy + (wh / 2), wh])
    xyxy = np.hstack([xy, xy + wh])

    out_xywh = xyxy2xywh(xyxy)
    out_xyxy = xywh2xyxy(xywh)

    assert np.isclose(xywh, out_xywh, rtol=1e-24).all()
    assert np.isclose(xyxy, out_xyxy, rtol=1e-24).all()


def test_xyxy2xywh_pixel() -> None:
    if random.random() > 0.5:
        return
    img_size = np.random.randint(300, 800, 2)
    xy = np.random.uniform(0, 1, (100, 2))
    wh = (1 - xy) * np.random.uniform(0.2, 0.8, (100, 1))

    xywh = np.hstack([xy + (wh / 2), wh])
    xyxy = np.hstack([xy, xy + wh])

    xywh_pixel = np.hstack([xy + (wh / 2), wh]) * np.concatenate([img_size] * 2)
    xyxy_pixel = np.hstack([xy, xy + wh]) * np.concatenate([img_size] * 2)

    out_xywh = xyxy2xywh(xyxy_pixel, wh=img_size)
    out_xyxy = xywh2xyxy(xywh_pixel, wh=1 / img_size)

    assert np.isclose(xywh, out_xywh, rtol=1e-24).all()
    assert np.isclose(xyxy, out_xyxy, rtol=1e-24).all()


if __name__ == "__main__":
    test_xyxy2xywh()
    test_xyxy2xywh_pixel()
