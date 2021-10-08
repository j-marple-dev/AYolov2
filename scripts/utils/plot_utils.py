"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.utils.constants import PLOT_COLOR
from scripts.utils.general import xywh2xyxy


def hist2d(x: np.ndarray, y: np.ndarray, n: int = 100) -> np.ndarray:
    """Draw 2D histogram.

    Args:
        x: x values.
        y: y values.
        n: linspace step.

    Returns:
        a numpy array which contains histogram 2d.
    """
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def plot_one_box(
    x: np.ndarray,
    img: np.ndarray,
    color: Union[Tuple[int, int, int], List[int]] = None,
    label: str = None,
    line_thickness: float = None,
) -> None:
    """Plot one bounding box on image.

    Args:
        x: box coordinates.
        img: base image to plot label.
        color: box edge color.
        label: label to plot.
        line_thickness: line thickness.
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_label_histogram(labels: np.ndarray, save_dir: Union[str, Path] = "") -> None:
    """Plot dataset labels.

    Args:
        labels: image labels.
        save_dir: save directory for saving the histogram.
    """
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


def plot_images(
    images: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    paths: Optional[List[str]] = None,
    fname: str = "images.jpg",
    names: Optional[Union[tuple, list]] = None,
    max_size: int = 640,
    max_subplots: int = 16,
) -> np.ndarray:
    """Plot images."""
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    if isinstance(targets, torch.Tensor):
        np_targets = targets.cpu().numpy()
    else:
        np_targets = targets

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb

    def hex2rgb(h):  # noqa
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))  # noqa

    # hex2rgb = lambda h: tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()["color"]]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y : block_y + h, block_x : block_x + w, :] = img  # noqa
        if len(np_targets) > 0:
            image_targets = np_targets[np_targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype("int")  # type: ignore
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            conf: Optional[np.ndarray] = (
                None if gt else image_targets[:, 6]
            )  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                # 0.3 conf thresh
                if gt or conf[j] > 0.3:  # type: ignore
                    label = "%s" % cls if gt else "%s %.1f" % (cls, conf[j])  # type: ignore
                    plot_one_box(
                        box, mosaic, label=label, color=color, line_thickness=tl
                    )

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(
                mosaic,
                label,
                (block_x + 5, block_y + t_size[1] + 5),
                0,
                tl / 3,
                [220, 220, 220],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

        # Image border
        cv2.rectangle(
            mosaic,
            (block_x, block_y),
            (block_x + w, block_y + h),
            (255, 255, 255),
            thickness=3,
        )

    if fname is not None:
        mosaic = cv2.resize(
            mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


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
