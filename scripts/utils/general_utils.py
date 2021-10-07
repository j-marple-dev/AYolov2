"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import math
import random
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.utils.plot_utils import hist2d
from scripts.utils.torch_utils import init_torch_seeds


def init_seeds(seed: int = 0) -> None:
    """Initialize random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


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


def plot_labels(labels: np.ndarray, save_dir: Union[str, Path] = "") -> None:
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
