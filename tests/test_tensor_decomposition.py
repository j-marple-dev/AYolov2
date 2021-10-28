"""Unit test for Tensor Decomposition.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import os
import random
from copy import deepcopy

import torch
from kindle import YOLOModel

from scripts.tensor_decomposition.decomposition import decompose_model
from scripts.utils.logger import get_logger
from scripts.utils.torch_utils import count_param, prune
from scripts.utils.wandb_utils import get_ckpt_path_from_wandb

LOGGER = get_logger(__name__)


def test_tensor_decomposition(p: float = 0.5) -> None:
    if random.random() > p:
        return

    test_input = torch.rand((1, 3, 320, 320))

    ckpt = torch.load(os.path.join("tests", "res", "weights", "yolov5s_kindle.pt"))

    model = ckpt["model"].float()

    decomposed_model = deepcopy(model)
    decompose_model(decomposed_model, loss_thr=0.13, prune_step=0.1)

    model.export().eval()
    decomposed_model.export().eval()

    LOGGER.info(f"Origin:     # parameter: {count_param(model):,d}")
    LOGGER.info(f"Decomposed: # parameter: {count_param(decomposed_model):,d}")

    origin_out = model(test_input)
    decomposed_out = decomposed_model(test_input)

    loss = (origin_out[0] - decomposed_out[0]).abs().sum() / origin_out[0].numel()

    LOGGER.info(f"Full forward loss: {loss}")

    assert count_param(model) == 7266973
    assert count_param(decomposed_model) == 6336589
    assert loss < 0.015


def test_wandb_loader_for_tensor_decomposition(
    force: bool = False, p: float = 0.5
) -> None:
    if not force:
        return

    if random.random() > p:
        return

    # You should define wandb run path that you want to load.
    # e.g. wandb_path = "j-marple/AYolov2/3a1r9rb"
    wandb_path = "j-marple/AYolov2/5v1o0e54"
    ckpt_path = get_ckpt_path_from_wandb(wandb_path)
    ckpt = torch.load(ckpt_path)
    if isinstance(ckpt, YOLOModel):
        model = ckpt.float()
    elif "ema" in ckpt.keys() and ckpt["ema"] is not None:
        model = ckpt["ema"].float()
    elif "model" in ckpt.keys():
        model = ckpt["model"]

    assert model is not None


if __name__ == "__main__":
    test_tensor_decomposition(p=1.0)
    test_wandb_loader_for_tensor_decomposition()
