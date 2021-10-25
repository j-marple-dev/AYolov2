"""Unit test for Tensor Decomposition.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import os
from copy import deepcopy

import torch
from kindle import YOLOModel

from scripts.tensor_decomposition.decomposition import decompose_model
from scripts.utils.logger import get_logger
from scripts.utils.torch_utils import count_param, prune

LOGGER = get_logger(__name__)


def test_tensor_decomposition() -> None:
    test_input = torch.rand((1, 3, 320, 320))

    ckpt = torch.load(os.path.join("tests", "res", "weights", "yolov5s_kindle.pt"))

    model = ckpt["model"].float()
    prune(model, 0.2)

    decomposed_model = deepcopy(model)
    decompose_model(decomposed_model, loss_thr=0.13)

    LOGGER.info(f"Origin:     # parameter: {count_param(model):,d}")
    LOGGER.info(f"Decomposed: # parameter: {count_param(decomposed_model):,d}")

    model.export().eval()
    decomposed_model.export().eval()

    origin_out = model(test_input)
    decomposed_out = decomposed_model(test_input)

    loss = (origin_out[0] - decomposed_out[0]).abs().sum() / origin_out[0].numel()

    LOGGER.info(f"Full forward loss: {loss}")

    assert count_param(model) == 7266973
    assert count_param(decomposed_model) == 6328573
    assert loss < 0.015


if __name__ == "__main__":
    test_tensor_decomposition()
