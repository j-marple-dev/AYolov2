"""Unit test for augmentation.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import gc
import os
import random

import torch
import yaml
from kindle import YOLOModel

from scripts.data_loader.data_loader_utils import create_dataloader
from scripts.train.train_model_builder import TrainModelBuilder
from scripts.utils.constants import probably_run
from scripts.utils.model_manager import YOLOModelManager


@probably_run()
def test_model_manager(p: float = 0.5) -> None:
    with open(
        os.path.join("tests", "res", "configs", "train_config_sample.yaml"), "r"
    ) as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["epochs"] = 1
    cfg["train"]["n_skip"] = 4
    cfg["train"]["image_size"] = 320

    if not torch.cuda.is_available():
        cfg["train"]["device"] = "cpu"  # Switch to CPU mode

    model = YOLOModel(
        os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=True
    )
    train_builder = TrainModelBuilder(model, cfg, "exp")
    train_builder.ddp_init()

    stride_size = int(max(model.stride))  # type: ignore

    train_loader, train_dataset = create_dataloader(
        "tests/res/datasets/coco/images/train2017", cfg, stride_size, prefix="[Train] ",
    )

    model_manager = YOLOModelManager(
        model, cfg, train_builder.device, train_builder.wdir
    )

    model = model_manager.load_model_weights()
    model = model_manager.set_model_params(train_dataset)
    model = model_manager.freeze(5)

    n_frozen = sum([not v.requires_grad for k, v in model.named_parameters()])
    n_trainable = sum([v.requires_grad for k, v in model.named_parameters()])

    del model, train_builder, train_loader, train_dataset, model_manager
    gc.collect()

    assert n_frozen == 45
    assert n_trainable == 132


if __name__ == "__main__":
    test_model_manager()
