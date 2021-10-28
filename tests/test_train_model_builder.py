"""Unit test TrainModelBuilder.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import gc
import os
import random

import numpy as np
import torch
import yaml
from kindle import YOLOModel
from torch.utils.data import DataLoader

from scripts.data_loader.data_loader import LoadImagesAndLabels
from scripts.data_loader.data_loader_utils import create_dataloader
from scripts.train.train_model_builder import TrainModelBuilder
from scripts.train.yolo_trainer import YoloTrainer
from scripts.utils.general import get_logger
from scripts.utils.model_manager import YOLOModelManager
from scripts.utils.torch_utils import select_device

LOGGER = get_logger(__name__)
RANK = int(os.getenv("RANK", -1))


def test_train_model_builder(p: float = 0.5) -> None:
    if random.random() > p:
        return

    with open(
        os.path.join("tests", "res", "configs", "train_config_sample.yaml"), "r"
    ) as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["epochs"] = 1
    cfg["train"]["n_skip"] = 5
    cfg["train"]["image_size"] = 320
    if not torch.cuda.is_available():
        cfg["train"]["device"] = "cpu"  # Switch to CPU mode

    model = YOLOModel(
        os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=True
    )
    train_builder = TrainModelBuilder(model, cfg, "exp")
    train_builder.ddp_init()

    model_manager = YOLOModelManager(
        model, cfg, train_builder.device, train_builder.wdir
    )

    stride_size = int(max(model.stride))  # type: ignore

    train_loader, train_dataset = create_dataloader(
        "tests/res/datasets/coco/images/train2017", cfg, stride_size, prefix="[Train] "
    )
    model, ema, device = train_builder.prepare()
    model_manager.model = model
    model = model_manager.set_model_params(train_dataset, ema=ema)

    del model, train_builder, model_manager, ema
    gc.collect()


def test_train(p: float = 0.5) -> None:
    if random.random() > p:
        return

    with open(
        os.path.join("tests", "res", "configs", "train_config_sample.yaml"), "r"
    ) as f:
        cfg = yaml.safe_load(f)

    cfg["train"]["epochs"] = 1
    cfg["train"]["n_skip"] = 5
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
        "tests/res/datasets/coco/images/train2017", cfg, stride_size, prefix="[Train] "
    )

    cfg["train"]["rect"] = True
    val_loader, val_dataset = create_dataloader(
        "tests/res/datasets/coco/images/val2017",
        cfg,
        stride_size,
        prefix="[Val] ",
        pad=0.5,
        validation=False,  # This is supposed to be True.
    )

    model_manager = YOLOModelManager(
        model, cfg, train_builder.device, train_builder.wdir
    )

    model, ema, device = train_builder.prepare()
    model = model_manager.set_model_params(train_dataset, ema=ema)

    trainer = YoloTrainer(
        model,
        cfg,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        ema=ema,
        device=device,
    )
    trainer.train()

    del (
        model,
        train_builder,
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        model_manager,
        trainer,
    )
    gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    opt = parser.parse_args()
    # test_train_model_builder()
    test_train()
