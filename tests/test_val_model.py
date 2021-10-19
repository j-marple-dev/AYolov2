"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""

import os

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from kindle import YOLOModel
from torch.utils.data import DataLoader

from scripts.data_loader.data_loader_utils import create_dataloader
from scripts.train.train_model_builder import TrainModelBuilder
from scripts.train.yolo_plmodule import YoloPLModule
from scripts.train.yolo_trainer import YoloTrainer
from scripts.utils.general import get_logger
from scripts.utils.model_manager import YOLOModelManager
from scripts.utils.torch_utils import select_device
from scripts.utils.train_utils import YoloValidator

LOGGER = get_logger(__name__)


def test_model_validator() -> None:
    with open(
        os.path.join("tests", "res", "configs", "train_config_sample.yaml"), "r"
    ) as f:
        cfg = yaml.safe_load(f)

    if not torch.cuda.is_available():
        cfg["train"]["device"] = "cpu"  # Switch to CPU mode

    model = YOLOModel(
        os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=True
    )
    cfg["train"]["epochs"] = 1

    train_builder = TrainModelBuilder(model, cfg, "exp")
    train_builder.ddp_init()

    model_manager = YOLOModelManager(
        model, cfg, train_builder.device, train_builder.wdir
    )

    stride_size = int(max(model.stride))  # type: ignore

    train_loader, train_dataset = create_dataloader(
        "tests/res/datasets/coco/images/train2017", cfg, stride_size, prefix="[Train] "
    )
    val_loader, val_dataset = create_dataloader(
        "tests/res/datasets/coco/images/val2017",
        cfg,
        stride_size,
        prefix="[Val] ",
        pad=0.5,
        validation=True,
    )

    model = model_manager.set_model_params(train_dataset)
    model, ema, device = train_builder.prepare()
    model = model_manager.set_model_params(train_dataset)

    model.eval()
    validator = YoloValidator(model, val_loader, device, cfg)

    validator.validation()


if __name__ == "__main__":
    test_model_validator()
