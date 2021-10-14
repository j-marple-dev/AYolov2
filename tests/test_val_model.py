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

from scripts.data_loader.data_loader import LoadImagesAndLabels
from scripts.train.train_model_builder import TrainModelBuilder
from scripts.train.yolo_plmodule import YoloPLModule
from scripts.train.yolo_trainer import YoloTrainer
from scripts.utils.general import get_logger
from scripts.utils.torch_utils import select_device
from scripts.utils.train_utils import YoloValidator

LOGGER = get_logger(__name__)


def test_model_validator() -> None:
    with open(
        os.path.join("tests", "res", "configs", "train_config_sample.yaml"), "r"
    ) as f:
        cfg = yaml.safe_load(f)

    model = YOLOModel(
        os.path.join("tests", "res", "configs", "model_yolov5s.yaml"), verbose=True
    )

    cfg["train"]["epochs"] = 1

    train_dataset = LoadImagesAndLabels(
        "tests/res/datasets/coco/images/train2017",
        cache_images=cfg["train"]["cache_image"],
        n_skip=cfg["train"]["n_skip"],
        batch_size=cfg["train"]["batch_size"],
        preprocess=lambda x: (x / 255.0).astype(np.float32),
        rect=False,
        pad=0,
        mosaic_prob=cfg["hyper_params"]["mosaic"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        # num_workers=multiprocessing.cpu_count() - 1,
        num_workers=8,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    val_dataset = LoadImagesAndLabels(
        "tests/res/datasets/coco/images/val2017",
        cache_images=cfg["train"]["cache_image"],
        n_skip=cfg["train"]["n_skip"],
        batch_size=cfg["train"]["batch_size"],
        preprocess=lambda x: (x / 255.0).astype(np.float32),
        rect=False,
        pad=0,
        mosaic_prob=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        # num_workers=multiprocessing.cpu_count() - 1,
        num_workers=8,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )

    if not torch.cuda.is_available():
        cfg["train"]["device"] = "cpu"  # Switch to CPU mode

    device = select_device(cfg["train"]["device"], cfg["train"]["batch_size"])

    model, ema = TrainModelBuilder(model, cfg, device, "exp")(
        train_dataset, train_loader
    )
    validator = YoloValidator(model, val_loader, device, cfg)

    validator.validation()


if __name__ == "__main__":
    test_model_validator()
