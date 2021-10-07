"""Unit test TrainModelBuilder.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
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

LOGGER = get_logger(__name__)


def test_train_model_builder() -> None:
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
        mosaic_prob=cfg["hyper_params"]["mosaic"],
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

    # pl_model = YoloPLModule(model, cfg)
    # TODO(jeikeilim): DP does not work but DDP work for some reason.
    # trainer = pl.Trainer(
    #     gpus=cfg["train"]["device"],
    #     accelerator="ddp",
    #     max_epochs=cfg["train"]["epochs"],
    #     check_val_every_n_epoch=cfg["train"]["validate_period"],
    # )
    # val_result0 = trainer.validate(pl_model, val_loader)
    # trainer.fit(pl_model, train_loader, val_loader)
    # val_result1 = trainer.validate(pl_model, val_loader)

    # assert (val_result0[0]["val_loss"] - val_result1[0]["val_loss"]) > 10


def test_train() -> None:
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
        mosaic_prob=cfg["hyper_params"]["mosaic"],
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

    trainer = YoloTrainer(
        model, cfg, train_dataloader=train_loader, val_dataloader=val_loader
    )
    trainer.train()
    # pl_model = YoloPLModule(model, cfg)
    # TODO(jeikeilim): DP does not work but DDP work for some reason.
    # trainer = pl.Trainer(
    #     gpus=cfg["train"]["device"],
    #     accelerator="ddp",
    #     max_epochs=cfg["train"]["epochs"],
    #     check_val_every_n_epoch=cfg["train"]["validate_period"],
    # )
    # val_result0 = trainer.validate(pl_model, val_loader)
    # trainer.fit(pl_model, train_loader, val_loader)
    # val_result1 = trainer.validate(pl_model, val_loader)

    # assert (val_result0[0]["val_loss"] - val_result1[0]["val_loss"]) > 10


if __name__ == "__main__":
    test_train()
