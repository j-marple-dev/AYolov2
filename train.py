"""Main script for your project.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import argparse
import multiprocessing
import os

import numpy as np
import yaml
from kindle import YOLOModel
from torch.utils.data import DataLoader

from scripts.augmentation.augmentation import MultiAugmentationPolicies
from scripts.data_loader.data_loader import LoadImagesAndLabels
from scripts.train.train_model_builder import TrainModelBuilder
from scripts.train.yolo_trainer import YoloTrainer
from scripts.utils.torch_utils import select_device


def get_parser() -> argparse.Namespace:
    """Get argument parser.

    Modify this function as your porject needs
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("res", "configs", "model", "yolov5s.yaml"),
        help="Model config file path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("res", "configs", "data", "coco.yaml"),
        help="Dataset config file path",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=os.path.join("res", "configs", "cfg", "train_config.yaml"),
        help="Training config file path",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    with open(args.data, "r") as f:
        data_cfg = yaml.safe_load(f)

    with open(args.cfg, "r") as f:
        train_cfg = yaml.safe_load(f)

    aug_policy = MultiAugmentationPolicies(train_cfg["augmentation"])

    train_dataset = LoadImagesAndLabels(
        data_cfg["train_path"],
        cache_images=train_cfg["train"]["cache_image"],
        n_skip=train_cfg["train"]["n_skip"],
        batch_size=train_cfg["train"]["batch_size"],
        preprocess=lambda x: (x / 255.0).astype(np.float32),
        rect=train_cfg["train"]["rect"],
        pad=0,
        mosaic_prob=train_cfg["hyper_params"]["mosaic"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["train"]["batch_size"],
        num_workers=multiprocessing.cpu_count() - 1,
        # num_workers=0,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    val_dataset = LoadImagesAndLabels(
        data_cfg["val_path"],
        cache_images=train_cfg["train"]["cache_image"],
        n_skip=train_cfg["train"]["n_skip"],
        batch_size=train_cfg["train"]["batch_size"],
        preprocess=lambda x: (x / 255.0).astype(np.float32),
        rect=False,
        pad=0,
        mosaic_prob=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["train"]["batch_size"],
        num_workers=multiprocessing.cpu_count() - 1,
        # num_workers=0,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )

    device = select_device(
        train_cfg["train"]["device"], train_cfg["train"]["batch_size"]
    )
    model = YOLOModel(args.model, verbose=True)
    model, ema = TrainModelBuilder(model, train_cfg, device, "exp")

    trainer = YoloTrainer(
        model,
        train_cfg,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        ema=ema,
        device=device,
    )
    trainer.train()
