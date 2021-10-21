"""Main script for your project.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""

import argparse
import multiprocessing
import os

import numpy as np
import torch
import yaml
from kindle import Model
from torch.utils.data import DataLoader

from scripts.augmentation.augmentation import AugmentationPolicy
from scripts.data_loader.data_loader_rl import LoadImagesForRL
from scripts.train.yolo_rl_trainer import YoloRepresentationLearningTrainer
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

    aug_policy = AugmentationPolicy(train_cfg["augmentation"])

    train_dataset = LoadImagesForRL(
        data_cfg["train_path"],
        batch_size=train_cfg["train"]["batch_size"],
        rect=train_cfg["train"]["rect"],
        cache_images=train_cfg["train"]["cache_image"],
        n_skip=train_cfg["train"]["n_skip"],
        augmentation=aug_policy,
        preprocess=lambda x: (x / 255.0).astype(np.float32),
        representation_learning=True,
        n_trans=train_cfg["train"]["n_trans"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["train"]["batch_size"],
        num_workers=0,  # multiprocessing.cpu_count() - 1,
        collate_fn=LoadImagesForRL.collate_fn,
    )
    val_dataset = LoadImagesForRL(
        data_cfg["val_path"],
        batch_size=train_cfg["train"]["batch_size"],
        rect=False,
        cache_images=train_cfg["train"]["cache_image"],
        n_skip=train_cfg["val"]["n_skip"],
        augmentation=aug_policy,
        preprocess=lambda x: (x / 255.0).astype(np.float32),
        representation_learning=True,
        n_trans=train_cfg["train"]["n_trans"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["train"]["batch_size"],
        num_workers=multiprocessing.cpu_count() - 1,
        collate_fn=LoadImagesForRL.collate_fn,
    )

    device = select_device(
        train_cfg["train"]["device"], train_cfg["train"]["batch_size"]
    )
    model = Model(args.model, verbose=True)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    trainer = YoloRepresentationLearningTrainer(
        model,
        train_cfg,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        n_trans=train_cfg["train"]["n_trans"],
    )
    trainer.train()
