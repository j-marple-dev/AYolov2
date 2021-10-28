"""Unit test for representation learning.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""

import gc
import multiprocessing
import os
import random
import shutil
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from kindle import Model
from torch.utils.data import DataLoader

from scripts.augmentation.augmentation import (AugmentationPolicy,
                                               MultiAugmentationPolicies)
from scripts.data_loader.data_loader_repr import (LoadImagesForRL,
                                                  LoadImagesForSimCLR)
from scripts.representation_learning.crop_bboxes import crop_and_save_bboxes
from scripts.train.yolo_repr_trainer import YoloRepresentationLearningTrainer
from scripts.utils.torch_utils import select_device


def test_crop_bboxes(show_gui: bool = False, force: bool = False, p: float = 0.5):
    if not force:
        return

    if random.random() > p:
        return

    MIN_SIZE = 32
    img_dir = "tests/res/datasets/coco/images/val2017"
    save_dir = "tests/res/datasets/coco/images/val2017_cropped"
    crop_and_save_bboxes(img_dir, save_dir)

    num_cropped_imgs = len(glob(f"{save_dir}/*"))
    shutil.rmtree(save_dir)

    target_dir = img_dir.replace("images", "labels")
    target_paths = glob(f"{target_dir}/*")
    num_targets = 0
    for target_path in target_paths:
        # Load an image
        img_path = target_path.replace("labels", "images").replace("txt", "jpg")
        img = cv2.imread(img_path)
        img_w, img_h = img.shape[:2][::-1]
        img_bbox = img.copy()

        with open(target_path, "r") as f:
            targets = f.read().splitlines()
            for target in targets:
                # Get bounding box coordinates
                _, cx, cy, w, h = map(float, target.split())
                x, w = int((cx - w / 2) * img_w), int(w * img_w)
                y, h = int((cy - h / 2) * img_h), int(h * img_h)

                if w >= MIN_SIZE and h >= MIN_SIZE:
                    num_targets += 1
                    img_bbox = cv2.rectangle(
                        img_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )

            if show_gui:
                cv2.imshow(os.path.basename(img_path), img_bbox)
                cv2.waitKey(0)

        del img, img_bbox
        gc.collect()

    # Check all whether all targets are cropped well or not
    assert num_cropped_imgs == num_targets

    
def test_train_repr(force: bool = False, p: float = 0.5) -> None:
    if not force:
        return

    if random.random() > p:
        return

    with open(
        os.path.join("tests", "res", "configs", "train_config_repr.yaml"), "r"
    ) as f:
        cfg = yaml.safe_load(f)

    cfg["train"]["epochs"] = 1
    cfg["train"]["n_skip"] = 10
    cfg["train"]["image_size"] = 320
    if not torch.cuda.is_available():
        cfg["train"]["device"] = "cpu"

    device = select_device(cfg["train"]["device"], cfg["train"]["batch_size"])

    aug_policy = AugmentationPolicy(cfg["augmentation"])

    train_dataset = LoadImagesForRL(
        "tests/res/datasets/coco/images/train2017",
        img_size=cfg["train"]["image_size"],
        batch_size=cfg["train"]["batch_size"],
        n_skip=cfg["val"]["n_skip"],
        augmentation=aug_policy,
        preprocess=lambda x: (x / 255.0).astype(np.float32),
        representation_learning=True,
        n_trans=2,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        num_workers=min(cfg["train"]["batch_size"], multiprocessing.cpu_count()),
        collate_fn=LoadImagesForRL.collate_fn,
    )
    val_dataset = LoadImagesForRL(
        "tests/res/datasets/coco/images/val2017",
        img_size=cfg["train"]["image_size"],
        batch_size=cfg["train"]["batch_size"],
        n_skip=cfg["val"]["n_skip"],
        augmentation=aug_policy,
        preprocess=lambda x: (x / 255.0).astype(np.float32),
        representation_learning=True,
        n_trans=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        num_workers=min(cfg["train"]["batch_size"], multiprocessing.cpu_count()),
        collate_fn=LoadImagesForRL.collate_fn,
    )

    model = Model(
        os.path.join("tests", "res", "configs", "model_yolov5s_repr.yaml"),
        verbose=True,
    )

    temperature = cfg["train"].get("temperature", 0.0)
    trainer = YoloRepresentationLearningTrainer(
        model,
        cfg,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        n_trans=2,
        rl_type="base",
        temperature=temperature,
    )
    trainer.train()

    del (
        device,
        aug_policy,
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        model,
        trainer,
    )
    gc.collect()


def test_train_simclr(force: bool = False) -> None:
    if not force:
        return

    with open(
        os.path.join("tests", "res", "configs", "train_config_simclr.yaml"), "r"
    ) as f:
        cfg = yaml.safe_load(f)

    cfg["train"]["epochs"] = 1
    cfg["train"]["n_skip"] = 10
    cfg["train"]["image_size"] = 320
    if not torch.cuda.is_available():
        cfg["train"]["device"] = "cpu"

    device = select_device(cfg["train"]["device"], cfg["train"]["batch_size"])

    aug_policy = MultiAugmentationPolicies(cfg["augmentation"])

    train_dataset = LoadImagesForSimCLR(
        "tests/res/datasets/coco/images/train2017",
        img_size=cfg["train"]["image_size"],
        batch_size=cfg["train"]["batch_size"],
        n_skip=cfg["val"]["n_skip"],
        augmentation=aug_policy,
        preprocess=lambda x: (x / 255.0).astype(np.float32),
        representation_learning=True,
        n_trans=2,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        num_workers=min(cfg["train"]["batch_size"], multiprocessing.cpu_count()),
        collate_fn=LoadImagesForSimCLR.collate_fn,
    )
    val_dataset = LoadImagesForSimCLR(
        "tests/res/datasets/coco/images/val2017",
        img_size=cfg["train"]["image_size"],
        batch_size=cfg["train"]["batch_size"],
        n_skip=cfg["val"]["n_skip"],
        augmentation=aug_policy,
        preprocess=lambda x: (x / 255.0).astype(np.float32),
        representation_learning=True,
        n_trans=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        num_workers=min(cfg["train"]["batch_size"], multiprocessing.cpu_count()),
        collate_fn=LoadImagesForSimCLR.collate_fn,
    )

    model = Model(
        os.path.join("tests", "res", "configs", "model_simclr.yaml"), verbose=True,
    )

    temperature = cfg["train"].get("temperature", 0.0)
    trainer = YoloRepresentationLearningTrainer(
        model,
        cfg,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        n_trans=2,
        rl_type="simclr",
        temperature=temperature,
    )
    trainer.train()

    del (
        device,
        aug_policy,
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        model,
        trainer,
    )
    gc.collect()


if __name__ == "__main__":
    test_crop_bboxes()
    test_train_repr()
    test_train_simclr()
