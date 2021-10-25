"""Unit test for augmentation.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import gc
import math
import os
import random

import cv2
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.augmentation.augmentation import (AugmentationPolicy,
                                               MultiAugmentationPolicies)
from scripts.data_loader.data_loader import LoadImages, LoadImagesAndLabels
from scripts.utils.constants import LABELS
from scripts.utils.plot_utils import draw_labels


def test_multi_aug_policies(show_gui: bool = False, p: float = 0.5):
    if random.random() > p:
        return

    label2str = LABELS["COCO"]
    batch_size = 8
    minimum_pixel = 4

    with open(os.path.join("tests", "res", "configs", "train_config_sample.yaml")) as f:
        cfg = yaml.safe_load(f)

    aug_policy = MultiAugmentationPolicies(cfg["augmentation"])

    dataset = LoadImagesAndLabels(
        "tests/res/datasets/coco/images/val2017",
        cache_images=None,
        n_skip=5,
        img_size=320,
        batch_size=batch_size,
        rect=False,
        augmentation=aug_policy,
        yolo_augmentation=cfg["yolo_augmentation"],
        label_type="segments",
    )
    dataset_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=LoadImagesAndLabels.collate_fn
    )

    pbar = tqdm(dataset_loader, desc="Load image test.")
    n_run = 0
    for (img, labels, path, shapes) in pbar:
        n_run += 1
        minimum_box_size = minimum_pixel / math.sqrt(img.shape[2] * img.shape[3])

        for i in range(img.shape[0]):
            np_image = img[i].numpy()[::-1].transpose((1, 2, 0))
            label_list = labels[labels[:, 0] == i][:, 1:]

            smaller_width_than_minimum_pixel = label_list[:, 3] < minimum_box_size
            smaller_height_than_minimum_pixel = label_list[:, 4] < minimum_box_size

            smaller_box_idx = torch.logical_or(
                smaller_width_than_minimum_pixel, smaller_height_than_minimum_pixel
            )

            # label_list = label_list[torch.logical_not(smaller_box_idx)]
            # label_list = label_list[smaller_box_idx]

            np_image = draw_labels(np_image, label_list.numpy(), label2str)

            if show_gui:
                cv2.imshow("test", np_image)
                cv2.waitKey(500)

            if smaller_box_idx.sum() > 0:
                print(
                    f"Smaller than {minimum_pixel}x{minimum_pixel} box size detected! Total {smaller_box_idx.sum()} boxes."
                )

            del np_image
        gc.collect()
    del dataset, dataset_loader
    gc.collect()


def test_augmentation(show_gui: bool = False, p: float = 0.5):
    if random.random() > p:
        return

    label2str = LABELS["VOC"]
    batch_size = 16
    aug_prob = 0.5
    aug_policy = AugmentationPolicy(
        {
            "Blur": {"p": aug_prob},
            "Flip": {"p": aug_prob},
            "ToGray": {"p": aug_prob},
            "BoxJitter": {"p": aug_prob, "jitter": 0.2},
        },
        prob=0.5,
    )

    dataset = LoadImages(
        "tests/res/datasets/VOC/images/train",
        cache_images=None,
        n_skip=4,
        img_size=320,
        batch_size=batch_size,
        rect=False,
        augmentation=aug_policy,
    )

    dataset_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=LoadImages.collate_fn
    )

    pbar = tqdm(dataset_loader, desc="Load image test.")
    n_run = 0
    for (img, path, shapes) in pbar:
        n_run += 1
        for i in range(img.shape[0]):
            np_image = img[i].numpy()[::-1].transpose((1, 2, 0))
            # label_list = labels[labels[:, 0] == i][:, 1:]

            # np_image = draw_labels(np_image, label_list.numpy(), label2str)

            if show_gui:
                cv2.imshow("test", np_image)
                cv2.waitKey(500)

            del np_image
        gc.collect()

    del dataset, dataset_loader
    gc.collect()


if __name__ == "__main__":
    test_augmentation(show_gui=True)
    # test_multi_aug_policies(show_gui=False)
