"""Unit test for augmentation.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import os

import cv2
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.augmentation.augmentation import (AugmentationPolicy,
                                               MultiAugmentationPolicies)
from scripts.data_loader.data_loader import LoadImages, LoadImagesAndLabels
from scripts.utils.constants import LABELS
from scripts.utils.general import draw_labels


def test_multi_aug_policies(show_gui: bool = False):
    label2str = LABELS["COCO"]
    batch_size = 16

    with open(os.path.join("tests", "res", "configs", "augmentation.yaml")) as f:
        aug_cfg = yaml.safe_load(f)

    aug_policy = MultiAugmentationPolicies(aug_cfg["augmentation"])

    dataset = LoadImagesAndLabels(
        "tests/res/datasets/coco/images/val2017",
        cache_images=None,
        n_skip=0,
        batch_size=batch_size,
        rect=False,
        augmentation=aug_policy,
        mosaic_prob=0.5,
    )
    dataset_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=LoadImagesAndLabels.collate_fn
    )

    pbar = tqdm(dataset_loader, desc="Load image test.")
    n_run = 0
    for (img, labels, path, shapes) in pbar:
        n_run += 1
        for i in range(img.shape[0]):
            np_image = img[i].numpy()[::-1].transpose((1, 2, 0))
            label_list = labels[labels[:, 0] == i][:, 1:]

            np_image = draw_labels(np_image, label_list.numpy(), label2str)

            if show_gui:
                cv2.imshow("test", np_image)
                cv2.waitKey(500)


def test_augmentation(show_gui: bool = False):
    label2str = LABELS["VOC"]
    batch_size = 16
    aug_prob = 0.5
    aug_policy = AugmentationPolicy(
        {"Blur": {"p": aug_prob}, "Flip": {"p": aug_prob}, "ToGray": {"p": aug_prob}},
        prob=0.5,
    )

    dataset = LoadImages(
        "tests/res/datasets/VOC/images/train",
        cache_images=None,
        n_skip=0,
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


if __name__ == "__main__":
    # test_augmentation(show_gui=True)
    test_multi_aug_policies(show_gui=True)
