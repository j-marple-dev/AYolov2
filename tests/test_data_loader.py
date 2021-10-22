"""Unit test for dataset loader.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import gc
import multiprocessing

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.augmentation.augmentation import AugmentationPolicy
from scripts.data_loader.data_loader import LoadImages, LoadImagesAndLabels
from scripts.utils.constants import LABELS, PLOT_COLOR
from scripts.utils.general import xywh2xyxy
from scripts.utils.plot_utils import draw_labels


def test_load_images(show_gui: bool = False):
    batch_size = 16
    dataset = LoadImages(
        "tests/res/datasets/VOC/images/train",
        cache_images=None,
        img_size=320,
        n_skip=2,
        batch_size=batch_size,
        rect=False,
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

            if show_gui:
                cv2.imshow("test", np_image)
                cv2.waitKey(100)
            del np_image
        gc.collect()

    assert n_run == 4
    del dataset, dataset_loader
    gc.collect()


def test_load_images_and_labels(show_gui: bool = False):
    batch_size = 16
    label2str = LABELS["COCO"]

    aug_policy = AugmentationPolicy({"BoxJitter": {"p": 1.0, "jitter": 0.2}})

    dataset = LoadImagesAndLabels(
        # "tests/res/datasets/VOC/images/train",
        # "tests/res/datasets/coco/images/val2017",
        "tests/res/datasets/coco/images/train2017",
        cache_images=None,
        img_size=320,
        n_skip=3,
        batch_size=batch_size,
        preprocess=lambda x: (x / 255.0).astype(np.float32),
        rect=False,
        pad=0,
        label_type="segments",
        augmentation=aug_policy,
    )

    dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # num_workers=multiprocessing.cpu_count() - 1,
        num_workers=0,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )

    pbar = tqdm(dataset_loader, desc="Load image test.")
    n_run = 0
    for (img, labels, path, shapes) in pbar:
        n_run += 1
        pixel_label = labels.clone()

        for i in range(img.shape[0]):
            np_image = (img[i].numpy()[::-1].transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
            label_list = labels[labels[:, 0] == i][:, 1:]

            np_image = draw_labels(np_image, label_list.numpy(), label2str)

            if show_gui:
                cv2.imshow("test", np_image)
                cv2.waitKey(0)
            del np_image
            gc.collect()

    del dataset, dataset_loader
    gc.collect()

    assert n_run == 3


if __name__ == "__main__":
    # test_load_images(show_gui=False)
    test_load_images_and_labels(show_gui=True)
