"""Unit test for dataset loader.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import multiprocessing

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.data_loader.data_loader import LoadImages, LoadImagesAndLabels
from scripts.utils.constants import LABELS, PLOT_COLOR
from scripts.utils.general import draw_labels, xywh2xyxy


def test_load_images(show_gui: bool = False):
    batch_size = 16
    dataset = LoadImages(
        "tests/res/datasets/VOC/images/train",
        cache_images=None,
        n_skip=0,
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

    assert n_run == 7


def test_load_images_and_labels(show_gui: bool = False):
    batch_size = 16
    label2str = LABELS["COCO"]

    dataset = LoadImagesAndLabels(
        # "tests/res/datasets/VOC/images/train",
        # "tests/res/datasets/coco/images/val2017",
        "tests/res/datasets/coco/images/train2017",
        cache_images=None,
        n_skip=0,
        batch_size=batch_size,
        preprocess=lambda x: (x / 255.0),
        rect=False,
        pad=0,
        label_type="segments",
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
        pad = (torch.tensor([img.shape[2:]]) - torch.tensor(shapes)[:, 1, :]) / 2

        for i in range(img.shape[0]):
            np_image = (img[i].numpy()[::-1].transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
            label_list = labels[labels[:, 0] == i][:, 1:]

            np_image = draw_labels(np_image, label_list.numpy(), label2str)

            if show_gui:
                cv2.imshow("test", np_image)
                cv2.waitKey(0)

    assert n_run == 7


if __name__ == "__main__":
    # test_load_images(show_gui=False)
    test_load_images_and_labels(show_gui=True)
