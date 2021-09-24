"""Unit test for dataset loader.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.data_loader.data_loader import LoadImages, LoadImagesAndLabels


def test_load_images(show_gui: bool = False):
    batch_size = 16
    dataset = LoadImages(
        "tests/res/datasets/VOC/images/train",
        cache_images=None,
        n_skip=0,
        batch_size=batch_size,
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
                cv2.waitKey(1)

    assert n_run == 7


def test_load_images_and_labels(show_gui: bool = False):
    batch_size = 16
    dataset = LoadImagesAndLabels(
        "tests/res/datasets/VOC/images/train",
        cache_images=None,
        n_skip=0,
        batch_size=batch_size,
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

            if show_gui:
                cv2.imshow("test", np_image)
                cv2.waitKey(1)

    assert n_run == 7


if __name__ == "__main__":
    # test_load_images_and_labels()
    test_load_images()
