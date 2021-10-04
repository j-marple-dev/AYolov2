"""Unit test for dataset loader.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import multiprocessing

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.data_loader.data_loader import LoadImages, LoadImagesAndLabels
from scripts.utils.constants import LABELS, PLOT_COLOR
from scripts.utils.general import xywh2xyxy


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
    label2str = LABELS["COCO"]

    dataset = LoadImagesAndLabels(
        # "tests/res/datasets/VOC/images/train",
        # "tests/res/datasets/coco/images/val2017",
        "tests/res/datasets/coco/images/train2017",
        cache_images=None,
        n_skip=0,
        batch_size=batch_size,
    )

    dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=multiprocessing.cpu_count() - 1,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )

    pbar = tqdm(dataset_loader, desc="Load image test.")
    n_run = 0
    for (img, labels, path, shapes) in pbar:
        n_run += 1
        pixel_label = labels.clone()
        pad = (torch.tensor([img.shape[2:]]) - torch.tensor(shapes)[:, 1, :]) / 2

        for i in range(img.shape[0]):
            np_image = img[i].numpy()[::-1].transpose((1, 2, 0))
            label_list = labels[labels[:, 0] == i][:, 1:]
            label_list[:, 1:] = xywh2xyxy(label_list[:, 1:], wh=img.shape[2:])

            # TODO(jeikeilim): Make this as plot class or function.
            for label in label_list:
                class_id = int(label[0])
                class_str = label2str[class_id]

                xy1 = tuple(label[1:3].numpy().astype("int"))
                xy2 = tuple(label[3:5].numpy().astype("int"))
                plot_color = tuple(map(int, PLOT_COLOR[class_id]))
                overlay_alpha = 0.3

                overlay = np_image.copy()
                overlay = cv2.rectangle(overlay, xy1, xy2, plot_color, -1)
                np_image = cv2.addWeighted(
                    overlay, overlay_alpha, np_image, 1 - overlay_alpha, 0
                )
                np_image = cv2.rectangle(np_image, xy1, xy2, plot_color, 1)

                (text_width, text_height), baseline = cv2.getTextSize(
                    class_str, 3, 0.5, 1
                )
                overlay = np_image.copy()
                overlay = cv2.rectangle(
                    overlay,
                    (xy1[0], xy1[1] - text_height),
                    (xy1[0] + text_width, xy1[1]),
                    (plot_color[0] // 0.3, plot_color[1] // 0.3, plot_color[2] // 0.3),
                    -1,
                )
                np_image = cv2.addWeighted(
                    overlay, overlay_alpha + 0.2, np_image, 0.8 - overlay_alpha, 0
                )
                cv2.putText(
                    np_image,
                    class_str,
                    xy1,
                    3,
                    0.5,
                    (plot_color[0] // 3, plot_color[1] // 3, plot_color[2] // 3),
                    1,
                    cv2.LINE_AA,
                )

            if show_gui:
                cv2.imshow("test", np_image)
                cv2.waitKey(500)

    assert n_run == 7


if __name__ == "__main__":
    # test_load_images_and_labels()
    test_load_images_and_labels(show_gui=True)
