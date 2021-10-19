"""Unit test for representation learning.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""

import os
import shutil
from glob import glob

import cv2

from scripts.representation_learning.crop_bboxes import crop_and_save_bboxes


def test_crop_bboxes(show_gui: bool = False):
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

    # Check all whether all targets are cropped well or not
    assert num_cropped_imgs == num_targets
