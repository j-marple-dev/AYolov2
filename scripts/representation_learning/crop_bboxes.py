"""Crop bounding box module.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""

import argparse
import os
from glob import glob

import cv2
from tqdm import tqdm


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="datasets/coco/images/train2017",
        required=True,
        help="Image data directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="datasets/coco/images/train2017_cropped",
        required=True,
        help="Cropped image data directory",
    )

    return parser.parse_args()


def crop_and_save_bboxes(img_dir: str, save_dir: str) -> None:
    """Crop and save cropped images by bounding boxes.

    Args:
        img_dir: Directory for original images
        save_dir: Directory to save cropped images
    """
    MIN_SIZE = 32
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    target_dir = img_dir.replace("images", "labels")
    target_paths = sorted(glob(f"{target_dir}/*"))
    for target_path in tqdm(target_paths, desc=img_dir.split("/")[-1]):
        # Load an image
        img_path = target_path.replace("labels", "images").replace("txt", "jpg")
        img = cv2.imread(img_path)
        img_w, img_h = img.shape[:2][::-1]

        with open(target_path, "r") as f:
            targets = f.read().splitlines()
            idx = 0
            for target in targets:
                # Get bounding box coordinates
                _, cx, cy, w, h = map(float, target.split())
                minx, w = int((cx - w / 2) * img_w), int(w * img_w)
                miny, h = int((cy - h / 2) * img_h), int(h * img_h)

                # Crop and save images by bounding boxes
                if w >= MIN_SIZE and h >= MIN_SIZE:
                    maxx, maxy = minx + w, miny + h
                    cropped_img = img[miny:maxy, minx:maxx]
                    img_name = os.path.basename(img_path)
                    img_name = img_name.replace(".jpg", f"_{idx:03d}.jpg")
                    cv2.imwrite(f"{save_dir}/{img_name}", cropped_img)
                    idx += 1


if __name__ == "__main__":
    args = get_parser()
    crop_and_save_bboxes(args.img_dir, args.save_dir)
