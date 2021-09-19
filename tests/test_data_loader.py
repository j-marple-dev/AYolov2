"""Unit test for dataset loader.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

from scripts.data_loader.data_loader import LoadImagesAndLabels


def test_load_images_and_labels():
    dataset_loader = LoadImagesAndLabels("../datasets/VOC/images/train2012")


if __name__ == "__main__":
    test_load_images_and_labels()
