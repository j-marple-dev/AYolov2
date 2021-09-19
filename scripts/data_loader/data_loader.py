"""Dataset loader.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import ExifTags, Image
from torch.utils.data import Dataset
from tqdm import tqdm

IMG_EXTS = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng"]
EXIF_REVERSE_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
CACHE_VERSION = "v0.1.0"


def get_files_hash(files: List[str]) -> float:
    """Return a single hash value of a list of files."""
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


class LoadImagesAndLabels(Dataset):  # for training/testing
    """Load Images and Labels Dataset."""

    def __init__(
        self,
        path: Union[str, List[str]],
        img_size: int = 640,
        batch_size: int = 16,
        augment: bool = False,
        hyp: Optional[dict] = None,
        rect: bool = False,
        image_weights: bool = False,
        cache_images: bool = False,
        cache_images_multiprocess: bool = False,
        single_cls: bool = False,
        stride: int = 32,
        pad: float = 0.0,
        rank: int = -1,
        n_skip: int = 0,
    ) -> None:
        """Initialize LoadImageAndLabels.

        Args:
            path: Root directory of images. The path must include `images` directory in the middle.
                And label files must include `labels` directory that matches with the image file path.
                Ex) ./data/COCO/images/train/image001.jpg,
                    ./data/COCO/images/train/image002.jpg,
                    ./data/COCO/images/train/image003.jpg, ...

                    ./data/COCO/labels/train/image001.txt,
                    ./data/COCO/labels/train/image002.txt,
                    ./data/COCO/labels/train/image003.txt, ...

            img_size: Image size (width)
            batch_size: Batch size
        """
        # Get image paths
        self.img_files = self._grep_all_images(path)

        # Skip n step if given.
        if n_skip > 0:
            self.img_files = [
                self.img_files[i] for i in range(0, len(self.img_files), n_skip)
            ]

        self.n_img = len(self.img_files)
        assert self.n_img > 0, f"No images found in {path}"

        # Define label paths
        substring_a = f"{os.sep}images{os.sep}"
        substring_b = f"{os.sep}labels{os.sep}"
        self.label_files = [
            x.replace(substring_a, substring_b, 1).replace(
                os.path.splitext(x)[-1], ".txt"
            )
            for x in self.img_files
        ]

        # Check cache
        cache_path = str(Path(self.label_files[0]).parent) + ".cache"

        cache: Optional[
            Dict[
                str,
                Union[
                    Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]],
                    Dict[str, Union[float, str, Dict[str, str]]],
                ],
            ]
        ] = None
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)
            assert cache is not None

            # Check if dataset has changed or version matches.
            if (
                "info" not in cache
                or "hash" not in cache["info"]
                or cache["info"]["hash"]  # type: ignore
                != get_files_hash(self.label_files + self.img_files)
                or cache["info"]["version"] != CACHE_VERSION  # type: ignore
            ):
                cache = None

        if cache is None:
            label_cache = self._cache_labels(cache_path)
        else:
            label_cache = cache

        labels, shapes = zip(*[label_cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        self.batch_idx = np.floor(np.arange(self.n_img) / batch_size).astype(int)
        self.total_n_batch = self.batch_idx[-1] + 1

    def _grep_all_images(self, path: Union[str, List[str]]) -> List[str]:
        """Grep all image files in the path.

        Args:
            path: Root directory of the images

        Return:
            Every image file paths in the given path.
        """
        try:
            img_files = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = str(Path(p))  # os-agnostic
                parent = str(Path(p).parent) + os.sep
                if os.path.isfile(p):  # file
                    with open(p, "r") as t:
                        text = t.read().splitlines()
                        img_files += [
                            x.replace("./", parent) if x.startswith("./") else x
                            for x in text
                        ]  # local to global path
                elif os.path.isdir(p):  # folder
                    img_files += glob.iglob(p + os.sep + "*.*")
                else:
                    raise Exception("%s does not exist" % p)

            return sorted(
                [
                    x.replace("/", os.sep)
                    for x in img_files
                    if os.path.splitext(x)[-1].lower() in IMG_EXTS
                ]
            )
        except Exception as e:
            raise Exception(f"Error loading data from {path}: {e}\n")

    def _cache_labels(self, path: str = "labels.cache") -> Dict:
        """Read labels and save cache file.

        Args:
            path: cache file path.

        Return:
            Read labels.
            {
                $IMG_PATH: (labels(n x 5), image_size(width, height)),
                ...
            }
        """
        pbar = tqdm(
            zip(self.img_files, self.label_files),
            desc="Scanning images ...",
            total=len(self.img_files),
        )

        labels: Dict[
            str,
            Union[
                Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]],
                Dict[str, Union[float, str, Dict[str, str]]],
            ],
        ] = {"info": {"version": CACHE_VERSION, "msgs": {}}}

        for (img, label) in pbar:
            try:
                image = Image.open(img)
                image.verify()

                shape = image.size
                img_exif = image._getexif()

                if (
                    img_exif is not None
                    and EXIF_REVERSE_TAGS["Orientation"] in img_exif.keys()
                ):
                    orientation = img_exif[EXIF_REVERSE_TAGS["Orientation"]]

                    if orientation in (6, 8):  # Rotation 270 or 90 degree
                        shape = shape[::-1]

                assert (shape[0] > 9) and (
                    shape[1] > 9
                ), f"Image size <10 pixels, ({shape[0]}, {shape[1]})"

                if os.path.isfile(label):
                    with open(label, "r") as f:
                        label_list = np.array(
                            [x.split() for x in f.read().splitlines()], dtype=np.float32
                        )

                    if len(label_list) == 0:
                        label_list = np.zeros((0, 5), dtype=np.float32)

                labels[img] = (label_list, shape)
            except Exception as e:
                err_msg = f"[LoadImagesAndLabels] WARNING: {img}: {e}"

                labels[img] = (None, None)
                labels["info"]["msgs"][img] = err_msg  # type: ignore

                print(err_msg)

        pbar.close()

        labels["info"]["hash"] = get_files_hash(self.label_files + self.img_files)  # type: ignore
        torch.save(labels, path)

        return labels
