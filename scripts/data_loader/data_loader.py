"""Dataset loader.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import glob
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import ExifTags, Image
from torch.utils.data import Dataset
from tqdm import tqdm

IMG_EXTS = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng"]
EXIF_REVERSE_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
CACHE_VERSION = "v0.1.0"
NUM_THREADS = os.cpu_count()


def get_files_hash(files: List[str]) -> float:
    """Return a single hash value of a list of files."""
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


class LoadImages(Dataset):
    """Load images only."""

    def __init__(
        self,
        path: Union[str, List[str]],
        img_size: int = 640,
        batch_size: int = 16,
        # rect: bool = False,
        cache_images: Optional[str] = None,
        stride: int = 32,
        # pad: float = 0.0,
        # rank: int = -1,
        n_skip: int = 0,
        prefix: str = "",
        preprocess: Optional[Callable] = None,
    ) -> None:
        """Initialize LoadImages instance.

        Args:
            path: Image root directory.
            img_size: Minimum width or height size.
            batch_size: batch size to use in iterator.
            cache_images: use caching images. if None, caching will not be used.
                'mem': Caching images in memory.
                'disk': Caching images in disk.
            stride: Stride value
            n_skip: Skip n images per one image. Ex) If we have 1024 images and n_skip is 1, then total 512 images will be used.
            prefix: logging prefix message
            preprocess: preprocess function.
        """
        self.stride = stride
        self.img_size = img_size
        self.preprocess = preprocess

        # Get image paths
        self.img_files = self.__grep_all_images(path)

        # Skip n step if given.
        if n_skip > 0:
            self.img_files = [
                self.img_files[i] for i in range(0, len(self.img_files), n_skip)
            ]

        self.n_img = len(self.img_files)
        assert self.n_img > 0, f"No images found in {path}"

        self.batch_idx = np.floor(np.arange(self.n_img) / batch_size).astype(int)
        self.total_n_batch = self.batch_idx[-1] + 1
        """Image indices for shuffling and image weight purpose"""
        self.indices = range(self.n_img)
        self.imgs = [None] * self.n_img
        self.img_npy: List[Optional[Path]] = [None] * self.n_img

        if cache_images:
            if cache_images == "disk":
                self.im_cache_dir = Path(
                    Path(self.img_files[0]).parent.as_posix() + "_npy"
                )
                self.img_npy = [
                    self.im_cache_dir / Path(f).with_suffix(".npy").name
                    for f in self.img_files
                ]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * self.n_img, [None] * self.n_img
            results = ThreadPool(NUM_THREADS).imap(
                lambda x: self._load_image(x), range(self.n_img)
            )
            pbar = tqdm(enumerate(results), total=self.n_img)
            for i, x in pbar:
                if cache_images == "disk":
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    (
                        self.imgs[i],
                        self.img_hw0[i],
                        self.img_hw[i],
                    ) = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.imgs[i].nbytes
                pbar.desc = f"{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})"
            pbar.close()

    def __grep_all_images(self, path: Union[str, List[str]]) -> List[str]:
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

    def _load_image(
        self, index: int
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Load image with index number.

        Return:
            OpenCv Image(NumPy) with resize only.
        """
        # loads 1 image from dataset index 'i', returns im, original hw, resized hw
        im = self.imgs[index]
        if im is None:  # not cached in ram
            npy = self.img_npy[index]
            if npy and npy.exists():  # load npy
                im = np.load(npy)
            else:  # read image
                path = self.img_files[index]
                im = cv2.imread(path)  # BGR
                assert im is not None, "Image Not Found " + path
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(
                    im,
                    (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR,
                )
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        else:
            return (
                self.imgs[index],
                self.img_hw0[index],
                self.img_hw[index],
            )  # im, hw_original, hw_resized

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.img_files)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, str, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get item from given index.

        Args:
            index: Index number for the image.

        Return:
            PyTorch image (CHW),
            Image path,
            Image shapes (Original, Resized)
        """
        index = self.indices[index]
        img, (h0, w0), (h1, w1) = self._load_image(index)

        if self.preprocess:
            img = self.preprocess(img)

        img = self._letterbox(img, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        shapes = (h0, w0), (h1, w1)
        return torch.from_numpy(img), self.img_files[index], shapes

    def _letterbox(
        self,
        im: np.ndarray,
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = True,
        scale_fill: bool = False,
        scale_up: bool = True,
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        """Make letterbox around the given image.

        Args:
            im: Image to apply letterbox.
            color: color to fill letterbox.
            auto: Auto apply to minimum rectagular letterbox.
                Ex) If image is 640x432 and stride is 32,
                    output image will be 640x448 when auto is True,
                    otherwise, 640x640 when auto is False.
            scale_fill: stretch image size
            scale_up: scale up image. If False, scale down only.

        Return:
            letter-boxed image,
            Scale ratio used to resize,
            padding size of the letterbox.
        """
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = (self.img_size, self.img_size)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scale_up:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw: float = new_shape[1] - new_unpad[0]  # w padding
        dh: float = new_shape[0] - new_unpad[1]  # h padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border

        return im, ratio, (dw, dh)

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, str, Tuple[Tuple[int, int], Tuple[int, int]]]]
    ) -> Tuple[
        torch.Tensor,
        Tuple[str, ...],
        Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...],
    ]:
        """Handle Collate in PyTorch.

        Args:
            batch: collated batch item.
        """
        img, path, shapes = zip(*batch)  # transposed
        return torch.stack(img, 0), path, shapes


class LoadImagesAndLabels(LoadImages):  # for training/testing
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
        cache_images: Optional[str] = None,
        single_cls: bool = False,
        stride: int = 32,
        pad: float = 0.0,
        rank: int = -1,
        n_skip: int = 0,
        prefix: str = "",
        preprocess: Optional[Callable] = None,
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
        super().__init__(
            path,
            img_size=img_size,
            batch_size=batch_size,
            cache_images=cache_images,
            stride=stride,
            n_skip=n_skip,
            prefix=prefix,
            preprocess=preprocess,
        )

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

    def __getitem__(  # type: ignore
        self, index: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, str, Tuple[Tuple[int, int], Tuple[int, int]]
    ]:
        """Get item from given index.

        Args:
            index: Index number for the image.

        Return:
            PyTorch image (CHW),
            Labels,
            Image path,
            Image shapes (Original, Resized)
        """
        index = self.indices[index]
        img, (h0, w0), (h1, w1) = self._load_image(index)

        if self.preprocess:
            img = self.preprocess(img)

        # TODO(jeikeilim): Add mosaic augmentation and other augmentations.
        # TODO(jeikeilim): Fix labels xywh to xyxy.
        img, ratio, pad = self._letterbox(img, auto=False)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        shapes = (h0, w0), (h1, w1)

        # TODO(jeikeilim): Fix labels to letterbox applied
        labels = self.labels[index].copy()
        n_labels = len(labels)
        labels_out = torch.zeros((n_labels, 6))
        if n_labels > 0:
            labels_out[:, 1:] = torch.from_numpy(labels)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(  # type: ignore
        batch: List[
            Tuple[
                torch.Tensor, torch.Tensor, str, Tuple[Tuple[int, int], Tuple[int, int]]
            ]
        ]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Tuple[str, ...],
        Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...],
    ]:
        """Handle Collate in PyTorch.

        Args:
            batch: collated batch item.
        """
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

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
