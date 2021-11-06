"""Dataset loader.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import glob
import os
import random
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from p_tqdm import p_map
from PIL import ExifTags, Image
from torch.utils.data import Dataset
from tqdm import tqdm

from scripts.augmentation.yolo_augmentation import (augment_hsv, copy_paste,
                                                    copy_paste2, mixup,
                                                    random_perspective)
from scripts.utils.constants import LABELS
from scripts.utils.general import segments2boxes, xyn2xy, xywh2xyxy, xyxy2xywh
from scripts.utils.logger import get_logger

IMG_EXTS = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng"]
EXIF_REVERSE_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
CACHE_VERSION = "v0.2.4"
NUM_THREADS = os.cpu_count()

LOGGER = get_logger(__name__)


def get_files_hash(files: List[str]) -> float:
    """Return a single hash value of a list of files.

    Args:
        files: list of file paths

    Return:
        hash value
    """
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


class LoadImages(Dataset):
    """Load images only."""

    def __init__(
        self,
        path: Union[str, List[str]],
        img_size: int = 640,
        batch_size: int = 16,
        rect: bool = False,
        cache_images: Optional[str] = None,
        stride: int = 32,
        pad: float = 0.0,
        n_skip: int = 0,
        prefix: str = "",
        preprocess: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
        use_mp: bool = True,
    ) -> None:
        """Initialize LoadImages instance.

        Args:
            path: Image root directory.
            img_size: Minimum width or height size.
            batch_size: batch size to use in iterator.
            rect: use rectangular image.
            cache_images: use caching images. if None, caching will not be useda
                'dynamic_mem': Caching images once it has been loaded.
                'mem': Caching images in memory.
                'disk': Caching images in disk.
            stride: Stride value
            pad: pad size for rectangular image. This applies only when rect is True
            n_skip: Skip n images per one image. Ex) If we have 1024 images and n_skip is 1, then total 512 images will be used.
            prefix: logging prefix message
            preprocess: preprocess function which takes (x: np.ndarray) and returns (np.ndarray)
            augmentation: augmentation function which takes (x: np.ndarray) and returns (np.ndarray)
            use_mp: use multi process to read image shapes.
        """
        self.stride = stride
        self.img_size = img_size
        self.preprocess = preprocess
        self.rect = rect
        self.pad = pad
        self.augmentation = augmentation
        self.cache_images = cache_images
        self.n_skip = n_skip
        self.use_mp = use_mp

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
        self.img_hw0: List[Optional[Tuple[int, int]]] = [None] * self.n_img
        self.img_hw: List[Optional[Tuple[int, int]]] = [None] * self.n_img

        self.shapes = self._get_shapes()

        if self.rect:
            self.batch_shapes, indices = self._get_batch_shapes()
            self.img_files = [self.img_files[i] for i in indices]
            self.shapes = self.shapes[indices]

        if cache_images and cache_images.endswith("disk"):
            self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + "_npy")
            self.img_npy = [
                self.im_cache_dir / Path(f).with_suffix(".npy").name
                for f in self.img_files
            ]
            self.im_cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_images and not cache_images.startswith("dynamic"):
            gb = 0  # Gigabytes of cached images
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

    def shuffle(self) -> None:
        """Shufle image indices for random image order."""
        indices = np.arange(0, self.n_img)
        random.shuffle(indices)
        self.indices = indices

    def no_shuffle(self) -> None:
        """Un-shufle image indices for original order."""
        self.indices = range(self.n_img)

    def _get_shapes(self) -> np.ndarray:
        def __get_img_shape(_path: str) -> Tuple[int, int]:
            _err_msg = ""
            try:
                _image = Image.open(_path)
                _image.verify()
                _shape = _image.size

                try:
                    _img_exif = _image._getexif()
                except AttributeError as e:
                    _err_msg += f"[LoadImages] WARNING: Get EXIF failed on {_path}: {e}"
                    _img_exif = None

                if (
                    _img_exif is not None
                    and EXIF_REVERSE_TAGS["Orientation"] in _img_exif.keys()
                ):
                    _orientation = _img_exif[EXIF_REVERSE_TAGS["Orientation"]]

                    if _orientation in (6, 8):  # Rotation 270 or 90 degree
                        _shape = _shape[::-1]

                    assert (_shape[0] > 9) and (
                        _shape[1] > 9
                    ), f"Image size <10 pixels, ({_shape[0]}, {_shape[1]})"
            except Exception as e:
                _err_msg += f"[LoadImages] WARNING: {_path}: {e}"

            if _err_msg != "":
                LOGGER.warn(_err_msg)

            return _shape

        cache_path = str(Path(self.img_files[0]).parent) + f"_{self.n_skip}_skip.cache"

        cache: Optional[
            Dict[
                str,
                Union[
                    Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]],
                    Dict[str, Union[float, str, Dict[str, str]]],
                ],
            ]
        ] = None

        files_hash = get_files_hash(self.img_files)
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)

            if (
                "info" not in cache  # type: ignore
                or "hash" not in cache["info"]  # type: ignore
                or cache["info"]["hash"] != files_hash  # type: ignore
                or cache["info"]["version"] != CACHE_VERSION  # type: ignore
            ):
                cache = None

        if cache is None:
            if self.use_mp:
                shapes = p_map(
                    __get_img_shape, self.img_files, desc="Getting image shapes ...",
                )
            else:
                shapes = [
                    __get_img_shape(x)
                    for x in tqdm(self.img_files, "Getting image shapes ...")
                ]

            cache = {
                "info": {"version": CACHE_VERSION, "hash": files_hash},
                "shapes": shapes,
            }

            try:
                torch.save(cache, cache_path)
            except PermissionError as e:
                LOGGER.warn(f"Saving cache to {cache_path} has failed! {e}")
        else:
            shapes = cache["shapes"]

        return np.array(shapes)

    def _get_batch_shapes(self) -> Tuple[np.ndarray, np.ndarray]:
        aspect_ratio = self.shapes[:, 1] / self.shapes[:, 0]
        indices = aspect_ratio.argsort()

        aspect_ratio = aspect_ratio[indices]

        shapes = [[1, 1]] * self.total_n_batch
        for i in range(self.total_n_batch):
            aspect_ratio_i = aspect_ratio[self.batch_idx == i]
            min_aspect_ratio = aspect_ratio_i.min()
            max_aspect_ratio = aspect_ratio_i.max()

            if max_aspect_ratio < 1:
                shapes[i] = [max_aspect_ratio, 1]
            elif min_aspect_ratio > 1:
                shapes[i] = [1, 1 / min_aspect_ratio]

        batch_shapes = (
            np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(
                "int"
            )
            * self.stride
        )

        return batch_shapes, indices

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
                try:
                    im = np.load(npy)
                except ValueError:
                    LOGGER.warn(
                        f"Load npy cache filed. {npy}. Removing the cache and load from disk."
                    )
                    os.remove(npy)

            if im is None:
                path = self.img_files[index]
                im = cv2.imread(path)  # BGR
                assert im is not None, "Image Not Found " + path

            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(
                    im,
                    (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_AREA
                    if r < 1 and not self.augmentation
                    else cv2.INTER_LINEAR,
                )

            if self.cache_images is not None and self.cache_images.startswith(
                "dynamic"
            ):
                if self.cache_images.endswith("mem"):
                    self.imgs[index] = im
                    self.img_hw0[index] = (h0, w0)
                    self.img_hw[index] = im.shape[:2]
                elif self.cache_images.endswith("disk") and npy and not npy.exists():
                    npy_path = self.img_npy[index]
                    if npy_path is not None:
                        np.save(npy_path.as_posix(), im)

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
    ) -> Tuple[
        torch.Tensor,
        str,
        Tuple[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]],
    ]:
        """Get item from given index.

        Args:
            index: Index number for the image.

        Return:
            PyTorch image (CHW),
            Image path,
            Image shapes (Original, (ratio(new/original), pad(h,w)))
        """
        index = self.indices[index]
        img, (h0, w0), (h1, w1) = self._load_image(index)

        shape = (
            self.batch_shapes[self.batch_idx[index]]
            if self.rect
            else (self.img_size, self.img_size)
        )
        img, ratio, pad = self._letterbox(img, new_shape=shape, auto=False)

        if self.augmentation:
            img = self.augmentation(img)

        if self.preprocess:
            img = self.preprocess(img)

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        shapes = (h0, w0), ((h1 / h0, w1 / w0), pad)
        torch_img = torch.from_numpy(img)
        return torch_img, self.img_files[index], shapes

    def _letterbox(
        self,
        im: np.ndarray,
        new_shape: Optional[Tuple[int, int]] = None,
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

        if new_shape is None:
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
        rect: bool = False,
        label_type: str = "segments",
        # image_weights: bool = False,
        cache_images: Optional[str] = None,
        single_cls: bool = False,
        stride: int = 32,
        pad: float = 0.0,
        n_skip: int = 0,
        prefix: str = "",
        yolo_augmentation: Optional[Dict[str, Any]] = None,
        preprocess: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
        dataset_name: str = "COCO",
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

            img_size: Minimum width or height size.
            batch_size: Batch size
            rect: use rectangular image.
            label_type: label directory name. This should be either 'labels' or 'segments'
            cache_images: use caching images. if None, caching will not be used.
                'mem': Caching images in memory.
                'disk': Caching images in disk.
            stride: Stride value
            pad: pad size for rectangular image. This applies only when rect is True
            n_skip: Skip n images per one image. Ex) If we have 1024 images and n_skip is 1, then total 512 images will be used.
            yolo_augmentation: augmentation parameters for YOLO augmentation.
            prefix: logging prefix message
            preprocess: preprocess function which takes (x: np.ndarray) and returns (np.ndarray)
            augmentation: augmentation function which takes (x: np.ndarray, label: np.ndarray)
                    and returns (np.ndarray, np.ndarray).
                    label format is xyxy with pixel coordinates.
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
            augmentation=augmentation,
            rect=rect,
            pad=pad,
        )

        self.label_type = label_type
        # Define label paths
        substring_a = f"{os.sep}images{os.sep}"
        self.names = LABELS[dataset_name]
        substring_b = f"{os.sep}{self.label_type}{os.sep}"

        self.label_files = [
            x.replace(substring_a, substring_b, 1).replace(
                os.path.splitext(x)[-1], ".txt"
            )
            for x in self.img_files
        ]

        self.yolo_augmentation = (
            yolo_augmentation if yolo_augmentation is not None else {}
        )

        # Disable mosaic in rectangular images
        if self.rect:
            self.yolo_augmentation["mosaic"] = 0.0

        # Check cache
        cache_path = (
            str(Path(self.label_files[0]).parent) + f"_{self.n_skip}_skip.cache"
        )

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

        labels, segments = zip(*[label_cache[x] for x in self.img_files])
        self.labels = list(labels)
        self.segments = segments

        if single_cls:
            for label in self.labels:
                label[:, 0] = 0

    def __getitem__(  # type: ignore
        self, index: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        str,
        Tuple[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]],
    ]:
        """Get item from given index.

        Args:
            index: Index number for the image.

        Return:
            PyTorch image (CHW),
            Normalized(0.0 ~ 1.0) xywh labels,
            Image path,
            Image shapes (Original, (ratio(new/original), pad(h,w)))
        """
        index = self.indices[index]

        shape = (
            self.batch_shapes[self.batch_idx[index]]
            if self.rect
            else (self.img_size, self.img_size)
        )

        if random.random() < self.yolo_augmentation.get("mosaic", 0.0):
            img, labels = self._load_mosaic(index)
            shapes = (0, 0), ((0.0, 0.0), (0.0, 0.0))
            if random.random() < self.yolo_augmentation.get("mixup", 1.0):
                img, labels = mixup(
                    img,
                    labels,
                    *self._load_mosaic(random.randint(0, len(self.img_files) - 1)),
                )

        else:
            img, (h0, w0), (h1, w1) = self._load_image(index)

            img, ratio, pad = self._letterbox(
                img,
                new_shape=shape,
                auto=False,
                scale_fill=False,
                scale_up=self.yolo_augmentation.get("augment", False),
            )
            shapes = (h0, w0), ((h1 / h0, w1 / w0), pad)

            if self.labels[index].shape[0] == 0:
                labels = np.empty((0, 5), dtype=np.float32)
                segments = []
            else:
                labels = self.labels[index].copy()
                segments = self.segments[index].copy()

            # Adjust bboxes to the letterbox.
            if labels.size:
                labels[:, 1:] = xywh2xyxy(
                    labels[:, 1:], ratio=ratio, wh=(w1, h1), pad=pad
                )
                segments = [xyn2xy(x, wh=(w1, h1), pad=pad) for x in segments]

            # Copy-paste 2
            copy_paste_cfg = self.yolo_augmentation.get("copy_paste2", {})
            if copy_paste_cfg.get("p", 0.0) > 0.0:
                for _ in range(copy_paste_cfg.get("n_img", 3)):
                    img, labels, segments = self._load_copy_paste(
                        img=img, label=labels, seg=segments
                    )

            if self.yolo_augmentation.get("augment", False):
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=self.yolo_augmentation.get("degrees", 0.0),
                    translate=self.yolo_augmentation.get("translate", 0.1),
                    scale=self.yolo_augmentation.get("scale", 0.5),
                    shear=self.yolo_augmentation.get("shear", 0.0),
                    perspective=self.yolo_augmentation.get("perspective", 0.0),
                )  # border to remove

        # Normalize bboxes
        if labels.size:
            labels[:, 1:] = xyxy2xywh(
                labels[:, 1:], wh=img.shape[:2][::-1], clip_eps=1e-3
            )

        if self.augmentation:
            img, labels = self.augmentation(img, labels)

            if self.yolo_augmentation.get("augment", False):
                augment_hsv(
                    img,
                    self.yolo_augmentation.get("hsv_h", 0.015),
                    self.yolo_augmentation.get("hsv_s", 0.7),
                    self.yolo_augmentation.get("hsv_v", 0.4),
                )

        if self.preprocess:
            img = self.preprocess(img)

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        n_labels = len(labels)
        labels_out = torch.zeros((n_labels, 6))
        if n_labels > 0:
            labels_out[:, 1:] = torch.from_numpy(labels)
        torch_img = torch.from_numpy(img)

        return torch_img, labels_out, self.img_files[index], shapes

    def _load_mosaic(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        img_half_size = self.img_size // 2
        # height, width
        mosaic_center = [
            int(random.uniform(img_half_size, 2 * self.img_size - img_half_size))
            for _ in range(2)
        ]

        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        loaded_imgs = [self._load_image(idx) for idx in indices]

        mosaic_img = np.full(
            (
                int(self.img_size * 2),
                int(self.img_size * 2),
                loaded_imgs[0][0].shape[2],
            ),
            114,
            dtype=np.uint8,
        )
        mosaic_labels = []
        mosaic_segments = []

        mc_h, mc_w = mosaic_center
        for i, (img, _, (h, w)) in enumerate(loaded_imgs):
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(mc_w - w, 0), max(mc_h - h, 0), mc_w, mc_h
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = (
                    mc_w,
                    max(mc_h - h, 0),
                    min(mc_w + w, self.img_size * 2),
                    mc_h,
                )
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = (
                    max(mc_w - w, 0),
                    mc_h,
                    mc_w,
                    min(self.img_size * 2, mc_h + h),
                )
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = (
                    mc_w,
                    mc_h,
                    min(mc_w + w, self.img_size * 2),
                    min(self.img_size * 2, mc_h + h),
                )
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            if self.labels[indices[i]].shape[0] == 0:
                labels = np.empty((0, 5), dtype=np.float32)
                segments = []
            else:
                labels = self.labels[indices[i]].copy()
                segments = self.segments[indices[i]].copy()

            if labels.size:
                labels[:, 1:] = xywh2xyxy(labels[:, 1:], wh=(w, h), pad=(pad_w, pad_h))
                segments = [xyn2xy(x, wh=(w, h), pad=(pad_w, pad_h)) for x in segments]
            mosaic_labels.append(labels)
            mosaic_segments.extend(segments)

        mosaic_labels_np = np.concatenate(mosaic_labels, 0)

        for x in (mosaic_labels_np[:, 1:], *mosaic_segments):
            np.clip(x, 1e-3, 2 * self.img_size, out=x)

        mosaic_img, mosaic_labels_np, mosaic_segments = copy_paste(
            mosaic_img,
            mosaic_labels_np,
            mosaic_segments,
            p=self.yolo_augmentation.get("copy_paste", 0.0),
        )

        # Copy-paste 2
        copy_paste_cfg = self.yolo_augmentation.get("copy_paste2", {})
        if copy_paste_cfg.get("p", 0.0) > 0.0:
            for _ in range(copy_paste_cfg.get("n_img", 3)):
                mosaic_img, mosaic_labels_np, mosaic_segments = self._load_copy_paste(
                    mosaic_img, mosaic_labels_np, mosaic_segments
                )

        mosaic_img, mosaic_labels_np = random_perspective(
            mosaic_img,
            mosaic_labels_np,
            mosaic_segments,
            degrees=self.yolo_augmentation.get("degrees", 0.0),
            translate=self.yolo_augmentation.get("translate", 0.1),
            scale=self.yolo_augmentation.get("scale", 0.5),
            shear=self.yolo_augmentation.get("shear", 0.0),
            perspective=self.yolo_augmentation.get("perspective", 0.0),
            border=(-img_half_size, -img_half_size),
        )  # border to remove

        return mosaic_img, mosaic_labels_np

    def _load_copy_paste(
        self, img: np.ndarray, label: np.ndarray, seg: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Load copy paste augmentation.

        This method will copy paste objects from another image file.

        Args:
            img: input image.
            labels: image labels.
            seg: image object segmentations.

        Returns:
            Copy-pasted image.
            Copy-pasted labels.
            Copy-pasted segmentations.
        """
        img_idx_for_copy = random.choice(self.indices)
        img_for_copy, _, (h, w) = self._load_image(img_idx_for_copy)

        if self.labels[img_idx_for_copy].shape[0] == 0:
            labels_for_copy = np.empty((0, 5), dtype=np.float32)
            seg_for_copy = []
        else:
            labels_for_copy = self.labels[img_idx_for_copy].copy()
            seg_for_copy = self.segments[img_idx_for_copy].copy()

        if labels_for_copy.size:
            labels_for_copy[:, 1:] = xywh2xyxy(
                labels_for_copy[:, 1:], wh=(w, h), pad=(0, 0)
            )
            seg_for_copy = [xyn2xy(x, wh=(w, h), pad=(0, 0)) for x in seg_for_copy]

        copy_paste_cfg = (
            self.yolo_augmentation["copy_paste2"]
            if "copy_paste2" in self.yolo_augmentation.keys()
            else {}
        )

        copy_paste_img, copy_paste_label, copy_paste_seg = copy_paste2(
            im1=img,
            labels1=label,
            seg1=seg,
            im2=img_for_copy,
            labels2=labels_for_copy,
            seg2=seg_for_copy,
            scale_min=copy_paste_cfg.get("scale_min", 0.9),
            scale_max=copy_paste_cfg.get("scale_max", 1.1),
            p=copy_paste_cfg.get("p", 0.0),
            area_thr=copy_paste_cfg.get("area_thr", 10),
            ioa_thr=copy_paste_cfg.get("ioa_thr", 0.3),
        )

        return copy_paste_img, copy_paste_label, copy_paste_seg

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

        def _get_label(_img_path: str, _label_path: str) -> Dict[str, Any]:
            _label: Dict[str, Any] = {}
            _segments: List[np.ndarray] = []
            _err_msg = ""
            try:
                if os.path.isfile(_label_path):
                    with open(_label_path, "r") as _f:
                        _label_split_list = [
                            _x.split() for _x in _f.read().splitlines()
                        ]

                    if any([len(x) > 8 for x in _label_split_list]):  # is segment
                        _classes = np.array(
                            [_x[0] for _x in _label_split_list], dtype=np.float32
                        )
                        _segments = [
                            np.array(_x[1:], dtype=np.float32).reshape(-1, 2)
                            for _x in _label_split_list
                        ]  # (cls, xy1...)
                        _label_list = np.concatenate(
                            (_classes.reshape(-1, 1), segments2boxes(_segments)), 1
                        )  # (cls, xywh)
                    else:
                        _label_list = np.array(_label_split_list, dtype=np.float32)
                    if len(_label_list):
                        assert (
                            _label_list.shape[1] == 5
                        ), "labels require 5 columns each"
                        assert (_label_list >= 0).all(), "negative labels"
                        assert (
                            _label_list[:, 1:] <= 1
                        ).all(), "non-normalized or out of bounds coordinate labels"
                        assert (
                            np.unique(_label_list, axis=0).shape[0]
                            == _label_list.shape[0]
                        ), "duplicate labels"
                    else:
                        _label_list = np.zeros((0, 5), dtype=np.float32)

                    _label[_img_path] = (_label_list, _segments)
                else:
                    _label[_img_path] = (np.zeros((0, 5), dtype=np.float32), None)
            except Exception as e:
                _err_msg += f"[LoadImagesAndLabels] WARNING: {_img_path}: {e}"

                _label[_img_path] = (np.zeros((0, 5), dtype=np.float32), None)
                _label["info"] = {"msgs": {_img_path: _err_msg}}  # type: ignore

                LOGGER.warn(_err_msg)

            return _label

        # Multi core
        label_list = p_map(
            _get_label, self.img_files, self.label_files, desc="Scanning images ..."
        )
        # Single core
        # label_list = [_get_label(img_path, label_path) for img_path, label_path in tqdm(zip(self.img_files, self.label_files), desc="Scanning images ...")]

        labels: Dict[
            str,
            Union[
                Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]],
                Dict[str, Union[float, str, Dict[str, str]]],
            ],
        ] = {"info": {"version": CACHE_VERSION, "msgs": {}}}
        for label in label_list:
            if "info" in label:
                labels["info"]["msgs"].update(label["info"]["msgs"])  # type: ignore
                label.pop("info")
            labels.update(label)

        labels["info"]["hash"] = get_files_hash(self.label_files + self.img_files)  # type: ignore
        torch.save(labels, path)

        return labels
