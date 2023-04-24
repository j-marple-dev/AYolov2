"""Dataset loader.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import glob
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from p_tqdm import p_map
from PIL import ExifTags, Image
from torch.utils.data import Dataset
from tqdm import tqdm
from turbojpeg import TurboJPEG

IMG_EXTS = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng"]
EXIF_REVERSE_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
CACHE_VERSION = "v0.2.4"
NUM_THREADS = os.cpu_count()


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
        stride: int = 32,
        pad: float = 0.0,
        prefix: str = "",
        preprocess: Optional[Callable] = None,
        use_mp: bool = True,
    ) -> None:
        """Initialize LoadImages instance.

        Args:
            path: Image root directory.
            img_size: Minimum width or height size.
            batch_size: batch size to use in iterator.
            rect: use rectangular image.
            stride: Stride value
            pad: pad size for rectangular image. This applies only when rect is True
            prefix: logging prefix message
            preprocess: preprocess function which takes (x: np.ndarray) and returns (np.ndarray)
            use_mp: use multi process to read image shapes.
        """
        self.stride = stride
        self.img_size = img_size
        self.preprocess = preprocess
        self.rect = rect
        self.pad = pad
        self.use_mp = use_mp
        self.turbo_jpeg = TurboJPEG()

        # Get image paths
        self.img_files = self.__grep_all_images(path)

        self.n_img = len(self.img_files)
        assert self.n_img > 0, f"No images found in {path}"

        self.batch_idx = np.floor(np.arange(self.n_img) / batch_size).astype(int)
        self.total_n_batch = self.batch_idx[-1] + 1
        """Image indices for shuffling and image weight purpose"""
        self.indices = range(self.n_img)
        self.imgs = [None] * self.n_img
        self.img_hw0: List[Optional[Tuple[int, int]]] = [None] * self.n_img
        self.img_hw: List[Optional[Tuple[int, int]]] = [None] * self.n_img

        self.shapes = self._get_shapes()

        if self.rect:
            self.batch_shapes, indices = self._get_batch_shapes()
            self.img_files = [self.img_files[i] for i in indices]
            self.shapes = self.shapes[indices]

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
                print(_err_msg)

            return _shape

        if self.use_mp:
            shapes = p_map(
                __get_img_shape, self.img_files, desc="Getting image shapes ...",
            )
        else:
            shapes = [
                __get_img_shape(x)
                for x in tqdm(self.img_files, "Getting image shapes ...")
            ]

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
            path = self.img_files[index]

            with open(path, 'rb') as f:
                im = self.turbo_jpeg.decode(f.read())

            assert im is not None, "Image Not Found " + path

            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(
                    im,
                    (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_AREA
                    if r < 1 else cv2.INTER_LINEAR,
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