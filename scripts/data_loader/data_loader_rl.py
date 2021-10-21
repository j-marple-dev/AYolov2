"""Dataset loader for representation learning.

- Author: Hyunwook Kim
- Contact: hwkim@jmarple.ai
"""

import os
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from scripts.data_loader.data_loader import LoadImages
from scripts.utils.logger import get_logger

LOGGER = get_logger(__name__)


class LoadImagesForRL(LoadImages):
    """Load images only for representation learning."""

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
        representation_learning: bool = True,
        n_trans: int = 2,
    ) -> None:
        """Initialize LoadImages instance.

        Args:
            path: Image root directory.
            img_size: Minimum width or height size.
            batch_size: batch size to use in iterator.
            rect: use rectangular image.
            cache_images: use caching images. if None, caching will not be used.
                'mem': Caching images in memory.
                'disk': Caching images in disk.
            stride: Stride value
            pad: pad size for rectangular image. This applies only when rect is True
            n_skip: Skip n images per one image. Ex) If we have 1024 images and n_skip is 1, then total 512 images will be used.
            prefix: logging prefix message
            preprocess: preprocess function which takes (x: np.ndarray) and returns (np.ndarray)
            augmentation: augmentation function which takes (x: np.ndarray) and returns (np.ndarray)
            representation_learning: bool value which is whether applying representation learning or not.
            n_trans: the number of times to apply transformations for representation learning.
        """
        super().__init__(
            path,
            img_size,
            batch_size,
            rect,
            cache_images,
            stride,
            pad,
            n_skip,
            prefix,
            preprocess,
            augmentation,
        )

        self.representation_learning = representation_learning
        self.n_trans = n_trans

    def __getitem__(
        self, index: int
    ) -> Tuple[
        Union[List[torch.Tensor], torch.Tensor],
        str,
        Tuple[Tuple[int, int], Tuple[int, int]],
    ]:
        """Get item from given index.

        Args:
            index: Index number for the image.

        Return:
            List of PyTorch Images or PyTorch image (CHW),
            Image path,
            Image shapes (Original, Resized)
        """
        index = self.indices[index]
        img, (h0, w0), (h1, w1) = self._load_image(index)  # BGR

        shape = (
            self.batch_shapes[self.batch_idx[index]]
            if self.rect
            else (self.img_size, self.img_size)
        )
        img = self._letterbox(img, new_shape=shape, auto=False)[0]
        shapes = (h0, w0), (h1, w1)

        if self.representation_learning:
            augmented_imgs = []
            for _ in range(self.n_trans):
                augmented_img = self.augmentation(img)
                augmented_img = augmented_img.transpose((2, 0, 1))[::-1]
                augmented_img = np.ascontiguousarray(augmented_img)
                augmented_imgs.append(augmented_img)

            if self.preprocess:
                augmented_imgs = [self.preprocess(i) for i in augmented_imgs]

            torch_imgs = [torch.from_numpy(i) for i in augmented_imgs]

            return torch_imgs, self.img_files[index], shapes
        else:
            if self.augmentation:
                img = self.augmentation(img)

            if self.preprocess:
                img = self.preprocess(img)

            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            torch_img = torch.from_numpy(img)

            return torch_img, self.img_files[index], shapes

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
        Returns:
            splitted information for collated batch item
        """
        img, path, shapes = zip(*batch)  # transposed
        img_flatten = []
        for i1, i2 in img:
            img_flatten.append(i1)
            img_flatten.append(i2)

        return torch.stack(img_flatten, 0), path, shapes


class LoadImagesForSimCLR(LoadImagesForRL):
    """Load images only for SimCLR."""

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
        representation_learning: bool = True,
        n_trans: int = 2,
    ) -> None:
        """Initialize LoadImagesForSimCLR instance.

        Args:
            path: Image root directory.
            img_size: Minimum width or height size.
            batch_size: batch size to use in iterator.
            rect: use rectangular image.
            cache_images: use caching images. if None, caching will not be used.
                'mem': Caching images in memory.
                'disk': Caching images in disk.
            stride: Stride value
            pad: pad size for rectangular image. This applies only when rect is True
            n_skip: Skip n images per one image. Ex) If we have 1024 images and n_skip is 1, then total 512 images will be used.
            prefix: loggin prefix message
            preprocess: preprocess function which takes (x: np.ndarray) and returns (np.ndarray)
            augmentation: augmentation function which takes (x: np.ndarray) and returns (np.ndarray)
            representation_learning: bool value which is whether applying representation learning or not.
            n_trans: the number of times to apply transformations for representation learning.
        """
        super().__init__(
            path,
            img_size,
            batch_size,
            rect,
            cache_images,
            stride,
            pad,
            n_skip,
            prefix,
            preprocess,
            augmentation,
            representation_learning,
            n_trans,
        )

    def _load_image(self, index: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Load image with index number.

        Returns:
            OpenCv Image (NumPy)
        """
        im = self.imgs[index]
        if im is None:
            npy = self.img_npy[index]
            if npy and npy.exists():
                try:
                    im = np.load(npy)
                except ValueError:
                    LOGGER.warn(
                        f"Load npy cache filed. {npy}. Removing the cache and load from disk."
                    )
                    os.remove(npy)

            if im is None:
                path = self.img_files[index]
                im = cv2.imread(path)
                assert im is not None, f"Image Not Found {path}"

            h0, w0 = im.shape[:2]  # original hw
            if self.cache_images is not None and self.cache_images.startswith(
                "dynamic"
            ):
                if self.cache_images.endswith("mem"):
                    self.imgs[index] = im
                    self.img_hw0[index] = (h0, w0)
                elif self.cache_images.endswith("disk") and npy and not npy.exists():
                    npy_path = self.img_npy[index]
                    if npy_path is not None:
                        np.save(npy_path.as_posix(), im)

            return im, (h0, w0)  # im, hw_original
        else:
            return (
                self.imgs[index],
                self.img_hw0[index],
            )  # im, hw_original

    def __getitem__(
        self, index: int
    ) -> Tuple[
        Union[List[torch.Tensor], torch.Tensor],
        str,
        Tuple[Tuple[int, int], Tuple[int, int]],
    ]:
        """Get item from given index.

        Args:
            index: Index number for the image.

        Returns:
            torch_imgs: List of PyTorch Images (CHW),
            Image path,
            shapes: Image shapes (Original, Resized)
        """
        index = self.indices[index]
        img, (h0, w0) = self._load_image(index)

        torch_imgs = []
        for _ in range(self.n_trans):
            augmented_img, _ = self.augmentation(img)
            augmented_img = augmented_img.transpose((2, 0, 1))[::-1]
            augmented_img = np.ascontiguousarray(augmented_img)
            if self.preprocess:
                augmented_img = self.preprocess(augmented_img)
            torch_imgs.append(torch.from_numpy(augmented_img))

        h1, w1 = augmented_img.shape[1:]
        shapes = ((h0, w0), (h1, w1))
        return torch_imgs, self.img_files[index], shapes
