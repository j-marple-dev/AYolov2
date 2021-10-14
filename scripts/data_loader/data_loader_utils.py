import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from scripts.augmentation.augmentation import MultiAugmentationPolicies
from scripts.data_loader.data_loader import LoadImagesAndLabels
from scripts.utils.torch_utils import torch_distributed_zero_first

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def create_dataloader(
    path: str,
    cfg: Dict[str, Any],
    stride: int,
    pad: float = 0.0,
    validation: bool = False,
    quad: bool = False,
    preprocess: Optional[Callable] = None,
    prefix="",
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Dataset]:
    """"""

    rank = LOCAL_RANK if not validation else -1
    batch_size = cfg["train"]["batch_size"] // WORLD_SIZE * (2 if validation else 1)
    workers = cfg["train"]["workers"]
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path,
            img_size=cfg["train"]["image_size"],
            batch_size=batch_size,
            rect=cfg["train"]["rect"]
            if not validation
            else True,  # rectangular training
            label_type=cfg["train"]["label_type"],
            cache_images=cfg["train"]["cache_image"] if not validation else None,
            single_cls=cfg["train"]["single_cls"],
            stride=int(stride),
            pad=pad,
            n_skip=cfg["train"]["n_skip"] if not validation else 0,
            prefix=prefix,
            # image_weights=image_weights,
            yolo_augmentation=cfg["yolo_augmentation"] if not validation else None,
            augmentation=MultiAugmentationPolicies(cfg["augmentation"])
            if not validation
            else None,
            preprocess=preprocess,
        )

    batch_size = min(batch_size, len(dataset))
    n_workers = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, workers]
    )  # number of workers
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    )
    loader = (
        torch.utils.data.DataLoader
        if cfg["train"]["image_weights"]
        else InfiniteDataLoader
    )
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """Dataloader that reuses workers Uses same syntax as
    torch.utils.data.dataloader.DataLoader."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever."""

    def __init__(self, sampler):
        """Initialize repeat sampler.

        Args:
            sampler (Sampler)
        """
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
