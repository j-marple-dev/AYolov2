"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

if TYPE_CHECKING:
    from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader


class AbstractTrainer(ABC):
    """Abstract trainer class."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
    ) -> None:
        """Initialize AbstractTrainer class."""
        super().__init__()
        self.model = model
        self.cfg_train = cfg["train"]
        self.cfg_hyp = cfg["hyper_params"]
        self.epochs = self.cfg_train["epochs"]
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.cuda = self.device.type != "cpu"
        self.start_epoch = 0
        self.pbar: tqdm

    @abstractmethod
    def training_step(
        self,
        batch: Union[List[torch.Tensor], torch.Tensor, Tuple[torch.Tensor, ...]],
        batch_idx: int,
        epoch: int,
    ) -> torch.Tensor:
        """Train a step.

        Args:
            batch: batch data.
            batch_idx: batch index.

        Returns:
            Result of loss function.
        """
        pass

    @abstractmethod
    def validation(self) -> None:
        """Validate model."""
        pass

    def validation_step(
        self, batch: Union[List[torch.Tensor], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validate a step (a batch).

        Args:
            batch: batch data.
            batch_idx: batch index.

        Returns:
            Result of loss function.
        """
        pass

    @abstractmethod
    def _lr_function(self, x: float) -> float:
        """Learning rate scheduler function."""

    def configure_optimizers(
        self,
    ) -> List[Dict[str, Union[torch.optim.Optimizer, Dict[str, Any]]]]:
        """Configure optimizers and return optimizer."""
        optimizers, schedulers = self._init_optimizer()
        return [
            {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "metric_to_track"},
            }
            for optimizer, scheduler in zip(optimizers, schedulers)
        ]

    @abstractmethod
    def _init_optimizer(self) -> Any:
        """Initialize optimizer."""
        pass

    def train(self) -> None:
        """Train model."""
        self.on_train_start()

        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs):
            is_final_epoch = epoch + 1 == self.epochs
            self.on_start_epoch(epoch)
            self.model.train()
            for i, batch in (
                self.pbar if self.pbar else enumerate(self.train_dataloader)
            ):
                self.training_step(batch, i, epoch)
            self.on_end_epoch(epoch)
            if is_final_epoch or epoch % self.cfg_train["validate_period"] == 0:
                self.model.eval()
                self.validation()

    def on_start_epoch(self, epoch: int) -> None:
        """Run on epoch starts."""
        pass

    def on_end_epoch(self, epoch: int) -> None:
        """Run on epoch ends."""
        pass

    def on_train_start(self) -> None:
        """Run on start training."""
        self.start_epoch = 0
        pass

    def prepare_img(self, img: torch.Tensor) -> torch.Tensor:
        """Prepare image to float32 and normalize image.

        Args:
            img: input image tensor.
        """
        img = img.to(self.device, non_blocking=True)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        return img
