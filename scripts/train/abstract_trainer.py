"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.utils.torch_utils import select_device


class AbstractTrainer(ABC):
    """Abstract trainer class."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        """Initialize AbstractTrainer class."""
        super().__init__()
        self.model = model
        self.cfg_train = cfg["train"]
        self.cfg_hyp = cfg["hyper_params"]
        self.epochs = self.cfg_train["epochs"]
        self.device = select_device(self.cfg_train["device"])
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    @abstractmethod
    def training_step(
        self,
        batch: Union[List[torch.Tensor], torch.Tensor, Tuple[torch.Tensor, ...]],
        batch_idx: int,
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
        for epoch in range(self.epochs):
            print(f"Epoch {epoch} starts.")
            self.on_start_epoch()
            self.model.train()
            for i, batch in enumerate(self.train_dataloader):
                self.training_step(batch, i)
            self.model.eval()
            for i, batch in enumerate(self.val_dataloader):
                self.validation_step(batch, i)
            self.on_end_epoch()

    def on_start_epoch(self) -> None:
        """Run on epoch starts."""
        pass

    def on_end_epoch(self) -> None:
        """Run on epoch ends."""
        pass
