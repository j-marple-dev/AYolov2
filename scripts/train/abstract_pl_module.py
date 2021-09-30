"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
from abc import abstractmethod
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn


class AbstractPLModule(pl.LightningModule):
    """Abstract traniner class."""

    def __init__(
        self, model: nn.Module, hyp: Dict[str, Any], epochs: int, device: torch.device
    ) -> None:
        """Initialize AbstractTrainer class.

        Args:
            model: torch model to train.
            hyp: hyper parameter config.
            epochs: number of epochs to train.
            device: torch device to train on.
        """
        super().__init__()
        self.model = model
        self.hyp = hyp
        self.epochs = epochs
        self.torch_device = device

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore
        """Feed forward the model."""
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers and return optimizer."""
        pass

    @abstractmethod
    def training_step(  # type: ignore
        self,
        train_batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        """Train a step (a batch).

        Args:
            train_batch: train batch in tuple (input_x, y_true).
            batch_idx: current batch index.

        Returns:
            Result of loss function.
        """
        pass

    @abstractmethod
    def validation_step(  # type: ignore
        self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validate a step (a batch).

        Args:
            val_batch: validation data batch in tuple (input_x, true_y).
            batch_idx: current batch index.

        Returns:
            Result of loss function.
        """
        pass

    # def load_pretrained_model(self, state_dict_dir: str) -> None:
    #     """Load pretrained model from .pt file."""
    #     self.model.load_state_dict(torch.load(state_dict_dir))
