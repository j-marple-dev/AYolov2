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
        self,
        model: nn.Module,
        hyp: Dict[str, Any],
        epochs: int,
        device: torch.device,
        batch_size: int,
    ) -> None:
        """Initialize AbstractTrainer class.

        Args:
            model: Torch model to train. The pretrained weights should be loaded before if you attempt to use the weights.
            hyp: Hyper parameter config.
            epochs: Number of epochs to train.
            device: Torch device to train on.
        """
        super().__init__()
        self.model = model
        self.hyp = hyp
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore
        """Feed forward the model."""
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers and return optimizer."""
        if self.automatic_optimization:
            return self.init_optimizer()
        else:
            pass

    @abstractmethod
    def init_optimizer(self) -> Any:
        """Initialize optimizer and scheduler.

        Returns:
            Any of these 6 options.
            A Single optimizer.
            Two lists - The first list has multiple optimizers, and the second has multiple LR schedulers (for multiple lr_scheduler_config).
            Dictionary - with and "optimizer" key, and (optionally) a "lr_scheduler" key whose value is a single LR scheduler or lr_sceduler_config.
            Tuple of dictionaries - as described above, with an optional "frequency" key.
            None - Fit will run without any optimizer.
        """
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
            train_batch: Train batch in tuple (input_x, y_true).
            batch_idx: Current batch index.

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
            val_batch: Validation data batch in tuple (input_x, true_y).
            batch_idx: Current batch index.

        Returns:
            Result of loss function.
        """
        pass

    # def load_pretrained_model(self, state_dict_dir: str) -> None:
    #     """Load pretrained model from .pt file."""
    #     self.model.load_state_dict(torch.load(state_dict_dir))
