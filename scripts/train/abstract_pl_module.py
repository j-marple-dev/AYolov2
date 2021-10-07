"""Abstract PyTorch Lightning Module.

- Author: Haneol Kim, Jongkuk Lim
- Contact: hekim@jmarple.ai, limjk@jmarple.ai
"""
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import lr_scheduler


class AbstractPLModule(pl.LightningModule):
    """Abstract traniner class."""

    def __init__(self, model: nn.Module, cfg: Dict[str, Any]) -> None:
        """Initialize AbstractTrainer class.

        Args:
            model: torch model to train.
            cfg: model trainining hyper parameter config.
        """
        super().__init__()
        self.model = model
        self.cfg = cfg

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore
        """Feed forward the model."""
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
    def _init_optimizer(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[lr_scheduler.LambdaLR]]:
        """Initialize optimizer and scheduler.

        Returns:
            Any of these 6 options.
            A Single optimizer.
            Two lists - The first list has multiple optimizers, and the second has multiple LR schedulers (for multiple lr_scheduler_config).
            Dictionary - with and "optimizer" key, and (optionally) a "lr_scheduler" key whose value is a single LR scheduler or lr_sceduler_config.
            Tuple of dictionaries - as described above, with an optional "frequency" key.
            None - Fit will run without any optimizer.
        """

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
