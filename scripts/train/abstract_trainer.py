"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn


class AbstractTrainer(ABC):
    """Abstract trainer class."""

    def __init__(
        self, model: nn.Module, hyp: Dict[str, Any], epochs: int, device: torch.device
    ) -> None:
        """Initialize AbstractTrainer class."""
        super().__init__()
        self.model = model
        self.hyp = hyp
        self.epochs = epochs
        self.device = device

    @abstractmethod
    def training_step(
        self, batch: Union[List[torch.Tensor], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Train a step.

        Args:
            batch: batch data.
            batch_idx: batch index.
        """
        pass

    @abstractmethod
    def validation_step(
        self, batch: Union[List[torch.Tensor], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validate a step.

        Args:
            batch: batch data.
            batch_idx: batch index.
        """
        pass

    @abstractmethod
    def init_optimizer(self) -> Any:
        """Initialize optimizer."""
        pass
