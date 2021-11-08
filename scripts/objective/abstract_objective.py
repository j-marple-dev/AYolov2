"""Abstract optuna objective module.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
import argparse
from abc import ABC, abstractmethod
from typing import Any, Dict

import optuna
import yaml


class AbstractObjective(ABC):
    """Abstract objective class for optuna optimization."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        optim_cfg: str,
        data_cfg: Dict[str, Any],
        args: argparse.Namespace,
    ) -> None:
        """Initialize AbstractObjective class.

        Args:
            model: torch model to validate and optimize params.
            device: a torch device.
            cfg: config to create validator.
            optim_cfg: optimizer config file path.
            data_cfg: dataset config.
            args: system arguments.
        """
        with open(optim_cfg, "rb") as f:
            self.optim_cfg = yaml.safe_load(f)
        self.data_cfg = data_cfg
        self.cfg = cfg
        self.args = args

    @abstractmethod
    def get_param(self, trial: optuna.trial.Trial) -> None:
        """Get objective parameters."""
        pass

    @abstractmethod
    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Calculate results for optimize."""
        pass
