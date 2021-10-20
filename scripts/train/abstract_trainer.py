"""Module Description.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from tqdm import tqdm
    import wandb.sdk.wandb_run.Run

import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.utils.general import increment_path
from scripts.utils.logger import colorstr, get_logger

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

LOGGER = get_logger(__name__)


class AbstractTrainer(ABC):
    """Abstract trainer class.

    The running cycle of train() method is as following.

    --> on_train_start()
        for epoch in ...:
    ------> on_epoch_start()

            for i, batch in ...:
    ----------> training_step(i, batch, epoch)
    ------> on_epoch_end(epoch)
    ------> on_validation_start()
    ------> validation()
    ------> on_validation_end()
    --> on_train_end()
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        device: torch.device,
        log_dir: str = "exp",
        incremental_log_dir: bool = False,
        wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    ) -> None:
        """Initialize AbstractTrainer class."""
        super().__init__()
        self.model = model
        self.cfg_train = cfg["train"]
        self.cfg_hyp = cfg["hyper_params"]
        self.epochs = self.cfg_train["epochs"]
        self.current_epoch = 0
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.cuda = self.device.type != "cpu"
        self.start_epoch = 0
        self.pbar: Optional[tqdm] = None
        self.state: Dict[str, Any] = {
            "is_train": False,
            "epoch": self.start_epoch,
            "step": 0,
            "train_log": {},
            "val_log": {},
        }
        if incremental_log_dir:
            self.log_dir = increment_path(
                os.path.join(log_dir, "train", datetime.now().strftime("%Y_%m%d_runs"))
            )
        else:
            self.log_dir = log_dir
        self.wdir = Path(os.path.join(self.log_dir, "weights"))

        if RANK in [-1, 0]:
            os.makedirs(self.wdir, exist_ok=True)
            LOGGER.info("Log directory: " + colorstr("bold", f"{self.log_dir}"))
        self.wandb_run = wandb_run

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

    @torch.no_grad()
    def validation(self) -> None:
        """Run validation."""
        if self.val_dataloader is None:
            return

        if self.pbar is None:
            pbar = enumerate(self.val_dataloader)
        else:
            pbar = self.pbar

        for i, batch in pbar:
            self.state["step"] = i
            self.validation_step(batch, i)

    def validation_step(
        self,
        batch: Union[List[torch.Tensor], torch.Tensor, Tuple[torch.Tensor, ...]],
        batch_idx: int,
    ) -> None:
        """Evaluate one batch of validation dataset."""
        # NOTE: this is a dummy method for linting.
        # In fact, the `Validator` in `train_utils.py` is used for the validation.
        pass

    def train(self, start_epoch: int = 0) -> None:
        """Train model."""
        self.start_epoch = start_epoch
        self.on_train_start()

        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            self.state.update(
                {
                    "epoch": epoch,
                    "is_train": True,
                    "step": 0,
                    "train_log": {},
                    "val_log": {},
                }
            )

            is_final_epoch = epoch + 1 == self.epochs
            self.on_epoch_start(epoch)
            self.model.train()

            for i, batch in (
                self.pbar if self.pbar else enumerate(self.train_dataloader)
            ):
                self.state["step"] = i
                self.training_step(batch, i, epoch)

            self.on_epoch_end(epoch)

            self.state.update({"is_train": False, "step": 0})
            if self.val_dataloader is not None and (
                is_final_epoch or (epoch + 1) % self.cfg_train["validate_period"] == 0
            ):
                self.on_validation_start()
                self.model.eval()
                self.validation()
                self.on_validation_end()

            if self.wandb_run and RANK in [-1, 0]:
                self.log_wandb()

        self.on_train_end()

    def log_wandb(self) -> None:
        """Log metrics to WanDB."""
        pass

    def log_dict(self, data: Dict[str, Any]) -> None:
        """Log dictionary data.

        This method will log {data} into
        self.state["train_log"] or self.state["val_log"]
        depending on self.state["is_train"] state.
        Also, logged data will be reset on each end of the epoch.

        Args:
            data: dictionary log data.
        """
        key = "train_log" if self.state["is_train"] else "val_log"
        self.state[key].update(data)

    def on_epoch_start(self, epoch: int) -> None:
        """Run on epoch starts."""
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Run on epoch ends."""
        pass

    def on_train_start(self) -> None:
        """Run on start training."""
        pass

    def on_train_end(self) -> None:
        """Run on the end of training."""
        pass

    def on_validation_start(self) -> None:
        """Run on validation start."""
        pass

    def on_validation_end(self) -> None:
        """Run on validation end."""
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
