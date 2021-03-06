"""PyTorch Model manager module.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from scripts.utils.torch_utils import ModelEMA

import torch
from torch import nn

import wandb
from scripts.utils.general import check_img_size, labels_to_class_weights
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.torch_utils import is_parallel, load_model_weights

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

LOGGER = get_logger(__name__)


class AbstractModelManager(ABC):
    """Abstract model manager."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        device: torch.device,
        weight_dir: Path,
    ) -> None:
        """Initialize model manager.

        Args:
            model: PyTorch model.
            cfg: training config.
            device: device to load the model weight.
            weight_dir: weight directory path.
        """
        self.model = model
        self.cfg = cfg
        self.device = device
        self.weight_dir = weight_dir
        self.start_epoch = 0

        if hasattr(self.model, "model_parser"):
            self.yaml = self.model.model_parser.cfg  # type: ignore
        else:
            self.yaml = None

    def load_model_weights(self, path: Optional[str] = None) -> nn.Module:
        """Load model weight.

        Args:
            path: model weight path. if it is None,
                self.cfg["train"]["weights"] will be used instead.

        Return:
            weights loaded model.
        """
        if path is None:
            return self._load_weight(self.cfg["train"]["weights"])
        else:
            return self._load_weight(path)

    @abstractmethod
    def _load_weight(self, path: str) -> nn.Module:
        """Abstract Load model weight.

        Read weights from the path and load them on to the model.

        Return:
            weights loaded model.
        """
        pass


class YOLOModelManager(AbstractModelManager):
    """YOLO Model Manager."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Dict[str, Any],
        device: torch.device,
        weight_dir: Path,
    ) -> None:
        """Initialize YOLO Model manager.

        Args:
            model: PyTorch model.
            cfg: training config.
            device: device to load the model weight.
            weight_dir: weight directory path.
        """
        super().__init__(model, cfg, device, weight_dir)

    def _load_weight(self, path: str) -> nn.Module:
        """Load weights from the model.

        Also, reads parameters to decide whether the model
        training has been completed or needs to be resumed.

        Return:
            weights loaded model.
        """
        start_epoch = 0
        pretrained = path.endswith(".pt")
        if path and not pretrained:
            best_weight = wandb.restore("best.pt", run_path=path)
            if not best_weight:
                LOGGER.warn(f"Failed downloading weight from wandb run path {path}")
            else:
                path = best_weight.name
                pretrained = path.endswith(".pt")

        if pretrained:
            ckpt = torch.load(path, map_location=self.device)
            # TODO(jeikeilim): Re-visit here.
            # exclude = (
            #     ["anchor"]
            #     if self.cfg["cfg"] or self.cfg["hyper_params"].get("anchors")
            #     else []
            # )
            exclude: List[str] = []
            self.model = load_model_weights(self.model, weights=ckpt, exclude=exclude)
            start_epoch = ckpt["epoch"] + 1
            if self.cfg["train"]["resume"]:
                assert start_epoch > 0, (
                    "%s training to %g epochs is finished, nothing to resume."
                    % (self.cfg["train"]["weights"], self.cfg["train"]["epochs"],)
                )
                if RANK in [-1, 0]:
                    LOGGER.info(
                        "Copying "
                        + colorstr("bold", f"{Path(path).parent.parent}")
                        + " to "
                        + colorstr("bold", f"{self.weight_dir.parent} ...")
                    )
                    shutil.copytree(
                        Path(path).parent.parent,
                        self.weight_dir.parent,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.yaml"),
                    )  # save previous files
                    shutil.copytree(
                        Path(path).parent.parent,
                        self.weight_dir.parent / f"backup_epoch{start_epoch - 1}",
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.pt"),
                    )  # save previous weights
                    shutil.copytree(
                        Path(path).parent,
                        self.weight_dir.parent / f"weight_epoch{start_epoch - 1}",
                        dirs_exist_ok=True,
                    )  # save previous weights
                self.start_epoch = start_epoch
            if self.cfg["train"]["epochs"] < start_epoch:
                LOGGER.info(
                    "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                    % (
                        self.cfg["train"]["weights"],
                        ckpt["epoch"],
                        self.cfg["train"]["epochs"],
                    )
                )
                self.start_epoch = 0

        return self.model

    def freeze(self, freeze_n_layer: int) -> nn.Module:
        """Freeze layers from the top.

        Args:
            freeze_n_layer: freeze from the top to n'th layer.
                0 will set all parameters to be trainable.
                i.e. freeze_n_layer = 3 will freeze
                model.0.*
                model.1.*
                model.2.*

        Return:
            frozen model.
        """
        freeze_list = [f"model.{x}." for x in range(freeze_n_layer)]
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze_list):
                LOGGER.info(f"freezing {k}")
                v.requires_grad = False

        return self.model

    def set_model_params(
        self, dataset: torch.utils.data.Dataset, ema: Optional["ModelEMA"] = None
    ) -> nn.Module:
        """Set necessary model parameters required in YOLO.

        Args:
            dataset: torch dataset which includes labels and names.
                names contain class names ex) ['person', 'cup', ...]

        Return:
            self.model with parameters
        """
        head = (
            self.model.module.model[-1] if is_parallel(self.model) else self.model.model[-1]  # type: ignore
        )  # YOLOHead module

        models = [self.model]

        nl = self.model.module.model[-1].nl if is_parallel(self.model) else self.model.model[-1].nl  # type: ignore
        # grid_size = int(max(self.model.stride))
        grid_size = (
            int(max(self.model.module.stride))  # type: ignore
            if is_parallel(self.model)
            else int(max(self.model.stride))  # type: ignore
        )

        imgsz = check_img_size(self.cfg["train"]["image_size"], grid_size)

        if ema:
            models.append(ema.ema)

        if is_parallel(self.model):
            models.append(self.model.module)  # type: ignore

        for model in models:
            model.nc = len(dataset.names)  # type: ignore
            model.hyp = self.cfg["hyper_params"]
            model.gr = 1.0  # type: ignore
            model.class_weights = labels_to_class_weights(dataset.labels, len(dataset.names)).to(  # type: ignore
                self.device
            )
            model.names = dataset.names  # type: ignore
            model.stride = head.stride  # type: ignore
            model.cfg = self.cfg  # type: ignore
            model.yaml = self.yaml  # type: ignore

            # Update loss weight hyper params
            # scale box loss with the number of head
            model.hyp["box"] *= 3.0 / nl  # type: ignore
            model.hyp["cls"] *= (  # type: ignore
                model.nc / 80.0 * 3.0 / nl  # type: ignore
            )  # scale to classes and layers
            model.hyp["obj"] *= (  # type: ignore
                (imgsz / 640) ** 2 * 3.0 / nl
            )  # scale box loss with image size

        return self.model
