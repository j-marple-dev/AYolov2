"""Validation hyper-parameters optimizer for YOLO.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import argparse
import os
import time
from typing import Optional, Union

import optuna
import torch
import yaml
from kindle import YOLOModel
from torch import nn

from scripts.data_loader.data_loader import LoadImagesAndLabels
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.torch_utils import (count_param, load_pytorch_model,
                                       select_device)
from scripts.utils.train_utils import YoloValidator
from scripts.utils.wandb_utils import load_model_from_wandb

LOGGER = get_logger(__name__)


class Objective:
    """Objective class for validation parameters optimizer."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cfg: dict,
        optim_cfg: str,
        data_cfg: dict,
        args: argparse.Namespace,
    ) -> None:
        """Objective optimizer.

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
        self.model = model
        self.model_params = count_param(self.model)
        self.device = device
        self.args = args

        # original yolov5x model
        self.baseline_model: nn.Module = load_model_from_wandb(
            "j-marple/AYolov2/1gxaqgk4"
        ).to(device).fuse().eval()
        self.baseline_n_params = count_param(self.baseline_model)
        self.baesline_map50: float
        self.baseline_t: float
        self.test_baseline()

    def get_param(self, trial: optuna.trial.Trial) -> None:
        """Get hyper params."""
        self.cfg["train"]["img_size"] = trial.suggest_int(
            "img_width", **self.optim_cfg["img_width"]
        )
        self.cfg["hyper_params"]["conf_t"] = trial.suggest_float(
            "conf_thr", **self.optim_cfg["conf_thr"]
        )
        self.cfg["hyper_params"]["iou_t"] = trial.suggest_float(
            "iou_thr", **self.optim_cfg["iou_thr"]
        )

    def test_baseline(self) -> None:
        """Validate baseline model network."""
        baseline_dataset = LoadImagesAndLabels(
            self.data_cfg["val_path"],
            img_size=640,
            batch_size=self.cfg["train"]["batch_size"],
            rect=self.cfg["train"]["rect"],
            label_type="labels",
            cache_images=None,
            single_cls=False,
            stride=int(max(self.baseline_model.stride)),  # type: ignore
            pad=0.5,
            n_skip=0,
            prefix="[val]",
            yolo_augmentation=None,
            augmentation=None,
            preprocess=None,
        )

        baseline_dataloader = torch.utils.data.DataLoader(
            baseline_dataset,
            batch_size=self.cfg["train"]["batch_size"],
            num_workers=min(os.cpu_count(), self.cfg["train"]["batch_size"]),  # type: ignore
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn,
        )

        baseline_validator = YoloValidator(
            self.baseline_model,
            baseline_dataloader,
            self.device,
            self.cfg,
            compute_loss=isinstance(self.baseline_model, YOLOModel)
            and hasattr(model, "hyp"),
            hybrid_label=self.args.hybrid_label,
            log_dir=None,
            incremental_log_dir=False,
            export=False,
        )
        t0 = time.monotonic()
        baseline_result = baseline_validator.validation()
        self.baseline_t = time.monotonic() - t0
        self.baseline_map50 = baseline_result[0][2]

    def calc_objective_fn(self, time: float, map50: float) -> float:
        """Calculate objective function.

        Objective function based on AIGC.

        Args:
            time: the time which the model(target) tooks.
            map50: the mAP@.5 result of the model(target).

        Returns:
            the result based on AIGC metric.
        """
        alpha, beta, gamma = 0.1, 0.3, 0.6
        return (
            alpha * (self.baseline_n_params / self.model_params)
            + beta * (self.baseline_t / time)
            + gamma * (map50 / self.baseline_map50)
        )

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Run model and compare with baseline model.

        Calculate the AIGC metirc, and optimize image size, IoU(intersection of union) threshold, confidence threshold.

        Args:
            trial: an optuna trial.

        Returns:
            a result of AIGC metric.
        """
        self.get_param(trial)
        if args.weights.endswith(".pt"):
            log_dir = os.path.join(args.weights[:-3])
        else:
            log_dir = os.path.join("wandb", "downloads", args.weights)

        val_dataset = LoadImagesAndLabels(
            self.data_cfg["val_path"],
            img_size=self.cfg["train"]["img_size"],
            batch_size=self.cfg["train"]["batch_size"],
            rect=self.cfg["train"]["rect"],
            label_type="labels",
            cache_images=None,
            single_cls=False,
            stride=int(max(self.model.stride)),  # type: ignore
            pad=0.5,
            n_skip=0,
            prefix="[val]",
            yolo_augmentation=None,
            augmentation=None,
            preprocess=None,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg["train"]["batch_size"],
            num_workers=min(os.cpu_count(), self.cfg["train"]["batch_size"]),  # type: ignore
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn,
        )

        validator = YoloValidator(
            self.model,
            val_loader,
            self.device,
            self.cfg,
            compute_loss=isinstance(self.model, YOLOModel)
            and hasattr(self.model, "hyp"),
            hybrid_label=self.args.hybrid_label,
            half=self.args.half,
            log_dir=log_dir,
            incremental_log_dir=True,
            export=True,
        )
        t0 = time.monotonic()
        val_result = validator.validation()
        time_took = time.monotonic() - t0
        map50_result = val_result[0][2]
        print(f"map50: {map50_result}")

        if map50_result < 0.688:
            return 0
        else:
            return self.calc_objective_fn(time_took, map50_result)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--weights", type=str, default="", help="Model weight path.")
    parser.add_argument(
        "--model-cfg", type=str, default="", help="Model config file path."
    )
    parser.add_argument(
        "--optim-cfg",
        type=str,
        default="./res/configs/cfg/val_optimizer.yaml",
        help="Optimize parameter config file path.",
    )
    parser.add_argument(
        "--data-cfg",
        type=str,
        default="res/configs/data/coco.yaml",
        help="Validation image root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device id. '' will use all GPUs. EX) '0,2' or 'cpu'",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of trials for optimization."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=512,
        help="Use top-k objects in NMS layer (TensorRT only)",
    )
    parser.add_argument(
        "-ktk",
        "--keep-top-k",
        default=100,
        help="Keep top-k after NMS. This must be less or equal to top-k (TensorRT only)",
    )
    parser.add_argument(
        "--rect",
        action="store_true",
        dest="rect",
        default=True,
        help="Use rectangular image",
    )
    parser.add_argument(
        "--no-rect", action="store_false", dest="rect", help="Use squared image.",
    )
    parser.add_argument(
        "--single-cls",
        action="store_true",
        default=False,
        help="Validate as single class only.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Save validation result plot.",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="Run half preceision model (PyTorch only)",
    )
    parser.add_argument(
        "--hybrid-label",
        action="store_true",
        default=False,
        help="Run NMS with hybrid information (ground truth label + predicted result.) (PyTorch only) This is for auto-labeling purpose.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    # if args.img_height < 0:
    #     args.img_height = args.img_width

    # Either weights or model_cfg must beprovided.
    if args.weights == "" and args.model_cfg == "":
        LOGGER.error(
            "Either "
            + colorstr("bold", "--weight")
            + " or "
            + colorstr("bold", "--model-cfg")
            + " must be provided."
        )
        exit(1)

    device = select_device(args.device, args.batch_size)

    # Unpack model from ckpt dict if the model has been saved during training.
    model: Optional[Union[nn.Module]] = None

    if args.weights == "":
        LOGGER.warning(
            "Providing "
            + colorstr("bold", "no weights path")
            + " will validate a randomly initialized model. Please use only for a experiment purpose."
        )
    elif args.weights.endswith(".pt"):
        model = load_pytorch_model(args.weights, args.model_cfg, load_ema=True)
        stride_size = int(max(model.stride))  # type: ignore

    else:  # load model from wandb
        model = load_model_from_wandb(args.weights)
        stride_size = int(max(model.stride))  # type: ignore

    if model is None:
        LOGGER.error(
            f"Load model from {args.weights} with config {args.model_cfg if args.model_cfg != '' else 'None'} has failed."
        )
        exit(1)

    with open(args.data_cfg, "r") as f:
        data_cfg = yaml.safe_load(f)

    # TODO(jeikeilim): config structure should be changed.
    cfg = {
        "train": {
            "single_cls": args.single_cls,
            "plot": args.plot,
            "batch_size": args.batch_size,
            "image_size": 0,
            "rect": args.rect,
        },
        "hyper_params": {"conf_t": 0, "iou_t": 0},
    }

    model.to(device).fuse().eval()  # type: ignore
    LOGGER.info(f"# of parameters: {count_param(model):,d}")
    if args.half:
        model.half()

    objective = Objective(model, device, cfg, args.optim_cfg, data_cfg, args)
    if args.weights.endswith(".pt"):
        os.makedirs(args.weights[:-3], exist_ok=True)
        save_path = os.path.join(args.weights[:-3], "params.yaml")
    else:
        log_dir = os.path.join("wandb", "downloads", args.weights)
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, "params.yaml")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    with open(save_path) as f:
        yaml.dump(study.best_params, f)
