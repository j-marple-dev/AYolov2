"""Validation hyper-parameters optimizer for YOLO.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Union

import optuna
import torch
import yaml
from torch import nn

from scripts.data_loader.data_loader import LoadImagesAndLabels
from scripts.utils.general import increment_path
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.torch_utils import count_param, load_pytorch_model
from scripts.utils.train_utils import YoloValidator
from scripts.utils.wandb_utils import load_model_from_wandb

LOGGER = get_logger(__name__)


class Objective:
    """Objective class for validation parameters optimizer."""

    """Base mAP50 threshold.

    If the mAP50 of the test model is under this value,
    trial objective score becomes zero.
    """
    BASE_mAP50: float = 0.688

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
        ).to(device).eval()
        self.baseline_n_params = count_param(self.baseline_model)

        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        self.baesline_map50: float
        self.baseline_t: float

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
            # compute_loss=isinstance(self.baseline_model, YOLOModel)
            # and hasattr(model, "hyp"),
            log_dir=None,
            incremental_log_dir=False,
            export=False,
        )
        LOGGER.info(
            f"Validating baseline model (n_parm: {self.baseline_n_params:,d}) ..."
        )
        t0 = time.monotonic()
        baseline_result = baseline_validator.validation(verbose=False)
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
        param_score = self.alpha * (self.baseline_n_params / self.model_params)
        time_score = self.beta * (self.baseline_t / time)
        map50_score = self.gamma * (map50 / self.baseline_map50)
        return param_score + time_score + map50_score

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Run model and compare with baseline model.

        Calculate the AIGC metirc, and optimize image size, IoU(intersection of union) threshold, confidence threshold.

        Args:
            trial: an optuna trial.

        Returns:
            a result of AIGC metric.
        """
        assert hasattr(self, "baseline_t") and hasattr(
            self, "baseline_map50"
        ), "Baseline information is missing. Required action: either baseline_t and baseline_map50 are set or self.test_baseline() has been called."

        LOGGER.info(f"Starting {trial.number}th trial.")
        if len(trial.study.best_trials):
            LOGGER.info("Showing previous best trials ...")

            for i, best_trial in enumerate(trial.study.best_trials):
                LOGGER.info(
                    f"[{(i+1):02d}:Best] {best_trial.number}th trial, Params: {best_trial.params}, Score: {best_trial.values}, Attr: {best_trial.user_attrs}"
                )

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
            # compute_loss=isinstance(self.model, YOLOModel)
            # and hasattr(self.model, "hyp"),
            half=self.args.half,
            log_dir=log_dir,
            incremental_log_dir=True,
            export=True,
        )

        LOGGER.info(
            f"{trial.number}th trial, Running validation (conf_thr: {self.cfg['hyper_params']['conf_t']}, iou_thr: {self.cfg['hyper_params']['iou_t']}, img_size: {self.cfg['train']['img_size']})"
        )

        t0 = time.monotonic()
        val_result = validator.validation(verbose=False)
        time_took = time.monotonic() - t0
        map50_result = val_result[0][2]

        LOGGER.info(
            f"   |----- Finished with mAP50: {map50_result}, time_took: {time_took:.3f}s"
        )

        trial.set_user_attr("map50", map50_result)
        trial.set_user_attr("time_took", time_took)

        if map50_result < Objective.BASE_mAP50:
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
        "--load-study",
        action="store_true",
        default=False,
        help="Load previous study if exists.",
    )
    parser.add_argument(
        "--study-name", type=str, default="val_optim", help="Optuna study name."
    )
    parser.add_argument(
        "--base-map50",
        type=float,
        default=0,
        help="Baseline mAP50 metric value. If base-map50 and base-time are given, baseline model validation will be skipped and use these values instead.",
    )
    parser.add_argument(
        "--base-time",
        type=float,
        default=0,
        help="Baseline validation time value. If base-map50 and base-time are given, baseline model validation will be skipped and use these values instead.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Score weight for parameter. Optuna study score will be computed by (alpha * param_score + beta * time_score + gamma * map50_score)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Score weight for parameter. Optuna study score will be computed by (alpha * param_score + beta * time_score + gamma * map50_score)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.6,
        help="Score weight for parameter. Optuna study score will be computed by (alpha * param_score + beta * time_score + gamma * map50_score)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

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

    if args.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

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

    if args.base_map50 > 0 and args.base_time > 0:
        objective.baseline_t = args.base_time
        objective.baseline_map50 = args.base_map50
    else:
        objective.test_baseline()

    db_file_name = ".val_optim_optuna.db"

    if not args.load_study and os.path.isfile(db_file_name):
        backup_db_file_name = Path(db_file_name).stem + "_backup.db"
        backup_db_file_name = increment_path(backup_db_file_name)
        LOGGER.info(
            f"Previous study has been found!, previous {colorstr('bold', db_file_name)} has been moved to {colorstr('bold', backup_db_file_name)}"
        )
        shutil.move(db_file_name, backup_db_file_name)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{db_file_name}",
        direction="maximize",
        load_if_exists=args.load_study,
    )
    study.optimize(objective, n_trials=args.n_trials)

    with open(save_path, "w") as f:
        yaml.dump(study.best_params, f)

    LOGGER.info(f"Optimized parameter has been saved to {colorstr('bold', save_path)}")
