"""Validation hyper-parameters optimizer for YOLO.

- Author: Haneol Kim, Jongkuk Lim
- Contact: hekim@jmarple.ai, limjk@jmarple.ai
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import optuna
import torch
import yaml
from torch import nn

from scripts.objective.objective_validator import ObjectiveValidator
from scripts.utils.general import increment_path
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.torch_utils import count_param, load_pytorch_model
from scripts.utils.wandb_utils import load_model_from_wandb

LOGGER = get_logger(__name__)


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
        default=0.681,
        help="Baseline mAP50 metric value. If base-map50 and base-time are given, baseline model validation will be skipped and use these values instead.",
    )
    parser.add_argument(
        "--base-time",
        type=float,
        default=331.63,
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
        help="Score weight for time. Optuna study score will be computed by (alpha * param_score + beta * time_score + gamma * map50_score)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.6,
        help="Score weight for mAP50. Optuna study score will be computed by (alpha * param_score + beta * time_score + gamma * map50_score)",
    )
    parser.add_argument(
        "--run-json",
        action="store_true",
        default=False,
        help="Optimize parameters with json or not.",
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
        "hyper_params": {"conf_t": 0.001, "iou_t": 0.65},
    }

    model.to(device).fuse().eval()  # type: ignore
    LOGGER.info(f"# of parameters: {count_param(model):,d}")
    if args.half:
        model.half()

    objective = ObjectiveValidator(model, device, cfg, args.optim_cfg, data_cfg, args)
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
