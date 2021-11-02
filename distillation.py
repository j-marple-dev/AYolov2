"""Train object detection model with Knowledge-Distillation.

- Author: Hyung-Seok Shin
- Contact: hsshin@jmarple.ai
"""
import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from kindle import YOLOModel

from scripts.data_loader.data_loader_utils import create_dataloader
from scripts.train.kd_trainer import SoftTeacherTrainer
from scripts.utils.logger import get_logger
from scripts.utils.model_manager import YOLOModelManager
from scripts.utils.torch_utils import load_pytorch_model, select_device
from scripts.utils.wandb_utils import load_model_from_wandb

LOGGER = get_logger(__name__)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("res", "configs", "model", "yolov5s.yaml"),
        help="Model config file path",
    )
    parser.add_argument(
        "--teacher", type=str, help="Teacher model checkpoint file path",
    )
    parser.add_argument(
        "--teacher_cfg",
        type=str,
        default="",
        help="Model config filepath of Teacher model.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("res", "configs", "data", "coco.yaml"),
        help="Dataset config file path",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=os.path.join("res", "configs", "cfg", "train_config.yaml"),
        help="Training config file path",
    )
    parser.add_argument("--device", type=str, default="0", help="GPU device id.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    with open(args.data, "r") as f:
        data_cfg = yaml.safe_load(f)

    with open(args.cfg, "r") as f:
        train_cfg = yaml.safe_load(f)

    if args.model.endswith(".pt"):
        model_cfg = args.model
    else:
        with open(args.model, "r") as f:
            model_cfg = yaml.safe_load(f)

    # Load models
    if isinstance(model_cfg, dict):
        model = YOLOModel(model_cfg, verbose=True)
    else:
        ckpt = torch.load(model_cfg)
        if isinstance(ckpt, nn.Module):
            model = ckpt.float()
        elif "ema" in ckpt.keys() and ckpt["ema"] is not None:
            model = ckpt["ema"].float()
        else:
            model = ckpt["model"].float()
    stride_size = int(max(model.stride))  # type: ignore

    # Load teacher model
    teacher: Optional[nn.Module] = None
    if not args.teacher:
        teacher = None
    elif args.teacher.endswith(".pt"):
        teacher = load_pytorch_model(args.weights, args.teacher_cfg)
    else:  # load model from wandb
        teacher = load_model_from_wandb(args.teacher)
    if teacher is None:
        LOGGER.error(
            f"Load model from {args.teacher} with config {args.teacher_cfg if args.teacher_cfg != '' else 'None'} has failed."
        )
        exit(1)
    if not stride_size == int(max(teacher.stride)):
        LOGGER.error("Teacher and Student have different strides.")
        exit(1)
    # Freeze teacher weights
    for _, v in teacher.model.named_parameters():
        v.requires_grad = False

    # Create Dataloaders
    train_loader, train_dataset = create_dataloader(
        data_cfg["train_path"], train_cfg, stride_size, prefix="[Train] "
    )
    # TODO(hsshin): revisit this part
    unlabeled_loader, unlabeled_dataset = create_dataloader(
        data_cfg["train_path"], train_cfg, stride_size, prefix="[Unlab] "
    )
    val_loader, val_dataset = create_dataloader(
        data_cfg["val_path"],
        train_cfg,
        stride_size,
        prefix="[Val] ",
        validation=True,
        pad=0.5,
    )

    device = select_device(args.device)
    wdir = Path(os.path.join("exp", "weights"))

    # student
    model_manager = YOLOModelManager(model, train_cfg, device, wdir)
    model = model_manager.load_model_weights()
    model = model_manager.freeze(train_cfg["train"]["freeze"])

    model_manager.model = model
    model = model_manager.set_model_params(train_dataset)

    # # teacher
    # # TODO(hsshin) revisit here
    # model_manager = YOLOModelManager(teacher, teacher_cfg, device, wdir)
    # teacher = model_manager.load_model_weights()
    # # freeze all params in teacher model
    # teacher = model_manager.freeze(-1)

    model_manager.model = teacher
    teacher = model_manager.set_model_params(train_dataset)

    trainer = SoftTeacherTrainer(
        model, teacher, train_cfg, train_loader, unlabeled_loader, val_loader, device,
    )
    trainer.train()
