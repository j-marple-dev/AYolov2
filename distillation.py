"""Train object detection model with Knowledge-Distillation.

- Author: Hyung-Seok Shin
- Contact: hsshin@jmarple.ai
"""
import argparse
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from kindle import YOLOModel

from scripts.data_loader.data_loader_utils import create_dataloader
from scripts.train.kd_trainer import SoftTeacherTrainer
from scripts.utils.model_manager import YOLOModelManager
from scripts.utils.torch_utils import select_device


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
    parser.add_argument(
        "--teacher", type=str, default="", help="Teacher model config file path",
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
    if not args.teacher:
        teacher_cfg = deepcopy(model_cfg)

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

    if isinstance(teacher_cfg, dict):
        teacher = YOLOModel(teacher_cfg, verbose=True)
    else:
        ckpt = torch.load(teacher_cfg)
        if isinstance(ckpt, nn.Module):
            teacher = ckpt.float()
        elif "ema" in ckpt.keys() and ckpt["ema"] is not None:
            teacher = ckpt["ema"].float()
        else:
            teacher = ckpt["model"].float()

    stride_size = int(max(model.stride))  # type: ignore
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

    # teacher
    # TODO(hsshin) revisit here
    model_manager = YOLOModelManager(teacher, train_cfg, device, wdir)
    teacher = model_manager.load_model_weights()
    teacher = model_manager.freeze(train_cfg["train"]["freeze"])

    model_manager.model = teacher
    teacher = model_manager.set_model_params(train_dataset)

    trainer = SoftTeacherTrainer(
        model, teacher, train_cfg, train_loader, unlabeled_loader, val_loader, device,
    )
    trainer.train()
