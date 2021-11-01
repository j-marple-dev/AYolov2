"""Tensor decomposition YOLO model.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import argparse
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from kindle import YOLOModel
from torch import nn

from scripts.data_loader.data_loader import LoadImagesAndLabels
from scripts.tensor_decomposition.decomposition import decompose_model
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.torch_utils import count_param, select_device
from scripts.utils.train_utils import YoloValidator
from scripts.utils.wandb_utils import get_ckpt_path

LOGGER = get_logger(__name__)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--weights", type=str, default="", help="Model weight path.")
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
    parser.add_argument(
        "--dst",
        type=str,
        default=os.path.join("exp", "decompose"),
        help="Export directory. Directory will be {dst}/decompose/{DATE}_runs1, ...",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size to test")
    parser.add_argument("-iw", "--img-width", type=int, default=640, help="Image width")
    parser.add_argument(
        "-ih",
        "--img-height",
        type=int,
        default=-1,
        help="Image height. (-1 will set image height to be identical to image width.)",
    )
    parser.add_argument(
        "--prune-step",
        default=0.01,
        type=float,
        help="Prunning trial max step. Maximum step while searching prunning ratio with binary search. Pruning will be applied priro to decomposition. If prune-step is equal or smaller than 0.0, prunning will not be applied.",
    )
    parser.add_argument(
        "--loss-thr",
        default=0.1,
        type=float,
        help="Loss value to compare original model output and decomposed model output to judge to switch to decomposed conv.",
    )
    parser.add_argument(
        "-ct", "--conf-t", type=float, default=0.001, help="Confidence threshold."
    )
    parser.add_argument(
        "-it", "--iou-t", type=float, default=0.65, help="IoU threshold."
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
    return parser.parse_args()


def run_decompose(
    args: argparse.Namespace,
    model: nn.Module,
    validator: YoloValidator,
    device: torch.device,
) -> Tuple[nn.Module, Tuple[Tuple[list, ...], np.ndarray, tuple]]:
    """Run tensor decomposition on given model.

    Args:
        args: arguments for the tensor decomposition.
            args.prune_step(float): prune step.
            args.loss_thr(float): Loss threshold for decomposition.
        model: Original model.
        validator: validation runner.
        device: device to run validation.

    Return:
        decomposed_model,
        (
            (mP, mR, mAP0.5, mAP0.5:0.95, 0, 0, 0),
            mAP0.5 by classes,
            time measured (pre-processing, inference, NMS)
        )
    """
    decomposed_model = deepcopy(model.cpu())
    decompose_model(
        decomposed_model, loss_thr=args.loss_thr, prune_step=args.prune_step
    )

    LOGGER.info(
        f"Decomposed with prunning step: {args.prune_step}, decomposition loss threshold: {args.loss_thr}"
    )
    LOGGER.info(f"  Original # of param: {count_param(model)}")
    LOGGER.info(f"Decomposed # of param: {count_param(decomposed_model)}")

    decomposed_model.to(device)
    decomposed_model.eval()

    validator.model = decomposed_model
    t0 = time.monotonic()
    val_result = validator.validation()
    time_took = time.monotonic() - t0

    LOGGER.info(f"Time took: {time_took:.5f}s")

    return decomposed_model, val_result


if __name__ == "__main__":
    args = get_parser()

    if args.img_height < 0:
        args.img_height = args.img_width

    if not args.weights:
        LOGGER.error(
            "Either "
            + colorstr("bold", "--weights")
            + " must be provided. (Current value: "
            + colorstr("bold", f"{args.weights}")
            + ")"
        )
        exit(1)

    device = select_device(args.device, args.batch_size)
    ckpt_path = get_ckpt_path(args.weights)
    ckpt = torch.load(ckpt_path)

    if isinstance(ckpt, YOLOModel):
        model = ckpt.float()
    elif "ema" in ckpt.keys() and ckpt["ema"] is not None:
        model = ckpt["ema"].float()
    else:
        model = ckpt["model"].float()

    with open(args.data_cfg, "r") as f:
        data_cfg = yaml.safe_load(f)

    # TODO(jeikeilim): config structure should be changed.
    cfg = {
        "train": {
            "single_cls": args.single_cls,
            "plot": args.plot,
            "batch_size": args.batch_size,
            "image_size": args.img_width,
        },
        "hyper_params": {"conf_t": args.conf_t, "iou_t": args.iou_t},
    }
    stride_size = int(max(model.stride))  # type: ignore

    val_dataset = LoadImagesAndLabels(
        data_cfg["val_path"],
        img_size=args.img_width,
        batch_size=args.batch_size,
        rect=args.rect,
        label_type="labels",
        cache_images=None,
        single_cls=False,
        stride=stride_size,
        pad=0.5,
        n_skip=0,
        prefix="[val]",
        yolo_augmentation=None,
        augmentation=None,
        preprocess=None,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=min(os.cpu_count(), args.batch_size),  # type: ignore
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )

    validator = YoloValidator(
        model.to(device).eval(),
        val_loader,
        device,
        cfg,
        compute_loss=False,
        half=False,
        log_dir=args.dst,
        incremental_log_dir=True,
        export=True,
    )

    LOGGER.info("Validating original model ...")
    t0 = time.monotonic()
    original_result = validator.validation()
    original_time = time.monotonic() - t0

    decomposed_model, decomposed_result = run_decompose(args, model, validator, device)

    LOGGER.info(
        f"[  Original] # param: {count_param(model):,d}, mAP0.5: {original_result[0][2]}, Speed(pre-process, inference, NMS): {original_result[2][0]:.3f}, {original_result[2][1]:.3f}, {original_result[2][2]:.3f}"
    )
    LOGGER.info(
        f"[Decomposed] # param: {count_param(decomposed_model):,d}, mAP0.5: {decomposed_result[0][2]}, Speed(pre-process, inference, NMS): {decomposed_result[2][0]:.3f}, {decomposed_result[2][1]:.3f}, {decomposed_result[2][2]:.3f}"
    )

    result = vars(args)
    result["result"] = {
        "original": {
            "n_param": count_param(model),
            "metric": {
                "mR": original_result[0][0],
                "mP": original_result[0][1],
                "mAP50": original_result[0][2],
                "mAP": original_result[0][3],
            },
            "mAP50_by_class": original_result[1],
            "time": {
                "preprocessing": original_result[2][0],
                "inference": original_result[2][1],
                "NMS": original_result[2][2],
            },
        },
        "decomposed": {
            "n_param": count_param(decomposed_model),
            "metric": {
                "mR": decomposed_result[0][0],
                "mP": decomposed_result[0][1],
                "mAP50": decomposed_result[0][2],
                "mAP": decomposed_result[0][3],
            },
            "mAP50_by_class": decomposed_result[1],
            "time": {
                "preprocessing": decomposed_result[2][0],
                "inference": decomposed_result[2][1],
                "NMS": decomposed_result[2][2],
            },
        },
    }

    cfg_path = os.path.join(validator.log_dir, "args.yaml")

    with open(cfg_path, "w") as f:
        yaml.dump(result, f)
        LOGGER.info("Decomposition config saved to " + colorstr("bold", cfg_path))

    weight_path = (
        str(Path(validator.log_dir) / Path(args.weights).stem) + "_decomposed.pt"
    )
    torch.save(
        {"model": decomposed_model.cpu().half(), "decomposed": True}, weight_path
    )
    LOGGER.info("Decomposed model saved to " + colorstr("bold", weight_path))
