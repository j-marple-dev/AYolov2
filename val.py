"""Validation for YOLO.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import argparse
import importlib
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import yaml
from kindle import YOLOModel
from torch import nn

from scripts.data_loader.data_loader import LoadImagesAndLabels
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.torch_utils import load_pytorch_model, select_device
from scripts.utils.train_utils import YoloValidator
from scripts.utils.wandb_utils import load_model_from_wandb

if importlib.util.find_spec("tensorrt") is not None:
    from scripts.utils.tensorrt_runner import TrtWrapper

LOGGER = get_logger(__name__)


def load_trt_model(
    weight_path: str, device: torch.device
) -> Tuple[Optional["TrtWrapper"], Optional[dict]]:
    """Load tensorRT model.

    Args:
        weight_path: TensorRT engine file path with .trt extension.
            i.e. if weight_path='path/trt/model.trt',
                TensorRT config file also should exist at 'path'/trt/model_trt.yaml
        device: PyTorch device to run TensorRT model.

    Return:
        TrtWrapper which has same interface with PyTorch model, and
        TensorRT config yaml.

        None, None if loading TensorRT model has failed.
    """
    if importlib.util.find_spec("tensorrt") is None:
        LOGGER.error("TensorRT can not be found.")
        return None, None

    cfg_path = str(Path(weight_path).parent / Path(weight_path).stem) + "_trt.yaml"
    LOGGER.info(f"Reading TensorRT config from {cfg_path}")
    with open(cfg_path, "r") as f:
        trt_cfg = yaml.safe_load(f)
    model = TrtWrapper(weight_path, trt_cfg["batch_size"], device, torch_input=True)

    return model, trt_cfg


def load_torchscript_model(weight_path: str) -> Tuple[torch.jit.ScriptModule, dict]:
    """Load TorchScript model.

    Args:
        weight_path: TorchScript file path with .ts extension.
            i.e. if weight_path='path/ts/model.ts',
                TorchScript config file also should exist at 'path'/ts/model_ts.yaml
    Return:
        TorchScript model and,
        TorchScript config yaml
    """
    model = torch.jit.load(weight_path)

    cfg_path = str(Path(weight_path).parent / Path(weight_path).stem) + "_ts.yaml"
    LOGGER.info(f"Reading TorchScript config from {cfg_path}")

    with open(cfg_path, "r") as f:
        ts_cfg = yaml.safe_load(f)
    ts_cfg["half"] = ts_cfg["dtype"] == "fp16"

    return model, ts_cfg


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
        default="exp",
        help="Export directory. Directory will be {dst}/val/{DATE}_runs1, ...",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("-iw", "--img-width", type=int, default=640, help="Image width")
    parser.add_argument(
        "-ih",
        "--img-height",
        type=int,
        default=-1,
        help="Image height. (-1 will set image height to be identical to image width.)",
    )
    parser.add_argument(
        "-ct", "--conf-t", type=float, default=0.001, help="Confidence threshold."
    )
    parser.add_argument(
        "-it", "--iou-t", type=float, default=0.65, help="IoU threshold."
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
        "--profile",
        action="store_true",
        default=False,
        help="Run profiling before validation.",
    )
    parser.add_argument(
        "--n-profile",
        type=int,
        default=100,
        help="Number of n iteration for profiling.",
    )
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
    parser.add_argument(
        "--no_weight_wandb",
        action="store_true",
        default=False,
        help="load weights from wandb run path",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    if args.img_height < 0:
        args.img_height = args.img_width

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
    trt_cfg: Optional[dict] = None
    ts_cfg: Optional[dict] = None

    if args.weights == "":
        LOGGER.warning(
            "Providing "
            + colorstr("bold", "no weights path")
            + " will validate a randomly initialized model. Please use only for a experiment purpose."
        )
    elif args.weights.endswith(".pt"):
        model = load_pytorch_model(args.weights, args.model_cfg, load_ema=True)
        stride_size = int(max(model.stride))  # type: ignore
    elif args.weights.endswith(".trt"):
        model, trt_cfg = load_trt_model(args.weights, device)
        if trt_cfg:
            for k in [
                "batch_size",
                "conf_t",
                "iou_t",
                "img_width",
                "img_height",
                "rect",
            ]:
                LOGGER.info(f"Overriding {k} from TensorRT config value {trt_cfg[k]}.")
                args.__setattr__(k, trt_cfg[k])

            stride_size = trt_cfg["stride_size"] if "stride" in trt_cfg.keys() else 32
    elif args.weights.endswith(".ts"):
        model, ts_cfg = load_torchscript_model(args.weights)
        stride_size = ts_cfg["stride_size"]
        if ts_cfg:
            for k in [
                "batch_size",
                "conf_t",
                "iou_t",
                "img_width",
                "img_height",
                "rect",
                "half",
            ]:
                LOGGER.info(
                    f"Overriding {k} from TorchScript config value {ts_cfg[k]}."
                )
                args.__setattr__(k, ts_cfg[k])
    else:  # load model from wandb
        model = load_model_from_wandb(
            args.weights, load_weights=not args.no_weight_wandb
        )
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
            "image_size": args.img_width,
        },
        "hyper_params": {"conf_t": args.conf_t, "iou_t": args.iou_t},
    }

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

    if isinstance(model, torch.jit.ScriptModule):
        model.to(device).eval()
    elif isinstance(model, nn.Module):
        model.to(device).fuse().eval()  # type: ignore
        if args.half:
            model.half()

    # TODO(jeikeilim): Implement TensorRT profiling.
    if args.profile and isinstance(model, YOLOModel):
        model.profile(
            input_size=(args.img_width, args.img_height),
            batch_size=args.batch_size,
            n_run=args.n_profile,
        )

    validator = YoloValidator(
        model,
        val_loader,
        device,
        cfg,
        compute_loss=isinstance(model, YOLOModel) and hasattr(model, "hyp"),
        hybrid_label=args.hybrid_label,
        half=args.half,
        log_dir=args.dst,
        incremental_log_dir=True,
        export=True,
    )
    t0 = time.monotonic()
    val_result = validator.validation()
    time_took = time.monotonic() - t0

    LOGGER.info(f"Time took: {time_took:.5f}s")

    with open(os.path.join(validator.log_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    if trt_cfg:
        with open(os.path.join(validator.log_dir, "trt_cfg.yaml"), "w") as f:
            yaml.dump(trt_cfg, f)

    if ts_cfg:
        with open(os.path.join(validator.log_dir, "ts_cfg.yaml"), "w") as f:
            yaml.dump(ts_cfg, f)
