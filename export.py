"""Export trained model to TorchScript, ONNX, TensorRT.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

import argparse
import os
from typing import Optional

import torch
import yaml
from kindle import YOLOModel
from torch import nn

from scripts.model_converter.model_converter import ModelConverter
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.torch_utils import load_model_weights

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
        "--type",
        type=str,
        default="tensorrt",
        help="Model type to convert. (torchscript, ts, onnx, tensorrt, trt",
    )
    parser.add_argument("--dst", type=str, default="export", help="Export directory")
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
        "--no-rect",
        action="store_false",
        dest="rect",
        default=False,
        help="Use squared image.",
    )
    parser.add_argument(
        "--rect",
        action="store_true",
        dest="rect",
        default=False,
        help="Use rectangular image",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        help="Data type to convert. (fp16 or int8) (int8: TensorRT only.)",
    )
    parser.add_argument(
        "--opset", type=int, default=11, help="opset version. (ONNX and TensorRT only)"
    )
    parser.add_argument(
        "--gpu-mem",
        type=int,
        default=6,
        help="Target GPU memory restriction (GiB) (TensorRT only)",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    if args.img_height < 0:
        args.img_height = args.img_width

    if args.weights == "" and args.model_cfg == "":
        LOGGER.error(
            "Either "
            + colorstr("bold", "--weight")
            + " or "
            + colorstr("bold", "--model-cfg")
            + " must be provided."
        )
        exit(1)

    ckpt_model: Optional[nn.Module] = None
    if args.weights == "":
        LOGGER.warning(
            "Providing "
            + colorstr("bold", "no weights path")
            + " will convert randomly initialized model. Please use only for a experiment purpose."
        )
    else:
        ckpt = torch.load(args.weights)
        if isinstance(ckpt, dict):
            ckpt_model = ckpt["ema"] if "ema" in ckpt.keys() else ckpt["model"]
        else:
            ckpt_model = ckpt

        if ckpt_model:
            ckpt_model = ckpt_model.cpu().float()

    if ckpt_model is None and args.model_cfg == "":
        LOGGER.warning("No weights and no model_cfg has been found.")
        exit(1)

    if args.model_cfg != "" and ckpt_model:
        model = YOLOModel(args.model_cfg, verbose=args.verbose > 0)
        model = load_model_weights(model, {"model": ckpt_model}, exclude=[])
    else:
        model = ckpt_model

    args.stride_size = int(max(model.stride))  # type: ignore
    model = model.eval().export(verbose=args.verbose > 0)
    converter = ModelConverter(
        model, args.batch_size, (args.img_height, args.img_width), verbose=args.verbose
    )
    converter.dry_run()

    model_name = (
        f"model_{args.dtype}_{args.batch_size}_{args.img_width}_{args.img_height}"
    )
    model_ext = ""

    if args.type in ("torchscript", "ts"):
        # TODO(jeikeilim): Add NMS layer
        converter.to_torch_script(
            os.path.join(args.dst, f"{model_name}.ts"), half=args.dtype == "fp16"
        )
        model_ext = "ts"
    elif args.type in ("onnx",):
        converter.to_onnx(
            os.path.join(args.dst, f"{model_name}.onnx"), opset_version=args.opset
        )
        model_ext = "onnx"
    elif args.type in ("tensorrt", "trt"):
        model.model[-1].out_xyxy = True
        converter.to_tensorrt(
            os.path.join(args.dst, f"{model_name}.trt"),
            opset_version=args.opset,
            fp16=args.dtype == "fp16",
            int8=args.dtype == "int8",
            workspace_size_gib=args.gpu_mem,
            conf_thres=args.conf_t,
            iou_thres=args.iou_t,
            top_k=args.top_k,
            keep_top_k=args.keep_top_k,
        )
        model_ext = "trt"
    else:
        LOGGER.warn(
            f"Wrong model type. Please specify model type among ('torchscript', 'ts', 'onnx', 'tensorrt', 'trt'). Given type: {args.type}"
        )

    with open(os.path.join(args.dst, f"{model_name}_{model_ext}.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    LOGGER.info(
        f"Converted model has been saved to {os.path.join(args.dst, model_name)}.{model_ext}"
    )
