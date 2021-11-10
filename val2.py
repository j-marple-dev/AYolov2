"""Validation for YOLO.

- Author: Jongkuk Lim, Haneol Kim
- Contact: limjk@jmarple.ai, hekim@jmarple.ai
"""

import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import nn
from tqdm import tqdm

from scripts.data_loader.data_loader import LoadImages
from scripts.utils.general import TimeChecker
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.metrics import COCOmAPEvaluator
from scripts.utils.multi_queue import ResultWriterTorch
from scripts.utils.nms import batched_nms
from scripts.utils.torch_utils import (count_param, load_pytorch_model,
                                       select_device)
from scripts.utils.tta_utils import inference_with_tta
from scripts.utils.wandb_utils import load_model_from_wandb

torch.set_grad_enabled(False)
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
        "--data",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "coco", "images", "val2017"),
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
        "--nms-box",
        type=int,
        default=500,
        help="Number of boxes to use before check confidecne threshold.",
    )
    parser.add_argument(
        "--agnostic",
        action="store_true",
        default=False,
        help="Separate bboxes by classes for NMS with class separation.",
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
        "--check-map",
        action="store_true",
        default=True,
        help="Check mAP after inference.",
    )
    parser.add_argument(
        "--export",
        type=str,
        default="",
        help="Export all inference results if path is given.",
    )
    parser.add_argument(
        "--nms_type",
        type=str,
        default="nms",
        help="NMS type (e.g. nms, batched_nms, fast_nms, matrix_nms, merge_nms)",
    )
    parser.add_argument(
        "--no_coco",
        action="store_true",
        default=False,
        help="Validate with pycocotools.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Apply TTA (Test Time Augmentation)",
    )
    parser.add_argument(
        "--tta-cfg",
        type=str,
        default="res/configs/cfg/tta.yaml",
        help="TTA config file path",
    )

    return parser.parse_args()


if __name__ == "__main__":
    time_checker = TimeChecker("val2", ignore_thr=0.0)
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

    time_checker.add("args")
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

    with open(args.tta_cfg, "r") as f:
        tta_cfg = yaml.safe_load(f)

    time_checker.add("load_model")

    val_dataset = LoadImages(
        args.data,
        img_size=args.img_width,
        batch_size=args.batch_size,
        rect=args.rect,
        cache_images=None,
        stride=stride_size,
        pad=0.5,
        n_skip=0,
        prefix="[val]",
        augmentation=None,
        preprocess=lambda x: (x / 255.0).astype(
            np.float16 if args.half else np.float32
        ),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=min(os.cpu_count(), args.batch_size),  # type: ignore
        pin_memory=True,
        collate_fn=LoadImages.collate_fn,
        prefetch_factor=5,
    )

    time_checker.add("create_dataloader")

    model.to(device).fuse().eval()  # type: ignore
    LOGGER.info(f"# of parameters: {count_param(model):,d}")
    if args.half:
        model.half()

    time_checker.add("Fuse model")

    result_writer = ResultWriterTorch("answersheet_4_04_000000.json")
    result_writer.start()

    time_checker.add("Prepare run")
    for _, (img, path, shape) in tqdm(
        enumerate(val_loader), "Inference ...", total=len(val_loader)
    ):
        if args.tta:
            out = inference_with_tta(
                model,
                img.to(device, non_blocking=True),
                tta_cfg["scales"],
                tta_cfg["flips"],
            )[0]
        else:
            out = model(img.to(device, non_blocking=True))[0]

        # TODO(jeikeilim): Find better and faster NMS method.
        outputs = batched_nms(
            out,
            conf_thres=args.conf_t,
            iou_thres=args.iou_t,
            nms_box=args.nms_box,
            agnostic=args.agnostic,
            nms_type=args.nms_type,
        )
        result_writer.add_outputs(path, outputs, img.shape[2:4], shapes=shape)
    time_checker.add("Inference")

    result_writer.close()
    time_checker.add("End")

    LOGGER.info(str(time_checker))

    # Check mAP
    if args.check_map:
        gt_path = os.path.join("tests", "res", "instances_val2017.json")
        json_path = "answersheet_4_04_000000.json"

        is_export = args.export != ""

        coco_eval = COCOmAPEvaluator(
            gt_path,
            img_root=args.data if is_export else None,
            export_root=args.export if is_export else None,
        )
        result = coco_eval.evaluate(json_path, debug=is_export)
        LOGGER.info(f"mAP50: {result['map50']}, mAP50:95: {result['map50_95']}")

    if not args.no_coco:
        anno = COCO(gt_path)
        pred = anno.loadRes(json_path)
        cocotools_eval = COCOeval(anno, pred, "bbox")

        cocotools_eval.evaluate()
        cocotools_eval.accumulate()
        cocotools_eval.summarize()
        # if need values
        # use cocotools_eval.stats
