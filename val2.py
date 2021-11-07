"""Validation for YOLO.

- Author: Jongkuk Lim, Haneol Kim
- Contact: limjk@jmarple.ai, hekim@jmarple.ai
"""

import argparse
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
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
from scripts.utils.wandb_utils import load_model_from_wandb

torch.set_grad_enabled(False)
LOGGER = get_logger(__name__)


def export_model_to_handwritten_model(model: nn.Module, path: str = "aigc") -> None:
    """Export model class file for submitting AIGC.

    Args:
        model: target model to export to .py file.
        path: model file path to export.
    """
    contents_header = (
        '"""AIGC2021 submission model.\n\nNote: This is auto-generated .py DO NOT modify.\n"""\n\n'
        "import torch\n"
        "from torch import nn\n\n"
        "framework = 'torch'  # type: ignore\n\n\n"
    )
    contents_class = (
        "class CompressionModel(nn.Module):  # type: ignore\n"
        '    """CompressedModel for AIGC2021."""\n\n'
        "    def __init__(self) -> None:  # type: ignore\n"
        '        """Initialize model."""\n'
        "        super().__init__()\n"
    )
    contents_forward = (
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore\n"
        '        """Run the model.\n\n'
        "        Caution: This method will not work since its purpose\n"
        "        is to compute number of parameters of the model.\n"
        '        """\n'
    )

    model_str = str(model)
    for i, line in enumerate(model_str.split("\n")):
        line_split = line.split(":")
        if len(line_split) > 1:
            module_name = line_split[1][: line_split[1].find("(")].replace(" ", "")
            if module_name not in ("Sequential", "ModuleList") and hasattr(
                nn, module_name
            ):
                module_str = (
                    line_split[1].replace(" ", "").replace("nearest", "'nearest'")
                )
                contents_class += (
                    f"        self.module_{i:03d} = nn.{module_str}  # type: ignore\n"
                )
                contents_forward += (
                    f"        x = self.module_{i:03d}(x)  # type: ignore\n"
                )
                LOGGER.info(line_split[1])

    root = Path(path)
    py_path = root / "answer_model" / "model.py"
    weight_path = root / "weights" / "model.pt"

    with open(py_path, "w") as f:
        f.write(contents_header)
        f.write(contents_class)
        f.write("\n\n")
        f.write(contents_forward)

    model_to_save = deepcopy(model)
    torch.save(model_to_save.cpu().half(), weight_path)


class ModelLoader(threading.Thread):
    """Parallel model loader with threading."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        """Initialize ModelLoader for parallel model load.

        Args:
            args: Namespace from __main__
            device: torch device to run model.
        """
        super().__init__()
        self.args = args
        self.device = device

        """self.model will be loaded once self.start() has been finished."""
        self.model: Optional[nn.Module] = None
        """Default stride_size is 32 but this might change by the model."""
        self.stride_size = 32

    def run(self) -> None:
        """Run model load thread.

        Loaded model can be accessed after self.join()
        """
        if self.args.weights.endswith(".pt"):
            self.model = load_pytorch_model(
                self.args.weights, self.args.model_cfg, load_ema=True
            )
        else:
            self.model = load_model_from_wandb(self.args.weights)

        if self.model is not None:
            self.model.to(self.device).fuse().eval().float()  # type: ignore
            if self.args.half:
                self.model.half()

            self.stride_size = int(max(self.model.stride))  # type: ignore

            LOGGER.info(f"# of parameters: {count_param(self.model):,d}")


class DataLoaderGenerator(threading.Thread):
    """Parallel dataset and dataloader generator with threading.."""

    def __init__(self, args: argparse.Namespace, stride_size: int = 32) -> None:
        """Initialize DataLoaderGenerator for parallel dataloader initialization.

        Args:
            args: Namespace from __main__
            stride_size: max stride size of the model to decide image sizes to load.
        """
        super().__init__()
        self.args = args
        self.stride_size = stride_size

        """LoadImages dataset."""
        self.dataset: Optional[LoadImages] = None
        """PyTorch dataloader."""
        self.dataloader: Optional[torch.utils.data.DataLoader] = None
        """Dataloader iterator to prefetch before actually running the model."""
        self.iterator: Optional[Iterator] = None

    def run(self) -> None:
        """Initialize dataset and dataloader with threading."""
        self.dataset = LoadImages(
            self.args.data,
            img_size=self.args.img_width,
            batch_size=self.args.batch_size,
            rect=self.args.rect,
            cache_images=None,
            stride=self.stride_size,
            pad=0.5,
            n_skip=0,
            prefix="[val]",
            augmentation=None,
            preprocess=lambda x: (x / 255.0).astype(
                np.float16 if self.args.half else np.float32
            ),
            use_mp=False,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            num_workers=min(os.cpu_count(), self.args.batch_size),  # type: ignore
            pin_memory=True,
            collate_fn=LoadImages.collate_fn,
            prefetch_factor=5,
        )
        self.iterator = tqdm(
            enumerate(self.dataloader), "Inference ...", total=len(self.dataloader)
        )


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
        "--no-check-map", action="store_false", dest="check_map", help="Skip mAP check."
    )
    parser.add_argument(
        "--export",
        type=str,
        default="",
        help="Export all inference results if path is given.",
    )
    parser.add_argument(
        "-emp",
        "--export-model-py",
        action="store_true",
        default=False,
        help="Export model.py for to follow AIGC standard.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    time_checker = TimeChecker("val2", ignore_thr=0.0)
    args = get_parser()
    time_checker.add("get_argparse")

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
    time_checker.add("device")

    # Parallel load model and dataloader
    model_loader = ModelLoader(args, device)
    dataloader_generator = DataLoaderGenerator(args, stride_size=32)

    model_loader.start()
    dataloader_generator.start()

    model_loader.join()
    dataloader_generator.join()

    model = model_loader.model
    iterator = dataloader_generator.iterator

    assert (
        iterator is not None and model is not None
    ), "Either dataloader or model has not been initialized!"

    if args.export_model_py:
        export_model_to_handwritten_model(model)

    result_writer = ResultWriterTorch("answersheet_4_04_000000.json")
    result_writer.start()

    time_checker.add("Prepare model")
    for _, (img, path, shape) in iterator:
        out = model(img.to(device, non_blocking=True))[0]
        # TODO(jeikeilim): Find better and faster NMS method.
        outputs = batched_nms(
            out,
            conf_thres=args.conf_t,
            iou_thres=args.iou_t,
            nms_box=args.nms_box,
            agnostic=args.agnostic,
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

        anno = COCO(gt_path)
        pred = anno.loadRes(json_path)
        cocotools_eval = COCOeval(anno, pred, "bbox")

        cocotools_eval.evaluate()
        cocotools_eval.accumulate()
        cocotools_eval.summarize()
        # if need values
        # use cocotools_eval.stats
