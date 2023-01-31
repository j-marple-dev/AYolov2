"""Inference code for AYolov2.

- Author: Haneol Kim
- Contact: hekim@jmarple.ai
"""

import argparse
from typing import Optional, Union

import cv2
import torch
import torch.nn as nn
import yaml

from scripts.utils.logger import colorstr, get_logger
from scripts.utils.nms import batched_nms
# from scripts.utils.draw_utils import draw_result
from scripts.utils.plot_utils import draw_result
from scripts.utils.torch_utils import (count_param, load_pytorch_model,
                                       select_device)
from scripts.utils.tta_utils import inference_with_tta
from scripts.utils.wandb_utils import load_model_from_wandb

LOGGER = get_logger(__name__)
torch.set_grad_enabled(False)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--weights", type=str, default="", help="Model weight path.")
    parser.add_argument(
        "--model_cfg", type=str, default="", help="Model config file path."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="res/test_video.mp4",
        help="Inference video root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
    )
    parser.add_argument("-iw", "--img-width", type=int, default=640, help="Image width")
    parser.add_argument(
        "-ih",
        "--img-height",
        type=int,
        default=-1,
        help="Image height. (-1 will set image height to be identical to image width.)",
    )
    parser.add_argument(
        "-ct", "--conf-t", type=float, default=0.5, help="Confidence threshold."
    )
    parser.add_argument(
        "-it", "--iou-t", type=float, default=0.65, help="IoU threshold."
    )
    parser.add_argument(
        "--nms-box",
        type=int,
        default=1000,
        help="Number of boxes to use before check confidecne threshold.",
    )
    parser.add_argument(
        "--agnostic",
        action="store_true",
        default=True,
        help="Separate bboxes by classes for NMS with class separation.",
    )
    parser.add_argument(
        "--tta", action="store_true", default=False, help="Use tta on inference."
    )
    parser.add_argument("--tta_cfg", type=str, default="", help="TTA config file path.")
    parser.add_argument(
        "--nms_type",
        type=str,
        default="nms",
        help="NMS type (e.g. nms, batched_nms, fast_nms, matrix_nms, merge_nms)",
    )
    parser.add_argument(
        "--save-dir",
        "-o",
        type=str,
        default="",
        help="Directory which the inference result video will be saved.",
    )
    parser.add_argument(
        "--dataset-type",
        "-dtype",
        type=str,
        default="COCO",
        help="Dataset type. COCO, VOC, etc.",
    )

    opt = parser.parse_args()
    return opt


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

    device = select_device(
        args.device, 1
    )  # batch size 1 for video or camera inference.

    model: Optional[Union[nn.Module]] = None

    if args.tta:
        with open(args.tta_cfg, "r") as f:
            tta_cfg = yaml.safe_load(f)

    if args.weights == "":
        LOGGER.warning(
            "Providing "
            + colorstr("bold", "no weights path")
            + " will inference with a randomly initialized model. Please use only for a experiment purpose."
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

    cap = cv2.VideoCapture(args.data)  # type:ignore
    model.to(device).fuse().eval()  # type: ignore
    LOGGER.info(f"# of parameters: {count_param(model):,d}")
    writer = None

    if args.save_dir:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        save_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(args.save_dir, fourcc, fps, (save_width, save_height))

    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break
        ratio_x, ratio_y = (
            image.shape[1] / args.img_width,
            image.shape[0] / args.img_height,
        )
        resize_ratio = ratio_x if ratio_x >= ratio_y else ratio_y
        np_img = cv2.resize(
            image,
            [int(image.shape[1] / resize_ratio), int(image.shape[0] / resize_ratio)],
        )
        height_border = int((args.img_height - np_img.shape[0]) / 2)
        np_img = cv2.copyMakeBorder(
            np_img,
            height_border,
            height_border,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[114, 114, 114],
        )
        img = torch.from_numpy(np_img).float().to(device, non_blocking=True) / 255.0
        img = torch.permute(img, [2, 0, 1])
        img = torch.unsqueeze(img, dim=0)

        if args.tta:
            out = inference_with_tta(
                model,
                img,
                tta_cfg["scales"],
                tta_cfg["flips"],
            )
        else:
            out = model(img)[0]

        outputs = batched_nms(
            out,
            conf_thres=args.conf_t,
            iou_thres=args.iou_t,
            nms_box=args.nms_box,
            agnostic=args.agnostic,
            nms_type=args.nms_type,
        )[0]

        # for outs in outputs:
        #     if outs[4] > args.conf_t:
        #         cv2.rectangle(np_img, [int(outs[0]), int(outs[1])], [int(outs[2]), int(outs[3])], [255, 0, 0])
        draw_img = draw_result(
            np_img,
            outputs,
            args.conf_t,
            dataset=args.dataset_type,
            x_border=0,
            y_border=height_border,
            orig_img=image,
        )
        if args.save_dir and writer:
            writer.write(draw_img)

        cv2.imshow("test", draw_img)

        if cv2.waitKey(0) == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
