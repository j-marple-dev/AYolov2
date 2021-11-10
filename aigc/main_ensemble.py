"""Weighted Ensemble inference main script.

- Author: Hyung-Seok Shin
- Contact: hsshin@jmarple.ai
"""
import argparse
import json
import os
from typing import Any, Dict, List

import torch
import yaml
from lib.answer_queue import ResultWriterTorch
from lib.nms_utils import batched_nms
from main import DataLoaderGenerator, ModelLoader


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--configs", nargs="+", default=[])
    return parser.parse_args()


def write_result(
    cfg: Dict[str, Any],
    filename: str,
    device: torch.device,
    write_orig_shape: bool = False,
) -> None:
    """Write result json file.

    This function is a minor modification of main logic in the `main.py`.
    """
    # Parallel load model and dataloader
    model_loader = ModelLoader(cfg["model"], device)
    dataloader_generator = DataLoaderGenerator(cfg, device)

    model_loader.start()
    dataloader_generator.start()

    model_loader.join()
    dataloader_generator.join()

    model = model_loader.model
    iterator = dataloader_generator.iterator

    assert (
        iterator is not None and model is not None
    ), "Either dataloader or model has not been initialized!"

    result_writer = ResultWriterTorch(filename, model_loader.n_param)
    result_writer.start()

    conf_thres = cfg["inference"].get("conf_t", 0.001)
    iou_thres = cfg["inference"].get("iou_t", 0.65)
    nms_box = cfg["inference"].get("nms_box", 500)
    agnostic = cfg["inference"].get("agnostic", False)

    # time_checker.add("Prepare model")
    orig_shapes: Dict[int, List[int, int]] = {}
    for img, path, shape in iterator:
        out = model(img.to(device, non_blocking=True))[0]

        # TODO(jeikeilim): Find better and faster NMS method.
        outputs = batched_nms(
            out,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nms_box=nms_box,
            agnostic=agnostic,
        )

        result_writer.add_outputs(path, outputs, img.shape[2:4], shapes=shape)
        if write_orig_shape:
            for p, sh in zip(path, shape):
                p = p.rsplit(os.path.sep, 1)[-1]
                p = p.rsplit(".", 1)[0]
                if p[:3] == "t4_":
                    p = p[3:]  # AIGC "t4_"
                orig_shapes[int(p)] = sh[0]

    # Write original shapes
    if write_orig_shape:
        with open("original_shapes.json", "w") as f:
            json.dump(orig_shapes, f)


if __name__ == "__main__":
    args = get_args()

    for cfg in args.configs:
        assert os.path.exists(cfg), f"Config `{cfg}` does not exist"

    device = torch.device("cuda:0")

    write_orig_shape = True
    for idx, cfg in enumerate(args.configs):
        # Load Dataloader
        with open(cfg, "r") as f:
            config = yaml.safe_load(f)
        write_result(config, f"{idx}.json", device, write_orig_shape)
        write_orig_shape = False
