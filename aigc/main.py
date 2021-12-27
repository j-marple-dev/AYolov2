"""Validation for YOLO.

- Author: Jongkuk Lim, Haneol Kim
- Contact: limjk@jmarple.ai, hekim@jmarple.ai
"""
import time

t0 = time.monotonic()

import torch  # noqa: E402

torch.cuda.init()
import threading  # noqa: E402

# Pre-allocate CUDA memory.
threading.Thread(
    target=lambda _: torch.zeros((1,), device=torch.device("cuda:0")), args=(0,)
).start()

if True:  # noqa: E402
    import argparse
    import os
    from functools import partial
    from typing import Any, Dict, Iterator, Optional, Union

    import cv2
    import numpy as np
    import yaml
    from lib.aigc_loader import LoadImages
    from lib.answer_queue import ResultWriterTorch
    from lib.general_utils import TimeChecker, count_param
    from lib.nms_utils import batched_nms
    from lib.tta import inference_with_tta
    from torch import nn
    from tqdm import tqdm

# Settings START
torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(
    linewidth=320, formatter={"float_kind": "{:11.5g}".format}
)  # format short g, %precision=5
# pd.options.display.max_columns = 10
cv2.setNumThreads(
    0
)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(min(os.cpu_count(), 8))  # NumExpr max threads

torch.set_grad_enabled(False)
# Settings END


class ModelLoader(threading.Thread):
    """Parallel model loader with threading."""

    def __init__(self, cfg: Dict[str, Any], device: torch.device) -> None:
        """Initialize ModelLoader for parallel model load.

        Args:
            args: Namespace from __main__
            device: torch device to run model.
        """
        super().__init__()
        self.cfg = cfg
        self.device = device

        """self.model will be loaded once self.start() has been finished."""
        self.model: Optional[nn.Module] = None
        """Default stride_size is 32 but this might change by the model."""
        self.stride_size = cfg.get("stride_size", 32)
        self.n_param = 0

    def run(self) -> None:
        """Run model load thread.

        Loaded model can be accessed after self.join()
        """
        self.model = torch.load(self.cfg.get("weights", "weights/model.pt"))

        if self.model is not None:
            self.model.to(self.device).eval()  # type: ignore

            if self.cfg.get("half", False):
                self.model.half()
            else:
                self.model.float()

            self.stride_size = int(max(self.model.stride))  # type: ignore
            self.n_param = count_param(self.model)

            print(f"# of parameters: {self.n_param:,d}")


class DataLoaderGenerator(threading.Thread):
    """Parallel dataset and dataloader generator with threading.."""

    def __init__(self, cfg: Dict[str, Any], device: torch.device) -> None:
        """Initialize DataLoaderGenerator for parallel dataloader initialization.

        Args:
            args: Namespace from __main__
            stride_size: max stride size of the model to decide image sizes to load.
        """
        super().__init__()
        self.data_cfg = cfg["data"]
        self.model_cfg = cfg["model"]
        self.infer_cfg = cfg["inference"]

        """LoadImages dataset."""
        self.dataset: Optional[LoadImages] = None
        """PyTorch dataloader."""
        self.dataloader: Optional[torch.utils.data.DataLoader] = None
        """Dataloader iterator to prefetch before actually running the model."""
        self.iterator: Optional[Iterator] = None
        self.device = device

    def run(self) -> None:
        """Initialize dataset and dataloader with threading."""
        self.dataset = LoadImages(
            self.data_cfg.get("path", os.path.join("/", "home", "agc2021", "dataset")),
            img_size=self.data_cfg.get("img_size", 640),
            batch_size=self.infer_cfg.get("batch_size", 8),
            rect=self.data_cfg.get("rect", True),
            stride=self.model_cfg.get("stride_size", 32),
            pad=self.data_cfg.get("pad", 0.5),
            prefix="[val]",
            preprocess=lambda x: (x / 255.0).astype(
                np.float16 if self.model_cfg.get("half", False) else np.float32
            ),
            use_mp=self.data_cfg.get("use_mp", False),
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.infer_cfg.get("batch_size", 8),
            num_workers=min(os.cpu_count(), self.infer_cfg.get("batch_size", 8)),  # type: ignore
            pin_memory=True,
            collate_fn=LoadImages.collate_fn,
            prefetch_factor=5,
        )
        self.iterator = tqdm(self.dataloader, "Inference ...")


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=os.path.join("configs", "submit_config.yaml"),
        help="Submission model config file path.",
    )
    parser.add_argument(
        "-ct",
        "--check-time",
        action="store_true",
        default=False,
        help="Check time consumption.",
    )
    parser.add_argument(
        "-cm",
        "--check-map",
        action="store_true",
        default=False,
        help="Check mAP after inference.",
    )
    parser.add_argument(
        "-gm",
        "--gpu-mem",
        default=6.0,
        type=float,
        help="GPU Memory restriction in GiB.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    ANSWER_PATH = "answersheet_4_04_jmarple.json"

    args = get_parser()

    init_time = time.monotonic() - t0

    time_checker = TimeChecker(
        "aigc", ignore_thr=0.0, cuda_sync=True, enable=args.check_time
    )

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0")

    # 6 GiB GPU memory limit.
    memory_fraction = (args.gpu_mem * 1024 ** 3) / torch.cuda.get_device_properties(
        device
    ).total_memory
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device)

    time_checker.add("init")

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

    result_writer = ResultWriterTorch(ANSWER_PATH, model_loader.n_param)
    result_writer.start()

    conf_thres = cfg["inference"].get("conf_t", 0.001)
    iou_thres = cfg["inference"].get("iou_t", 0.65)
    nms_box = cfg["inference"].get("nms_box", 1000)
    agnostic = cfg["inference"].get("agnostic", True)
    tta_cfg = cfg["tta"]

    time_checker.add("Prepare model")

    # Choose normal inference or TTA inference
    if cfg["inference"].get("tta", False):
        run_model: Union[partial, nn.Module] = partial(
            inference_with_tta, model, s=tta_cfg["scales"], f=tta_cfg["flips"]
        )
    else:
        run_model = model

    for img, path, shape in iterator:
        out = run_model(img.to(device, non_blocking=True))[0]

        # TODO(jeikeilim): Find better and faster NMS method.
        outputs = batched_nms(
            out,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nms_box=nms_box,
            agnostic=agnostic,
        )

        result_writer.add_outputs(path, outputs, img.shape[2:4], shapes=shape)

    time_checker.add("Inference")

    result_writer.close()
    time_checker.add("End")

    if args.check_time:
        print(f"Init time: {init_time:.3f}s")
        print(str(time_checker))

    if args.check_map:
        import importlib

        if importlib.util.find_spec("pycocotools") is None:
            print("pycocotools can not be found. Exit.")
            exit(0)

        gt_path = os.path.join("res", "instances_val2017.json")
        if not os.path.isfile(gt_path):
            print(f"Ground truth file does not exist on {gt_path}. Exit.")
            exit(0)

        import json

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        with open(ANSWER_PATH, "r") as f:
            pred = json.load(f)

        gt_anno = COCO(gt_path)
        pred_anno = gt_anno.loadRes(pred[2:])

        coco_eval = COCOeval(gt_anno, pred_anno, "bbox")

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
