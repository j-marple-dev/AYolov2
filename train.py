"""Training YOLO model.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""
import argparse
import os
import pprint

import torch
import wandb
import yaml
from kindle import YOLOModel
from torch import nn

from scripts.data_loader.data_loader_utils import create_dataloader
from scripts.train.train_model_builder import TrainModelBuilder
from scripts.train.yolo_trainer import YoloTrainer
from scripts.utils.logger import colorstr, get_logger
from scripts.utils.model_manager import YOLOModelManager

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOGGER = get_logger(__name__)


def get_parser() -> argparse.Namespace:
    """Get argument parser.

    Modify this function as your porject needs
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("res", "configs", "model", "yolov5s.yaml"),
        help="Model "
        + colorstr("config")
        + " or "
        + colorstr("weight")
        + "  file path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("res", "configs", "data", "coco.yaml"),
        help=colorstr("Dataset config") + " file path",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=os.path.join("res", "configs", "cfg", "train_config.yaml"),
        help=colorstr("Training config") + " file path",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="DDP parameter. " + colorstr("red", "bold", "Do not modify"),
    )
    parser.add_argument(
        "--wlog", action="store_true", default=False, help="Use Wandb logger."
    )
    parser.add_argument(
        "--wlog_name", type=str, default="", help="The run id for Wandb log."
    )
    parser.add_argument("--log_dir", type=str, default="", help="Log root directory.")

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

    if args.log_dir:
        train_cfg["train"]["log_dir"] = args.log_dir
    log_dir = train_cfg["train"]["log_dir"]
    if not log_dir:
        log_dir = "exp"
        train_cfg["train"]["log_dir"] = log_dir

    cfg_all = {
        "data_cfg": data_cfg,
        "train_cfg": train_cfg,
        "model_cfg": model_cfg,
        "args": vars(args),
    }

    LOGGER.info(
        "\n"
        + colorstr("red", "bold", f"{'-'*30} Training Configs START {'-'*30}")
        + "\n"
        + pprint.pformat(cfg_all, indent=4)
        + "\n"
        + colorstr("red", "bold", f"{'-'*30} Training Configs END {'-'*30}")
    )

    # TODO(jeikeilim): Need to implement
    #   loading a  model from saved ckpt['model'].yaml

    # WanDB Logger
    wandb_run = None
    if args.wlog and RANK in [-1, 0]:
        wandb_run = wandb.init(project="AYolov2", name=args.wlog_name)
        assert isinstance(
            wandb_run, wandb.sdk.wandb_run.Run
        ), "Failed initializing WanDB"
        for config_fp in [args.data, args.cfg, args.model]:
            wandb_run.save(
                config_fp, base_path=os.path.dirname(config_fp), policy="now"
            )

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

    train_builder = TrainModelBuilder(model, train_cfg, "exp", full_cfg=cfg_all)
    train_builder.ddp_init()

    stride_size = int(max(model.stride))  # type: ignore

    train_loader, train_dataset = create_dataloader(
        data_cfg["train_path"], train_cfg, stride_size, prefix="[Train] "
    )

    if RANK in [-1, 0]:
        val_loader, val_dataset = create_dataloader(
            data_cfg["val_path"],
            train_cfg,
            stride_size,
            prefix="[Val] ",
            validation=True,
            pad=0.5,
        )
    else:
        val_loader, val_dataset = None, None

    model_manager = YOLOModelManager(
        model, train_cfg, train_builder.device, train_builder.wdir
    )
    model = model_manager.load_model_weights()
    model = model_manager.freeze(train_cfg["train"]["freeze"])

    model, ema, device = train_builder.prepare()
    model_manager.model = model
    model = model_manager.set_model_params(train_dataset, ema=ema)

    trainer = YoloTrainer(
        model,
        train_cfg,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        ema=ema,
        device=train_builder.device,
        log_dir=train_builder.log_dir,
        wandb_run=wandb_run,
    )

    trainer.train(start_epoch=model_manager.start_epoch)
